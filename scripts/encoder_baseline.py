"""Baseline: fine-tune an off-the-shelf small encoder (default MiniLM-L6, 22M params) on the same CLINC150
split, and save its logits in the same format as loopthink.evaluate, so scripts/cascade.py and
scripts/llm_validate.py run on it unchanged. Answers "why not just fine-tune BERT?".

  pip install transformers
  python scripts/encoder_baseline.py --setup official --out runs/minilm_official
  python scripts/encoder_baseline.py --setup heldout  --out runs/minilm_heldout
  python scripts/cascade.py --runs runs/kill/ft_official runs/minilm_official --unknown_share 0.05
  python scripts/llm_validate.py --run runs/minilm_official --model qwen2.5:7b --hint_topk 5

Other encoders: --model answerdotai/ModernBERT-base (149M), --model BAAI/bge-small-en-v1.5 (33M).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from loopthink.data import load_clinc, make_split  # noqa: E402
from loopthink.train_utils import get_device  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


class EncoderClassifier(nn.Module):
    def __init__(self, name, n_classes):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.enc.config.hidden_size, n_classes)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))   # mean pooling


def batches(tok, texts, bs, max_len, device):
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield t["input_ids"].to(device), t["attention_mask"].to(device)


@torch.no_grad()
def logits_for(model, tok, texts, bs, max_len, device):
    model.eval()
    out = [model(i, m).float().cpu().numpy() for i, m in batches(tok, texts, bs, max_len, device)]
    return np.concatenate(out) if out else np.zeros((0, model.head.out_features), np.float32)


def fpr_at_95(y, score):
    thr = np.percentile(score[y == 1], 5)
    return float((score[y == 0] >= thr).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--setup", default="official", choices=["official", "heldout"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--clinc", default="data", help="data dir (CLINC150) or a .json in the same format, e.g. data/hwu64.json")
    ap.add_argument("--n_heldout", type=int, default=30, help="intents hidden in the heldout setup (HWU64: 16)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--select", default="acc", choices=["acc", "loss"],
                    help="keep the epoch with the best validation accuracy (benchmark runs) or the lowest validation "
                         "loss (steadier on small datasets, where accuracy on a few dozen rows is noisy)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = get_device()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    split = make_split(load_clinc(a.clinc), a.setup, a.n_heldout, 0)
    C = len(split.intents)
    model = EncoderClassifier(a.model, C).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.model} | {n_params/1e6:.1f}M params | setup={a.setup} intents={C} | device={device}")

    tr_x = [t for t, _ in split.train]; tr_y = torch.tensor([y for _, y in split.train])
    va_x = [t for t, _ in split.val]; y_val = np.array([y for _, y in split.val])
    steps = a.epochs * ((len(tr_x) + a.bs - 1) // a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best, best_score, best_state, log = -1.0, None, None, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); perm = np.random.permutation(len(tr_x)); tot = 0.0
        for s in range(0, len(perm), a.bs):
            idx = perm[s:s + a.bs]
            t = tok([tr_x[i] for i in idx], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
            loss = nn.functional.cross_entropy(model(t["input_ids"].to(device), t["attention_mask"].to(device)),
                                               tr_y[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        zv = logits_for(model, tok, va_x, 256, a.max_len, device)
        acc = float((zv.argmax(-1) == y_val).mean())
        zs = zv - zv.max(-1, keepdims=True)
        vloss = float(-(zs - np.log(np.exp(zs).sum(-1, keepdims=True)))[np.arange(len(y_val)), y_val].mean())
        rec = dict(epoch=ep, train_loss=round(tot / len(tr_x), 4), val_acc=round(acc, 4), val_loss=round(vloss, 4),
                   minutes=round((time.time() - t0) / 60, 2))
        print(json.dumps(rec)); log.append(rec)
        score = acc if a.select == "acc" else -vloss
        if best_state is None or score > best_score:
            best, best_score = acc, score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    parts = {"val": va_x, "test": [t for t, _ in split.test], "unk_test": split.unknown_test}
    lg = {k: logits_for(model, tok, v, 256, a.max_len, device) for k, v in parts.items()}
    y_test = np.array([y for _, y in split.test])
    acc = float((lg["test"].argmax(-1) == y_test).mean())

    def scores(x):
        z = x - x.max(-1, keepdims=True); p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        return dict(msp=-p.max(-1), energy=-(np.log(np.exp(z).sum(-1)) + x.max(-1)))
    si, so = scores(lg["test"]), scores(lg["unk_test"])
    y = np.r_[np.zeros(len(y_test)), np.ones(len(split.unknown_test))]
    ood = ({k: dict(auroc=float(roc_auc_score(y, np.r_[si[k], so[k]])), fpr95=fpr_at_95(y, np.r_[si[k], so[k]])) for k in si}
           if len(split.unknown_test) else {})   # custom data may come without unknown requests

    # single-query CPU latency (batch 1), the number that matters for serving
    cpu = model.to("cpu").eval(); q = parts["test"][:200]; ms = []
    with torch.no_grad():
        for text in q:
            t = tok([text], truncation=True, max_length=a.max_len, return_tensors="pt")
            t0 = time.perf_counter(); cpu(t["input_ids"], t["attention_mask"]); ms.append((time.perf_counter() - t0) * 1000)
    R = dict(setup=a.setup, data=a.clinc, n_heldout=a.n_heldout, seed=a.seed, looped=False, model=a.model, n_params=n_params, cap=1, patience=2,
             n_in_test=len(y_test), n_unknown_test=len(split.unknown_test), acc_per_loop=[acc], acc_at_cap=acc,
             best_val_acc=best, ood=ood, cpu_latency_ms_bs1=dict(p50=float(np.median(ms)), p95=float(np.percentile(ms, 95))),
             train_log=log)
    ev = os.path.join(a.out, "eval"); os.makedirs(ev, exist_ok=True)
    json.dump(R, open(os.path.join(ev, "results.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(ev, "trajectories.npz"),
                        **{f"{k}_logits": v[:, None, :] for k, v in lg.items()},
                        **{f"{k}_hcos": np.ones((len(v), 1), np.float32) for k, v in lg.items()},
                        y_val=y_val, y_test=y_test)
    if a.save_model:
        cpu.enc.save_pretrained(os.path.join(a.out, "encoder")); tok.save_pretrained(os.path.join(a.out, "encoder"))
        torch.save(cpu.head.state_dict(), os.path.join(a.out, "head.pt"))
        zv = lg["val"] - lg["val"].max(-1, keepdims=True); pv = np.exp(zv) / np.exp(zv).sum(-1, keepdims=True)
        thr = {str(r): float(np.quantile(-pv.max(-1), 1 - r)) for r in (0.05, 0.10, 0.20)}  # escalate if -msp >= thr
        json.dump([[t, split.intents[y]] for t, y in split.train], open(os.path.join(a.out, "examples.json"), "w"))
        json.dump(dict(base_model=a.model, setup=a.setup, data=a.clinc, n_heldout=a.n_heldout, intents=split.intents, max_len=a.max_len,
                       signal="msp", thresholds_by_val_rate=thr, test_acc=acc),
                  open(os.path.join(a.out, "router.json"), "w"), indent=2)
    md = (f"# {a.model} on {a.setup}\n\n{n_params/1e6:.1f}M params. In-scope test accuracy **{acc:.3f}**.\n\n"
          "| detector | AUROC | FPR@95 |\n|---|---|---|\n" +
          "\n".join(f"| {k} | {v['auroc']:.3f} | {v['fpr95']:.3f} |" for k, v in ood.items()) +
          f"\n\nCPU latency, batch 1: p50 {R['cpu_latency_ms_bs1']['p50']:.1f} ms, p95 {R['cpu_latency_ms_bs1']['p95']:.1f} ms\n")
    open(os.path.join(ev, "summary.md"), "w").write(md); print("\n" + md)


if __name__ == "__main__":
    main()
