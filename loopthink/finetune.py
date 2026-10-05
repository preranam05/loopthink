"""Decision fine-tuning on CLINC150 (softmax CE over all intents, random r, deep supervision).

  # from a pretrained LM
  python -m loopthink.finetune --init /kaggle/working/pretrain/final.pt --tokenizer tok.json \
      --setup heldout --out runs/ft_heldout
  # from scratch (sandbox kill test / baselines without pretraining)
  python -m loopthink.finetune --preset tiny --setup official --out runs/tiny_official
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from .train_utils import get_device
import torch.nn.functional as F

from .data import encode_batch, label_text, load_clinc, load_tokenizer, make_split, train_tokenizer
from .model import PRESETS, ModelConfig, build_model
from .train_utils import make_optimizers, precision_setup, sample_r, seed_all, set_lr


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--init", default="", help="pretrained checkpoint (final.pt). Empty = from scratch")
    p.add_argument("--preset", default="tiny", choices=list(PRESETS), help="used when --init is empty")
    p.add_argument("--tokenizer", default="", help="tokenizer.json (trained on CLINC train text if missing)")
    p.add_argument("--scratch_vocab", type=int, default=4096)
    p.add_argument("--clinc", default="data")
    p.add_argument("--setup", default="official", choices=["official", "heldout"])
    p.add_argument("--n_heldout", type=int, default=30)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--muon", action="store_true")
    p.add_argument("--mean_r", type=float, default=6.0)
    p.add_argument("--max_r", type=int, default=16)
    p.add_argument("--fixed_r", type=int, default=0)
    p.add_argument("--ds", default="final_heavy", choices=["none", "final_heavy", "geometric", "uniform"],
                   help="deep-supervision schedule over intermediate loops")
    p.add_argument("--ds_weight", type=float, default=0.5, help="total weight of intermediate loops (final_heavy)")
    p.add_argument("--ds_lambda", type=float, default=0.3, help="geometric prior rate")
    p.add_argument("--bptt", type=int, default=0, help="0 = full backprop through all loops")
    p.add_argument("--max_len", type=int, default=48)
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args(argv)


def loop_weights(r: int, ds: str, ds_weight: float, lam: float) -> dict[int, float]:
    if ds == "none" or r == 1:
        return {r: 1.0}
    if ds == "final_heavy":
        w = {t: ds_weight / (r - 1) for t in range(1, r)}
        w[r] = 1.0
    elif ds == "uniform":
        w = {t: 1.0 / r for t in range(1, r + 1)}
    elif ds == "geometric":
        w = {t: lam * (1 - lam) ** (t - 1) for t in range(1, r)}
        w[r] = (1 - lam) ** (r - 1)
    s = sum(w.values())
    return {t: v / s for t, v in w.items()}


def label_batch(tok, intents, device):
    return encode_batch(tok, [label_text(i) for i in intents], 16, device)


@torch.no_grad()
def eval_acc(model, tok, rows, device, autocast, loops=(1, 4, 6, 8, 16), bs=256, max_len=48):
    model.eval()
    label_emb = model.intent_keys()
    correct = {r: 0 for r in loops}
    maxr = max(loops) if model.cfg.is_looped else 1
    for i in range(0, len(rows), bs):
        texts, ys = zip(*rows[i: i + bs])
        idx, pm = encode_batch(tok, texts, max_len, device)
        with autocast():
            traj = model.classify_trajectory(idx, pm, maxr, label_emb, score_loops=loops)
        y = torch.tensor(ys, device=device)
        for t, logits, _ in traj:
            correct[t if model.cfg.is_looped else loops[0]] += (logits.argmax(-1) == y).sum().item()
    model.train()
    return {f"acc_r{r}": correct[r] / len(rows) for r in (loops if model.cfg.is_looped else loops[:1])}


def main(argv=None):
    args = get_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = get_device()
    seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    raw = load_clinc(args.clinc)
    split = make_split(raw, args.setup, args.n_heldout, args.split_seed)

    # tokenizer
    tok_path = args.tokenizer or os.path.join(args.out, "tokenizer.json")
    if not os.path.exists(tok_path):
        assert not args.init, "pretrained model needs its own tokenizer (--tokenizer)"
        texts = [t for t, _ in raw["train"]] + [label_text(l) for l in {l for _, l in raw["train"]}]
        train_tokenizer(iter(texts), args.scratch_vocab, tok_path)
    tok = load_tokenizer(tok_path)
    if tok_path != os.path.join(args.out, "tokenizer.json"):
        tok.save(os.path.join(args.out, "tokenizer.json"))

    # model
    if args.init:
        st = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**st["cfg"])
        model = build_model(cfg)
        model.load_state_dict(st["model"])
        lr = args.lr or 3e-4
    else:
        cfg = ModelConfig(vocab_size=tok.get_vocab_size(), **PRESETS[args.preset])
        model = build_model(cfg)
        lr = args.lr or 1e-3
    model.to(device)
    lab = label_batch(tok, split.intents, device)
    model.eval()
    with torch.no_grad():
        model.set_label_table(model.encode_labels(*lab))   # computed once from label text, then trained
    model.train()
    print(f"params {model.num_params()/1e6:.2f}M | setup={args.setup} | intents={len(split.intents)} "
          f"| train={len(split.train)} | unknown_test={len(split.unknown_test)}")

    opts = make_optimizers(model, lr_adam=lr, lr_muon=0.02 if args.muon else 0, wd=0.01, use_muon=args.muon)
    autocast, scaler = precision_setup(args.precision, device)
    steps_per_epoch = math.ceil(len(split.train) / args.batch_size)
    total, warm = steps_per_epoch * args.epochs, min(200, steps_per_epoch)
    step, best, log = 0, -1.0, []
    t0 = time.time()
    rng = random.Random(args.seed)
    for ep in range(args.epochs):
        rows = split.train[:]
        rng.shuffle(rows)
        run_loss = 0.0
        for i in range(0, len(rows), args.batch_size):
            mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / (total - warm))
            set_lr(opts, mult)
            texts, ys = zip(*rows[i: i + args.batch_size])
            idx, pm = encode_batch(tok, texts, args.max_len, device)
            y = torch.tensor(ys, device=device)
            r = args.fixed_r or (sample_r(args.mean_r, hi=args.max_r, rng=rng) if cfg.is_looped else 1)
            w = loop_weights(r, args.ds, args.ds_weight, args.ds_lambda) if cfg.is_looped else {1: 1.0}
            with autocast():
                label_emb = model.intent_keys()
                traj = model.classify_trajectory(idx, pm, r, label_emb, score_loops=list(w),
                                                 grad_last_k=args.bptt or None)
                loss = sum(w[t] * F.cross_entropy(lg.float(), y) for t, lg, _ in traj)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                for o in opts:
                    scaler.unscale_(o)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for o in opts:
                scaler.step(o) if scaler else o.step()
                o.zero_grad(set_to_none=True)
            if scaler:
                scaler.update()
            run_loss += loss.item()
            step += 1
        ev = eval_acc(model, tok, split.val, device, autocast, max_len=args.max_len)
        rec = dict(epoch=ep + 1, train_loss=run_loss / steps_per_epoch, **ev, minutes=(time.time() - t0) / 60)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}))
        log.append(rec)
        score = max(v for k, v in ev.items())
        if score >= best:
            best = score
            save_classifier(model, split, args, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("best val acc", best)


def save_classifier(model, split, args, path):
    torch.save(dict(model=model.state_dict(), cfg=model.cfg.to_dict(), split=split.to_meta(),
                    args=vars(args)), path)


def load_classifier(run_dir: str, device="cpu"):
    """-> (model[eval], tokenizer, meta, intent keys [C,d])."""
    st = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
    model = build_model(ModelConfig(**st["cfg"]))
    model.set_label_table(st["model"]["label_table"])
    model.load_state_dict(st["model"])
    model.to(device).eval()
    tok = load_tokenizer(os.path.join(run_dir, "tokenizer.json"))
    with torch.no_grad():
        label_emb = model.intent_keys()
    return model, tok, st, label_emb


if __name__ == "__main__":
    main()
