"""Part 2 kill test: a small contrastive decision model (CLM-style) with a learned "none of these".

Idea. A classifier has one fixed output per trained intent, so it can never say "none" except through
low confidence, and it cannot score an intent it was not trained on. A contrastive decision model
embeds the query and each candidate ACTION (from its text) and scores the match, so
  (a) candidates can change at run time - new intents need only a text label, and
  (b) "none of these" can be TRAINED: in each batch a random part of the labels is hidden; queries
      whose label was hidden must then choose a learned `none` action. Unknowns are simulated from
      known intents only - no out-of-scope data is used in training.

Backbone is frozen (embeddings are computed once and cached); only two small projection heads, the
`none` vector and a temperature are trained - seconds per run on a laptop CPU.

  python scripts/clm.py                 # CLINC150 + HWU64, far-OOS + held-out intents, 3 seeds
  python scripts/clm.py --backbone Qwen/Qwen3-Embedding-0.6B --pool last     # bigger frozen backbone

Compared, on identical splits and seeds:
  zero-shot      cosine(query, label text) on raw frozen embeddings - no training at all
  linear probe   a classifier on the same frozen embeddings (fixed labels; max-softmax for unknowns)
  CLM no-none    contrastive heads, trained without hidden labels (ablation)
  CLM            contrastive heads + hidden-label training + `none`
  fine-tuned     the Part 1 MiniLM classifier, read from runs/minilm_*  (full fine-tuning, 22.8M trained)

Go / no-go (held-out intents = near-domain unknowns, the weakness Part 1 measured four ways):
  GO   CLM beats the fine-tuned classifier's unknown-detection AUROC on held-out intents on both
       datasets by more than seed noise, with in-scope accuracy within 2 points.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from loopthink.data import load_clinc, make_split  # noqa: E402

DATASETS = {"clinc": ("data", 30, "CLINC150"), "hwu": ("data/hwu64.json", 16, "HWU64")}


def label_text(intent):
    return intent.replace("_", " ")


# ----------------------------------------------------------------------------- frozen embeddings (cached)
def embed_all(texts, backbone, pool, device, bs=128, max_len=64):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone)
    enc = AutoModel.from_pretrained(backbone).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            t = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            h = enc(**t).last_hidden_state
            if pool == "last":
                idx = t["attention_mask"].sum(1) - 1
                v = h[torch.arange(len(idx)), idx]
            else:
                m = t["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(F.normalize(v.float(), dim=-1).cpu())
    return torch.cat(out).numpy()


def get_embeddings(ds, backbone, pool, device, cache_dir):
    path, _, _ = DATASETS[ds]
    raw = load_clinc(os.path.join(ROOT, path))
    texts = sorted({t for part in ("train", "val", "test", "oos_val", "oos_test") for t, _ in raw.get(part, [])} |
                   {label_text(l) for _, l in raw["train"]})
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, f"{ds}_{backbone.replace('/', '_')}_{pool}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if list(z["texts"]) == texts:
            return raw, dict(zip(texts, z["emb"]))
    print(f"embedding {len(texts)} texts for {ds} with frozen {backbone} ...")
    emb = embed_all(texts, backbone, pool, device)
    np.savez_compressed(f, texts=np.array(texts, dtype=object), emb=emb)
    return raw, dict(zip(texts, emb))


# ----------------------------------------------------------------------------- model
class CLM(nn.Module):
    """Residual adapters on both towers, zero-initialised: at step 0 the model IS the frozen backbone
    (zero-shot cosine), so training can only move away from that as far as the data justifies. This keeps
    the ability to score intents never seen in training."""

    def __init__(self, d_in, hidden=512):
        super().__init__()
        def mk():
            m = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_in))
            nn.init.zeros_(m[2].weight); nn.init.zeros_(m[2].bias)
            return m
        self.q, self.a = mk(), mk()
        self.none = nn.Parameter(torch.randn(d_in) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))

    def scores(self, q, a):
        """q: (B, d_in) query embeddings; a: (C, d_in) action embeddings -> (B, C+1); last column = none."""
        zq, za = F.normalize(q + self.q(q), dim=-1), F.normalize(a + self.a(a), dim=-1)
        zn = F.normalize(self.none, dim=-1)[None]
        return self.logit_scale.exp().clamp(max=100) * zq @ torch.cat([za, zn]).T

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_clm(Xq, y, A, seed, hide=True, epochs=40, bs=256, lr=1e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = CLM(Xq.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    Xq, y, A = torch.tensor(Xq), torch.tensor(y), torch.tensor(A)
    C = len(A); steps = epochs * ((len(y) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for s in range(0, len(perm), bs):
            idx = torch.tensor(perm[s:s + bs]); yb = y[idx]
            if hide and rng.random() < 0.7:             # hide 20-60% of the labels in this batch
                keep = torch.tensor(rng.random(C) >= rng.uniform(0.2, 0.6))
                if keep.sum() < 2:
                    keep[:] = True
            else:
                keep = torch.ones(C, dtype=torch.bool)
            sc = m.scores(Xq[idx], A)
            sc = torch.cat([sc[:, :C].masked_fill(~keep[None], -1e4), sc[:, C:]], 1)
            target = torch.where(keep[yb], yb, torch.full_like(yb, C))      # hidden label -> none
            if not hide:
                sc = sc[:, :C]                                                # ablation: no none at all
            loss = F.cross_entropy(sc, target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return m.eval()


def train_probe(Xq, y, C, seed, epochs=40, bs=256, lr=1e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = nn.Sequential(nn.Linear(Xq.shape[1], 512), nn.GELU(), nn.Linear(512, C))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    Xq, y = torch.tensor(Xq), torch.tensor(y)
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for s in range(0, len(perm), bs):
            idx = torch.tensor(perm[s:s + bs])
            loss = F.cross_entropy(m(Xq[idx]), y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m.eval()


# ----------------------------------------------------------------------------- metrics
def fpr95(y, s):
    thr = np.percentile(s[y == 1], 5)
    return float((s[y == 0] >= thr).mean())


def ood(score_in, score_unk):
    """score: higher = more likely unknown."""
    y = np.r_[np.zeros(len(score_in)), np.ones(len(score_unk))]
    s = np.r_[score_in, score_unk]
    return float(roc_auc_score(y, s)), fpr95(y, s)


def softmax(x):
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def run_one(ds, setup, seed, raw, E):
    path, nh, _ = DATASETS[ds]
    sp = make_split(raw, setup, nh, 0)
    C = len(sp.intents)
    em = lambda texts: np.stack([E[t] for t in texts]).astype(np.float32)
    Xtr, ytr = em([t for t, _ in sp.train]), np.array([y for _, y in sp.train])
    Xte, yte = em([t for t, _ in sp.test]), np.array([y for _, y in sp.test])
    Xun = em(sp.unknown_test)
    A = em([label_text(i) for i in sp.intents])
    held = sp.heldout_intents
    if held:   # true labels of the unknown queries, for the zero-shot test
        yun = np.array([held.index(l) for t, l in raw["test"] if l in held])
        Ah = em([label_text(i) for i in held])
    R = {}
    with torch.no_grad():
        # zero-shot: raw frozen embeddings, cosine to label text
        s_in, s_un = Xte @ A.T, Xun @ A.T
        R["zero-shot"] = dict(acc=float((s_in.argmax(1) == yte).mean()), **dict(zip(("auroc", "fpr95"), ood(-s_in.max(1), -s_un.max(1)))))
        if held:
            z = Xun @ np.r_[A, Ah].T
            R["zero-shot"].update(zs_all=float((z.argmax(1) == C + yun).mean()), zs_held=float(((Xun @ Ah.T).argmax(1) == yun).mean()))
    pr = train_probe(Xtr, ytr, C, seed)   # linear probe on frozen embeddings
    with torch.no_grad():
        p_in, p_un = softmax(pr(torch.tensor(Xte)).numpy()), softmax(pr(torch.tensor(Xun)).numpy())
        R["linear probe"] = dict(acc=float((p_in.argmax(1) == yte).mean()), **dict(zip(("auroc", "fpr95"), ood(-p_in.max(1), -p_un.max(1)))))
    for name, hide in (("CLM no-none", False), ("CLM", True)):
        m = train_clm(Xtr, ytr, A, seed, hide=hide)
        with torch.no_grad():
            sc_in, sc_un = m.scores(torch.tensor(Xte), torch.tensor(A)).numpy(), m.scores(torch.tensor(Xun), torch.tensor(A)).numpy()
            acc = float((sc_in[:, :C].argmax(1) == yte).mean())
            au_msp, fp_msp = ood(-softmax(sc_in[:, :C]).max(1), -softmax(sc_un[:, :C]).max(1))
            au_sim, _ = ood(-sc_in[:, :C].max(1), -sc_un[:, :C].max(1))          # raw best match, no softmax
            if hide:    # unknown score = probability of `none` among [known..., none]
                au, fp = ood(softmax(sc_in)[:, C], softmax(sc_un)[:, C])
            else:
                au, fp = au_msp, fp_msp
            R[name] = dict(acc=acc, auroc=au, fpr95=fp, auroc_msp=au_msp, auroc_sim=au_sim, params=m.n_params())
            if held:    # add the held-out intents as candidates by their text alone - no retraining
                z = m.scores(torch.tensor(Xun), torch.tensor(np.r_[A, Ah])).numpy()[:, :C + len(held)]
                zh = m.scores(torch.tensor(Xun), torch.tensor(Ah)).numpy()[:, :len(held)]
                R[name].update(zs_all=float((z.argmax(1) == C + yun).mean()), zs_held=float((zh.argmax(1) == yun).mean()))
    # Part 1 fine-tuned classifier on the same split/seed, if present
    ft = os.path.join(ROOT, "runs", f"minilm_{ds}_{setup}_s{seed}", "eval", "results.json")
    if os.path.exists(ft):
        J = json.load(open(ft))
        best = max(J["ood"].values(), key=lambda v: v["auroc"])
        R["fine-tuned (Part 1)"] = dict(acc=J["acc_at_cap"], auroc=best["auroc"], fpr95=best["fpr95"])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--datasets", nargs="*", default=["clinc", "hwu"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "runs", "clm_cache"))
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    ALL = {}
    for ds in a.datasets:
        if not os.path.exists(os.path.join(ROOT, DATASETS[ds][0] if ds != "clinc" else "data/data_full.json")):
            print(f"skip {ds}: data file missing"); continue
        raw, E = get_embeddings(ds, a.backbone, a.pool, device, a.cache)
        for setup in ("official", "heldout"):
            for seed in a.seeds:
                ALL.setdefault((DATASETS[ds][2], setup), []).append(run_one(ds, setup, seed, raw, E))
                print("done", ds, setup, "seed", seed)

    ms = lambda v, k=100, d=1: (f"{k * np.mean(v):.{d}f}" + (f" ± {k * np.std(v, ddof=1):.{d}f}" if len(v) > 1 else "")) if len(v) else "–"
    names = ["zero-shot", "linear probe", "CLM no-none", "CLM", "fine-tuned (Part 1)"]
    L = ["# Part 2 kill test: small contrastive decision model", "",
         f"Frozen backbone `{a.backbone}`; trained parts are two projection heads + a `none` vector. Mean ± std over seeds {a.seeds}.", "",
         "`unknown AUROC` uses each model's own unknown score: `none` probability for the CLM, max-softmax otherwise. "
         "The two extra AUROC columns show the CLM's other possible scores, for transparency.", "",
         "| dataset | unknowns | model | in-scope acc % | unknown AUROC | FPR@95 % | AUROC max-softmax | AUROC best-match | new intents by text only, among all % | among new only % |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    verdict = []
    for (dsn, setup), rs in sorted(ALL.items()):
        for n in names:
            vs = [r[n] for r in rs if n in r]
            if not vs:
                continue
            g = lambda f: [v[f] for v in vs if f in v]
            L.append(f"| {dsn} | {'held-out intents' if setup == 'heldout' else 'far-OOS'} | {n} | {ms(g('acc'))} | {ms(g('auroc'), 1, 3)} | "
                     f"{ms(g('fpr95'))} | {ms(g('auroc_msp'), 1, 3)} | {ms(g('auroc_sim'), 1, 3)} | {ms(g('zs_all'))} | {ms(g('zs_held'))} |")
        if setup == "heldout" and all("fine-tuned (Part 1)" in r for r in rs):
            c = np.array([r["CLM"]["auroc"] for r in rs]); f = np.array([r["fine-tuned (Part 1)"]["auroc"] for r in rs])
            ca = np.mean([r["CLM"]["acc"] for r in rs]); fa = np.mean([r["fine-tuned (Part 1)"]["acc"] for r in rs])
            noise = max(c.std(ddof=1), f.std(ddof=1)) if len(c) > 1 else 0.01
            verdict.append((dsn, c.mean() - f.mean(), noise, ca - fa))
    L += ["", "## Verdict", ""]
    if verdict:
        for dsn, dA, noise, dAcc in verdict:
            L.append(f"- {dsn} held-out intents: CLM vs fine-tuned classifier, unknown AUROC {dA:+.3f} (seed noise ~{noise:.3f}), in-scope accuracy {100 * dAcc:+.1f} points.")
        go = all(dA > 2 * noise and dAcc > -0.02 for _, dA, noise, dAcc in verdict)
        L.append("")
        L.append("**GO**: better unknown detection on both datasets, accuracy within 2 points." if go else
                 "**NO-GO as specified**: the CLM does not clearly beat the fine-tuned classifier at spotting held-out intents on both datasets.")
    else:
        L.append("Fine-tuned Part 1 runs not found for these seeds; compare against the linear probe above.")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "CLM_KILLTEST.md"), "w").write("\n".join(L) + "\n")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in ALL.items()}, open(os.path.join(a.out, "clm_killtest.json"), "w"), indent=1)
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
