"""Run every experiment off one saved trajectory per query.

  python -m loopthink.evaluate --run runs/ft_heldout --max_loops 32 --cap 16

For each query we record intent logits and the pooled latent state after every loop (1..max_loops),
then compute offline:
  1. out-of-scope detection AUROC / FPR@95 for: max-softmax, energy, loops-to-converge,
     late-trajectory drift, hidden-state drift, and combined (convergence + softmax) detectors
  2. accuracy vs average loops: adaptive exit (tau sweep) vs fixed r  (Pareto)
  3. test-time scaling: accuracy at loops 1..max_loops (beyond the training range)
  4. where it thinks harder: loops used per intent, confusable pairs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .train_utils import get_device
from sklearn.metrics import roc_auc_score

from .data import encode_batch, load_clinc, make_split
from .finetune import load_classifier


# ----------------------------------------------------------------------------- trajectories
@torch.no_grad()
def collect(model, tok, texts, label_emb, max_loops, device="cpu", bs=256, max_len=48):
    L = max_loops if model.cfg.is_looped else 1
    logits = np.zeros((len(texts), L, label_emb.shape[0]), dtype=np.float32)
    hcos = np.ones((len(texts), L), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx, pm = encode_batch(tok, texts[i: i + bs], max_len, device)
        traj = model.classify_trajectory(idx, pm, L, label_emb)
        prev = None
        for t, lg, pooled in traj:
            logits[i: i + len(idx), t - 1] = lg.float().cpu().numpy()
            if prev is not None:
                hcos[i: i + len(idx), t - 1] = torch.nn.functional.cosine_similarity(pooled, prev, dim=-1).cpu().numpy()
            prev = pooled
    return dict(logits=logits, hcos=hcos)


# ----------------------------------------------------------------------------- signals
def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def signals(tr):
    lg = tr["logits"]
    p = softmax(lg)
    srt = np.sort(p, -1)
    out = dict(top1=p.argmax(-1), msp=srt[..., -1], margin=srt[..., -1] - srt[..., -2],
               energy=-np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) - lg.max(-1),
               hcos=tr["hcos"])
    kl = np.zeros(p.shape[:2], dtype=np.float32)
    kl[:, 1:] = (p[:, 1:] * (np.log(p[:, 1:] + 1e-9) - np.log(p[:, :-1] + 1e-9))).sum(-1)
    out["kl"] = kl
    return out


def adaptive_exit(sig, tau: float, patience: int = 2, cap: int = 16):
    """Exit at first loop where top intent has been stable for `patience` loops and margin > tau.
    Queries that never satisfy it within `cap` loops are flagged not-converged."""
    top1, margin = sig["top1"][:, :cap], sig["margin"][:, :cap]
    N, L = top1.shape
    stable = np.ones_like(top1)
    for t in range(1, L):
        stable[:, t] = np.where(top1[:, t] == top1[:, t - 1], stable[:, t - 1] + 1, 1)
    ok = (stable >= patience) & (margin > tau)
    converged = ok.any(1)
    first = np.where(converged, ok.argmax(1), L - 1)
    return dict(loops=first + 1, converged=converged, pred=top1[np.arange(N), first])


def ood_scores(sig, cap, tau, patience):
    """Higher = more likely out-of-scope."""
    L = sig["top1"].shape[1]
    R = min(cap, L) - 1
    s = dict(msp=-sig["msp"][:, R], energy=sig["energy"][:, R])
    if L > 1:
        ex = adaptive_exit(sig, tau, patience, cap)
        lo = max(1, R // 2)
        s["loops_to_converge"] = ex["loops"].astype(np.float32)
        s["late_kl_drift"] = sig["kl"][:, lo: R + 1].mean(1)
        s["hidden_drift"] = 1 - sig["hcos"][:, max(1, R - 3): R + 1].mean(1)
    return s


def fpr_at_95(y, score):
    """FPR on in-scope queries when 95% of out-of-scope queries are caught."""
    thr = np.percentile(score[y == 1], 5)
    return float((score[y == 0] >= thr).mean())


def detect(scores_in, scores_out, z_ref=None):
    res = {}
    y = np.r_[np.zeros(len(next(iter(scores_in.values())))), np.ones(len(next(iter(scores_out.values()))))]
    names = list(scores_in)
    if "loops_to_converge" in names and z_ref is not None:
        for conv in ("loops_to_converge", "late_kl_drift"):
            name = f"combined_msp+{conv}"
            zi = lambda d, k: (d[k] - z_ref[k][0]) / (z_ref[k][1] + 1e-9)
            scores_in[name] = zi(scores_in, "msp") + zi(scores_in, conv)
            scores_out[name] = zi(scores_out, "msp") + zi(scores_out, conv)
            names.append(name)
    for k in names:
        sc = np.r_[scores_in[k], scores_out[k]]
        res[k] = dict(auroc=float(roc_auc_score(y, sc)), fpr95=fpr_at_95(y, sc))
    return res


# ----------------------------------------------------------------------------- main
def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--clinc", default="data")
    p.add_argument("--max_loops", type=int, default=32)
    p.add_argument("--cap", type=int, default=16, help="N: loops before a query is flagged not-converged")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min_converge_frac", type=float, default=0.9,
                   help="tau selection uses in-scope val only: largest tau with >= this fraction converging")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    device = get_device()
    model, tok, st, label_emb = load_classifier(args.run, device)
    meta = st["split"]
    split = make_split(load_clinc(args.clinc), meta["name"], len(meta["heldout_intents"]) or 30,
                       st["args"].get("split_seed", 0))
    assert split.intents == meta["intents"]
    looped = model.cfg.is_looped
    L = args.max_loops if looped else 1
    cap = min(args.cap, L)

    tr = {k: collect(model, tok, t, label_emb, L, device) for k, t in [
        ("val", [x for x, _ in split.val]), ("test", [x for x, _ in split.test]),
        ("unk_test", split.unknown_test)]}
    y_val = np.array([y for _, y in split.val]); y_test = np.array([y for _, y in split.test])
    sig = {k: signals(v) for k, v in tr.items()}
    R = {"setup": meta["name"], "looped": looped, "cap": cap, "patience": args.patience,
         "n_in_test": len(y_test), "n_unknown_test": len(split.unknown_test),
         "data": args.clinc, "n_heldout": len(meta["heldout_intents"])}

    # --- test-time scaling
    acc_per_loop = [(sig["test"]["top1"][:, t] == y_test).mean() for t in range(L)]
    R["acc_per_loop"] = [float(a) for a in acc_per_loop]
    R["acc_at_cap"] = float(acc_per_loop[cap - 1])

    taus = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
    if looped:
        # --- tau selection on in-scope val only
        conv_frac = {t: adaptive_exit(sig["val"], t, args.patience, cap)["converged"].mean() for t in taus}
        ok = [t for t in taus if conv_frac[t] >= args.min_converge_frac]
        tau = max(ok) if ok else 0.0
        R["tau"] = tau
        R["val_converged_frac_by_tau"] = {str(k): float(v) for k, v in conv_frac.items()}
        # --- Pareto: adaptive vs fixed
        par = []
        for t in taus:
            ex = adaptive_exit(sig["test"], t, args.patience, cap)
            par.append(dict(tau=t, avg_loops=float(ex["loops"].mean()), acc=float((ex["pred"] == y_test).mean()),
                            converged=float(ex["converged"].mean())))
        R["pareto_adaptive"] = par
        R["pareto_fixed"] = [dict(r=r, acc=float(acc_per_loop[r - 1])) for r in range(1, cap + 1)]
        # --- per-intent thinking
        ex = adaptive_exit(sig["test"], tau, args.patience, cap)
        per = {}
        for c, name in enumerate(split.intents):
            m = y_test == c
            per[name] = dict(loops=float(ex["loops"][m].mean()), acc=float((ex["pred"][m] == c).mean()))
        R["per_intent"] = per
        R["slowest_intents"] = sorted(per, key=lambda k: -per[k]["loops"])[:15]
        R["fastest_intents"] = sorted(per, key=lambda k: per[k]["loops"])[:15]
        pairs = [("book_flight", "flight_status"), ("timer", "alarm"), ("play_music", "next_song"),
                 ("calendar", "calendar_update"), ("reminder", "reminder_update")]
        R["confusable_pairs"] = {f"{a}|{b}": dict(loops=[per[a]["loops"], per[b]["loops"]],
                                                   acc=[per[a]["acc"], per[b]["acc"]])
                                 for a, b in pairs if a in per and b in per}
        R["unknown_loops_mean"] = float(adaptive_exit(sig["unk_test"], tau, args.patience, cap)["loops"].mean())
        R["in_scope_loops_mean"] = float(ex["loops"].mean())
        R["unknown_converged_frac"] = float(adaptive_exit(sig["unk_test"], tau, args.patience, cap)["converged"].mean())
        R["in_scope_converged_frac"] = float(ex["converged"].mean())

    # --- OOD detection
    tau = R.get("tau", 0.0)
    s_val = ood_scores(sig["val"], cap, tau, args.patience)
    z_ref = {k: (float(v.mean()), float(v.std())) for k, v in s_val.items()}
    s_in, s_out = ood_scores(sig["test"], cap, tau, args.patience), ood_scores(sig["unk_test"], cap, tau, args.patience)
    R["ood"] = detect(s_in, s_out, z_ref)
    if looped:  # AUROC of loops-to-converge across tau (so the choice is transparent)
        R["ood_loops_auroc_by_tau"] = {}
        for t in taus:
            a = adaptive_exit(sig["test"], t, args.patience, cap)["loops"]
            b = adaptive_exit(sig["unk_test"], t, args.patience, cap)["loops"]
            R["ood_loops_auroc_by_tau"][str(t)] = float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
        # does OOD get *more* loops?  (direction check from the kill test)
        R["direction_check"] = ("unknowns take MORE loops" if R["unknown_loops_mean"] > R["in_scope_loops_mean"]
                                else "unknowns converge as fast or FASTER (ACT failure mode)")

    out_dir = os.path.join(args.run, "eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(R, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "trajectories.npz"),
                        **{f"{k}_{kk}": vv for k, v in tr.items() for kk, vv in v.items()},
                        y_val=y_val, y_test=y_test)
    write_summary(R, out_dir)
    if not args.no_plots:
        plot_all(R, sig, y_test, out_dir, cap)
    print(open(os.path.join(out_dir, "summary.md")).read())


def write_summary(R, out_dir):
    lines = [f"# Results — setup `{R['setup']}` ({'looped' if R['looped'] else 'dense'})", ""]
    lines.append(f"In-scope test accuracy at {R['cap']} loops: **{R['acc_at_cap']:.3f}**")
    if R["looped"]:
        lines.append(f"Selected tau (in-scope val only): {R['tau']} | mean loops in-scope "
                     f"{R['in_scope_loops_mean']:.2f} vs unknown {R['unknown_loops_mean']:.2f} → {R['direction_check']}")
        lines.append(f"Converged within cap: in-scope {R['in_scope_converged_frac']:.3f}, unknown {R['unknown_converged_frac']:.3f}")
    lines += ["", "| detector | AUROC ↑ | FPR@95 ↓ |", "|---|---|---|"]
    for k, v in sorted(R["ood"].items(), key=lambda kv: -kv[1]["auroc"]):
        lines.append(f"| {k} | {v['auroc']:.3f} | {v['fpr95']:.3f} |")
    if R["looped"]:
        lines += ["", "Accuracy vs loops (test-time scaling): " +
                  ", ".join(f"r{t+1}={a:.3f}" for t, a in enumerate(R["acc_per_loop"]) if t + 1 in (1, 2, 4, 8, 16, 24, 32))]
        lines += ["", "Slowest intents: " + ", ".join(R["slowest_intents"][:8])]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_all(R, sig, y_test, out_dir, cap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not R["looped"]:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot([p["avg_loops"] for p in R["pareto_adaptive"]], [p["acc"] for p in R["pareto_adaptive"]], "o-", label="adaptive exit (τ sweep)")
    ax[0].plot([p["r"] for p in R["pareto_fixed"]], [p["acc"] for p in R["pareto_fixed"]], "s--", label="fixed r")
    ax[0].set(xlabel="average loops", ylabel="in-scope accuracy", title="Accuracy vs compute"); ax[0].legend()
    ax[1].plot(range(1, len(R["acc_per_loop"]) + 1), R["acc_per_loop"], "o-")
    ax[1].axvline(16, ls=":", c="gray"); ax[1].text(16.3, min(R["acc_per_loop"]), "max train r", color="gray")
    ax[1].set(xlabel="loops", ylabel="accuracy", title="Test-time scaling")
    ks = list(R["ood"]); ax[2].barh(ks, [R["ood"][k]["auroc"] for k in ks]); ax[2].axvline(0.5, c="gray", ls=":")
    ax[2].set(xlim=(0.4, 1.0), title=f"Out-of-scope AUROC ({R['setup']})")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "overview.png"), dpi=120); plt.close(fig)
    # KL trajectories in-scope vs unknown
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, lab in (("test", "in-scope"), ("unk_test", "unknown")):
        ax.plot(range(2, cap + 1), sig[k]["kl"][:, 1:cap].mean(0), label=lab)
    ax.set(yscale="log", xlabel="loop", ylabel="KL(p_t || p_t-1)", title="How much the answer is still moving")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "convergence_curves.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
