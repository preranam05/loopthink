"""Conformal decision router: turn the classifier's probabilities into a prediction SET with a
coverage guarantee, and let the set size decide what happens.

    one label   -> answer now
    several     -> escalate, and the set IS the shortlist the LLM chooses from
    empty       -> no known intent fits (out of scope)

  python scripts/conformal.py                      # all runs -> results/CONFORMAL.md + conformal_coverage.png

Method (split conformal, LAC score s = 1 - p_true): on n calibration queries take
q = the ceil((n+1)(1-alpha))/n quantile of s; the set for a new query is {y : p_y >= 1 - q}.
Guarantee: P(true intent in set) >= 1 - alpha for queries exchangeable with the calibration data.
The guarantee says nothing about unknown intents - this script measures exactly how it breaks there.

Two calibration protocols are reported:
  val   : calibrate on the in-scope validation split, test on the test split (what you would deploy;
          the checkpoint was picked on validation accuracy, so exchangeability holds only approximately)
  split : 200 random halves of the test split, calibrate on one half, test on the other (clean check)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ALPHAS = (0.01, 0.02, 0.05, 0.10)


def softmax(x):
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def lac_threshold(p_cal, y_cal, alpha):
    """Probability cut t such that sets {y: p_y >= t} cover the truth with prob >= 1 - alpha."""
    s = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return 0.0                                   # not enough calibration data: include everything
    return float(1.0 - np.sort(s)[k - 1])


def describe(p, t, y=None):
    """Set statistics for probabilities p at cut t. y=None for unknown queries."""
    in_set = p >= t
    size = in_set.sum(1)
    d = dict(size=float(size.mean()), single=float((size == 1).mean()), multi=float((size > 1).mean()),
             empty=float((size == 0).mean()), p90_size=float(np.percentile(size, 90)))
    if y is not None:
        cov = in_set[np.arange(len(y)), y]
        d["coverage"] = float(cov.mean())
        one = size == 1
        d["single_acc"] = float((p.argmax(1)[one] == y[one]).mean()) if one.any() else float("nan")
        d["multi_has_truth"] = float(cov[size > 1].mean()) if (size > 1).any() else float("nan")
    return d


def router_error(p_in, y, p_unk, t, share):
    """System error with a perfect LLM on escalated sets (upper bound on what routing can achieve).
    in-scope wrong iff truth not in set; unknown wrong iff the set is a single (confident) label."""
    size_in = (p_in >= t).sum(1)
    e_in = 1.0 - (p_in >= t)[np.arange(len(y)), y].mean()
    size_unk = (p_unk >= t).sum(1)
    e_unk = float((size_unk == 1).mean())
    esc = (1 - share) * float((size_in > 1).mean()) + share * float((size_unk > 1).mean())
    return float((1 - share) * e_in + share * e_unk), esc, float(e_in), e_unk


def threshold_router_error(p_in, y, p_unk, esc_target, share):
    """Baseline: escalate the lowest-confidence queries (max prob), same LLM budget, perfect LLM."""
    conf = np.r_[p_in.max(1), p_unk.max(1)]
    w = np.r_[np.full(len(y), (1 - share) / len(y)), np.full(len(p_unk), share / len(p_unk))]
    order = np.argsort(conf)
    cut = np.searchsorted(np.cumsum(w[order]), esc_target)
    esc = np.zeros(len(conf), bool); esc[order[:cut]] = True
    wrong = np.r_[p_in.argmax(1) != y, np.ones(len(p_unk), bool)] & ~esc
    return float((w * wrong).sum())


def analyse(run, share, splits, rng):
    d = np.load(os.path.join(run, "eval", "trajectories.npz"))
    R = json.load(open(os.path.join(run, "eval", "results.json")))
    pv, pt, pu = softmax(d["val_logits"][:, 0]), softmax(d["test_logits"][:, 0]), softmax(d["unk_test_logits"][:, 0])
    yv, yt = d["y_val"], d["y_test"]
    out = {}
    for a in ALPHAS:
        t = lac_threshold(pv, yv, a)
        ins, unk = describe(pt, t, yt), describe(pu, t)
        err, esc, e_in, e_unk = router_error(pt, yt, pu, t, share)
        base = threshold_router_error(pt, yt, pu, esc, share)
        cov = []
        for _ in range(splits):                       # clean exchangeable check on the test split
            perm = rng.permutation(len(yt)); h = len(yt) // 2
            c, e = perm[:h], perm[h:]
            ts = lac_threshold(pt[c], yt[c], a)
            cov.append(float((pt[e] >= ts)[np.arange(len(e)), yt[e]].mean()))
        out[a] = dict(t=t, n_cal=len(yv), ins=ins, unk=unk, sys_err=err, escalated=esc, err_in=e_in, err_unk=e_unk,
                      baseline_err=base, split_cov_mean=float(np.mean(cov)),
                      split_cov_lo=float(np.percentile(cov, 5)), split_cov_hi=float(np.percentile(cov, 95)))
    model = "MiniLM-L6 (22.8M)" if not R.get("looped") else "looped (28.8M, ours)"
    data = "HWU64" if "hwu" in str(R.get("data", "")) else "CLINC150"
    setup = {"official": "far-OOS", "heldout": "held-out intents"}[R["setup"]]
    return (model, data, setup), out


def ms(vals, pct=True, d=1):
    v = np.array(vals, float); k = 100 if pct else 1
    return f"{k * v.mean():.{d}f}" + (f" ± {k * v.std(ddof=1):.{d}f}" if len(v) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--unknown_share", type=float, default=0.05)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    runs = a.runs or sorted(set(glob.glob(os.path.join(ROOT, "runs", "minilm_*_s*")) +
                                glob.glob(os.path.join(ROOT, "runs", "kill", "ft_official")) +
                                glob.glob(os.path.join(ROOT, "runs", "kill", "ft_heldout"))))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "eval", "trajectories.npz"))]
    rng = np.random.default_rng(0)
    G = {}
    for r in runs:
        k, o = analyse(r, a.unknown_share, a.splits, rng)
        G.setdefault(k, []).append(o)

    L = ["# Conformal decision router", "",
         "The classifier returns a set of intents guaranteed to contain the right one with probability ≥ 1 − α. "
         "One label → answer; several → escalate with that set as the shortlist; none → out of scope.",
         f"Mean ± std over seeds. Unknown traffic share for system numbers: {a.unknown_share:.0%}.", "",
         "## 1. Does the guarantee hold on in-scope queries?", "",
         "Target coverage is 1 − α. `val` = calibrated on validation, measured on test. "
         "`split` = 200 random calibration/test halves of the test set, mean [5th–95th percentile].", "",
         "| model | dataset | α | target % | coverage (val) % | coverage (split) % | avg set size | 90th pct size |",
         "|---|---|---|---|---|---|---|---|"]
    keys = sorted({(k[0], k[1]) for k in G})
    for model, data in keys:
        rs = [o for k, v in G.items() if (k[0], k[1]) == (model, data) and k[2] == "far-OOS" for o in v] or \
             [o for k, v in G.items() if (k[0], k[1]) == (model, data) for o in v]
        for al in ALPHAS:
            L.append(f"| {model} | {data} | {al} | {100 * (1 - al):.0f} | {ms([o[al]['ins']['coverage'] for o in rs])} | "
                     f"{100 * np.mean([o[al]['split_cov_mean'] for o in rs]):.1f} "
                     f"[{100 * np.mean([o[al]['split_cov_lo'] for o in rs]):.1f}–{100 * np.mean([o[al]['split_cov_hi'] for o in rs]):.1f}] | "
                     f"{ms([o[al]['ins']['size'] for o in rs], pct=False, d=2)} | {np.mean([o[al]['ins']['p90_size'] for o in rs]):.1f} |")
    L += ["", "## 2. What the router does with each query", "",
          "In-scope queries: answered directly (one label), escalated (several), or rejected (none). "
          "`direct acc` = accuracy of the directly answered ones; `truth in shortlist` = how often an escalated set contains the answer.", "",
          "| model | dataset | unknowns | α | answered directly % | direct acc % | escalated % | truth in shortlist % | rejected % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(G):
        for al in ALPHAS:
            rs = G[k]; g = lambda f: ms([o[al]["ins"][f] for o in rs])
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {al} | {g('single')} | {g('single_acc')} | {g('multi')} | {g('multi_has_truth')} | {g('empty')} |")
    L += ["", "## 3. Where the guarantee breaks: unknown intents", "",
          "The guarantee covers in-scope queries only. For unknown queries the right outcome is *rejected* or *escalated*; "
          "a single confident label is an error the guarantee does not cover.", "",
          "| model | dataset | unknowns | α | rejected % | escalated % | **confidently wrong %** |", "|---|---|---|---|---|---|---|"]
    for k in sorted(G):
        for al in ALPHAS:
            rs = G[k]; g = lambda f: ms([o[al]["unk"][f] for o in rs])
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {al} | {g('empty')} | {g('multi')} | **{g('single')}** |")
    L += ["", "## 4. System error with a perfect LLM, versus a confidence threshold at the same LLM budget", "",
          "`in-scope error` should be ≤ α. `system error` adds unknown traffic. "
          "`threshold router` escalates the least-confident queries, using the same share of LLM calls.", "",
          "| model | dataset | unknowns | α | in-scope error % | system error % | sent to LLM % | threshold router error % |",
          "|---|---|---|---|---|---|---|---|"]
    for k in sorted(G):
        for al in ALPHAS:
            rs = G[k]; g = lambda f: ms([o[al][f] for o in rs])
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {al} | {g('err_in')} | {g('sys_err')} | {g('escalated')} | {g('baseline_err')} |")
    L += ["", "## Caveats", "",
          "- A perfect LLM is assumed in section 4; real LLM errors on escalated sets add to these numbers.",
          "- `val` calibration is approximate: the checkpoint was selected on validation accuracy. `split` is the clean check.",
          "- Coverage is marginal (averaged over queries), not per intent.", ""]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "CONFORMAL.md"), "w").write("\n".join(L))
    json.dump({" | ".join(k): [{str(al): o[al] for al in ALPHAS} for o in v] for k, v in G.items()},
              open(os.path.join(a.out, "conformal.json"), "w"), indent=1)
    plot(G, os.path.join(a.out, "conformal_coverage.png"))
    print("\n".join(L))


def plot(G, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    al = np.array(ALPHAS) * 100
    for k in sorted(G):
        lab = f"{k[0].split(' ')[0]} · {k[1]} · {k[2]}"
        ax[0].plot(al, [100 * np.mean([o[a]["err_in"] for o in G[k]]) for a in ALPHAS], "o-", label=lab)
        ax[1].plot(al, [100 * np.mean([o[a]["unk"]["single"] for o in G[k]]) for a in ALPHAS], "o-", label=lab)
    ax[0].plot([0, 10], [0, 10], "k--", lw=1); ax[0].text(6.2, 6.9, "guarantee", rotation=38, fontsize=9)
    ax[0].set_xlabel("error budget α (%)"); ax[0].set_ylabel("measured in-scope error (%)")
    ax[0].set_title("In-scope: at or below the line = guarantee held")
    ax[1].set_xlabel("error budget α (%)"); ax[1].set_ylabel("unknown queries answered confidently (%)")
    ax[1].set_title("Unknown intents: not covered by the guarantee")
    for x in ax:
        x.grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
