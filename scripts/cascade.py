"""Escalation cascade: small decision model answers, uncertain queries go to an LLM.

Reads the trajectories that `loopthink.evaluate` already saved (no model, no GPU needed) and asks:
at a given system accuracy, what fraction of traffic still has to hit the LLM, and what does it cost?

  python scripts/cascade.py --runs runs/kill/ft_official runs/kill/ft_heldout
  python scripts/cascade.py --runs runs/kill/ft_official --llm_acc 0.95 --llm_cost 2.0 --llm_ms 800

Ground truth: in-scope queries need the right intent; unknown queries (CLINC OOS / held-out intents)
count as correct only if escalated and the LLM gets them right (the small model cannot say "unknown"
except by escalating). The LLM is simulated: correct with probability --llm_acc on every escalated
query (1.0 = oracle). Replace with measured numbers from scripts/llm_validate.py (week 2).

Escalation signals compared: energy, max-softmax (msp), top-2 margin, loops-to-converge (looped model),
and random escalation (the floor any signal must beat).

Outputs per run in <run>/cascade/: cascade.json, cascade.md, cascade.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def load(run, loop):
    d = np.load(os.path.join(run, "eval", "trajectories.npz"))
    R = json.load(open(os.path.join(run, "eval", "results.json")))
    L = d["test_logits"].shape[1]
    r = min(loop, L) - 1
    out = {}
    for part in ("val", "test", "unk_test"):
        full = d[f"{part}_logits"]
        lg = full[:, r]
        p = softmax(lg)
        srt = np.sort(p, -1)
        m = lg.max(-1, keepdims=True)
        out[part] = dict(pred=p.argmax(-1),
                         energy=-(np.log(np.exp(lg - m).sum(-1)) + m[:, 0]),   # higher = more unknown
                         msp=-srt[:, -1], margin=-(srt[:, -1] - srt[:, -2]),
                         full=full)
    if L > 1:  # loops-to-converge, same rule as evaluate.adaptive_exit
        tau, cap, pat = R.get("tau", 0.0), R.get("cap", 16), R.get("patience", 2)
        for part in out:
            p = softmax(out[part]["full"][:, :cap])
            srt = np.sort(p, -1)
            top1, margin = p.argmax(-1), srt[..., -1] - srt[..., -2]
            stable = np.ones_like(top1)
            for t in range(1, top1.shape[1]):
                stable[:, t] = np.where(top1[:, t] == top1[:, t - 1], stable[:, t - 1] + 1, 1)
            ok = (stable >= pat) & (margin > tau)
            loops = np.where(ok.any(1), ok.argmax(1) + 1, cap).astype(np.float32)
            out[part]["loops"] = loops + 1e-3 * out[part]["energy"]  # break ties among cap-hitters
    for part in out:
        out[part].pop("full")
    return out, d["y_val"], d["y_test"], R


def cascade_curve(score, correct_small, is_unknown, llm_acc, rng, w):
    """Escalate the highest-score queries first; returns (traffic fraction escalated, system accuracy),
    both weighted by w so the unknown share can match real traffic."""
    n = len(score)
    order = np.argsort(-score, kind="stable")
    llm_right = rng.random(n) < llm_acc
    kept_ok = (correct_small & ~is_unknown).astype(float)[order]
    esc_ok = llm_right.astype(float)[order]
    wo = w[order] / w.sum()
    # escalating the first k (in score order): acc = sum_{<k} w*esc_ok + sum_{>=k} w*kept_ok
    esc_cum = np.r_[0, np.cumsum(wo * esc_ok)]
    kept_tail = np.r_[np.cumsum((wo * kept_ok)[::-1])[::-1], 0]
    frac = np.r_[0, np.cumsum(wo)]
    return frac, esc_cum + kept_tail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--loop", type=int, default=1, help="read the small model's answer after this many loops")
    ap.add_argument("--llm_acc", type=float, default=1.0, help="simulated LLM accuracy on escalated queries")
    ap.add_argument("--llm_cost", type=float, default=1.0, help="$ per 1,000 LLM calls (placeholder; measure it)")
    ap.add_argument("--small_cost", type=float, default=0.01, help="$ per 1,000 small-model calls")
    ap.add_argument("--llm_ms", type=float, default=600.0, help="LLM latency per call (placeholder)")
    ap.add_argument("--small_ms", type=float, default=5.0, help="small-model latency per call")
    ap.add_argument("--targets", default="0.90,0.95,0.97,0.98", help="system accuracies to report")
    ap.add_argument("--val_rates", default="0.05,0.10,0.20",
                    help="deployable thresholds: escalate this fraction of IN-SCOPE VALIDATION queries")
    ap.add_argument("--unknown_share", type=float, default=0.0,
                    help="reweight so unknowns are this share of traffic (e.g. 0.05); 0 = keep the test mix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_plot", action="store_true")
    a = ap.parse_args(argv)

    for run in a.runs:
        S, y_val, y_test, R = load(run, a.loop)
        name = f"{os.path.basename(os.path.normpath(run))} ({'looped' if R['looped'] else 'dense'})"
        n_in, n_unk = len(y_test), len(S["unk_test"]["pred"])
        is_unknown = np.r_[np.zeros(n_in, bool), np.ones(n_unk, bool)]
        pred = np.r_[S["test"]["pred"], S["unk_test"]["pred"]]
        correct_small = np.r_[S["test"]["pred"] == y_test, np.zeros(n_unk, bool)]
        share = a.unknown_share or n_unk / (n_in + n_unk)
        w = np.r_[np.full(n_in, (1 - share) / n_in), np.full(n_unk, share / n_unk)]
        signals = [k for k in ("energy", "msp", "margin", "loops") if k in S["test"]]
        rng0 = np.random.default_rng(a.seed)
        score = {k: np.r_[S["test"][k], S["unk_test"][k]] for k in signals}
        score["random"] = rng0.random(n_in + n_unk)
        targets = [float(t) for t in a.targets.split(",")]

        res = dict(run=run, name=name, loop=a.loop, n_in_scope=n_in, n_unknown=n_unk,
                   unknown_share=share, small_only_acc=float((w * correct_small).sum() / w.sum()),
                   llm_only_acc=a.llm_acc, assumptions=dict(llm_acc=a.llm_acc, llm_cost_per_1k=a.llm_cost,
                   small_cost_per_1k=a.small_cost, llm_ms=a.llm_ms, small_ms=a.small_ms), signals={})
        curves = {}
        for k, sc in score.items():
            fr, acc = cascade_curve(sc, correct_small, is_unknown, a.llm_acc, np.random.default_rng(a.seed + 1), w)
            curves[k] = (fr, acc)
            cost = a.small_cost + fr * a.llm_cost
            at = {}
            for t in targets:
                hit = np.nonzero(acc >= t - 1e-9)[0]
                if len(hit):
                    f = float(fr[hit[0]])
                    at[str(t)] = dict(escalated=f, cost_per_1k=float(a.small_cost + f * a.llm_cost),
                                      saving_vs_llm_only=float(1 - (a.small_cost + f * a.llm_cost) / a.llm_cost),
                                      mean_ms=float(a.small_ms + f * a.llm_ms))
                else:
                    at[str(t)] = None
            # area under accuracy-vs-escalation curve (higher = better signal)
            res["signals"][k] = dict(at_target=at, auc=float(np.trapezoid(acc, fr)))
            curves[k] = (fr, acc, cost)

        # deployable operating points: threshold fixed on in-scope validation only (no unknowns seen)
        dep = {}
        for k in signals:
            dep[k] = {}
            for rate in [float(x) for x in a.val_rates.split(",")]:
                thr = np.quantile(S["val"][k], 1 - rate)
                esc = score[k] >= thr
                llm_right = np.random.default_rng(a.seed + 2).random(len(esc)) < a.llm_acc
                acc = float((w * np.where(esc, llm_right, correct_small)).sum() / w.sum())
                f = float((w * esc).sum() / w.sum())
                dep[k][str(rate)] = dict(test_escalated=f, system_acc=acc,
                                         unknown_caught=float(esc[is_unknown].mean()),
                                         in_scope_escalated=float(esc[~is_unknown].mean()),
                                         cost_per_1k=float(a.small_cost + f * a.llm_cost),
                                         mean_ms=float(a.small_ms + f * a.llm_ms))
        res["deployable"] = dep

        out = os.path.join(run, "cascade" + (f"_unk{a.unknown_share:g}" if a.unknown_share else ""))
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, "cascade.json"), "w"), indent=2)
        md = write_md(res, targets)
        open(os.path.join(out, "cascade.md"), "w").write(md)
        print(md)
        if not a.no_plot:
            plot(curves, res, os.path.join(out, "cascade.png"))


def write_md(res, targets):
    A = res["assumptions"]
    L = [f"# Escalation cascade: {res['name']}", "",
         f"Traffic: {res['n_in_scope']} in-scope + {res['n_unknown']} unknown queries "
         f"({res['unknown_share']:.0%} unknown). Small model alone: **{res['small_only_acc']:.1%}** system accuracy. "
         f"LLM alone: **{res['llm_only_acc']:.1%}** at ${A['llm_cost_per_1k']:.2f}/1k queries.", "",
         f"_Assumptions (simulated until measured): LLM accuracy {A['llm_acc']:.0%}, LLM ${A['llm_cost_per_1k']}/1k calls, "
         f"{A['llm_ms']:.0f} ms/call; small model ${A['small_cost_per_1k']}/1k, {A['small_ms']:.0f} ms._", "",
         "## % of traffic that must go to the LLM to reach a target system accuracy", "",
         "| signal | " + " | ".join(f"acc ≥ {t:.0%}" for t in targets) + " | curve area ↑ |",
         "|---|" + "---|" * (len(targets) + 1)]
    for k, v in sorted(res["signals"].items(), key=lambda kv: -kv[1]["auc"]):
        cells = []
        for t in targets:
            x = v["at_target"][str(t)]
            cells.append("n/a" if x is None else f"{x['escalated']:.0%} (saves {x['saving_vs_llm_only']:.0%})")
        L.append(f"| {k} | " + " | ".join(cells) + f" | {v['auc']:.4f} |")
    L += ["", "## Deployable thresholds (set on in-scope validation only)", "",
          "| signal | val escalation rate | test escalated | system acc | unknowns caught | in-scope escalated | $/1k | mean ms |",
          "|---|---|---|---|---|---|---|---|"]
    for k, d in res["deployable"].items():
        for rate, x in d.items():
            L.append(f"| {k} | {float(rate):.0%} | {x['test_escalated']:.1%} | {x['system_acc']:.1%} | "
                     f"{x['unknown_caught']:.1%} | {x['in_scope_escalated']:.1%} | {x['cost_per_1k']:.2f} | {x['mean_ms']:.0f} |")
    return "\n".join(L) + "\n"


def plot(curves, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    style = dict(energy="-", msp="-", margin="--", loops="-.", random=":")
    for k, (fr, acc, cost) in curves.items():
        ax[0].plot(fr * 100, acc * 100, style.get(k, "-"), label=k)
        ax[1].plot(cost, acc * 100, style.get(k, "-"), label=k)
    for x in ax:
        x.axhline(res["llm_only_acc"] * 100, color="gray", lw=0.8, ls=":")
        x.set_ylabel("system accuracy (%)"); x.grid(alpha=0.3)
    ax[0].set_xlabel("% of queries escalated to the LLM")
    ax[1].set_xlabel("$ per 1,000 queries")
    ax[0].set_title("Accuracy vs escalation"); ax[1].set_title("Accuracy vs cost")
    ax[0].legend(); fig.suptitle(res["name"])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
