"""One credible results table: every headline number with seed variance and a 95% bootstrap CI.

  python scripts/report.py                 # scans runs/, writes results/REPORT.md + results/report.json

For each run it reads the saved logits (answer after loop 1 = the deployed setting) and computes
  - in-scope accuracy, OOD AUROC (max-softmax, energy), FPR@95
  - the deployable cascade: threshold on in-scope validation only (escalate 10%), traffic reweighted
    to 5% unknown, perfect LLM (upper bound)
  - where a run has real LLM calls (scripts/llm_validate.py): the measured cascade vs LLM-only
CIs: bootstrap over test queries (and over the sampled LLM calls), pooled across seeds, so they include
both test-set noise and training-seed noise.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
_spec = importlib.util.spec_from_file_location("cascade", os.path.join(HERE, "cascade.py"))
cascade = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cascade)


def fpr95(y, s):
    thr = np.percentile(s[y == 1], 5)
    return float((s[y == 0] >= thr).mean())


def run_metrics(run, rng, B, val_rate, unk_share):
    S, y_val, y_test, R = cascade.load(run, 1)
    ok = S["test"]["pred"] == y_test
    n_in, n_unk = len(y_test), len(S["unk_test"]["pred"])
    thr = float(np.quantile(S["val"]["msp"], 1 - val_rate))
    esc_in, esc_unk = S["test"]["msp"] >= thr, S["unk_test"]["msp"] >= thr

    def stats(ii, uu):
        y = np.r_[np.zeros(len(ii)), np.ones(len(uu))]
        m = dict(acc=ok[ii].mean())
        for k in ("msp", "energy"):
            sc = np.r_[S["test"][k][ii], S["unk_test"][k][uu]]
            m[f"auroc_{k}"] = roc_auc_score(y, sc)
        m["fpr95_msp"] = fpr95(y, np.r_[S["test"]["msp"][ii], S["unk_test"]["msp"][uu]])
        e_in, e_unk = esc_in[ii].mean(), esc_unk[uu].mean()
        kept = ok[ii][~esc_in[ii]].mean() if (~esc_in[ii]).any() else 0.0
        m["small_only_sys_acc"] = (1 - unk_share) * ok[ii].mean()
        m["cascade_oracle_sys_acc"] = (1 - unk_share) * ((1 - e_in) * kept + e_in) + unk_share * e_unk
        m["escalated"] = (1 - unk_share) * e_in + unk_share * e_unk
        m["unknowns_caught"] = e_unk
        return m

    point = stats(np.arange(n_in), np.arange(n_unk))
    boots = [stats(rng.integers(0, n_in, n_in), rng.integers(0, n_unk, n_unk)) for _ in range(B)]
    return R, point, boots, dict(S=S, y_test=y_test, ok=ok, esc_in=esc_in, esc_unk=esc_unk, thr=thr)


def measured(run, ctx, rng, B, unk_share):
    out = []
    for summ in sorted(glob.glob(os.path.join(run, "llm_validate", "*", "summary.json"))):
        tag = os.path.basename(os.path.dirname(summ))
        if "fake" in tag or "v1_noschema" in tag or "oldprompt" in tag:
            continue
        sm = json.load(open(summ))
        calls = [json.loads(l) for l in open(os.path.join(os.path.dirname(summ), "calls.jsonl"))]
        thr = sm["threshold"]
        esc_in = ctx["S"]["test"]["msp"] >= thr; esc_unk = ctx["S"]["unk_test"]["msp"] >= thr
        g = dict(ein=np.array([c["correct"] for c in calls if c["part"] == "in" and esc_in[c["i"]]], float),
                 eunk=np.array([c["correct"] for c in calls if c["part"] == "unk" and esc_unk[c["i"]]], float))
        ok, n_in, n_unk = ctx["ok"], len(ctx["ok"]), len(esc_unk)
        # LLM-only baseline = the random samples only; rebuild them exactly as llm_validate.py drew them
        L_ = sm["llm"]
        n_esc = max(L_["acc_escalated_in_scope"][1], L_["acc_escalated_unknown"][1])
        n_rand = max(L_["acc_random_in_scope"][1], L_["acc_random_unknown"][1])
        prng = np.random.default_rng(sm.get("seed", 0))
        pick = lambda idx, k: sorted(prng.choice(idx, min(k, len(idx)), replace=False).tolist()) if len(idx) else []
        pick(np.nonzero(esc_in)[0], n_esc); pick(np.nonzero(esc_unk)[0], n_esc)
        rand_in, rand_unk = set(pick(np.arange(n_in), n_rand)), set(pick(np.arange(n_unk), n_rand))
        byk = {(c["part"], c["i"]): c for c in calls}
        rin = [byk[("in", i)] for i in rand_in if ("in", i) in byk]; runk = [byk[("unk", i)] for i in rand_unk if ("unk", i) in byk]

        def sys_acc(ii, uu, a_ein, a_eunk, share):
            e_in, e_unk = esc_in[ii].mean(), esc_unk[uu].mean()
            kept = ok[ii][~esc_in[ii]].mean()
            return (1 - share) * ((1 - e_in) * kept + e_in * a_ein) + share * e_unk * a_eunk
        shortlist = bool(sm["llm"].get("shortlist_k"))
        pt = sys_acc(np.arange(n_in), np.arange(n_unk), g["ein"].mean(), g["eunk"].mean(), unk_share)
        bs = [sys_acc(rng.integers(0, n_in, n_in), rng.integers(0, n_unk, n_unk),
                      rng.choice(g["ein"], len(g["ein"])).mean(), rng.choice(g["eunk"], len(g["eunk"])).mean(), unk_share)
              for _ in range(B)]
        rows = dict(tag=tag, shortlist=shortlist, cascade=pt, cascade_ci=list(np.percentile(bs, [2.5, 97.5])),
                    escalated=sm["cascade"]["escalated"], llm_p50_ms=sm["llm"]["p50_ms"],
                    cascade_mean_ms=sm["cascade"]["mean_ms"], llm_only_mean_ms=None if shortlist else sm["llm_only"]["mean_ms"],
                    cascade_cost_1k=sm["cascade"]["cost_per_1k"], llm_only_cost_1k=None if shortlist else sm["llm_only"]["cost_per_1k"],
                    n_llm_calls=len(calls),
                    by_share={str(s): sys_acc(np.arange(n_in), np.arange(n_unk), g["ein"].mean(), g["eunk"].mean(), s)
                              for s in (0.02, 0.05, 0.10, 0.18)})
        if not shortlist:
            rin_ok = np.array([c["correct"] for c in rin], float); runk_ok = np.array([c["correct"] for c in runk], float)
            lo = lambda a, b, s: (1 - s) * a + s * b
            rows["llm_only"] = lo(rin_ok.mean(), runk_ok.mean(), unk_share)
            rows["llm_only_ci"] = list(np.percentile([lo(rng.choice(rin_ok, len(rin_ok)).mean(),
                                                         rng.choice(runk_ok, len(runk_ok)).mean(), unk_share) for _ in range(B)], [2.5, 97.5]))
            rows["llm_only_by_share"] = {str(s): lo(rin_ok.mean(), runk_ok.mean(), s) for s in (0.02, 0.05, 0.10, 0.18)}
        out.append(rows)
    return out


def group_key(run, R):
    model = "MiniLM-L6 (22.8M)" if not R.get("looped") else "looped (28.8M, ours)"
    d = str(R.get("data", ""))
    data = "HWU64" if "hwu" in d else "CLINC150"
    setup = {"official": "far-OOS", "heldout": "held-out intents"}[R["setup"]]
    return model, data, setup


def na(x, spec):
    return "–" if x is None else format(x, spec)


def fmt(vals, ci, pct=True, d=1):
    f = (lambda x: f"{100 * x:.{d}f}") if pct else (lambda x: f"{x:.3f}")
    m = np.mean(vals)
    s = f"{f(m)}"
    if len(vals) > 1:
        s += f" ± {f(np.std(vals, ddof=1))}"
    return s + f" [{f(ci[0])}–{f(ci[1])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--val_rate", type=float, default=0.10)
    ap.add_argument("--unknown_share", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    runs = a.runs or sorted(set(glob.glob(os.path.join(ROOT, "runs", "minilm_*_s*")) +
                                glob.glob(os.path.join(ROOT, "runs", "kill", "ft_official")) +
                                glob.glob(os.path.join(ROOT, "runs", "kill", "ft_heldout")) +
                                glob.glob(os.path.join(ROOT, "runs", "looped_hwu_*"))))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "eval", "trajectories.npz"))]
    rng = np.random.default_rng(0)
    groups, meas = {}, []
    for run in runs:
        R, pt, bs, ctx = run_metrics(run, rng, a.boot, a.val_rate, a.unknown_share)
        k = group_key(run, R)
        groups.setdefault(k, []).append(dict(run=os.path.relpath(run, ROOT), point=pt, boots=bs))
        for m in measured(run, ctx, rng, a.boot, a.unknown_share):
            meas.append(dict(group=k, run=os.path.relpath(run, ROOT), **m))
        print("read", os.path.relpath(run, ROOT))

    cols = [("acc", "In-scope acc %"), ("auroc_msp", "OOD AUROC (msp)"), ("auroc_energy", "OOD AUROC (energy)"),
            ("fpr95_msp", "FPR@95 %")]
    L = ["# Results", "",
         f"Mean ± std over seeds, [95% bootstrap CI pooled over seeds and test queries]. Answer read after loop 1. "
         f"Cascade: threshold set on in-scope validation only (escalate {a.val_rate:.0%}), traffic reweighted to "
         f"{a.unknown_share:.0%} unknown.", "",
         "## 1. Classifier quality", "",
         "| model | dataset | unknowns | seeds | " + " | ".join(c[1] for c in cols) + " |",
         "|---|---|---|---|" + "---|" * len(cols)]
    summary = {}
    for k in sorted(groups):
        rs = groups[k]
        pooled = [b for r in rs for b in r["boots"]]
        row = []
        for c, _ in cols:
            vals = [r["point"][c] for r in rs]; ci = np.percentile([b[c] for b in pooled], [2.5, 97.5])
            row.append(fmt(vals, ci, pct=not c.startswith("auroc")))
            summary.setdefault(" | ".join(k), {})[c] = dict(mean=float(np.mean(vals)), ci=[float(x) for x in ci], n=len(vals))
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | " + " | ".join(row) + " |")
    L += ["", "## 2. Cascade with a perfect LLM (upper bound)", "",
          "| model | dataset | unknowns | small model alone % | cascade % | sent to LLM % | unknowns caught % |",
          "|---|---|---|---|---|---|---|"]
    for k in sorted(groups):
        rs = groups[k]; pooled = [b for r in rs for b in r["boots"]]
        cell = lambda c: fmt([r["point"][c] for r in rs], np.percentile([b[c] for b in pooled], [2.5, 97.5]))
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {cell('small_only_sys_acc')} | {cell('cascade_oracle_sys_acc')} | "
                 f"{cell('escalated')} | {cell('unknowns_caught')} |")
    if meas:
        L += ["", "## 3. Measured cascade with a real LLM", "",
              "| model | dataset | unknowns | LLM (mode) | cascade % [CI] | LLM alone % [CI] | to LLM | mean latency cascade / LLM-only | ref $/1k cascade / LLM-only |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in meas:
            k = m["group"]; mode = ("top-5 shortlist" + (" + examples" if "_ex" in m["tag"] else "")) if m["shortlist"] else "all intents"
            lo = (f"{100 * m['llm_only']:.1f} [{100 * m['llm_only_ci'][0]:.1f}–{100 * m['llm_only_ci'][1]:.1f}]"
                  if "llm_only" in m else "n/a (needs shortlist)")
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {m['tag']} ({mode}) | {100 * m['cascade']:.1f} "
                     f"[{100 * m['cascade_ci'][0]:.1f}–{100 * m['cascade_ci'][1]:.1f}] | {lo} | {100 * m['escalated']:.1f}% | "
                     f"{m['cascade_mean_ms']:.0f} / {na(m['llm_only_mean_ms'], '.0f')} ms | "
                     f"{m['cascade_cost_1k']:.4f} / {na(m['llm_only_cost_1k'], '.4f')} |")
        L += ["", "### Sensitivity to the share of unknown traffic (measured cascade, point estimates)", "",
              "| run | LLM | 2% | 5% | 10% | 18% |", "|---|---|---|---|---|---|"]
        for m in meas:
            L.append(f"| {m['run']} | {m['tag']} | " + " | ".join(f"{100 * m['by_share'][s]:.1f}" for s in ("0.02", "0.05", "0.1", "0.18")) + " |")
            if "llm_only_by_share" in m:
                L.append(f"| ↳ LLM alone | {m['tag']} | " + " | ".join(f"{100 * m['llm_only_by_share'][s]:.1f}" for s in ("0.02", "0.05", "0.1", "0.18")) + " |")
    L += ["", "## Caveats", "",
          "- Unknown share of real traffic is an assumption; see the sensitivity table.",
          "- Cost is priced at reference API rates for the tokens used ($0.15 / $0.60 per 1M in/out); the local LLM itself is free.",
          "- Measured-LLM CIs resample the ~300 sampled LLM calls per group; LLM runs use seed 0 of the classifier only.",
          "- HWU64 far-OOS unknowns are CLINC150's out-of-scope queries (not home-assistant requests); held-out intents are the harder test.",
          "- The looped model has one seed per setup; its pretraining took ~10 h, so extra seeds were not run.", ""]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(L))
    json.dump(dict(classifier=summary, measured=[{k: v for k, v in m.items() if k != "group"} | {"group": list(m["group"])} for m in meas]),
              open(os.path.join(a.out, "report.json"), "w"), indent=2, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
