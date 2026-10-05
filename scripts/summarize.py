"""Collect eval/results.json from several runs into one markdown table.
  python scripts/summarize.py runs/*   >  results_table.md"""
import json, os, sys
print("| run | setup | acc@cap | τ | loops in / unknown | AUROC msp | energy | loops | KL drift | hidden | combined | FPR95 best |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for run in sys.argv[1:]:
    f = os.path.join(run, "eval", "results.json")
    if not os.path.exists(f):
        continue
    R = json.load(open(f)); o = R["ood"]; g = lambda k: f"{o[k]['auroc']:.3f}" if k in o else "–"
    comb = max((v["auroc"] for k, v in o.items() if k.startswith("combined")), default=None)
    best = min(o.values(), key=lambda v: v["fpr95"])["fpr95"]
    lo = f"{R['in_scope_loops_mean']:.1f} / {R['unknown_loops_mean']:.1f}" if R["looped"] else "–"
    print(f"| {run} | {R['setup']} | {R['acc_at_cap']:.3f} | {R.get('tau','–')} | {lo} | {g('msp')} | {g('energy')} | "
          f"{g('loops_to_converge')} | {g('late_kl_drift')} | {g('hidden_drift')} | {f'{comb:.3f}' if comb else '–'} | {best:.3f} |")
