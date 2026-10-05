"""Week-1 kill test: ~200M-token looped pretrain -> CLINC150 fine-tune (official + 120/30 held-out)
-> does "did the answer settle?" flag unknowns at least as well as max-softmax?

  # Mac (resumes from the latest checkpoint in runs/kill/pretrain automatically)
  caffeinate -dims python scripts/kill_test.py --data_dir data/tok200m --out runs/kill \
      --precision fp32 --batch_size 16 --grad_accum 4 --max_hours 1000

  # add --with_dense to also pretrain + fine-tune the param-matched dense-6 baseline (for "strong go")
  # add --skip_pretrain --preset tiny for the from-scratch CPU sanity version

Go/no-go (from the plan):
  GO         best convergence detector matches or beats max-softmax *within the looped model*
             (AUROC within 0.01 or better) AND unknowns use more loops than in-scope queries.
  STRONG GO  GO, and it also matches or beats dense + max-softmax (needs --with_dense or --dense_dir).
  NO-GO      no signal, or reversed direction (unknowns settle faster).
Extra flags not listed here are passed to pretraining.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from loopthink import pretrain, finetune, evaluate

CONV = ("loops_to_converge", "late_kl_drift", "hidden_drift")
TOL = 0.01

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="")
p.add_argument("--out", required=True)
p.add_argument("--tokens", default="2e8")
p.add_argument("--preset", default="looped-27m")
p.add_argument("--skip_pretrain", action="store_true")
p.add_argument("--epochs", default="8")
p.add_argument("--ds", default="geometric", help="deep supervision for fine-tuning (plan: geometric, lambda~0.3)")
p.add_argument("--precision", default="auto")
p.add_argument("--with_dense", action="store_true", help="also pretrain + fine-tune dense-6 on the same tokens")
p.add_argument("--dense_dir", default="", help="existing dir with dense ft_official/ft_heldout runs")
a, extra = p.parse_known_args()
os.makedirs(a.out, exist_ok=True)


def pretrain_if_needed(out, preset):
    final = os.path.join(out, "final.pt")
    if not os.path.exists(final):
        pretrain.main(["--data_dir", a.data_dir, "--out", out, "--preset", preset, "--tokens", a.tokens,
                       "--precision", a.precision, "--resume"] + extra)
    if not os.path.exists(final):
        sys.exit(f"\nPretraining stopped before the token budget (time limit or interruption); no {final} yet.\n"
                 "Re-run the exact same command: it resumes from the latest checkpoint.")
    return final


def finetune_eval(run_root, init, preset):
    res = {}
    for setup in ("official", "heldout"):
        run = os.path.join(run_root, f"ft_{setup}")
        rj = os.path.join(run, "eval", "results.json")
        if not os.path.exists(rj):
            args = ["--out", run, "--setup", setup, "--epochs", a.epochs, "--ds", a.ds, "--precision", a.precision]
            args += (["--init", init, "--tokenizer", os.path.join(a.data_dir, "tokenizer.json")] if init
                     else ["--preset", preset])
            finetune.main(args)
            evaluate.main(["--run", run])
        res[setup] = json.load(open(rj))
    return res


init = None if a.skip_pretrain else pretrain_if_needed(os.path.join(a.out, "pretrain"), a.preset)
looped = finetune_eval(a.out, init, a.preset)

dense = None
if a.with_dense:
    dinit = None if a.skip_pretrain else pretrain_if_needed(os.path.join(a.out, "dense6", "pretrain"), "dense-6")
    dense = finetune_eval(os.path.join(a.out, "dense6"), dinit, "dense-6")
elif a.dense_dir:
    dense = {s: json.load(open(os.path.join(a.dense_dir, f"ft_{s}", "eval", "results.json"))) for s in ("official", "heldout")}

report, go_all, strong_all, reversed_any = {}, True, dense is not None, False
for setup, R in looped.items():
    conv = {k: R["ood"][k]["auroc"] for k in CONV if k in R["ood"]}
    best_name = max(conv, key=conv.get)
    best, msp = conv[best_name], R["ood"]["msp"]["auroc"]
    more_loops = R["unknown_loops_mean"] > R["in_scope_loops_mean"]
    go = best >= msp - TOL and more_loops
    row = dict(in_scope_acc=R.get("acc_at_cap"), best_convergence=best_name, best_convergence_auroc=round(best, 4),
               looped_msp_auroc=round(msp, 4), combined_auroc=round(R["ood"].get("combined_msp+late_kl_drift", {}).get("auroc", float("nan")), 4),
               loops_in_scope=round(R["in_scope_loops_mean"], 2), loops_unknown=round(R["unknown_loops_mean"], 2),
               go=go)
    if dense is not None:
        D = dense[setup]["ood"]
        dbest = max(D["msp"]["auroc"], D.get("energy", {"auroc": 0})["auroc"])
        row.update(dense_best_auroc=round(dbest, 4), dense_acc=dense[setup].get("acc_at_cap"))
        row["strong_go"] = go and max(best, row["combined_auroc"]) >= dbest - TOL
        strong_all &= row["strong_go"]
    reversed_any |= not more_loops
    go_all &= go
    report[setup] = row

if go_all and strong_all:
    verdict = "STRONG GO: convergence matches/beats max-softmax in the looped model AND the dense baseline"
elif go_all:
    verdict = "GO: convergence matches/beats max-softmax within the looped model; unknowns use more loops" + (
        "" if dense is not None else " (run --with_dense to test strong go)")
elif reversed_any:
    verdict = "NO-GO (reversed direction): unknowns settle faster - pivot to the negative-result write-up or retrofit"
else:
    verdict = "NO-GO: convergence below max-softmax - pivot to negative result or retrofit recurrence"
report["VERDICT"] = verdict
print(json.dumps(report, indent=2))
json.dump(report, open(os.path.join(a.out, "kill_test_verdict.json"), "w"), indent=2)
