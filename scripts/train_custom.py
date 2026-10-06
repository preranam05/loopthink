"""Train and evaluate the router on your own intents.

  python scripts/train_custom.py --data my_requests.csv --out runs/mine
  python scripts/train_custom.py --data examples/helpdesk/requests.csv --unknown examples/helpdesk/unknown.csv --out runs/helpdesk

--data     CSV with a header row and two columns: the request text (text/request/query/...) and its
           intent (intent/label/category). At least 10 examples per intent; 30 or more is better.
--unknown  optional CSV of requests the router should decline (one per row). With it, the report also
           says how many unknown requests each threshold would send to the LLM.

Writes the model, a REPORT.md with accuracy on held-out examples, and prints how to serve it.
Training is the same code path as the benchmark runs (scripts/encoder_baseline.py).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from loopthink import custom  # noqa: E402


def auto_epochs(n_train, n_intents):
    """Small datasets need more passes: aim for roughly the number of updates the benchmark runs got per intent."""
    per = n_train / max(1, n_intents)
    return int(min(20, max(6, round(300 / max(per, 1)))))


def softmax(x):
    z = x - x.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def build_report(name, pairs, notes, warns, splits, unknown, intents, P_test, y_test, P_unk, presets, epochs, base_model):
    pred, conf = P_test.argmax(-1), P_test.max(-1)
    correct = (pred == y_test).astype(int).tolist()
    acc = float(np.mean(correct)); lo, hi = custom.bootstrap_ci(correct)
    rows, confusions = custom.per_intent(y_test.tolist(), pred.tolist(), intents)
    names = {"0.05": "Cheaper", "0.1": "Balanced", "0.2": "Safer"}
    thr = [(names.get(k, k), v) for k, v in sorted(presets.items(), key=lambda kv: float(kv[0]))]
    routing = custom.routing_table(conf.tolist(), correct, thr)
    unk_conf = P_unk.max(-1) if len(P_unk) else np.zeros(0)
    counts = collections.Counter(l for _, l in pairs)
    L = [f"# Router report: {name}", "",
         f"**{acc:.1%} of held-out requests get the right intent from the small model alone** "
         f"(95% interval {lo:.1%} to {hi:.1%}, {len(correct)} test requests).", "",
         "## Data", "",
         f"- {len(pairs)} labelled requests across {len(intents)} intents ({min(counts.values())} to {max(counts.values())} per intent)",
         f"- Split: {len(splits['train'])} to train, {len(splits['val'])} to set the threshold, {len(splits['test'])} held out for the scores below"]
    if unknown:
        L.append(f"- {len(unknown)} unknown requests supplied; half used here to test declining")
    L += [f"- {n}" for n in notes + warns]
    L += ["", "## What each threshold would do", "",
          "The small model answers when its confidence is above the threshold and sends the rest to the LLM.", "",
          "| Setting | Threshold | Sent to LLM | Small model's accuracy on what it keeps |"
          + (" Unknown requests sent to LLM |" if len(unk_conf) else ""),
          "|---|---|---|---|" + ("---|" if len(unk_conf) else "")]
    for r in routing:
        ka = "n/a (keeps nothing)" if r["kept"] == 0 else f"{r['kept_accuracy']:.1%} of {r['kept']}"
        row = f"| {r['name']} | {r['threshold']:.3f} | {r['escalated']:.0%} | {ka} |"
        if len(unk_conf):
            row += f" {float((unk_conf <= r['threshold']).mean()):.0%} of {len(unk_conf)} |"
        L.append(row)
    if len(unk_conf):
        missed = float((unk_conf > dict(thr).get("Balanced", thr[len(thr) // 2][1])).mean())
        L += ["", f"At the Balanced setting, {missed:.0%} of the unknown requests would get a confident wrong answer "
                  f"from the small model and never reach the LLM. The other {1 - missed:.0%} reach the LLM, which can "
                  "decline them. A higher threshold or more training examples lowers the first number."]
    else:
        L += ["", "No unknown requests were supplied, so this report cannot say how well out-of-scope requests "
                  "are caught. Add `--unknown` with real examples of requests you do not handle."]
    L += ["", "## Weakest intents", "", "| Intent | Test requests | Accuracy |", "|---|---|---|"]
    L += [f"| {i} | {n} | {a:.0%} |" for i, n, a in rows[:6]]
    if confusions:
        L += ["", "Most common mix-ups (true intent, what the model said, times):", ""]
        L += [f"- {t} read as {p}: {n}" for t, p, n in confusions[:5]]
    else:
        L += ["", "No test request was misclassified."]
    L += ["", "## How to read this", "",
          f"- The test set is small ({len(correct)} requests), so the interval above is wide. More labelled data narrows it.",
          "- Scores describe requests that look like the ones in your file. Real traffic is messier, so expect lower numbers until the model has seen real examples.",
          "- Two intents that are mixed up often usually need clearer definitions or more examples, not a bigger model.",
          f"- Trained for {epochs} passes from `{base_model}`. The LLM stage is not scored here; `scripts/llm_validate.py --run <this folder>` measures it with real calls.", ""]
    return "\n".join(L), dict(test_accuracy=acc, ci95=[lo, hi], n_test=len(correct), routing=routing,
                              weakest=[dict(intent=i, n=n, accuracy=a) for i, n, a in rows[:6]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--unknown", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="", help="shown in the report and the console (default: the data file name)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=0, help="default: chosen from the dataset size")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    name = a.name or os.path.splitext(os.path.basename(a.data))[0]
    try:
        pairs, notes = custom.read_examples(a.data)
        warns = custom.check(pairs)
        unknown = custom.read_texts(a.unknown) if a.unknown else []
    except (custom.DataError, OSError) as e:
        sys.exit(f"Cannot use this data: {e}")
    known = {t.lower() for t, _ in pairs}
    unknown = [u for u in unknown if u.lower() not in known]
    splits = custom.split_examples(pairs, seed=a.seed)
    n_int = len({l for _, l in pairs})
    epochs = a.epochs or auto_epochs(len(splits["train"]), n_int)
    print(f"{len(pairs)} requests, {n_int} intents" + (f", {len(unknown)} unknown" if unknown else ""))
    for n in notes + warns:
        print("  note:", n)
    os.makedirs(a.out, exist_ok=True)
    data_path = os.path.join(a.out, "data.json")
    json.dump(custom.to_dataset(splits, unknown, a.seed), open(data_path, "w"))

    spec = importlib.util.spec_from_file_location("encoder_baseline", os.path.join(ROOT, "scripts", "encoder_baseline.py"))
    eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
    eb.main(["--clinc", data_path, "--setup", "official", "--out", a.out, "--save_model", "--model", a.model,
             "--epochs", str(epochs), "--bs", "16" if len(splits["train"]) < 2000 else "64", "--seed", str(a.seed), "--select", "loss"])

    z = np.load(os.path.join(a.out, "eval", "trajectories.npz"))
    cfg_path = os.path.join(a.out, "router.json")
    cfg = json.load(open(cfg_path))
    presets = {k: round(-float(v), 3) for k, v in cfg["thresholds_by_val_rate"].items()}
    P_test, y_test = softmax(z["test_logits"][:, 0, :]), z["y_test"]
    P_unk = softmax(z["unk_test_logits"][:, 0, :]) if z["unk_test_logits"].shape[0] else np.zeros((0, len(cfg["intents"])))
    report, summary = build_report(name, pairs, notes, warns, splits, unknown, cfg["intents"], P_test, y_test,
                                   P_unk, presets, epochs, a.model)
    open(os.path.join(a.out, "REPORT.md"), "w").write(report)

    # requests the console offers as examples: held-out ones the model has not trained on, plus unknowns
    rng = random.Random(a.seed)
    by = {}
    for t, l in splits["test"]:
        by.setdefault(l, t)
    ex = [[t, 0] for t in list(by.values())[:5]]
    unk_test = [t for t, _ in json.load(open(data_path))["oos_test"]]
    ex += [[t, 1] for t in unk_test[:2]]
    sample = [t for t, _ in splits["test"]][:16 if unk_test else 20] + unk_test[2:6]
    rng.shuffle(sample)
    cfg.update(custom=True, dataset=name, ui=dict(examples=ex, sample=sample), report=summary)
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    print("\n" + report)
    print(f"Saved to {a.out}. Report: {os.path.join(a.out, 'REPORT.md')}\n\nServe it:\n"
          f"  ROUTER_DIR={a.out} LLM_MODEL= uvicorn serve.router_api:app --port 8000      (small model only)\n"
          f"  ROUTER_DIR={a.out} uvicorn serve.router_api:app --port 8000                 (with a local Ollama LLM)\n"
          "then open http://localhost:8000")


if __name__ == "__main__":
    main()
