"""Send the queries the cascade would escalate to a real LLM (Ollama, local) and measure what it
actually gets right, how long it takes and how many tokens it uses. Replaces the simulated LLM in
scripts/cascade.py with measured numbers.

  ollama pull qwen2.5:3b                      # once (~2 GB)
  python scripts/llm_validate.py --run runs/kill/ft_official --model qwen2.5:3b
  python scripts/llm_validate.py --run runs/kill/ft_heldout  --model qwen2.5:3b

What it measures (all at a *deployable* threshold, set on in-scope validation only):
  - LLM accuracy on escalated in-scope queries and on escalated unknowns
  - LLM accuracy on a random sample of all traffic  -> the LLM-only baseline
  - p50 / p95 latency and prompt/output tokens per call
then combines them with the small model's exact results into measured system accuracy,
% escalated and cost for "small model + LLM" vs "LLM only".

Ground truth for the LLM: in-scope -> the intent; official OOS -> out_of_scope; held-out intents ->
their true intent (the LLM sees all 150 intents, which is the point of a general fallback).
Resumable: calls are cached in <run>/llm_validate/<model>/calls.jsonl.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from loopthink.data import load_clinc, make_split  # noqa: E402

_spec = importlib.util.spec_from_file_location("cascade", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascade.py"))
cascade = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cascade)

from loopthink.llm import OOS, ExampleIndex, make_asker, ask_ollama, build_shortlist_prompt, build_system_prompt, schema, shortlist_message  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "openai", "groq", "openrouter", "gemini", "custom"],
                    help="hosted providers use an OpenAI-compatible API; key from OPENAI_API_KEY / GROQ_API_KEY / ...")
    ap.add_argument("--host", default=None, help="Ollama URL, or base URL for --provider custom")
    ap.add_argument("--signal", default="msp", help="escalation signal (msp, margin, energy, loops)")
    ap.add_argument("--val_rate", type=float, default=0.10, help="escalate this share of in-scope validation queries")
    ap.add_argument("--unknown_share", type=float, default=0.05)
    ap.add_argument("--n_escalated", type=int, default=300, help="per group (in-scope / unknown) to send")
    ap.add_argument("--n_random", type=int, default=150, help="per group, for the LLM-only baseline")
    ap.add_argument("--price_in", type=float, default=0.15, help="reference $ per 1M input tokens (API-equivalent)")
    ap.add_argument("--price_out", type=float, default=0.60, help="reference $ per 1M output tokens")
    ap.add_argument("--small_ms", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--hint_topk", type=int, default=0,
                    help="0 = LLM sees all intents; K>0 = LLM picks from the small model's top-K + out_of_scope")
    ap.add_argument("--examples", type=int, default=0,
                    help="with --hint_topk: show this many nearest TRAINING examples under each candidate label")
    ap.add_argument("--no_schema", action="store_true", help="plain JSON mode instead of a label-enum schema")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    S, y_val, y_test, R = cascade.load(a.run, 1)
    data = R.get("data", "data")
    raw = load_clinc(data if os.path.isabs(data) else os.path.join(ROOT, data))
    split = make_split(raw, R["setup"], R.get("n_heldout", 30), 0)
    all_intents = sorted({l for _, l in raw["train"]})
    in_texts = [t for t, _ in split.test]
    in_truth = [split.intents[y] for y in y_test]
    unk_texts = split.unknown_test
    unk_truth = ([OOS] * len(unk_texts) if R["setup"] == "official"
                 else [l for t, l in raw["test"] if l in split.heldout_intents])
    assert len(in_texts) == len(y_test) and len(unk_texts) == len(S["unk_test"]["pred"])

    # deployable threshold: fixed on in-scope validation only
    thr = float(np.quantile(S["val"][a.signal], 1 - a.val_rate))
    esc_in = S["test"][a.signal] >= thr
    esc_unk = S["unk_test"][a.signal] >= thr
    small_ok_in = S["test"]["pred"] == y_test

    rng = np.random.default_rng(a.seed)
    pick = lambda idx, k: sorted(rng.choice(idx, min(k, len(idx)), replace=False).tolist()) if len(idx) else []
    jobs = ([("esc_in", i) for i in pick(np.nonzero(esc_in)[0], a.n_escalated)] +
            [("esc_unk", i) for i in pick(np.nonzero(esc_unk)[0], a.n_escalated)] +
            [("rand_in", i) for i in pick(np.arange(len(in_texts)), a.n_random)] +
            [("rand_unk", i) for i in pick(np.arange(len(unk_texts)), a.n_random)])

    tag = (a.provider + "_" if a.provider != "ollama" else "") + a.model.replace(":", "_").replace("/", "_") + (f"_top{a.hint_topk}" if a.hint_topk else "") + (f"_ex{a.examples}" if a.examples else "") + ("_noschema" if a.no_schema else "")
    out_dir = os.path.join(a.run, "llm_validate", tag)
    topk = {}
    if a.hint_topk:
        d = np.load(os.path.join(a.run, "eval", "trajectories.npz"))
        for part, key in (("in", "test_logits"), ("unk", "unk_test_logits")):
            topk[part] = np.argsort(-d[key][:, 0], -1)[:, :a.hint_topk]
        # with a shortlist the LLM can only choose known intents, so an unknown is right iff it says out_of_scope
        unk_truth = [OOS] * len(unk_texts)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "calls.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line); cache[(r["part"], r["i"])] = r
    if a.examples and not a.hint_topk:
        sys.exit("--examples needs --hint_topk")
    ex_index = ExampleIndex(split.train, split.intents) if a.examples else None
    system = build_shortlist_prompt(bool(a.examples)) if a.hint_topk else build_system_prompt(all_intents)
    ask = make_asker(a.provider, a.host)
    if a.provider == "ollama":
        host = a.host or "http://localhost:11434"
        try:
            urllib.request.urlopen(f"{host}/api/tags", timeout=5)
        except Exception as e:
            sys.exit(f"Can't reach Ollama at {host} ({e}). Start it with `ollama serve` (or open the Ollama app).")

    todo, seen = [], set()
    for g, i in jobs:
        key = ("in" if g.endswith("in") else "unk", i)
        if key not in cache and key not in seen:
            seen.add(key); todo.append((g, i))
    print(f"{len(jobs)} queries ({len(jobs) - len(todo)} cached) -> {a.model}. threshold {a.signal} >= {thr:.4f}")
    with open(cache_path, "a") as f:
        for n, (g, i) in enumerate(todo, 1):
            part = "in" if g.endswith("in") else "unk"
            text, truth = (in_texts[i], in_truth[i]) if part == "in" else (unk_texts[i], unk_truth[i])
            try:
                if a.hint_topk:
                    cands = [split.intents[c] for c in topk[part][i]] + [OOS]
                    msg = shortlist_message(text, cands, ex_index.lookup(text, cands, a.examples) if ex_index else None)
                    r = ask(a.model, system, msg, a.timeout, None if a.no_schema else cands)
                else:
                    r = ask(a.model, system, text, a.timeout, None if a.no_schema else all_intents + [OOS])
            except Exception as e:
                print(f"  call failed ({e}); re-run to retry"); continue
            rec = dict(part=part, i=int(i), text=text, truth=truth, **r, correct=r["label"] == truth)
            cache[(part, i)] = rec
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n % 25 == 0 or n == len(todo):
                done = [c for c in cache.values()]
                print(f"  {n}/{len(todo)}  running acc {np.mean([c['correct'] for c in done]):.3f}  "
                      f"p50 {np.median([c['ms'] for c in done]):.0f} ms")

    missing = sum(1 for g, i in jobs if (("in" if g.endswith("in") else "unk"), i) not in cache)
    if missing:
        print(f"\nWARNING: {missing} of {len(jobs)} queries have no LLM answer yet (failed calls). "
              "Re-run the same command to fill them in; numbers below are partial.")

    def acc(group):
        rs = [cache.get(("in" if group.endswith("in") else "unk", i)) for g, i in jobs if g == group]
        rs = [r for r in rs if r]
        return (float(np.mean([r["correct"] for r in rs])) if rs else float("nan")), len(rs)

    a_ein, n_ein = acc("esc_in"); a_eunk, n_eunk = acc("esc_unk")
    a_rin, n_rin = acc("rand_in"); a_runk, n_runk = acc("rand_unk")
    calls = list(cache.values())
    if not calls:
        sys.exit("No LLM calls succeeded - see the errors above, then re-run the same command.")
    ms = np.array([c["ms"] for c in calls]); pt = np.array([c["prompt_tokens"] or 0 for c in calls])
    ot = np.array([c["output_tokens"] or 0 for c in calls])
    # Ollama reports only the non-cached part of the prompt, so bill every call at the largest prompt seen
    # (the first, uncached call). Hosted APIs report the full prompt, so use their mean.
    full_pt = (float(pt.max()) if a.provider == "ollama" else float(pt.mean())) if len(pt) else 0.0
    llm_cost_1k = 1000 * (full_pt * a.price_in + ot.mean() * a.price_out) / 1e6
    ms_typ = float(ms[ms <= 5 * np.median(ms)].mean())   # mean without rate-limit/retry outliers
    recall = None
    if a.hint_topk:
        ein = [i for g, i in jobs if g == "esc_in"]
        recall = float(np.mean([y_test[i] in topk["in"][i] for i in ein])) if ein else None

    w_unk = a.unknown_share or len(unk_texts) / (len(unk_texts) + len(in_texts)); w_in = 1 - w_unk
    e_in, e_unk = float(esc_in.mean()), float(esc_unk.mean())
    kept_in_acc = float(small_ok_in[~esc_in].mean()) if (~esc_in).any() else 0.0
    sys_acc = w_in * ((1 - e_in) * kept_in_acc + e_in * a_ein) + w_unk * (e_unk * a_eunk)
    esc_frac = w_in * e_in + w_unk * e_unk
    llm_only_acc = w_in * a_rin + w_unk * a_runk
    p50, p95 = float(np.percentile(ms, 50)), float(np.percentile(ms, 95))
    res = dict(run=a.run, provider=a.provider, model=a.model, seed=a.seed, signal=a.signal, val_rate=a.val_rate, threshold=thr,
               unknown_share=w_unk, n_calls=len(calls),
               llm=dict(acc_escalated_in_scope=[a_ein, n_ein], acc_escalated_unknown=[a_eunk, n_eunk],
                        acc_random_in_scope=[a_rin, n_rin], acc_random_unknown=[a_runk, n_runk],
                        p50_ms=p50, p95_ms=p95, mean_prompt_tokens=float(pt.mean()),
                        mean_output_tokens=float(ot.mean()), billed_prompt_tokens=full_pt,
                        ref_cost_per_1k_calls=llm_cost_1k, shortlist_k=a.hint_topk, examples_per_label=a.examples,
                        shortlist_recall_escalated_in_scope=recall,
                        invalid_label_rate=float(np.mean([c["label"] not in set(all_intents + [OOS]) for c in calls]))),
               cascade=dict(system_acc=sys_acc, escalated=esc_frac, cost_per_1k=esc_frac * llm_cost_1k,
                            mean_ms=a.small_ms + esc_frac * ms_typ,
                            in_scope_escalated=e_in, unknown_caught=e_unk),
               llm_only=dict(system_acc=llm_only_acc, escalated=1.0, cost_per_1k=llm_cost_1k,
                             mean_ms=ms_typ))
    json.dump(res, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    C, Lo = res["cascade"], res["llm_only"]
    md = "\n".join([
        f"# Measured cascade: {os.path.basename(os.path.normpath(a.run))} + {a.model}"
        + (f" (shortlist top-{a.hint_topk}" + (f", {a.examples} examples/label)" if a.examples else ")") if a.hint_topk else " (all intents)"), "",
        f"Threshold on `{a.signal}` set to escalate {a.val_rate:.0%} of in-scope validation queries; "
        f"traffic reweighted to {w_unk:.0%} unknown. {len(calls)} real LLM calls.", "",
        "| system | accuracy | % to LLM | ref $/1k queries | mean latency |", "|---|---|---|---|---|",
        f"| small model + {a.model} | **{C['system_acc']:.1%}** | {C['escalated']:.1%} | {C['cost_per_1k']:.4f} | {C['mean_ms']:.0f} ms |",
        (f"| {a.model} only | {Lo['system_acc']:.1%} | 100% | {Lo['cost_per_1k']:.4f} | {Lo['mean_ms']:.0f} ms |" if not a.hint_topk else
         f"| _(LLM-only baseline: use the all-intents run; shortlist mode needs the small model)_ | | | | |"), "",
        f"LLM accuracy: escalated in-scope {a_ein:.1%} (n={n_ein}), escalated unknowns {a_eunk:.1%} (n={n_eunk}), "
        f"random in-scope {a_rin:.1%} (n={n_rin}), random unknowns {a_runk:.1%} (n={n_runk}).",
        f"LLM latency p50 {p50:.0f} ms / p95 {p95:.0f} ms; ~{pt.mean():.0f} prompt + {ot.mean():.0f} output tokens per call "
        f"(billed at {full_pt:.0f} prompt tokens/call; ref price ${a.price_in}/${a.price_out} per 1M in/out).",
        f"Invalid labels: {res['llm']['invalid_label_rate']:.1%}."
        + (f" Shortlist top-{a.hint_topk} contains the true intent for {recall:.1%} of escalated in-scope queries "
           "(ceiling for the LLM on those)." if recall is not None else ""), ""])
    open(os.path.join(out_dir, "summary.md"), "w").write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
