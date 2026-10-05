---
license: apache-2.0
tags: [looped-transformer, recurrent-depth, adaptive-computation, out-of-distribution, intent-classification]
datasets: [HuggingFaceFW/fineweb-edu, clinc_oos]
---
# loopthink-27m

A ~27M-parameter looped (recurrent-depth) transformer that **decides how long to think** and
**flags queries that don't fit any intent it knows**.

- Architecture: tied 16k embedding → 2-layer prelude → 2-layer core looped *r* times with input
  injection → 2-layer coda. Intent decisions by cosine score of a pooled query vector against
  cached intent keys (initialised from label-text encodings).
- Pretraining: FineWeb-Edu, 1B tokens, random r ~ log-normal(mean 6, clipped 1–16), truncated BPTT (4), Muon + WSD, fp16.
- Fine-tuning: CLINC150, random r, deep supervision over intermediate loops.
- Inference: stop once the top intent is stable for 2 loops and its margin > τ; if it never stabilises
  within N loops, the answer is **"I don't know this"**.

## Results
See `eval/summary.md` in this repo (filled in by `loopthink.evaluate`).

## Intended use & limits
Research/demo. Intent coverage = CLINC150's 150 intents (10 domains). Not a production
classifier. Out-of-scope detection quality depends on how different unknown queries are from
training intents; near-OOS (semantically similar held-out intents) is much harder than far-OOS.

## Usage
```python
from loopthink.engine import AdaptiveClassifier
clf = AdaptiveClassifier.from_run("path/to/this/repo")
d = clf.classify(["is my flight to denver on time"], max_loops=16, tau=0.5)[0]
print(d.intent, d.loops_used, d.converged)
```
