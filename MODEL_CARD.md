---
license: mit
tags: [looped-transformer, recurrent-depth, intent-classification, out-of-distribution, negative-result]
datasets: [HuggingFaceFW/fineweb-edu, clinc_oos]
---
# loopthink looped transformer (28.8M)

A looped (recurrent-depth) transformer trained from scratch, used in this repository to test one
hypothesis: that a query whose answer never settles across loops is one the model does not know.
**The hypothesis did not hold.** This card describes the model as it is, not as it was hoped to be.
The model the router actually deploys is a fine-tuned MiniLM; see the [README](README.md).

- Architecture: tied 16k embedding, 2-layer prelude, 2-layer core looped *r* times with input
  injection, 2-layer coda. Intent scores are the cosine of a pooled query vector against intent keys.
- Pretraining: 200M tokens of FineWeb-Edu on a laptop (about 10 hours), random loop count 1-16,
  truncated backpropagation through the last 4 loops, Muon + AdamW, fp16.
- Fine-tuning: CLINC150, 8 epochs, random loop count.

## Results (CLINC150, one seed)

| | Far out-of-scope | Held-out intents |
|---|---|---|
| In-scope accuracy | 93.8% | 94.5% |
| Unknown detection AUROC, max-softmax | 0.943 | 0.904 |
| Unknown detection AUROC, convergence (late KL drift) | 0.544 | 0.516 |

Accuracy is the same from 1 to 32 loops: after pretraining the model reaches its final answer at
loop 1. Convergence therefore carries no information beyond ordinary confidence. Three
deep-supervision settings gave the same outcome. Full numbers: [results/REPORT.md](results/REPORT.md).

## Intended use and limits

Research only. One seed, one dataset, 150 intents. Not a production classifier. A fine-tuned MiniLM
of similar size is more accurate (95.9%) and trains in two minutes.

## Usage

```python
from loopthink.engine import AdaptiveClassifier
clf = AdaptiveClassifier.from_run("path/to/run")      # folder with model.pt + tokenizer.json
d = clf.classify(["is my flight to denver on time"], max_loops=16, tau=0.5)[0]
print(d.intent, d.loops_used, d.converged)
```
