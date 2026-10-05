# Part 2 kill test: small contrastive decision model

Frozen backbone `sentence-transformers/all-MiniLM-L6-v2`; trained parts are two projection heads + a `none` vector. Mean ± std over seeds [0, 1, 2].

`unknown AUROC` uses each model's own unknown score: `none` probability for the CLM, max-softmax otherwise. The two extra AUROC columns show the CLM's other possible scores, for transparency.

| dataset | unknowns | model | in-scope acc % | unknown AUROC | FPR@95 % | AUROC max-softmax | AUROC best-match | new intents by text only, among all % | among new only % |
|---|---|---|---|---|---|---|---|---|---|
| CLINC150 | held-out intents | zero-shot | 70.0 ± 0.0 | 0.773 ± 0.000 | 66.2 ± 0.0 | – | – | 72.4 ± 0.0 | 85.4 ± 0.0 |
| CLINC150 | held-out intents | linear probe | 96.1 ± 0.1 | 0.892 ± 0.000 | 40.5 ± 0.4 | – | – | – | – |
| CLINC150 | held-out intents | CLM no-none | 96.1 ± 0.1 | 0.908 ± 0.000 | 35.9 ± 0.6 | 0.908 ± 0.000 | 0.901 ± 0.000 | 51.1 ± 0.4 | 85.9 ± 0.3 |
| CLINC150 | held-out intents | CLM | 96.2 ± 0.1 | 0.903 ± 0.001 | 35.4 ± 1.1 | 0.908 ± 0.002 | 0.905 ± 0.001 | 55.8 ± 0.5 | 87.8 ± 0.5 |
| CLINC150 | held-out intents | fine-tuned (Part 1) | 96.6 ± 0.1 | 0.923 ± 0.005 | 33.2 ± 6.4 | – | – | – | – |
| CLINC150 | far-OOS | zero-shot | 68.4 ± 0.0 | 0.902 ± 0.000 | 40.5 ± 0.0 | – | – | – | – |
| CLINC150 | far-OOS | linear probe | 95.2 ± 0.1 | 0.943 ± 0.001 | 23.7 ± 1.1 | – | – | – | – |
| CLINC150 | far-OOS | CLM no-none | 95.1 ± 0.1 | 0.949 ± 0.000 | 16.6 ± 0.5 | 0.949 ± 0.000 | 0.967 ± 0.000 | – | – |
| CLINC150 | far-OOS | CLM | 95.3 ± 0.1 | 0.955 ± 0.000 | 15.3 ± 0.7 | 0.951 ± 0.000 | 0.967 ± 0.000 | – | – |
| CLINC150 | far-OOS | fine-tuned (Part 1) | 95.9 ± 0.1 | 0.973 ± 0.002 | 12.1 ± 0.3 | – | – | – | – |
| HWU64 | held-out intents | zero-shot | 61.3 ± 0.0 | 0.728 ± 0.000 | 80.8 ± 0.0 | – | – | 43.4 ± 0.0 | 73.0 ± 0.0 |
| HWU64 | held-out intents | linear probe | 90.9 ± 0.2 | 0.851 ± 0.002 | 41.9 ± 1.2 | – | – | – | – |
| HWU64 | held-out intents | CLM no-none | 89.6 ± 0.1 | 0.848 ± 0.003 | 41.3 ± 0.6 | 0.848 ± 0.003 | 0.836 ± 0.003 | 29.1 ± 0.9 | 61.4 ± 1.5 |
| HWU64 | held-out intents | CLM | 90.6 ± 0.1 | 0.853 ± 0.001 | 45.6 ± 0.3 | 0.853 ± 0.001 | 0.858 ± 0.001 | 35.5 ± 0.8 | 62.8 ± 0.0 |
| HWU64 | held-out intents | fine-tuned (Part 1) | 92.1 ± 0.1 | 0.879 ± 0.007 | 42.7 ± 6.6 | – | – | – | – |
| HWU64 | far-OOS | zero-shot | 54.0 ± 0.0 | 0.804 ± 0.000 | 57.3 ± 0.0 | – | – | – | – |
| HWU64 | far-OOS | linear probe | 90.0 ± 0.4 | 0.852 ± 0.002 | 53.5 ± 0.4 | – | – | – | – |
| HWU64 | far-OOS | CLM no-none | 88.5 ± 0.3 | 0.843 ± 0.001 | 57.8 ± 0.4 | 0.843 ± 0.001 | 0.896 ± 0.001 | – | – |
| HWU64 | far-OOS | CLM | 89.7 ± 0.2 | 0.848 ± 0.003 | 56.5 ± 2.3 | 0.861 ± 0.001 | 0.895 ± 0.001 | – | – |
| HWU64 | far-OOS | fine-tuned (Part 1) | 91.6 ± 0.2 | 0.895 ± 0.002 | 40.9 ± 3.3 | – | – | – | – |

## Verdict

- CLINC150 held-out intents: CLM vs fine-tuned classifier, unknown AUROC -0.021 (seed noise ~0.005), in-scope accuracy -0.3 points.
- HWU64 held-out intents: CLM vs fine-tuned classifier, unknown AUROC -0.027 (seed noise ~0.007), in-scope accuracy -1.5 points.

**NO-GO as specified**: the CLM does not clearly beat the fine-tuned classifier at spotting held-out intents on both datasets.
