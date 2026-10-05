# Results

Mean ± std over seeds, [95% bootstrap CI pooled over seeds and test queries]. Answer read after loop 1. Cascade: threshold set on in-scope validation only (escalate 10%), traffic reweighted to 5% unknown.

## 1. Classifier quality

| model | dataset | unknowns | seeds | In-scope acc % | OOD AUROC (msp) | OOD AUROC (energy) | FPR@95 % |
|---|---|---|---|---|---|---|---|
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 3 | 95.9 ± 0.1 [95.3–96.5] | 0.965 ± 0.002 [0.959–0.971] | 0.973 ± 0.002 [0.967–0.977] | 14.1 ± 1.0 [11.1–16.4] |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 3 | 96.6 ± 0.1 [96.0–97.1] | 0.920 ± 0.006 [0.905–0.931] | 0.923 ± 0.005 [0.910–0.934] | 34.2 ± 5.6 [25.6–47.4] |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 3 | 91.6 ± 0.2 [89.9–93.3] | 0.879 ± 0.002 [0.863–0.894] | 0.895 ± 0.002 [0.880–0.909] | 43.1 ± 1.9 [37.0–49.9] |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 3 | 92.1 ± 0.1 [90.1–93.9] | 0.871 ± 0.004 [0.848–0.892] | 0.879 ± 0.007 [0.853–0.902] | 42.9 ± 3.4 [30.7–52.9] |
| looped (28.8M, ours) | CLINC150 | far-OOS | 1 | 93.8 [93.1–94.5] | 0.943 [0.936–0.949] | 0.947 [0.941–0.954] | 21.2 [17.8–25.4] |
| looped (28.8M, ours) | CLINC150 | held-out intents | 1 | 94.5 [93.8–95.2] | 0.904 [0.894–0.913] | 0.892 [0.881–0.902] | 28.3 [24.8–36.9] |

## 2. Cascade with a perfect LLM (upper bound)

| model | dataset | unknowns | small model alone % | cascade % | sent to LLM % | unknowns caught % |
|---|---|---|---|---|---|---|
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 91.1 ± 0.1 [90.5–91.6] | 98.8 ± 0.0 [98.5–99.1] | 14.2 ± 0.4 [13.3–15.4] | 92.1 ± 0.1 [90.4–93.7] |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 91.7 ± 0.1 [91.2–92.3] | 98.6 ± 0.1 [98.3–98.9] | 14.6 ± 0.5 [13.4–15.8] | 81.6 ± 2.5 [77.3–85.9] |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 87.0 ± 0.2 [85.4–88.6] | 94.9 ± 0.2 [93.7–95.9] | 14.1 ± 0.1 [12.0–15.8] | 64.5 ± 2.2 [60.4–68.9] |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 87.5 ± 0.1 [85.6–89.2] | 95.5 ± 0.2 [94.4–96.6] | 13.7 ± 0.2 [11.7–15.8] | 59.1 ± 0.0 [53.3–64.6] |
| looped (28.8M, ours) | CLINC150 | far-OOS | 89.1 [88.5–89.8] | 97.2 [96.8–97.6] | 13.7 [12.8–14.5] | 84.0 [81.6–86.3] |
| looped (28.8M, ours) | CLINC150 | held-out intents | 89.8 [89.1–90.5] | 97.6 [97.2–97.9] | 14.7 [13.7–15.6] | 71.7 [68.8–74.3] |

## 3. Measured cascade with a real LLM

| model | dataset | unknowns | LLM (mode) | cascade % [CI] | LLM alone % [CI] | to LLM | mean latency cascade / LLM-only | ref $/1k cascade / LLM-only |
|---|---|---|---|---|---|---|---|---|
| looped (28.8M, ours) | CLINC150 | far-OOS | qwen2.5_7b_top5 (top-5 shortlist) | 92.7 [91.9–93.3] | n/a (needs shortlist) | 13.7% | 66 / – ms | 0.0042 / – |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | qwen2.5_7b (all intents) | 92.9 [92.0–93.6] | 78.7 [71.8–85.0] | 14.7% | 72 / 455 ms | 0.0145 / 0.0987 |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | qwen2.5_7b_top5 (top-5 shortlist) | 93.6 [92.9–94.4] | n/a (needs shortlist) | 14.7% | 71 / – ms | 0.0044 / – |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | qwen2.5_7b_top5_ex3 (top-5 shortlist + examples) | 95.2 [94.6–95.9] | n/a (needs shortlist) | 14.7% | 113 / – ms | 0.0122 / – |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | groq_openai_gpt-oss-120b (all intents) | 89.7 [87.8–91.4] | 84.6 [78.6–89.9] | 13.5% | 109 / 771 ms | 0.0178 / 0.1317 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | groq_openai_gpt-oss-120b_top5 (top-5 shortlist) | 89.1 [87.4–90.7] | n/a (needs shortlist) | 13.5% | 110 / – ms | 0.0095 / – |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | qwen2.5_7b (all intents) | 88.5 [86.8–90.4] | 81.5 [75.8–87.1] | 13.5% | 66 / 449 ms | 0.0084 / 0.0624 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | qwen2.5_7b_top5 (top-5 shortlist) | 87.9 [86.1–89.8] | n/a (needs shortlist) | 13.5% | 68 / – ms | 0.0041 / – |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | qwen2.5_7b_top5_ex3 (top-5 shortlist + examples) | 89.8 [88.0–91.6] | n/a (needs shortlist) | 13.5% | 98 / – ms | 0.0088 / – |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | qwen2.5_7b (all intents) | 87.6 [85.8–89.2] | 72.1 [65.2–78.8] | 14.1% | 71 / 465 ms | 0.0088 / 0.0620 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | qwen2.5_7b_top5 (top-5 shortlist) | 87.9 [86.2–89.4] | n/a (needs shortlist) | 14.1% | 71 / – ms | 0.0043 / – |

### Sensitivity to the share of unknown traffic (measured cascade, point estimates)

| run | LLM | 2% | 5% | 10% | 18% |
|---|---|---|---|---|---|
| runs/kill/ft_official | qwen2.5_7b_top5 | 93.2 | 92.7 | 91.8 | 90.5 |
| runs/minilm_clinc_official_s0 | qwen2.5_7b | 93.5 | 92.9 | 91.8 | 90.2 |
| ↳ LLM alone | qwen2.5_7b | 78.7 | 78.7 | 78.7 | 78.7 |
| runs/minilm_clinc_official_s0 | qwen2.5_7b_top5 | 94.0 | 93.6 | 93.0 | 92.0 |
| runs/minilm_clinc_official_s0 | qwen2.5_7b_top5_ex3 | 95.7 | 95.2 | 94.5 | 93.4 |
| runs/minilm_hwu_heldout_s0 | groq_openai_gpt-oss-120b | 91.0 | 89.7 | 87.7 | 84.4 |
| ↳ LLM alone | groq_openai_gpt-oss-120b | 84.7 | 84.6 | 84.6 | 84.5 |
| runs/minilm_hwu_heldout_s0 | groq_openai_gpt-oss-120b_top5 | 90.8 | 89.1 | 86.1 | 81.4 |
| runs/minilm_hwu_heldout_s0 | qwen2.5_7b | 89.9 | 88.5 | 86.1 | 82.3 |
| ↳ LLM alone | qwen2.5_7b | 81.8 | 81.5 | 81.0 | 80.2 |
| runs/minilm_hwu_heldout_s0 | qwen2.5_7b_top5 | 89.9 | 87.9 | 84.7 | 79.5 |
| runs/minilm_hwu_heldout_s0 | qwen2.5_7b_top5_ex3 | 91.7 | 89.8 | 86.7 | 81.8 |
| runs/minilm_hwu_official_s0 | qwen2.5_7b | 89.2 | 87.6 | 84.9 | 80.6 |
| ↳ LLM alone | qwen2.5_7b | 72.8 | 72.1 | 70.8 | 68.8 |
| runs/minilm_hwu_official_s0 | qwen2.5_7b_top5 | 89.1 | 87.9 | 85.9 | 82.6 |

## Caveats

- Unknown share of real traffic is an assumption; see the sensitivity table.
- Cost is priced at reference API rates for the tokens used ($0.15 / $0.60 per 1M in/out); the local LLM itself is free.
- Measured-LLM CIs resample the ~300 sampled LLM calls per group; LLM runs use seed 0 of the classifier only.
- HWU64 far-OOS unknowns are CLINC150's out-of-scope queries (not home-assistant requests); held-out intents are the harder test.
- The looped model has one seed per setup; its pretraining took ~10 h, so extra seeds were not run.
