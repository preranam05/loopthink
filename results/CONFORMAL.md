# Conformal decision router

The classifier returns a set of intents guaranteed to contain the right one with probability ≥ 1 − α. One label → answer; several → escalate with that set as the shortlist; none → out of scope.
Mean ± std over seeds. Unknown traffic share for system numbers: 5%.

## 1. Does the guarantee hold on in-scope queries?

Target coverage is 1 − α. `val` = calibrated on validation, measured on test. `split` = 200 random calibration/test halves of the test set, mean [5th–95th percentile].

| model | dataset | α | target % | coverage (val) % | coverage (split) % | avg set size | 90th pct size |
|---|---|---|---|---|---|---|---|
| MiniLM-L6 (22.8M) | CLINC150 | 0.01 | 99 | 99.0 ± 0.1 | 99.0 [98.5–99.5] | 1.62 ± 0.10 | 2.0 |
| MiniLM-L6 (22.8M) | CLINC150 | 0.02 | 98 | 97.9 ± 0.1 | 98.0 [97.3–98.7] | 1.09 ± 0.01 | 1.0 |
| MiniLM-L6 (22.8M) | CLINC150 | 0.05 | 95 | 94.7 ± 0.1 | 95.0 [93.9–96.0] | 0.97 ± 0.00 | 1.0 |
| MiniLM-L6 (22.8M) | CLINC150 | 0.1 | 90 | 90.0 ± 0.4 | 90.1 [88.5–91.5] | 0.91 ± 0.00 | 1.0 |
| MiniLM-L6 (22.8M) | HWU64 | 0.01 | 99 | 99.2 ± 0.2 | 99.1 [98.0–99.8] | 3.75 ± 0.62 | 7.7 |
| MiniLM-L6 (22.8M) | HWU64 | 0.02 | 98 | 97.8 ± 0.4 | 98.1 [96.6–99.3] | 1.80 ± 0.11 | 3.0 |
| MiniLM-L6 (22.8M) | HWU64 | 0.05 | 95 | 94.6 ± 0.3 | 95.2 [92.8–97.0] | 1.12 ± 0.01 | 1.2 |
| MiniLM-L6 (22.8M) | HWU64 | 0.1 | 90 | 88.8 ± 0.5 | 90.3 [87.4–93.1] | 0.94 ± 0.01 | 1.0 |
| looped (28.8M, ours) | CLINC150 | 0.01 | 99 | 98.9 | 99.0 [98.6–99.4] | 2.88 | 5.0 |
| looped (28.8M, ours) | CLINC150 | 0.02 | 98 | 97.9 | 98.0 [97.3–98.8] | 1.50 | 2.0 |
| looped (28.8M, ours) | CLINC150 | 0.05 | 95 | 94.5 | 95.0 [93.8–96.0] | 1.02 | 1.0 |
| looped (28.8M, ours) | CLINC150 | 0.1 | 90 | 89.1 | 89.9 [88.3–91.4] | 0.92 | 1.0 |

## 2. What the router does with each query

In-scope queries: answered directly (one label), escalated (several), or rejected (none). `direct acc` = accuracy of the directly answered ones; `truth in shortlist` = how often an escalated set contains the answer.

| model | dataset | unknowns | α | answered directly % | direct acc % | escalated % | truth in shortlist % | rejected % |
|---|---|---|---|---|---|---|---|---|
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.01 | 83.8 ± 1.3 | 99.5 ± 0.0 | 16.2 ± 1.3 | 96.8 ± 0.8 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.02 | 93.7 ± 0.4 | 98.6 ± 0.2 | 6.3 ± 0.4 | 87.1 ± 0.7 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.05 | 97.2 ± 0.5 | 97.5 ± 0.3 | 0.0 ± 0.0 | nan ± nan | 2.8 ± 0.5 |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.1 | 90.9 ± 0.4 | 99.0 ± 0.1 | 0.0 ± 0.0 | nan ± nan | 9.1 ± 0.4 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.01 | 86.3 ± 1.4 | 99.5 ± 0.0 | 13.7 ± 1.4 | 97.7 ± 0.4 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.02 | 96.2 ± 0.7 | 98.2 ± 0.3 | 3.8 ± 0.7 | 85.2 ± 2.9 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.05 | 95.7 ± 0.2 | 98.4 ± 0.1 | 0.0 ± 0.0 | nan ± nan | 4.3 ± 0.2 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.1 | 89.6 ± 0.5 | 99.4 ± 0.1 | 0.0 ± 0.0 | nan ± nan | 10.4 ± 0.5 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.01 | 54.7 ± 3.9 | 99.6 ± 0.2 | 45.3 ± 3.9 | 98.8 ± 0.2 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.02 | 73.0 ± 2.0 | 98.8 ± 0.2 | 27.0 ± 2.0 | 95.1 ± 0.8 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.05 | 90.2 ± 0.3 | 95.5 ± 0.5 | 9.8 ± 0.3 | 87.1 ± 1.9 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.1 | 94.2 ± 1.0 | 94.3 ± 0.9 | 0.0 ± 0.0 | nan ± nan | 5.8 ± 1.0 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.01 | 67.1 ± 3.2 | 99.5 ± 0.3 | 32.9 ± 3.2 | 97.0 ± 0.8 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.02 | 79.8 ± 2.7 | 98.4 ± 0.3 | 20.2 ± 2.7 | 94.6 ± 1.1 | 0.0 ± 0.0 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.05 | 93.1 ± 0.9 | 95.3 ± 0.2 | 6.9 ± 0.8 | 77.0 ± 3.3 | 0.0 ± 0.1 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.1 | 91.8 ± 0.3 | 96.2 ± 0.4 | 0.0 ± 0.0 | nan ± nan | 8.2 ± 0.3 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.01 | 69.4 | 99.4 | 30.6 | 98.0 | 0.0 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.02 | 82.8 | 98.8 | 17.2 | 93.3 | 0.0 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.05 | 97.3 | 95.0 | 2.2 | 93.1 | 0.4 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.1 | 91.6 | 97.2 | 0.0 | nan | 8.4 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.01 | 78.1 | 99.4 | 21.9 | 97.1 | 0.0 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.02 | 86.7 | 98.8 | 13.3 | 92.3 | 0.0 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.05 | 97.9 | 95.8 | 0.0 | nan | 2.1 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.1 | 89.7 | 98.5 | 0.0 | nan | 10.3 |

## 3. Where the guarantee breaks: unknown intents

The guarantee covers in-scope queries only. For unknown queries the right outcome is *rejected* or *escalated*; a single confident label is an error the guarantee does not cover.

| model | dataset | unknowns | α | rejected % | escalated % | **confidently wrong %** |
|---|---|---|---|---|---|---|
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.01 | 0.0 ± 0.0 | 93.4 ± 1.0 | **6.6 ± 1.0** |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.02 | 0.1 ± 0.1 | 72.3 ± 2.7 | **27.6 ± 2.6** |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.05 | 67.6 ± 4.0 | 0.0 ± 0.0 | **32.4 ± 4.0** |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.1 | 90.7 ± 0.1 | 0.0 ± 0.0 | **9.3 ± 0.1** |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.01 | 0.0 ± 0.0 | 82.3 ± 3.8 | **17.7 ± 3.8** |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.02 | 0.3 ± 0.1 | 39.7 ± 7.2 | **60.0 ± 7.1** |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.05 | 52.0 ± 3.6 | 0.0 ± 0.0 | **48.0 ± 3.6** |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.1 | 80.3 ± 2.5 | 0.0 ± 0.0 | **19.7 ± 2.5** |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.01 | 0.0 ± 0.0 | 93.9 ± 1.0 | **6.1 ± 1.0** |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.02 | 0.0 ± 0.0 | 84.5 ± 2.7 | **15.5 ± 2.7** |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.05 | 0.0 ± 0.1 | 48.6 ± 1.9 | **51.3 ± 1.9** |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.1 | 44.2 ± 4.0 | 0.0 ± 0.0 | **55.8 ± 4.0** |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.01 | 0.0 ± 0.0 | 90.0 ± 1.5 | **10.0 ± 1.5** |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.02 | 0.0 ± 0.0 | 78.3 ± 4.6 | **21.7 ± 4.6** |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.05 | 0.6 ± 0.8 | 33.6 ± 0.6 | **65.8 ± 1.1** |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.1 | 45.0 ± 1.3 | 0.0 ± 0.0 | **55.0 ± 1.3** |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.01 | 0.0 | 96.2 | **3.8** |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.02 | 0.0 | 90.6 | **9.4** |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.05 | 20.5 | 7.2 | **72.3** |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.1 | 80.1 | 0.0 | **19.9** |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.01 | 0.0 | 90.2 | **9.8** |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.02 | 0.0 | 75.7 | **24.3** |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.05 | 22.1 | 0.1 | **77.8** |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.1 | 66.8 | 0.0 | **33.2** |

## 4. System error with a perfect LLM, versus a confidence threshold at the same LLM budget

`in-scope error` should be ≤ α. `system error` adds unknown traffic. `threshold router` escalates the least-confident queries, using the same share of LLM calls.

| model | dataset | unknowns | α | in-scope error % | system error % | sent to LLM % | threshold router error % |
|---|---|---|---|---|---|---|---|
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.01 | 1.0 ± 0.1 | 1.2 ± 0.1 | 20.1 ± 1.3 | 0.7 ± 0.1 |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.02 | 2.1 ± 0.1 | 3.4 ± 0.3 | 9.6 ± 0.5 | 2.3 ± 0.3 |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.05 | 5.3 ± 0.1 | 6.6 ± 0.1 | 0.0 ± 0.0 | 8.9 ± 0.1 |
| MiniLM-L6 (22.8M) | CLINC150 | far-OOS | 0.1 | 10.0 ± 0.4 | 10.0 ± 0.4 | 0.0 ± 0.0 | 8.9 ± 0.1 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.01 | 0.8 ± 0.1 | 1.6 ± 0.2 | 17.2 ± 1.5 | 1.1 ± 0.1 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.02 | 2.3 ± 0.3 | 5.2 ± 0.6 | 5.6 ± 1.1 | 4.4 ± 0.6 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.05 | 5.9 ± 0.2 | 8.0 ± 0.1 | 0.0 ± 0.0 | 8.3 ± 0.1 |
| MiniLM-L6 (22.8M) | CLINC150 | held-out intents | 0.1 | 10.9 ± 0.5 | 11.4 ± 0.4 | 0.0 ± 0.0 | 8.3 ± 0.1 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.01 | 0.8 ± 0.2 | 1.0 ± 0.2 | 47.7 ± 3.8 | 0.5 ± 0.2 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.02 | 2.2 ± 0.4 | 2.9 ± 0.3 | 29.9 ± 2.1 | 1.4 ± 0.2 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.05 | 5.4 ± 0.3 | 7.7 ± 0.3 | 11.7 ± 0.3 | 6.0 ± 0.4 |
| MiniLM-L6 (22.8M) | HWU64 | far-OOS | 0.1 | 11.2 ± 0.5 | 13.4 ± 0.4 | 0.0 ± 0.0 | 13.0 ± 0.2 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.01 | 1.3 ± 0.2 | 1.7 ± 0.2 | 35.7 ± 3.1 | 0.7 ± 0.1 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.02 | 2.4 ± 0.1 | 3.3 ± 0.3 | 23.1 ± 2.8 | 2.2 ± 0.2 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.05 | 6.0 ± 0.2 | 9.0 ± 0.2 | 8.2 ± 0.8 | 7.1 ± 0.5 |
| MiniLM-L6 (22.8M) | HWU64 | held-out intents | 0.1 | 11.6 ± 0.6 | 13.8 ± 0.6 | 0.0 ± 0.0 | 12.5 ± 0.1 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.01 | 1.1 | 1.2 | 33.8 | 0.6 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.02 | 2.1 | 2.5 | 20.8 | 1.3 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.05 | 5.5 | 8.8 | 2.5 | 8.6 |
| looped (28.8M, ours) | CLINC150 | far-OOS | 0.1 | 10.9 | 11.4 | 0.0 | 10.9 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.01 | 1.1 | 1.5 | 25.4 | 0.8 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.02 | 2.0 | 3.1 | 16.4 | 2.0 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.05 | 6.2 | 9.7 | 0.0 | 10.2 |
| looped (28.8M, ours) | CLINC150 | held-out intents | 0.1 | 11.6 | 12.7 | 0.0 | 10.2 |

## Caveats

- A perfect LLM is assumed in section 4; real LLM errors on escalated sets add to these numbers.
- `val` calibration is approximate: the checkpoint was selected on validation accuracy. `split` is the clean check.
- Coverage is marginal (averaged over queries), not per intent.
