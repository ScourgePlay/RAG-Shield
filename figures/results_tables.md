# RAG-Shield benchmark results

Source: benchmark_results.csv, 432 cases.

## Table 1. Poisons detected, by attack type and check

Counted over the poisons that reached the top three passages.

| Attack | Poisons | Retrieved | Layer 1 | Layer 2 | Layer 3 | Integrity check | Layers 1-3 together | All four checks |
|---|---|---|---|---|---|---|---|---|
| Contradiction | 108 | 107 | 105/107 (98.1%) | 0/107 (0.0%) | 107/107 (100.0%) | 107/107 (100.0%) | 107/107 (100.0%) | 107/107 (100.0%) |
| Invented fact | 108 | 85 | 2/85 (2.4%) | 0/85 (0.0%) | 84/85 (98.8%) | 85/85 (100.0%) | 84/85 (98.8%) | 85/85 (100.0%) |
| Hijacking | 108 | 107 | 106/107 (99.1%) | 107/107 (100.0%) | 107/107 (100.0%) | 107/107 (100.0%) | 107/107 (100.0%) | 107/107 (100.0%) |

## Table 2. False alarms on clean questions

| Check | False alarms |
|---|---|
| Layer 1 | 5/108 (4.6%) |
| Layer 2 | 0/108 (0.0%) |
| Layer 3 | 0/108 (0.0%) |
| Integrity check | 0/108 (0.0%) |
| Layers 1-3 together | 5/108 (4.6%) |
| All four checks | 5/108 (4.6%) |

## Table 3. Layer 3 on the two subject areas it never trained on

| Passages | Flagged by Layer 3 |
|---|---|
| Clean questions (false alarms) | 0/36 (0.0%) |
| Contradiction poisons (detected) | 35/35 (100.0%) |
| Invented fact poisons (detected) | 26/27 (96.3%) |
| Hijacking poisons (detected) | 36/36 (100.0%) |

## Table 4. Time per question (median)

| Step | Milliseconds |
|---|---|
| Layer 1 | 871.6 |
| Layer 2 | 25.8 |
| Layer 3 | 63.8 |
| Integrity check | 0.0 |
| Writing the answer (no checks) | 1141.5 |
| Layers 2, 3 and integrity together | 89.9 |
| All four checks together | 965.6 |
