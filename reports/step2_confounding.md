# Step 2 - deliberate confounding

Selection runs along a cross-fitted prognostic score -- `P(visit=1 | X)` fitted on control units only, holdout AUC **0.9471**. Confounding along a direction that did not predict the outcome would create imbalance but no bias.

## Strength sweep

| k | n selected | treated share | max \|SMD\| | naive conversion ATE | vs target |
|---|---|---|---|---|---|
| 0 | 6,986,770 | 0.850 | 0.049 | +0.001165 | 1.16x |
| 1 | 7,000,515 | 0.849 | 0.461 | +0.003221 | 2.49x |
| 2 | 7,009,165 | 0.849 | 0.831 | +0.004636 | 3.10x |
| 3 | 7,014,516 | 0.849 | 1.102 | +0.005397 | 3.37x |
| 4 | 7,015,784 | 0.849 | 1.286 | +0.005742 | 3.47x |
| 6 | 7,015,566 | 0.849 | 1.486 | +0.005972 | 3.54x |

`k = 0` is the control condition: a random subsample, which must reproduce the RCT answer. It does, which is what licenses reading the rest of the table as induced bias rather than a bug.


## Scenario: `inflate` (k = 3.0)

7,014,516 rows retained (50.2%), 84.9% treated. Max |SMD| **1.102** (12 of 12 covariates above 0.10, 7 above 0.25).

| outcome | naive ATE | true ATE for this subsample | bias | ratio |
|---|---|---|---|---|
| `visit` | **+0.082373** | +0.011849 | +0.070524 | 6.95x |
| `conversion` | **+0.005397** | +0.001603 | +0.003794 | 3.37x |

## Scenario: `deflate` (k = 3.0)

6,963,206 rows retained (49.8%), 85.1% treated. Max |SMD| **1.081** (11 of 12 covariates above 0.10, 7 above 0.25).

| outcome | naive ATE | true ATE for this subsample | bias | ratio |
|---|---|---|---|---|
| `visit` | **-0.062595** | +0.003254 | -0.065849 | -19.24x |
| `conversion` | **-0.003148** | +0.000402 | -0.003550 | -7.82x |
