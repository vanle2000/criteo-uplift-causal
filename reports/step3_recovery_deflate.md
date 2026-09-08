# Step 3 - recovering the truth (`deflate` scenario)


## `conversion`

Subsample: 6,963,186 rows. RCT target ATE **+0.000402**, ATT +0.000166.

| estimator | estimate | 95% CI | target | bias | covers target |
|---|---|---|---|---|---|
| naive (no adjustment) | **-0.003148** | [-0.003658, -0.002638] | +0.000402 | -882.2% | NO |
| IPW (stabilized, lgbm PS) | **+0.000402** | [+0.000337, +0.000467] | +0.000402 | -0.1% | yes |
| AIPW doubly-robust (lgbm PS) | **+0.000369** | [+0.000304, +0.000434] | +0.000402 | -8.3% | yes |
| PSM 1:1 NN (lgbm PS) | **+0.000326** | [+0.000294, +0.000358] | +0.000402 | -19.0% | NO |
| IPW (stabilized, logit PS) | **+0.001135** | [+0.000951, +0.001319] | +0.000402 | +182.0% | NO |
| AIPW doubly-robust (logit PS) | **+0.000267** | [+0.000097, +0.000436] | +0.000402 | -33.8% | yes |
| PSM 1:1 NN (logit PS) | **+0.000050** | [+0.000018, +0.000082] | +0.000402 | -87.5% | NO |
| g-computation (outcome regression) | **+0.000541** | [+0.000490, +0.000592] | +0.000402 | +34.5% | NO |
| AIPW, trimmed to common support | **+0.000442** | [+0.000357, +0.000528] | +0.000491 | -9.9% | yes |
| IPW, trimmed to common support | **+0.000492** | [+0.000405, +0.000579] | +0.000491 | +0.2% | yes |
| DiD, parallel trends | **-0.000115** | [-0.000267, +0.000037] | +0.000166 | -169.3% | NO |
| DiD, violated trends | **+0.000022** | [-0.000129, +0.000173] | +0.000166 | -86.8% | yes |

**Propensity models.** `lgbm` AUC 0.8564, calibration slope 1.004; `logit` AUC 0.8323, calibration slope 1.000.

| covariate | \|SMD\| confounded | after IPW | after matching |
|---|---|---|---|
| `f0` | 0.9084 | 0.0098 | 0.0134 |
| `f1` | 0.1158 | 0.0014 | 0.0122 |
| `f2` | 0.8071 | 0.0164 | 0.0021 |
| `f3` | 0.3641 | 0.0099 | 0.0007 |
| `f4` | 0.2740 | 0.0003 | 0.0033 |
| `f5` | 0.1826 | 0.0008 | 0.0430 |
| `f6` | 0.0717 | 0.0052 | 0.0216 |
| `f7` | 0.1621 | 0.0026 | 0.0482 |
| `f8` | 1.0809 | 0.0055 | 0.0012 |
| `f9` | 0.6949 | 0.0033 | 0.0012 |
| `f10` | 0.2946 | 0.0021 | 0.0068 |
| `f11` | 0.1713 | 0.0024 | 0.0000 |

**Placebo tests.** Negative-control outcome (a pre-period that no treatment could have affected; true effect exactly 0):

| estimator | placebo estimate | 95% CI | passes |
|---|---|---|---|
| naive | -0.003010 | [-0.003500, -0.002520] | NO |
| ipw | +0.000014 | [-0.000076, +0.000104] | yes |
| aipw | +0.000015 | [-0.000075, +0.000105] | yes |

Permuted-treatment placebo over 20 draws: mean +2.83e-06, sd 3.35e-05, range [-5.49e-05, +8.24e-05].


## `visit`

Subsample: 6,963,186 rows. RCT target ATE **+0.003254**, ATT +0.001532.

| estimator | estimate | 95% CI | target | bias | covers target |
|---|---|---|---|---|---|
| naive (no adjustment) | **-0.062595** | [-0.072257, -0.052933] | +0.003254 | -2023.7% | NO |
| IPW (stabilized, lgbm PS) | **+0.003247** | [+0.002932, +0.003563] | +0.003254 | -0.2% | yes |
| AIPW doubly-robust (lgbm PS) | **+0.003082** | [+0.002810, +0.003354] | +0.003254 | -5.3% | yes |
| PSM 1:1 NN (lgbm PS) | **+0.002614** | [+0.002493, +0.002736] | +0.003254 | -19.7% | NO |
| IPW (stabilized, logit PS) | **+0.013953** | [+0.012439, +0.015467] | +0.003254 | +328.8% | NO |
| AIPW doubly-robust (logit PS) | **+0.003007** | [+0.002581, +0.003433] | +0.003254 | -7.6% | yes |
| PSM 1:1 NN (logit PS) | **-0.002731** | [-0.002854, -0.002607] | +0.003254 | -183.9% | NO |
| g-computation (outcome regression) | **+0.002913** | [+0.002720, +0.003107] | +0.003254 | -10.5% | NO |
| AIPW, trimmed to common support | **+0.003773** | [+0.003373, +0.004172] | +0.004020 | -6.2% | yes |
| IPW, trimmed to common support | **+0.004145** | [+0.003666, +0.004625] | +0.004020 | +3.1% | yes |
| DiD, parallel trends | **+0.001521** | [+0.000933, +0.002110] | +0.001532 | -0.7% | yes |
| DiD, violated trends | **+0.003534** | [+0.002940, +0.004127] | +0.001532 | +130.7% | NO |

**Propensity models.** `lgbm` AUC 0.8564, calibration slope 1.004; `logit` AUC 0.8323, calibration slope 1.000.

| covariate | \|SMD\| confounded | after IPW | after matching |
|---|---|---|---|
| `f0` | 0.9084 | 0.0098 | 0.0134 |
| `f1` | 0.1158 | 0.0014 | 0.0122 |
| `f2` | 0.8071 | 0.0164 | 0.0021 |
| `f3` | 0.3641 | 0.0099 | 0.0007 |
| `f4` | 0.2740 | 0.0003 | 0.0033 |
| `f5` | 0.1826 | 0.0008 | 0.0430 |
| `f6` | 0.0717 | 0.0052 | 0.0216 |
| `f7` | 0.1621 | 0.0026 | 0.0482 |
| `f8` | 1.0809 | 0.0055 | 0.0012 |
| `f9` | 0.6949 | 0.0033 | 0.0012 |
| `f10` | 0.2946 | 0.0021 | 0.0068 |
| `f11` | 0.1713 | 0.0024 | 0.0000 |

**Placebo tests.** Negative-control outcome (a pre-period that no treatment could have affected; true effect exactly 0):

| estimator | placebo estimate | 95% CI | passes |
|---|---|---|---|
| naive | -0.064181 | [-0.073908, -0.054455] | NO |
| ipw | +0.000070 | [-0.000159, +0.000300] | yes |

Permuted-treatment placebo over 20 draws: mean -4.31e-05, sd 1.20e-04, range [-2.82e-04, +2.09e-04].

