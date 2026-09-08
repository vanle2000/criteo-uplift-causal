# Step 3 - recovering the truth (`inflate` scenario)


## `conversion`

Subsample: 7,014,516 rows. RCT target ATE **+0.001603**, ATT +0.001861.

| estimator | estimate | 95% CI | target | bias | covers target |
|---|---|---|---|---|---|
| naive (no adjustment) | **+0.005397** | [+0.004619, +0.006175] | +0.001603 | +236.7% | NO |
| IPW (stabilized, lgbm PS) | **+0.000648** | [-0.000111, +0.001407] | +0.001603 | -59.6% | NO |
| AIPW doubly-robust (lgbm PS) | **+0.002085** | [+0.001208, +0.002962] | +0.001603 | +30.1% | yes |
| PSM 1:1 NN (lgbm PS) | **+0.001229** | [+0.001161, +0.001296] | +0.001603 | -23.3% | NO |
| IPW (stabilized, logit PS) | **-0.002991** | [-0.005734, -0.000247] | +0.001603 | -286.6% | NO |
| AIPW doubly-robust (logit PS) | **+0.014765** | [-0.002895, +0.032425] | +0.001603 | +821.3% | yes |
| PSM 1:1 NN (logit PS) | **+0.002607** | [+0.002543, +0.002672] | +0.001603 | +62.7% | NO |
| g-computation (outcome regression) | **+0.001404** | [+0.001214, +0.001594] | +0.001603 | -12.4% | NO |
| AIPW, trimmed to common support | **+0.000198** | [+0.000084, +0.000312] | +0.000224 | -11.4% | yes |
| IPW, trimmed to common support | **+0.000349** | [+0.000249, +0.000450] | +0.000224 | +56.2% | NO |
| DiD, parallel trends | **+0.002193** | [+0.001883, +0.002503] | +0.001861 | +17.9% | NO |
| DiD, violated trends | **+0.003287** | [+0.002830, +0.003744] | +0.001861 | +76.7% | NO |

**Propensity models.** `lgbm` AUC 0.8594, calibration slope 1.002; `logit` AUC 0.8363, calibration slope 1.000.

| covariate | \|SMD\| confounded | after IPW | after matching |
|---|---|---|---|
| `f0` | 0.9127 | 0.0028 | 0.0252 |
| `f1` | 0.1469 | 0.0268 | 0.0047 |
| `f2` | 0.7814 | 0.0053 | 0.0043 |
| `f3` | 0.4342 | 0.0271 | 0.0266 |
| `f4` | 0.2840 | 0.0418 | 0.0047 |
| `f5` | 0.2242 | 0.0319 | 0.0183 |
| `f6` | 0.1489 | 0.0241 | 0.0364 |
| `f7` | 0.1951 | 0.0209 | 0.0216 |
| `f8` | 1.1019 | 0.0327 | 0.0100 |
| `f9` | 0.7203 | 0.0339 | 0.0151 |
| `f10` | 0.3076 | 0.0236 | 0.0145 |
| `f11` | 0.1774 | 0.0338 | 0.0036 |

**Placebo tests.** Negative-control outcome (a pre-period that no treatment could have affected; true effect exactly 0):

| estimator | placebo estimate | 95% CI | passes |
|---|---|---|---|
| naive | +0.003207 | [+0.002733, +0.003681] | NO |
| ipw | -0.000568 | [-0.001180, +0.000043] | yes |
| aipw | -0.000076 | [-0.000750, +0.000597] | yes |

Permuted-treatment placebo over 20 draws: mean +6.18e-06, sd 8.09e-05, range [-1.16e-04, +1.48e-04].


## `visit`

Subsample: 7,014,516 rows. RCT target ATE **+0.011849**, ATT +0.013702.

| estimator | estimate | 95% CI | target | bias | covers target |
|---|---|---|---|---|---|
| naive (no adjustment) | **+0.082373** | [+0.070690, +0.094056] | +0.011849 | +595.2% | NO |
| IPW (stabilized, lgbm PS) | **+0.006555** | [+0.004271, +0.008839] | +0.011849 | -44.7% | NO |
| AIPW doubly-robust (lgbm PS) | **+0.011300** | [+0.009083, +0.013517] | +0.011849 | -4.6% | yes |
| PSM 1:1 NN (lgbm PS) | **+0.011372** | [+0.011140, +0.011605] | +0.011849 | -4.0% | NO |
| IPW (stabilized, logit PS) | **-0.158639** | [-0.178784, -0.138494] | +0.011849 | -1438.8% | NO |
| AIPW doubly-robust (logit PS) | **+0.006077** | [-0.027099, +0.039254] | +0.011849 | -48.7% | yes |
| PSM 1:1 NN (logit PS) | **+0.026508** | [+0.026281, +0.026734] | +0.011849 | +123.7% | NO |
| g-computation (outcome regression) | **+0.013135** | [+0.011519, +0.014751] | +0.011849 | +10.9% | yes |
| AIPW, trimmed to common support | **+0.002733** | [+0.002187, +0.003279] | +0.003303 | -17.3% | NO |
| IPW, trimmed to common support | **+0.003544** | [+0.002921, +0.004166] | +0.003303 | +7.3% | yes |
| DiD, parallel trends | **+0.013614** | [+0.011739, +0.015489] | +0.013702 | -0.6% | yes |
| DiD, violated trends | **+0.036120** | [+0.031268, +0.040973] | +0.013702 | +163.6% | NO |

**Propensity models.** `lgbm` AUC 0.8594, calibration slope 1.002; `logit` AUC 0.8363, calibration slope 1.000.

| covariate | \|SMD\| confounded | after IPW | after matching |
|---|---|---|---|
| `f0` | 0.9127 | 0.0028 | 0.0252 |
| `f1` | 0.1469 | 0.0268 | 0.0047 |
| `f2` | 0.7814 | 0.0053 | 0.0043 |
| `f3` | 0.4342 | 0.0271 | 0.0266 |
| `f4` | 0.2840 | 0.0418 | 0.0047 |
| `f5` | 0.2242 | 0.0319 | 0.0183 |
| `f6` | 0.1489 | 0.0241 | 0.0364 |
| `f7` | 0.1951 | 0.0209 | 0.0216 |
| `f8` | 1.1019 | 0.0327 | 0.0100 |
| `f9` | 0.7203 | 0.0339 | 0.0151 |
| `f10` | 0.3076 | 0.0236 | 0.0145 |
| `f11` | 0.1774 | 0.0338 | 0.0036 |

**Placebo tests.** Negative-control outcome (a pre-period that no treatment could have affected; true effect exactly 0):

| estimator | placebo estimate | 95% CI | passes |
|---|---|---|---|
| naive | +0.068738 | [+0.058861, +0.078614] | NO |
| ipw | -0.005608 | [-0.007798, -0.003419] | NO |

Permuted-treatment placebo over 20 draws: mean -1.80e-05, sd 2.31e-04, range [-3.94e-04, +3.61e-04].

