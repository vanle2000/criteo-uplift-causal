# Step 3 - recovering the truth (`inflate` scenario)


## `conversion`

Subsample: 7,014,508 rows. RCT target ATE **+0.001603**, ATT +0.001861.

| estimator | estimate | 95% CI | target | bias | covers target |
|---|---|---|---|---|---|
| naive (no adjustment) | **+0.005397** | [+0.004619, +0.006175] | +0.001603 | +236.7% | NO |
| IPW (stabilized, lgbm PS) | **+0.000127** | [-0.000727, +0.000981] | +0.001603 | -92.1% | NO |
| AIPW doubly-robust (lgbm PS) | **+0.001677** | [+0.000742, +0.002611] | +0.001603 | +4.6% | yes |
| PSM 1:1 NN (lgbm PS) | **+0.001433** | [+0.001367, +0.001500] | +0.001603 | -10.6% | NO |
| IPW (stabilized, logit PS) | **-0.002941** | [-0.005659, -0.000222] | +0.001603 | -283.5% | NO |
| AIPW doubly-robust (logit PS) | **+0.010385** | [-0.003401, +0.024170] | +0.001603 | +547.9% | yes |
| PSM 1:1 NN (logit PS) | **+0.002742** | [+0.002679, +0.002806] | +0.001603 | +71.1% | NO |
| g-computation (outcome regression) | **+0.001061** | [+0.000916, +0.001207] | +0.001603 | -33.8% | NO |
| AIPW, trimmed to common support | **+0.000254** | [+0.000089, +0.000419] | +0.000224 | +13.4% | yes |
| IPW, trimmed to common support | **+0.000346** | [+0.000244, +0.000447] | +0.000224 | +54.2% | NO |
| DiD, parallel trends | **+0.002267** | [+0.001943, +0.002591] | +0.001861 | +21.8% | NO |
| DiD, violated trends | **+0.003348** | [+0.002880, +0.003816] | +0.001861 | +79.9% | NO |

**Propensity models.** `lgbm` AUC 0.8593, calibration slope 1.002; `logit` AUC 0.8363, calibration slope 1.000.

| covariate | \|SMD\| confounded | after IPW | after matching |
|---|---|---|---|
| `f0` | 0.9127 | 0.0040 | 0.0277 |
| `f1` | 0.1469 | 0.0304 | 0.0058 |
| `f2` | 0.7814 | 0.0055 | 0.0016 |
| `f3` | 0.4342 | 0.0308 | 0.0229 |
| `f4` | 0.2840 | 0.0492 | 0.0042 |
| `f5` | 0.2242 | 0.0372 | 0.0158 |
| `f6` | 0.1489 | 0.0260 | 0.0341 |
| `f7` | 0.1951 | 0.0213 | 0.0210 |
| `f8` | 1.1019 | 0.0328 | 0.0108 |
| `f9` | 0.7203 | 0.0342 | 0.0134 |
| `f10` | 0.3076 | 0.0230 | 0.0091 |
| `f11` | 0.1774 | 0.0410 | 0.0024 |

**Placebo tests.** Negative-control outcome (a pre-period that no treatment could have affected; true effect exactly 0):

| estimator | placebo estimate | 95% CI | passes |
|---|---|---|---|
| naive | +0.003285 | [+0.002801, +0.003770] | NO |
| ipw | -0.000753 | [-0.001447, -0.000058] | NO |
| aipw | +0.000314 | [-0.000563, +0.001191] | yes |

Permuted-treatment placebo over 20 draws: mean -1.79e-05, sd 6.57e-05, range [-1.29e-04, +9.94e-05].


## `visit`

Subsample: 7,014,508 rows. RCT target ATE **+0.011849**, ATT +0.013702.

| estimator | estimate | 95% CI | target | bias | covers target |
|---|---|---|---|---|---|
| naive (no adjustment) | **+0.082373** | [+0.070691, +0.094056] | +0.011849 | +595.2% | NO |
| IPW (stabilized, lgbm PS) | **+0.006081** | [+0.003787, +0.008375] | +0.011849 | -48.7% | NO |
| AIPW doubly-robust (lgbm PS) | **+0.010849** | [+0.008680, +0.013019] | +0.011849 | -8.4% | yes |
| PSM 1:1 NN (lgbm PS) | **+0.013356** | [+0.013126, +0.013586] | +0.011849 | +12.7% | NO |
| IPW (stabilized, logit PS) | **-0.158601** | [-0.178593, -0.138609] | +0.011849 | -1438.5% | NO |
| AIPW doubly-robust (logit PS) | **+0.003915** | [-0.029955, +0.037785] | +0.011849 | -67.0% | yes |
| PSM 1:1 NN (logit PS) | **+0.026827** | [+0.026601, +0.027053] | +0.011849 | +126.4% | NO |
| g-computation (outcome regression) | **+0.013130** | [+0.011515, +0.014745] | +0.011849 | +10.8% | yes |
| AIPW, trimmed to common support | **+0.002830** | [+0.002285, +0.003375] | +0.003313 | -14.6% | yes |
| IPW, trimmed to common support | **+0.003708** | [+0.003077, +0.004339] | +0.003313 | +11.9% | yes |
| DiD, parallel trends | **+0.013749** | [+0.011879, +0.015619] | +0.013702 | +0.3% | yes |
| DiD, violated trends | **+0.036239** | [+0.031395, +0.041084] | +0.013702 | +164.5% | NO |

**Propensity models.** `lgbm` AUC 0.8593, calibration slope 1.002; `logit` AUC 0.8363, calibration slope 1.000.

| covariate | \|SMD\| confounded | after IPW | after matching |
|---|---|---|---|
| `f0` | 0.9127 | 0.0040 | 0.0277 |
| `f1` | 0.1469 | 0.0304 | 0.0058 |
| `f2` | 0.7814 | 0.0055 | 0.0016 |
| `f3` | 0.4342 | 0.0308 | 0.0229 |
| `f4` | 0.2840 | 0.0492 | 0.0042 |
| `f5` | 0.2242 | 0.0372 | 0.0158 |
| `f6` | 0.1489 | 0.0260 | 0.0341 |
| `f7` | 0.1951 | 0.0213 | 0.0210 |
| `f8` | 1.1019 | 0.0328 | 0.0108 |
| `f9` | 0.7203 | 0.0342 | 0.0134 |
| `f10` | 0.3076 | 0.0230 | 0.0091 |
| `f11` | 0.1774 | 0.0410 | 0.0024 |

**Placebo tests.** Negative-control outcome (a pre-period that no treatment could have affected; true effect exactly 0):

| estimator | placebo estimate | 95% CI | passes |
|---|---|---|---|
| naive | +0.068732 | [+0.058916, +0.078549] | NO |
| ipw | -0.004240 | [-0.006404, -0.002076] | NO |

Permuted-treatment placebo over 20 draws: mean -1.00e-04, sd 2.91e-04, range [-7.04e-04, +4.68e-04].

