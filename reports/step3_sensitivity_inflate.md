# Step 3, part two - sensitivity to unmeasured confounding (`inflate` scenario)

Rosenbaum's Gamma is an odds ratio on **treatment assignment**: how differently two units identical on X could have been selected before the matched-pair inference stops being significant. The E-value is the minimum association, on the risk-ratio scale, that a confounder would need with **both** treatment and outcome to explain the estimate away.


## `conversion`

5,952,926 matched pairs, 55,858 discordant (32,851 favouring treated, 23,007 favouring control). **Gamma\* = 1.41**.

| Gamma | upper-bound p | lower-bound p | still significant |
|---|---|---|---|
| 1.00 | 0 | 0 | yes |
| 1.10 | 4.83e-205 | 0 | yes |
| 1.25 | 1.15e-54 | 0 | yes |
| 1.50 | 1 | 0 | NO |
| 1.75 | 1 | 0 | NO |
| 2.00 | 1 | 0 | NO |
| 2.50 | 1 | 0 | NO |
| 3.00 | 1 | 0 | NO |
| 4.00 | 1 | 0 | NO |
| 5.00 | 1 | 0 | NO |
| 7.50 | 1 | 0 | NO |
| 10.00 | 1 | 0 | NO |

The RCT-derived target for this subsample is a risk ratio of 1.419 against a counterfactual untreated risk of 0.00382, an E-value of **2.19**. The *observed* control rate in the confounded subsample is only 0.00030; using that as the denominator would inflate every ratio below, because selection kept low-prognostic controls on purpose.

| estimator | estimate | implied RR | E-value (point) | E-value (CI) | bias vs target |
|---|---|---|---|---|---|
| naive (no adjustment) | +0.005397 | 2.412 | 4.26 | 3.84 | +236.7% |
| IPW (stabilized, lgbm PS) | +0.000127 | 1.033 | 1.22 | 1.00 | -92.1% |
| AIPW doubly-robust (lgbm PS) | +0.001677 | 1.439 | 2.23 | 1.68 | +4.6% |
| PSM 1:1 NN (lgbm PS) | +0.001433 | 1.375 | 2.09 | 2.05 | -10.6% |
| IPW (stabilized, logit PS) | -0.002941 | n/a | n/a | n/a | -283.5% |
| AIPW doubly-robust (logit PS) | +0.010385 | 3.717 | 6.90 | 1.00 | +547.9% |
| PSM 1:1 NN (logit PS) | +0.002742 | 1.718 | 2.83 | 2.79 | +71.1% |
| g-computation (outcome regression) | +0.001061 | 1.278 | 1.87 | 1.78 | -33.8% |

## `visit`

5,952,926 matched pairs, 667,295 discordant (379,464 favouring treated, 287,831 favouring control). **Gamma\* = 1.31**.

| Gamma | upper-bound p | lower-bound p | still significant |
|---|---|---|---|
| 1.00 | 0 | 0 | yes |
| 1.10 | 0 | 0 | yes |
| 1.25 | 1.95e-103 | 0 | yes |
| 1.50 | 1 | 0 | NO |
| 1.75 | 1 | 0 | NO |
| 2.00 | 1 | 0 | NO |
| 2.50 | 1 | 0 | NO |
| 3.00 | 1 | 0 | NO |
| 4.00 | 1 | 0 | NO |
| 5.00 | 1 | 0 | NO |
| 7.50 | 1 | 0 | NO |
| 10.00 | 1 | 0 | NO |

The RCT-derived target for this subsample is a risk ratio of 1.187 against a counterfactual untreated risk of 0.06329, an E-value of **1.66**. The *observed* control rate in the confounded subsample is only 0.00629; using that as the denominator would inflate every ratio below, because selection kept low-prognostic controls on purpose.

| estimator | estimate | implied RR | E-value (point) | E-value (CI) | bias vs target |
|---|---|---|---|---|---|
| naive (no adjustment) | +0.082373 | 2.302 | 4.03 | 3.65 | +595.2% |
| IPW (stabilized, lgbm PS) | +0.006081 | 1.096 | 1.42 | 1.31 | -48.7% |
| AIPW doubly-robust (lgbm PS) | +0.010849 | 1.171 | 1.62 | 1.53 | -8.4% |
| PSM 1:1 NN (lgbm PS) | +0.013356 | 1.211 | 1.72 | 1.71 | +12.7% |
| IPW (stabilized, logit PS) | -0.158601 | n/a | n/a | n/a | -1438.5% |
| AIPW doubly-robust (logit PS) | +0.003915 | 1.062 | 1.32 | 1.00 | -67.0% |
| PSM 1:1 NN (logit PS) | +0.026827 | 1.424 | 2.20 | 2.19 | +126.4% |
| g-computation (outcome regression) | +0.013130 | 1.207 | 1.71 | 1.65 | +10.8% |
