# Step 4 - uplift modelling (`conversion`)

Trained on 10,375,433 rows, evaluated on 3,604,159 held-out rows, split by cluster so no unit appears on both sides. Holdout base rate 0.00338.

## Ranking quality

| model | Qini coefficient | uplift @ top 10% | lift vs untargeted | captured share @ 10% | fit seconds |
|---|---|---|---|---|---|
| `two_model_tlearner` | **+0.0033** | +0.001868 | 1.35x | 13.5% | 66 |
| `transformed_outcome` | **+0.6590** | +0.009090 | 6.58x | 65.8% | 48 |
| `response_baseline` | **+0.8133** | +0.009902 | 7.17x | 71.7% | 67 |

## Precision@K


**`two_model_tlearner`**

| K | n targeted | uplift @ K | lift vs untargeted | captured share |
|---|---|---|---|---|
| 5% | 180,208 | +0.002279 | 1.65x | 8.3% |
| 10% | 360,416 | +0.001868 | 1.35x | 13.5% |
| 20% | 720,832 | +0.001696 | 1.23x | 24.6% |
| 30% | 1,081,248 | +0.001386 | 1.00x | 30.1% |

**`transformed_outcome`**

| K | n targeted | uplift @ K | lift vs untargeted | captured share |
|---|---|---|---|---|
| 5% | 180,208 | +0.014791 | 10.71x | 53.6% |
| 10% | 360,416 | +0.009090 | 6.58x | 65.8% |
| 20% | 720,832 | +0.005337 | 3.87x | 77.3% |
| 30% | 1,081,248 | +0.003830 | 2.77x | 83.2% |

**`response_baseline`**

| K | n targeted | uplift @ K | lift vs untargeted | captured share |
|---|---|---|---|---|
| 5% | 180,208 | +0.015964 | 11.56x | 57.8% |
| 10% | 360,416 | +0.009902 | 7.17x | 71.7% |
| 20% | 720,832 | +0.005998 | 4.35x | 86.9% |
| 30% | 1,081,248 | +0.004234 | 3.07x | 92.0% |

## Response model quality (class imbalance)

Base rate 0.00338. AUC-ROC 0.9354; AUC-PR 0.1834, which is **54.3x** the base rate a random ranker would achieve. Brier 0.003097; mean predicted 0.00338 against observed 0.00338 (calibration ratio 0.999).

Accuracy is not reported: predicting "never converts" for everyone scores 99.662% and is worthless.

