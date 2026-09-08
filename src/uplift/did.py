"""Difference-in-differences on a synthetic pre-period, and placebo tests.

The Criteo file has no time dimension, so a pre-period has to be constructed.
That is a real limitation and it is stated here rather than buried: because the
pre-period is simulated, parallel trends is *imposed by the construction*, not
tested against data. What follows therefore demonstrates the mechanics of DiD
and, more usefully, the exact shape of its failure -- it is not evidence that
DiD identifies anything on this dataset.

To keep that honest, two pre-periods are built:

  `parallel`  Y_pre ~ Bernoulli(mu_0(X)), the same mapping for both arms. The
              confounding is a level shift in X, and differencing removes any
              time-invariant level, so DiD should recover the effect.

  `violated`  the treated arm's pre-period is depressed by `trend_gap`, so the
              treated would have risen faster than controls even untreated.
              Parallel trends fails and DiD inherits the gap as bias, no matter
              how much data is thrown at it.

The estimand is worth care: with Y_pre carrying no treatment effect, DiD
recovers the ATT, not the ATE. Under confounding with heterogeneous effects
those differ substantially, so Step 3 scores DiD against a separately
re-standardised ATT target.
"""

from __future__ import annotations

import numpy as np

from . import stats


def make_pre_period(
    mu0: np.ndarray,
    t: np.ndarray,
    mode: str = "parallel",
    trend_gap: float = 0.30,
    seed: int = 0,
) -> np.ndarray:
    """Draw a synthetic pre-period outcome.

    `mu0` must be a *control-arm* outcome model P(Y=1 | X, T=0), i.e. the
    behaviour of each unit absent any campaign. Using an all-arms model would
    leak the treatment effect into the pre-period and quietly bias DiD toward
    zero.

    `trend_gap` is a multiplicative shortfall applied to the treated arm's
    pre-period probability: 0.30 means treated units were running 30% below
    their own prognostic level beforehand, so they would have caught up on
    their own.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(mu0, dtype=np.float64).copy()
    if mode == "violated":
        p = np.where(t == 1, p * (1.0 - trend_gap), p)
    elif mode != "parallel":
        raise ValueError(f"mode must be 'parallel' or 'violated', got {mode!r}")
    return (rng.random(p.size) < np.clip(p, 0.0, 1.0)).astype(np.int8)


def did_estimate(
    y_post: np.ndarray,
    y_pre: np.ndarray,
    t: np.ndarray,
    cluster: np.ndarray | None = None,
    name: str = "DiD",
) -> stats.Estimate:
    """Two-period, two-group DiD.

    With one observation per unit per period, DiD collapses to a difference in
    means of the within-unit change d_i = Y_post,i - Y_pre,i. Writing it that
    way is not a shortcut: it makes the unit the independent object, so the
    same cluster-robust machinery used in Step 1 applies unchanged.
    """
    d = np.asarray(y_post, dtype=np.float64) - np.asarray(y_pre, dtype=np.float64)
    if cluster is None:
        return stats.diff_in_means(d, t, name=name)
    return stats.cluster_robust_diff_in_means(d, t, cluster, name=name)


def permutation_placebo(
    y: np.ndarray, t: np.ndarray, n_draws: int = 20, seed: int = 0
) -> dict:
    """Re-randomise treatment and re-estimate the naive contrast.

    Destroys the association between treatment and covariates, so the true
    effect under the permuted labels is exactly zero. A naive estimator that
    does not centre on zero here is reporting something other than an effect.

    This validates the harness, not the estimators: it says the pipeline
    produces no signal when none exists.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float64)
    ests = []
    for _ in range(n_draws):
        tp = rng.permutation(t)
        ests.append(stats.diff_in_means(y, tp).estimate)
    ests = np.array(ests)
    return {
        "n_draws": n_draws,
        "mean": float(ests.mean()),
        "sd": float(ests.std(ddof=1)),
        "min": float(ests.min()),
        "max": float(ests.max()),
    }
