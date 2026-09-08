"""Tests for the Step 3 estimators.

The centrepiece is `test_aipw_is_doubly_robust`: AIPW must recover the truth
when the propensity is right and the outcome model is garbage, AND when the
outcome model is right and the propensity is garbage. That property is the
entire reason the estimator is in the project, so it is tested directly rather
than assumed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import did, estimators as E, stats  # noqa: E402

TRUE_ATE = 0.05


@pytest.fixture
def confounded():
    """A confounded binary-outcome world with a known constant effect.

    Treatment probability and outcome level both rise with x1, so the naive
    contrast is biased upward -- the same shape as the Step 2 construction, in
    miniature and with every nuisance function known in closed form.
    """
    rng = np.random.default_rng(7)
    n = 120_000
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["x1", "x2"])
    e = 1 / (1 + np.exp(-(0.9 * X.x1 - 0.5 * X.x2)))
    t = rng.binomial(1, e)
    mu0 = 1 / (1 + np.exp(-(-1.0 + 0.7 * X.x1 + 0.3 * X.x2)))
    mu1 = np.clip(mu0 + TRUE_ATE, 0, 1)
    y0 = rng.binomial(1, mu0)
    y1 = rng.binomial(1, mu1)
    y = np.where(t == 1, y1, y0).astype(float)
    return dict(X=X, t=t, y=y, e=e.to_numpy(), mu0=mu0.to_numpy(), mu1=mu1.to_numpy())


def test_simulation_is_actually_confounded(confounded):
    """Guard: if the naive estimate were unbiased the other tests prove nothing."""
    naive = stats.diff_in_means(confounded["y"], confounded["t"])
    assert naive.estimate - TRUE_ATE > 5 * naive.se


def test_ipw_with_true_propensity_recovers(confounded):
    c = confounded
    r = E.ipw_ate(c["y"], c["t"], c["e"])
    assert abs(r.estimate - TRUE_ATE) < 3 * r.se


def test_aipw_is_doubly_robust(confounded):
    """Either nuisance may be wrong, but not both."""
    c = confounded
    n = len(c["y"])
    junk = np.full(n, 0.5)

    # (a) propensity right, outcome model deliberately useless.
    a = E.aipw_ate(c["y"], c["t"], c["e"], junk, junk)
    assert abs(a.estimate - TRUE_ATE) < 3 * a.se, f"PS-right arm failed: {a.estimate}"

    # (b) outcome models right, propensity deliberately useless.
    b = E.aipw_ate(c["y"], c["t"], junk, c["mu0"], c["mu1"])
    assert abs(b.estimate - TRUE_ATE) < 3 * b.se, f"OM-right arm failed: {b.estimate}"

    # (c) both wrong -> no protection. This is the honest half of the claim.
    d = E.aipw_ate(c["y"], c["t"], junk, junk, junk)
    assert abs(d.estimate - TRUE_ATE) > 3 * d.se


def test_gcomputation_recovers_with_true_outcome_models(confounded):
    c = confounded
    r = E.gcomputation_ate(c["mu0"], c["mu1"])
    assert abs(r.estimate - TRUE_ATE) < 1e-9 + 3 * r.se


def test_psm_reduces_bias_versus_naive(confounded):
    c = confounded
    naive = stats.diff_in_means(c["y"], c["t"]).estimate
    r, diag = E.psm_ate(c["y"], c["t"], c["e"])
    assert abs(r.estimate - TRUE_ATE) < abs(naive - TRUE_ATE)
    assert 0 < diag["treated_match_rate"] <= 1.0
    assert diag["n_treated_matched"] > 0 and diag["n_control_matched"] > 0


def test_ipw_weighting_restores_balance(confounded):
    c = confounded
    before = stats.balance_table(c["X"], c["t"])["abs_smd"].max()
    after = E.weighted_balance(c["X"], c["t"], c["e"])["abs_smd"].max()
    assert before > 0.3
    assert after < 0.05


def test_influence_se_matches_plain_formula():
    rng = np.random.default_rng(1)
    psi = rng.normal(size=10_000)
    assert E.influence_se(psi) == pytest.approx(psi.std(ddof=1) / np.sqrt(10_000))
    # Singleton clusters must reduce to the unclustered answer.
    singleton = E.influence_se(psi, np.arange(psi.size))
    assert singleton == pytest.approx(E.influence_se(psi), rel=1e-3)


def test_influence_se_inflates_under_clustering():
    rng = np.random.default_rng(2)
    G, m = 2_000, 5
    g = np.repeat(np.arange(G), m)
    psi = np.repeat(rng.normal(size=G), m) + 0.1 * rng.normal(size=G * m)
    assert E.influence_se(psi, g) > 1.8 * E.influence_se(psi)


def test_overlap_mask_trims_extremes():
    e = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
    keep = E.overlap_mask(e, 0.02, 0.98)
    assert keep.tolist() == [False, True, True, True, False]


def test_propensity_calibration_slope_is_one_when_correct():
    rng = np.random.default_rng(3)
    n = 60_000
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    p = 1 / (1 + np.exp(-(0.8 * X.a - 0.4 * X.b)))
    t = rng.binomial(1, p)
    pr = E.fit_propensity(X, t, method="logit", n_folds=2)
    assert pr.calibration_slope == pytest.approx(1.0, abs=0.06)
    assert pr.auc > 0.65


# --------------------------------------------------------------------------
# DiD
# --------------------------------------------------------------------------

def test_did_recovers_att_under_parallel_trends(confounded):
    """Confounding is a level shift here, and differencing removes levels."""
    c = confounded
    y_pre = did.make_pre_period(c["mu0"], c["t"], mode="parallel", seed=5)
    r = did.did_estimate(c["y"], y_pre, c["t"])
    assert abs(r.estimate - TRUE_ATE) < 3 * r.se


def test_did_is_biased_when_trends_diverge(confounded):
    c = confounded
    y_pre = did.make_pre_period(c["mu0"], c["t"], mode="violated", trend_gap=0.30, seed=5)
    r = did.did_estimate(c["y"], y_pre, c["t"])
    assert r.estimate - TRUE_ATE > 3 * r.se, "violated trends should bias DiD upward"


def test_make_pre_period_rejects_bad_mode(confounded):
    with pytest.raises(ValueError):
        did.make_pre_period(confounded["mu0"], confounded["t"], mode="nonsense")


def test_permutation_placebo_centres_on_zero(confounded):
    c = confounded
    out = did.permutation_placebo(c["y"], c["t"], n_draws=30, seed=11)
    assert abs(out["mean"]) < 4 * out["sd"] / np.sqrt(out["n_draws"])
