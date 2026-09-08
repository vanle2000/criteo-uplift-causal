"""Tests for the Step 4 uplift metrics.

These check the properties the metrics are supposed to have -- a perfect
ranking must beat a random one, the transformed outcome must be unbiased for
the ATE, decile uplift must track a known effect surface -- rather than
pinning numbers that would break on any refactor.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import uplift as U  # noqa: E402

P_TREAT = 0.85


@pytest.fixture
def rct():
    """Randomised sample with an effect that rises steeply in one covariate.

    Mirrors the Criteo shape: 85/15 assignment, a rare outcome, and uplift
    concentrated in a small slice of the population.
    """
    rng = np.random.default_rng(4)
    n = 400_000
    x = rng.uniform(0, 1, size=n)
    t = rng.binomial(1, P_TREAT, size=n)
    base = 0.002 + 0.02 * x**3
    tau = 0.03 * x**4                      # near zero except at the very top
    p = base + tau * t
    y = rng.binomial(1, np.clip(p, 0, 1)).astype(float)
    return dict(x=x, t=t, y=y, tau=tau, n=n)


def test_transformed_outcome_is_unbiased_for_ate(rct):
    r = rct
    z = U.transformed_outcome(r["y"], r["t"], P_TREAT)
    true_ate = r["tau"].mean()
    se = z.std(ddof=1) / np.sqrt(r["n"])
    assert abs(z.mean() - true_ate) < 4 * se


def test_transformed_outcome_is_high_variance(rct):
    """The unbiasedness above is bought with variance; that is the tradeoff."""
    r = rct
    z = U.transformed_outcome(r["y"], r["t"], P_TREAT)
    assert z.std() > 20 * abs(r["tau"].mean())


def test_qini_prefers_a_perfect_ranking(rct):
    r = rct
    rng = np.random.default_rng(0)
    perfect = U.qini_curve(r["y"], r["t"], r["tau"])
    random_ = U.qini_curve(r["y"], r["t"], rng.normal(size=r["n"]))
    assert perfect.qini_coefficient > random_.qini_coefficient
    assert abs(random_.qini_coefficient) < 0.15
    assert perfect.qini_coefficient > 0.3


def test_qini_curve_shape(rct):
    r = rct
    q = U.qini_curve(r["y"], r["t"], r["tau"], n_points=50)
    assert q.x[0] > 0 and q.x[-1] == pytest.approx(1.0)
    assert np.all(np.diff(q.x) > 0)
    assert q.n == r["n"]


def test_uplift_by_decile_is_monotonic_for_a_good_score(rct):
    r = rct
    tbl = U.uplift_by_decile(r["y"], r["t"], r["tau"])
    assert len(tbl) == 10
    # Top bin must beat the bottom bin decisively.
    top, bottom = tbl.iloc[0], tbl.iloc[-1]
    assert top["observed_uplift"] - bottom["observed_uplift"] > 3 * top["se"]
    # Broadly decreasing: allow local noise, require a strong rank correlation.
    from scipy.stats import spearmanr

    rho = spearmanr(tbl["bin"], tbl["observed_uplift"]).statistic
    assert rho < -0.8


def test_precision_at_k_captures_most_uplift_at_the_top(rct):
    r = rct
    tbl = U.precision_at_k(r["y"], r["t"], r["tau"], ks=(0.10, 0.50, 1.0))
    top10 = tbl.iloc[0]
    assert top10["lift_vs_overall"] > 2.0
    assert 0 < top10["captured_share"] < 1.0
    # Targeting everyone must, by definition, capture all of it.
    assert tbl.iloc[-1]["captured_share"] == pytest.approx(1.0, abs=1e-9)
    assert tbl.iloc[-1]["lift_vs_overall"] == pytest.approx(1.0, abs=1e-9)


def test_precision_at_k_random_score_gives_no_lift(rct):
    r = rct
    rng = np.random.default_rng(3)
    tbl = U.precision_at_k(r["y"], r["t"], rng.normal(size=r["n"]), ks=(0.10,))
    assert abs(tbl.iloc[0]["lift_vs_overall"] - 1.0) < 0.6


def test_response_metrics_reports_lift_over_base_rate(rct):
    r = rct
    p = 0.002 + 0.02 * r["x"] ** 3
    m = U.response_metrics(r["y"], p)
    assert m["base_rate"] == pytest.approx(r["y"].mean())
    assert m["auc_pr"] > m["base_rate"]
    assert m["auc_pr_lift_over_base"] > 1.0
    assert 0.5 < m["auc_roc"] <= 1.0


def test_calibration_table_tracks_a_calibrated_model(rct):
    r = rct
    p = 0.002 + 0.02 * r["x"] ** 3 + r["tau"] * P_TREAT
    tbl = U.calibration_table(r["y"], p)
    assert len(tbl) == 10
    # A calibrated score should sit near the diagonal in aggregate.
    err = np.abs(tbl["mean_predicted"] - tbl["observed_rate"]).max()
    assert err < 0.01
