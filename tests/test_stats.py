"""Tests for the estimator primitives.

The chunked normal-equation solver in `lin_ate` is hand-rolled, so it is
checked against statsmodels rather than against itself.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import stats  # noqa: E402


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 8_000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    t = rng.binomial(1, 0.6, size=n)
    # True ATE = 0.5, with a genuine covariate effect and heteroskedasticity.
    y = 0.5 * t + 1.2 * X["a"] - 0.7 * X["b"] + rng.normal(scale=1 + np.abs(X["c"]), size=n)
    return X, t, y.to_numpy()


def test_diff_in_means_matches_manual():
    y = np.array([1.0, 0, 1, 1, 0, 0, 1, 0])
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    est = stats.diff_in_means(y, t)
    assert est.mean_treated == pytest.approx(0.75)
    assert est.mean_control == pytest.approx(0.25)
    assert est.estimate == pytest.approx(0.5)
    assert est.n_treated == 4 and est.n_control == 4
    # Welch SE with unpooled variances.
    expected = np.sqrt(np.var([1, 0, 1, 1], ddof=1) / 4 + np.var([0, 0, 1, 0], ddof=1) / 4)
    assert est.se == pytest.approx(expected)


def test_diff_in_means_relative_lift():
    y = np.array([1.0] * 3 + [0.0] * 7 + [1.0] * 1 + [0.0] * 9)
    t = np.array([1] * 10 + [0] * 10)
    est = stats.diff_in_means(y, t)
    # 0.30 vs 0.10 -> +200% relative.
    assert est.relative_lift == pytest.approx(2.0)


def test_diff_in_means_requires_both_arms():
    with pytest.raises(ValueError):
        stats.diff_in_means(np.ones(10), np.ones(10, dtype=int))


def test_smd_zero_when_identical():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    t = np.tile([0, 1], 500)
    # Same values in both arms by construction -> SMD ~ 0.
    x = np.repeat(rng.normal(size=500), 2)
    assert abs(stats.smd(x, t)) < 1e-12


def test_smd_known_value():
    x = np.concatenate([np.zeros(100), np.ones(100)])
    t = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    # Both arms are constant -> pooled sd is 0 -> guarded to 0.0.
    assert stats.smd(x, t) == 0.0
    x2 = np.concatenate([np.array([0.0, 2.0] * 50), np.array([1.0, 3.0] * 50)])
    d = stats.smd(x2, t)
    # Mean gap is exactly -1; pooled sd is sqrt(100/99) = 1.00504 because
    # both arm variances use ddof=1, so the SMD is -1/1.00504.
    assert d == pytest.approx(-1.0 / (100 / 99) ** 0.5, rel=1e-12)


def test_balance_table_shape_and_columns(toy):
    X, t, _ = toy
    tbl = stats.balance_table(X, t)
    assert list(tbl.index) == ["a", "b", "c"]
    assert {"smd", "abs_smd", "variance_ratio"} <= set(tbl.columns)
    # Randomly assigned -> well balanced.
    assert tbl["abs_smd"].max() < 0.05


def test_balance_table_weighted_improves_imbalance():
    rng = np.random.default_rng(2)
    n = 20_000
    x = rng.normal(size=n)
    # Confounded assignment: treatment probability rises with x.
    p = 1 / (1 + np.exp(-1.5 * x))
    t = rng.binomial(1, p)
    X = pd.DataFrame({"x": x})
    before = stats.balance_table(X, t)["abs_smd"].iloc[0]
    w = np.where(t == 1, 1 / p, 1 / (1 - p))  # true IPW weights
    after = stats.balance_table(X, t, weights=w)["abs_smd"].iloc[0]
    assert before > 0.5
    assert after < 0.05


def test_lin_ate_matches_statsmodels(toy):
    X, t, y = toy
    ate, se = stats.lin_ate(X, t, y, chunk=4096)  # small chunk -> exercises accumulation

    import statsmodels.api as sm

    Xc = X - X.mean(axis=0)
    D = np.column_stack([np.ones(len(y)), t, Xc.to_numpy(), t[:, None] * Xc.to_numpy()])
    ref = sm.OLS(y, D).fit(cov_type="HC1")

    assert ate == pytest.approx(ref.params[1], rel=1e-9)
    assert se == pytest.approx(ref.bse[1], rel=1e-7)


def test_lin_ate_recovers_true_effect(toy):
    X, t, y = toy
    ate, se = stats.lin_ate(X, t, y)
    assert abs(ate - 0.5) < 3 * se


def test_lin_ate_chunking_is_invariant(toy):
    X, t, y = toy
    a1, s1 = stats.lin_ate(X, t, y, chunk=1_000_000)
    a2, s2 = stats.lin_ate(X, t, y, chunk=997)
    assert a1 == pytest.approx(a2, rel=1e-10)
    assert s1 == pytest.approx(s2, rel=1e-10)


def test_wald_ratio_scales_itt():
    cace, se = stats.wald_ratio(itt=0.01, itt_se=0.001, compliance=0.5, compliance_se=0.0)
    assert cace == pytest.approx(0.02)
    assert se == pytest.approx(0.002)


def test_estimate_ci_and_serialisation():
    est = stats.diff_in_means(np.array([1.0, 1, 0, 0, 1, 0]), np.array([1, 1, 1, 0, 0, 0]))
    lo, hi = est.ci
    assert lo < est.estimate < hi
    d = est.as_dict()
    assert {"ci_low", "ci_high", "z", "relative_lift_ci_low"} <= set(d)
