"""Tests for the sensitivity analysis.

Checked against the properties these quantities are defined to have, and
against two published reference values, rather than against themselves.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import sensitivity as S  # noqa: E402


def pairs(n_plus: int, n_minus: int, n_tied: int = 0):
    """Matched pairs with a given discordance pattern."""
    yt = np.r_[np.ones(n_plus), np.zeros(n_minus), np.ones(n_tied)]
    yc = np.r_[np.zeros(n_plus), np.ones(n_minus), np.ones(n_tied)]
    return yt, yc


def test_gamma_one_reproduces_mcnemar():
    """At Gamma = 1 the bound is the ordinary sign test on discordant pairs."""
    from scipy.stats import binom

    yt, yc = pairs(60, 40)
    tbl = S.rosenbaum_bounds(yt, yc, gammas=(1.0,))
    expected = binom.sf(59, 100, 0.5)
    assert tbl.iloc[0]["p_upper"] == pytest.approx(expected)
    assert tbl.iloc[0]["p_lower"] == pytest.approx(expected)


def test_p_upper_increases_with_gamma():
    yt, yc = pairs(120, 60)
    tbl = S.rosenbaum_bounds(yt, yc, gammas=(1.0, 1.5, 2.0, 3.0, 5.0))
    assert list(tbl["p_upper"]) == sorted(tbl["p_upper"])
    assert list(tbl["p_lower"]) == sorted(tbl["p_lower"], reverse=True)


def test_concordant_pairs_are_ignored():
    """Adding pairs where both units responded must not change the bound."""
    a = S.rosenbaum_bounds(*pairs(80, 40), gammas=(2.0,))
    b = S.rosenbaum_bounds(*pairs(80, 40, n_tied=5_000), gammas=(2.0,))
    assert a.iloc[0]["p_upper"] == pytest.approx(b.iloc[0]["p_upper"])


def test_no_discordant_pairs_raises():
    yt, yc = pairs(0, 0, n_tied=100)
    with pytest.raises(ValueError, match="discordant"):
        S.rosenbaum_bounds(yt, yc)


def test_gamma_critical_is_one_when_not_significant():
    yt, yc = pairs(52, 48)          # nowhere near significant
    assert S.gamma_critical(yt, yc) == 1.0


def test_gamma_critical_grows_with_effect_strength():
    weak = S.gamma_critical(*pairs(120, 80))
    strong = S.gamma_critical(*pairs(180, 20))
    assert 1.0 < weak < strong


def test_gamma_critical_is_the_crossing_point():
    """Just below the critical Gamma is significant; just above is not."""
    yt, yc = pairs(150, 50)
    g = S.gamma_critical(yt, yc, alpha=0.05)
    below = S.rosenbaum_bounds(yt, yc, gammas=(g - 0.01,)).iloc[0]["p_upper"]
    above = S.rosenbaum_bounds(yt, yc, gammas=(g + 0.01,)).iloc[0]["p_upper"]
    assert below < 0.05 <= above


def test_e_value_reference_values():
    """VanderWeele & Ding (2017): RR = 2 -> ~3.41, RR = 3.9 -> ~7.26."""
    assert S.e_value(2.0) == pytest.approx(3.414, abs=1e-3)
    assert S.e_value(3.9) == pytest.approx(7.263, abs=1e-3)


def test_e_value_is_one_at_the_null():
    assert S.e_value(1.0) == pytest.approx(1.0)


def test_e_value_is_symmetric_under_inversion():
    assert S.e_value(0.5) == pytest.approx(S.e_value(2.0))


def test_e_value_rejects_nonpositive_rr():
    with pytest.raises(ValueError):
        S.e_value(0.0)


def test_e_value_ci_is_one_when_interval_spans_null():
    assert S.e_value_ci(0.8, 1.4) == 1.0


def test_e_value_ci_uses_the_limit_nearest_the_null():
    assert S.e_value_ci(1.5, 4.0) == pytest.approx(S.e_value(1.5))
    assert S.e_value_ci(0.2, 0.6) == pytest.approx(S.e_value(0.6))


def test_e_value_ci_is_never_above_the_point_estimate_e_value():
    assert S.e_value_ci(1.5, 4.0) <= S.e_value(2.5)


def test_risk_ratio_conversion():
    assert S.risk_ratio_from_ate(0.001152, 0.00194) == pytest.approx(1.5938, abs=1e-4)
    with pytest.raises(ValueError):
        S.risk_ratio_from_ate(0.01, 0.0)
