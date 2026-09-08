"""Sensitivity analysis: how much hidden bias would overturn the conclusion?

Steps 1-3 answer "which estimators recover a known answer". That question is
only available because the answer is known. In real observational work it is
not, and the useful question becomes the contrapositive: an unmeasured
confounder of what strength would be needed to explain away the effect I am
reporting?

Two complementary answers, because they assume different things:

* **Rosenbaum bounds** (`rosenbaum_bounds`) operate on matched pairs and ask
  how far the treatment-assignment odds could differ between two units that
  look identical on X before the inference stops being significant. The output
  is Gamma -- an odds ratio on *assignment*, saying nothing about how the
  confounder relates to the outcome.

* **E-values** (`e_value`) ask instead for the minimum strength of association,
  on the risk-ratio scale, that a confounder would need with BOTH treatment and
  outcome to explain the observed effect away. No matching required, and it is
  reported on the same scale practitioners already reason about.

Neither is a test for confounding. Both are statements of the form "a confounder
this strong would be required", and the reader still has to judge whether one
that strong is plausible in the domain. That judgement is not statistical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom


def rosenbaum_bounds(
    y_treated: np.ndarray,
    y_control: np.ndarray,
    gammas: tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rosenbaum bounds for matched pairs with a binary outcome.

    Uses discordant pairs only -- those where exactly one of the matched units
    responded -- which is McNemar's test. Concordant pairs carry no information
    about the direction of the effect and drop out of the statistic entirely.

    Under no hidden bias each discordant pair is equally likely to favour the
    treated or the control unit. If an unobserved confounder makes one unit up
    to `Gamma` times more likely to be treated than its match, that probability
    is only bounded:

        1 / (1 + Gamma)  <=  pi  <=  Gamma / (1 + Gamma)

    The upper-bound p-value takes pi at its maximum, which is the least
    favourable case for rejecting the null. Gamma = 1 reproduces the ordinary
    randomisation test.
    """
    d = np.asarray(y_treated, dtype=float) - np.asarray(y_control, dtype=float)
    n_plus = int((d > 0).sum())
    n_minus = int((d < 0).sum())
    n_disc = n_plus + n_minus
    if n_disc == 0:
        raise ValueError("no discordant pairs; the bound is undefined")

    rows = []
    for g in gammas:
        p_hi = g / (1.0 + g)
        p_lo = 1.0 / (1.0 + g)
        # P(X >= n_plus) with X ~ Binomial(n_disc, p) = survival at n_plus - 1.
        p_upper = float(binom.sf(n_plus - 1, n_disc, p_hi))
        p_lower = float(binom.sf(n_plus - 1, n_disc, p_lo))
        rows.append({
            "gamma": g,
            "p_upper": p_upper,
            "p_lower": p_lower,
            "significant_at_alpha": p_upper < alpha,
        })
    out = pd.DataFrame(rows)
    out.attrs.update({"n_discordant": n_disc, "n_plus": n_plus, "n_minus": n_minus,
                      "alpha": alpha})
    return out


def gamma_critical(
    y_treated: np.ndarray,
    y_control: np.ndarray,
    alpha: float = 0.05,
    hi: float = 100.0,
    tol: float = 1e-3,
) -> float:
    """The Gamma at which the upper-bound p-value first reaches `alpha`.

    Read as: "a hidden confounder would have to shift the odds of treatment by
    at least this factor, between two units identical on the observed
    covariates, before this result stops being significant."

    Returns 1.0 when the result is not significant even without hidden bias,
    and `hi` when it survives the whole search range.
    """
    d = np.asarray(y_treated, dtype=float) - np.asarray(y_control, dtype=float)
    n_plus = int((d > 0).sum())
    n_disc = n_plus + int((d < 0).sum())
    if n_disc == 0:
        raise ValueError("no discordant pairs; the bound is undefined")

    def p_up(g: float) -> float:
        return float(binom.sf(n_plus - 1, n_disc, g / (1.0 + g)))

    if p_up(1.0) >= alpha:
        return 1.0
    if p_up(hi) < alpha:
        return hi
    lo = 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if p_up(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return float(lo)


def e_value(rr: float) -> float:
    """E-value for a point estimate on the risk-ratio scale (VanderWeele & Ding).

        E = RR + sqrt(RR * (RR - 1))       for RR >= 1

    Protective effects are inverted first, so the result is always >= 1 and is
    read the same way in both directions. An E-value of 1 means no unmeasured
    confounding at all would be needed -- i.e. there is no effect to explain.
    """
    rr = float(rr)
    if rr <= 0:
        raise ValueError(f"risk ratio must be positive, got {rr}")
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def e_value_ci(rr_low: float, rr_high: float, null: float = 1.0) -> float:
    """E-value for the confidence limit nearest the null.

    Usually the more useful of the two numbers: it says what an unmeasured
    confounder would have to do to push the *interval* across the null, not
    merely to move the point estimate.

    Takes both limits so the side does not have to be inferred. Returns 1.0
    when the interval already contains the null -- there is then nothing for a
    confounder to explain away.
    """
    rr_low, rr_high = float(rr_low), float(rr_high)
    if rr_low <= 0 or rr_high <= 0:
        raise ValueError("risk-ratio limits must be positive")
    if rr_low <= null <= rr_high:
        return 1.0
    return e_value(rr_low if rr_low > null else rr_high)


def risk_ratio_from_ate(ate: float, control_rate: float) -> float:
    """Convert an absolute risk difference to a risk ratio.

    E-values are defined on the ratio scale, and every estimate in this project
    is a risk difference, so the conversion needs the baseline rate to be
    explicit rather than implied.
    """
    if control_rate <= 0:
        raise ValueError("control rate must be positive")
    return float((control_rate + ate) / control_rate)
