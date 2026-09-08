"""Estimator primitives shared across steps.

Deliberately small and dependency-light: every number the README quotes should
be traceable to a function here that a reader can check in one sitting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

Z95 = 1.959963984540054


@dataclass
class Estimate:
    """A point estimate with an analytic standard error."""

    name: str
    estimate: float
    se: float
    n_treated: int
    n_control: int
    mean_treated: float
    mean_control: float
    relative_lift: float
    relative_lift_se: float

    @property
    def ci(self) -> tuple[float, float]:
        return (self.estimate - Z95 * self.se, self.estimate + Z95 * self.se)

    @property
    def relative_lift_ci(self) -> tuple[float, float]:
        lo = self.relative_lift - Z95 * self.relative_lift_se
        hi = self.relative_lift + Z95 * self.relative_lift_se
        return (lo, hi)

    @property
    def z(self) -> float:
        return self.estimate / self.se if self.se > 0 else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ci_low"], d["ci_high"] = self.ci
        d["relative_lift_ci_low"], d["relative_lift_ci_high"] = self.relative_lift_ci
        d["z"] = self.z
        return d

    def __str__(self) -> str:
        lo, hi = self.ci
        rl_lo, rl_hi = self.relative_lift_ci
        return (
            f"{self.name:<30} ATE = {self.estimate:+.6f}  "
            f"95% CI [{lo:+.6f}, {hi:+.6f}]  "
            f"lift = {self.relative_lift:+.2%} [{rl_lo:+.2%}, {rl_hi:+.2%}]"
        )


def diff_in_means(y: np.ndarray, t: np.ndarray, name: str = "difference in means") -> Estimate:
    """Unadjusted difference in means with an unpooled (Welch) standard error.

    Under randomisation this is an unbiased estimator of the ATE. The unpooled
    SE is used rather than the pooled/equal-variance one because the arms here
    are wildly unequal in size (~85/15) and, for a binary outcome, unequal in
    variance by construction whenever the effect is non-zero.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    n1, n0 = y1.size, y0.size
    if n1 < 2 or n0 < 2:
        raise ValueError(f"need >=2 units per arm, got n1={n1} n0={n0}")

    m1, m0 = y1.mean(), y0.mean()
    var1 = y1.var(ddof=1) / n1          # variance OF THE MEAN, treated arm
    var0 = y0.var(ddof=1) / n0          # variance OF THE MEAN, control arm
    ate = m1 - m0
    se = np.sqrt(var1 + var0)

    # Relative lift (m1/m0 - 1) via the delta method. Reported because an
    # absolute lift of 0.0007 on a 0.2% base rate is a ~35% relative move, and
    # the second number is the one a marketer actually acts on.
    if m0 > 0:
        rel = m1 / m0 - 1.0
        rel_se = np.sqrt(var1 / m0**2 + (m1**2) * var0 / m0**4)
    else:
        rel = rel_se = float("nan")

    return Estimate(
        name=name,
        estimate=float(ate),
        se=float(se),
        n_treated=int(n1),
        n_control=int(n0),
        mean_treated=float(m1),
        mean_control=float(m0),
        relative_lift=float(rel),
        relative_lift_se=float(rel_se),
    )


def smd(x: np.ndarray, t: np.ndarray) -> float:
    """Standardised mean difference: (m1 - m0) / sqrt((v1 + v0)/2).

    The standard covariate-balance diagnostic. Unlike a t-test it does not
    drift towards "significant" as n grows, which is exactly the property
    needed at 14M rows where every trivial imbalance has p < 1e-10.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t)
    x1, x0 = x[t == 1], x[t == 0]
    pooled = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((x1.mean() - x0.mean()) / pooled)


def variance_ratio(x: np.ndarray, t: np.ndarray) -> float:
    """Rubin's second balance criterion: var(treated)/var(control), target ~1."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t)
    v0 = x[t == 0].var(ddof=1)
    return float(x[t == 1].var(ddof=1) / v0) if v0 > 0 else float("nan")


def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and reliability-corrected weighted variance."""
    sw = w.sum()
    m = float((w * x).sum() / sw)
    v = float((w * (x - m) ** 2).sum() / (sw - (w**2).sum() / sw))
    return m, v


def balance_table(
    X: pd.DataFrame, t: np.ndarray, weights: np.ndarray | None = None
) -> pd.DataFrame:
    """Per-covariate balance diagnostics, optionally weighted (for IPW).

    Rubin's rules of thumb: |SMD| < 0.10 is good balance, < 0.25 tolerable;
    variance ratio should sit inside [0.5, 2].
    """
    t = np.asarray(t)
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(dtype=np.float64)
        if weights is None:
            m1, m0 = x[t == 1].mean(), x[t == 0].mean()
            d, vr = smd(x, t), variance_ratio(x, t)
        else:
            w = np.asarray(weights, dtype=np.float64)
            m1, v1 = _weighted_mean_var(x[t == 1], w[t == 1])
            m0, v0 = _weighted_mean_var(x[t == 0], w[t == 0])
            pooled = np.sqrt((v1 + v0) / 2.0)
            d = float((m1 - m0) / pooled) if pooled > 0 else 0.0
            vr = float(v1 / v0) if v0 > 0 else float("nan")
        rows.append(
            {
                "covariate": col,
                "mean_treated": m1,
                "mean_control": m0,
                "smd": d,
                "abs_smd": abs(d),
                "variance_ratio": vr,
            }
        )
    return pd.DataFrame(rows).set_index("covariate")


def lin_ate(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    chunk: int = 2_000_000,
) -> tuple[float, float]:
    """Lin (2013) interaction-adjusted ATE with HC1 robust standard errors.

    Regresses y on [1, T, Xc, T*Xc] where Xc is the fully-centred covariate
    matrix; the coefficient on T is then the ATE. Centring plus the full
    treatment interaction is what makes this guaranteed not to hurt asymptotic
    precision relative to the raw difference in means -- the classic objection
    to naive ANCOVA in a randomised trial.

    Solved by accumulating the (k x k) normal equations in chunks rather than
    materialising a 14M x 26 float64 design matrix, which would cost ~3 GB
    before statsmodels copies it. The result is exact, not approximate.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = len(y), X.shape[1]
    k = 2 + 2 * p
    means = X.mean(axis=0).to_numpy(dtype=np.float64)

    def design(sl: slice) -> np.ndarray:
        xc = X.iloc[sl].to_numpy(dtype=np.float64) - means
        ts = t[sl][:, None]
        return np.hstack([np.ones((xc.shape[0], 1)), ts, xc, ts * xc])

    # Pass 1: accumulate X'X and X'y.
    XtX = np.zeros((k, k))
    Xty = np.zeros(k)
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        d = design(sl)
        XtX += d.T @ d
        Xty += d.T @ y[sl]
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ Xty

    # Pass 2: the HC1 sandwich "meat", sum_i e_i^2 * d_i d_i'.
    meat = np.zeros((k, k))
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        d = design(sl)
        e = y[sl] - d @ beta
        meat += (d * (e**2)[:, None]).T @ d
    V = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    return float(beta[1]), float(np.sqrt(V[1, 1]))


def wald_ratio(
    itt: float, itt_se: float, compliance: float, compliance_se: float
) -> tuple[float, float]:
    """Wald/IV estimator: CACE = ITT / first-stage, with a delta-method SE.

    Valid under (a) random assignment, (b) the exclusion restriction -- being
    assigned to targeting moves the outcome only by actually serving an ad --
    and (c) monotonicity. Numerator and denominator come from the same sample,
    so this SE ignores their covariance; with a first stage this strong the
    correction is immaterial, but it is an approximation.
    """
    cace = itt / compliance
    se = np.sqrt(
        itt_se**2 / compliance**2 + (itt**2) * compliance_se**2 / compliance**4
    )
    return float(cace), float(se)


def cluster_robust_diff_in_means(
    y: np.ndarray,
    t: np.ndarray,
    cluster: np.ndarray,
    name: str = "diff-in-means (cluster-robust)",
) -> Estimate:
    """Difference in means with a CR1 cluster-robust standard error.

    The Criteo file is one row per *impression*, not per user: 14.0M rows carry
    only ~9.2M distinct covariate vectors, and treated units recur more often
    than control ones. Rows within a cluster are therefore not independent, and
    the classical Welch SE understates the true sampling variability.

    This fits y = a + b*T by OLS -- so the point estimate is numerically
    identical to `diff_in_means` -- and replaces the variance with the
    Liang-Zeger sandwich, summing the score vector within each cluster before
    taking the outer product.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    cluster = np.asarray(cluster)

    base = diff_in_means(y, t, name=name)
    n = y.size
    n1 = int(t.sum())
    n0 = n - n1

    # Residuals from the two-parameter model are just within-arm demeaning.
    resid = y - np.where(t == 1, base.mean_treated, base.mean_control)

    # Cluster sums of the score vector d_i * e_i, where d_i = [1, T_i].
    order = np.argsort(cluster, kind="stable")
    c_sorted = cluster[order]
    s0 = np.add.reduceat(resid[order], np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1]]))
    s1 = np.add.reduceat(
        (resid * t)[order], np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1]])
    )
    G = s0.size

    meat = np.array([[np.dot(s0, s0), np.dot(s0, s1)], [np.dot(s0, s1), np.dot(s1, s1)]])
    XtX = np.array([[float(n), float(n1)], [float(n1), float(n1)]])
    bread = np.linalg.inv(XtX)
    corr = (G / (G - 1)) * ((n - 1) / (n - 2))
    V = bread @ meat @ bread * corr

    base.name = name
    base.se = float(np.sqrt(V[1, 1]))
    # Relative-lift SE rescaled by the same design effect.
    if np.isfinite(base.relative_lift_se):
        deff = base.se / np.sqrt(
            np.asarray(y)[t == 1].var(ddof=1) / n1 + np.asarray(y)[t == 0].var(ddof=1) / n0
        )
        base.relative_lift_se = float(base.relative_lift_se * deff)
    return base
