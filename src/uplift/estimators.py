"""Step 3 - estimators that try to recover the truth from confounded data.

Propensity modelling, matching, IPW, and an augmented (doubly-robust)
estimator, plus the balance diagnostics that say whether any of them had a
right to work.

Two conventions run through the module:

* **Cross-fitting.** Every nuisance function -- the propensity e(X) and the
  outcome regressions mu_0(X), mu_1(X) -- is fitted out-of-fold. Plugging
  in-sample ML predictions into a doubly-robust estimator reintroduces
  first-order bias through overfitting, which is exactly the failure the
  double-robustness property is supposed to protect against.

* **Influence functions.** Where an estimator has one, the standard error comes
  from it, which makes clustering a one-line change: sum the influence
  contributions within a cluster before taking the variance. Given the Criteo
  file is impression-level (Step 1), that correction is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import stats


# --------------------------------------------------------------------------
# variance helpers
# --------------------------------------------------------------------------

def influence_se(psi: np.ndarray, cluster: np.ndarray | None = None) -> float:
    """SE of a mean-of-influence-function estimator, optionally clustered.

    For an estimator with theta_hat = mean(psi_i), Var = Var(psi)/n. Under
    clustering the independent unit is the cluster, so influence contributions
    are summed within cluster first:  Var = (1/n^2) * sum_g (sum_{i in g} psi_i)^2
    after centring, with the usual G/(G-1) correction.
    """
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.size
    if cluster is None:
        return float(np.std(psi, ddof=1) / np.sqrt(n))

    cluster = np.asarray(cluster)
    centred = psi - psi.mean()
    order = np.argsort(cluster, kind="stable")
    cs = cluster[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    g_sums = np.add.reduceat(centred[order], starts)
    G = g_sums.size
    var = (g_sums**2).sum() / n**2 * (G / (G - 1))
    return float(np.sqrt(var))


@dataclass
class Result:
    """One estimate, with everything a reader needs to judge it."""

    method: str
    estimate: float
    se: float
    n_used: int
    outcome: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def ci(self) -> tuple[float, float]:
        return (self.estimate - stats.Z95 * self.se, self.estimate + stats.Z95 * self.se)

    def as_dict(self) -> dict:
        lo, hi = self.ci
        return {
            "method": self.method, "outcome": self.outcome,
            "estimate": self.estimate, "se": self.se,
            "ci_low": lo, "ci_high": hi, "n_used": self.n_used, **self.notes,
        }


# --------------------------------------------------------------------------
# propensity
# --------------------------------------------------------------------------

@dataclass
class Propensity:
    e: np.ndarray                 # out-of-fold P(T=1 | X)
    method: str
    auc: float
    brier: float
    calibration_slope: float
    n_folds: int


def fit_propensity(
    X: pd.DataFrame,
    t: np.ndarray,
    method: str = "lgbm",
    n_folds: int = 3,
    seed: int = C.SEED,
    n_estimators: int = 300,
) -> Propensity:
    """Cross-fitted propensity score.

    Two methods are offered because the comparison is the point. `logit` is a
    correctly-specified-if-you-are-lucky parametric model; `lgbm` is flexible.
    The Step 2 selection rule is a logistic function of the *rank* of a
    gradient-boosted score, which is monotone but very much not linear in X --
    so the linear model is misspecified by construction and its failure is
    informative rather than a bug.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(t)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=n)
    e = np.zeros(n, dtype=np.float64)

    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if method == "lgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators, num_leaves=63, learning_rate=0.08,
                min_child_samples=200, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=seed + k,
            )
        elif method == "logit":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0),
            )
        else:
            raise ValueError(f"unknown method {method!r}")
        model.fit(X[tr], t[tr])
        e[te] = model.predict_proba(X[te])[:, 1]

    e = np.clip(e, 1e-6, 1 - 1e-6)

    # Calibration slope: the coefficient from a LOGISTIC regression of T on
    # logit(e). Perfect calibration gives exactly 1; < 1 means the scores are
    # over-dispersed (too confident), > 1 under-dispersed. It must be the
    # logistic fit, not a linear one -- a linear probability model on the logit
    # scale has no reason to give 1 even for a perfectly calibrated score.
    #
    # IPW divides by these numbers, so calibration matters more here than
    # discrimination does: a model can rank perfectly and still ruin weights.
    lo = np.log(e / (1 - e))
    slope = float(
        LogisticRegression(max_iter=1000, C=1e9)
        .fit(lo.reshape(-1, 1), t)
        .coef_[0][0]
    )

    return Propensity(
        e=e, method=method, auc=float(roc_auc_score(t, e)),
        brier=float(brier_score_loss(t, e)), calibration_slope=slope, n_folds=n_folds,
    )


def overlap_mask(e: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    """Common-support trimming.

    Trimming is not free: it changes the estimand to the ATE on the trimmed
    population. Step 3 handles that by recomputing the RCT target on exactly
    the retained rows rather than pretending the estimand is unchanged.
    """
    return (e >= lo) & (e <= hi)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def ipw_ate(
    y: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    cluster: np.ndarray | None = None,
    stabilized: bool = True,
    outcome: str = "",
) -> Result:
    """Inverse-probability weighting (Hajek form when stabilized).

    The Hajek/stabilized version normalises by the sum of weights rather than
    n. It is what almost everyone should use: the Horvitz-Thompson form is
    unbiased but has no upper bound on variance when any e is near 0 or 1, and
    with 7M rows and an aggressive selection rule some always are.

    The SE treats e as known. Ignoring the estimation of e is conservative for
    IPW -- the true asymptotic variance with an estimated propensity is smaller
    -- so the intervals reported here are, if anything, too wide.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = y.size
    w1, w0 = t / e, (1 - t) / (1 - e)

    if stabilized:
        m1 = (w1 * y).sum() / w1.sum()
        m0 = (w0 * y).sum() / w0.sum()
        psi = w1 * (y - m1) / w1.mean() - w0 * (y - m0) / w0.mean()
    else:
        m1, m0 = (w1 * y).mean(), (w0 * y).mean()
        psi = w1 * y - w0 * y

    est = m1 - m0
    return Result(
        method=f"IPW ({'stabilized' if stabilized else 'Horvitz-Thompson'})",
        estimate=float(est), se=influence_se(psi, cluster), n_used=n, outcome=outcome,
        notes={"mean_treated": float(m1), "mean_control": float(m0),
               "max_weight": float(np.maximum(w1, w0).max()),
               "ess_treated": float(w1.sum() ** 2 / (w1**2).sum()),
               "ess_control": float(w0.sum() ** 2 / (w0**2).sum())},
    )


def aipw_ate(
    y: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    cluster: np.ndarray | None = None,
    outcome: str = "",
) -> Result:
    """Augmented IPW / doubly-robust estimator.

        psi_i = mu1_i - mu0_i
                + T_i (Y_i - mu1_i)/e_i
                - (1-T_i)(Y_i - mu0_i)/(1-e_i)

    Consistent if EITHER the propensity or the outcome model is right, and with
    cross-fitted nuisances it is asymptotically normal at root-n even when both
    are ML. The SE is the empirical standard deviation of psi, which is the
    genuine article rather than a plug-in: the influence function accounts for
    the nuisance estimation under cross-fitting.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    psi = (mu1 - mu0) + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return Result(
        method="AIPW (doubly robust)", estimate=float(psi.mean()),
        se=influence_se(psi, cluster), n_used=y.size, outcome=outcome,
        notes={"mean_mu1": float(mu1.mean()), "mean_mu0": float(mu0.mean()),
               "plugin_gcomp": float((mu1 - mu0).mean())},
    )


def gcomputation_ate(
    mu0: np.ndarray, mu1: np.ndarray, cluster: np.ndarray | None = None, outcome: str = ""
) -> Result:
    """Outcome-regression / g-computation plug-in: mean(mu1 - mu0).

    Included as the single-robustness counterpart to AIPW. Its SE is the naive
    one for a mean and does NOT account for having estimated mu; it is reported
    to show the point estimate, not to be trusted as an interval.
    """
    d = mu1 - mu0
    return Result(
        method="g-computation (outcome regression)", estimate=float(d.mean()),
        se=influence_se(d, cluster), n_used=d.size, outcome=outcome,
        notes={"se_understates": True},
    )


def fit_outcome_models(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
    seed: int = C.SEED,
    n_estimators: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted mu_0(X) = E[Y|X,T=0] and mu_1(X) = E[Y|X,T=1].

    Fitted as two separate models (a T-learner) rather than one model with T as
    a feature: with a 0.29% outcome rate and strong confounding, a single model
    tends to spend its splits on T and underfit the arm-specific surfaces.
    """
    import lightgbm as lgb

    n = len(y)
    rng = np.random.default_rng(seed + 7)
    fold = rng.integers(0, n_folds, size=n)
    mu0 = np.zeros(n, dtype=np.float64)
    mu1 = np.zeros(n, dtype=np.float64)

    for k in range(n_folds):
        tr, te = fold != k, fold == k
        for arm, out in ((0, mu0), (1, mu1)):
            sel = tr & (t == arm)
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators, num_leaves=63, learning_rate=0.08,
                min_child_samples=200, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=seed + k + arm,
            )
            model.fit(X[sel], y[sel])
            out[te] = model.predict_proba(X[te])[:, 1]
    return mu0, mu1


def psm_ate(
    y: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    caliper_sd: float = 0.2,
    outcome: str = "",
) -> tuple[Result, dict]:
    """1:1 nearest-neighbour propensity matching with replacement, on the logit.

    Matching happens on logit(e) rather than e because the logit scale spreads
    out the extremes where the interesting units live, and the conventional
    caliper (0.2 standard deviations of the logit) is defined there.

    Scales to millions of rows because the match is one-dimensional: sort the
    opposite arm once and use binary search, O(n log n), instead of a
    high-dimensional neighbour query.

    ATE is assembled from both directions -- treated matched to controls gives
    the ATT, controls matched to treated the ATC -- and combined by arm share.
    Reporting only the ATT, as most PSM write-ups do, would answer a different
    question from the one Step 1 poses.

    Caveat stated plainly: the SE below treats matched pairs as independent
    observations. It ignores both the estimation of e and the reuse of control
    units under matching-with-replacement, for which the bootstrap is known to
    be invalid (Abadie & Imbens 2008). Treat it as indicative.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t)
    lo = np.log(e / (1 - e))
    caliper = caliper_sd * lo.std(ddof=1)

    idx1 = np.flatnonzero(t == 1)
    idx0 = np.flatnonzero(t == 0)

    def match(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest neighbour in `dst` for every element of `src`, within caliper."""
        d_sorted = np.argsort(lo[dst], kind="stable")
        d_idx = dst[d_sorted]
        d_val = lo[d_idx]
        pos = np.searchsorted(d_val, lo[src])
        pos = np.clip(pos, 1, len(d_val) - 1)
        left, right = pos - 1, pos
        pick = np.where(
            np.abs(lo[src] - d_val[left]) <= np.abs(lo[src] - d_val[right]), left, right
        )
        partner = d_idx[pick]
        ok = np.abs(lo[src] - lo[partner]) <= caliper
        return src[ok], partner[ok]

    a_src, a_dst = match(idx1, idx0)          # ATT
    c_src, c_dst = match(idx0, idx1)          # ATC
    att_pairs = y[a_src] - y[a_dst]
    atc_pairs = y[c_dst] - y[c_src]

    n1m, n0m = att_pairs.size, atc_pairs.size
    att, atc = att_pairs.mean(), atc_pairs.mean()
    w1 = n1m / (n1m + n0m)
    ate = w1 * att + (1 - w1) * atc
    se = float(np.sqrt(
        w1**2 * att_pairs.var(ddof=1) / n1m + (1 - w1) ** 2 * atc_pairs.var(ddof=1) / n0m
    ))

    diag = {
        "caliper_logit": float(caliper),
        "n_treated": int(idx1.size), "n_treated_matched": int(n1m),
        "n_control": int(idx0.size), "n_control_matched": int(n0m),
        "treated_match_rate": float(n1m / idx1.size),
        "control_match_rate": float(n0m / idx0.size),
        "att": float(att), "atc": float(atc),
        "matched_treated_idx": a_src, "matched_control_idx": a_dst,
    }
    res = Result(
        method="PSM (1:1 NN, caliper, w/ replacement)", estimate=float(ate), se=se,
        n_used=int(n1m + n0m), outcome=outcome,
        notes={k: v for k, v in diag.items() if not k.startswith("matched_")},
    )
    return res, diag


def weighted_balance(X: pd.DataFrame, t: np.ndarray, e: np.ndarray) -> pd.DataFrame:
    """Covariate balance under ATE (inverse-probability) weights."""
    w = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    return stats.balance_table(X, t, weights=w)
