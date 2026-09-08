"""Step 4 - heterogeneous treatment effects and uplift ranking.

CATE estimation plus the evaluation machinery that decides whether a ranking is
worth acting on. Evaluation runs on a held-out slice of the *randomised*
sample: Qini and uplift-decile curves need unconfounded data to be meaningful,
because each point compares treated and control outcomes inside a score bucket
and only randomisation makes that comparison causal.

A note on metrics, since the outcome rate here is 0.29%. Accuracy is useless
(predicting "never converts" scores 99.7%), and plain AUC is only mildly more
informative. Response quality is therefore reported as AUC-PR against the base
rate, and calibration is reported separately, because a model can rank well and
still produce probabilities that are badly scaled -- which matters the moment
anyone multiplies them by a margin to pick a bid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


def transformed_outcome(y: np.ndarray, t: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    """Athey-Imbens transformed outcome:  Z = Y (T - p) / (p (1 - p)).

    Its conditional mean is the CATE, E[Z | X] = tau(X), so any ordinary
    regression of Z on X estimates the effect surface directly -- no need to
    model the outcome level at all.

    The catch, and the reason it is worth showing next to a two-model
    approach: Z is enormously noisy. With p = 0.85 and a binary Y, a treated
    responder contributes Y/p ~ 1.18 while a control responder contributes
    -Y/(1-p) ~ -6.67. The estimator is unbiased and almost unusably
    high-variance, which is exactly the tradeoff worth seeing.
    """
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return np.asarray(y, dtype=np.float64) * (t - p) / (p * (1.0 - p))


@dataclass
class QiniResult:
    x: np.ndarray               # cumulative share of population targeted
    y: np.ndarray               # incremental positive outcomes gained
    qini_coefficient: float     # area between the curve and random targeting
    auuc: float                 # area under the uplift curve, normalised
    n: int

    def as_dict(self) -> dict:
        return {"qini_coefficient": self.qini_coefficient, "auuc": self.auuc, "n": self.n}


def qini_curve(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_points: int = 200
) -> QiniResult:
    """Qini curve for a uplift ranking.

    Sorting the population by predicted uplift descending, the curve at depth k
    is the incremental number of positive outcomes attributable to treating the
    top k:

        Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)

    The control response is rescaled by the treated/control ratio inside the
    top k rather than globally, which is what keeps the curve honest when a
    score happens to correlate with treatment assignment.

    The Qini coefficient normalises the area between this curve and the
    diagonal (random targeting) by the same area for the overall effect, so
    0 is random and higher is better.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = y.size

    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    ys, ts = y[order], t[order]

    nt = np.cumsum(ts)
    nc = np.cumsum(1.0 - ts)
    yt = np.cumsum(ys * ts)
    yc = np.cumsum(ys * (1.0 - ts))

    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt - yc * np.where(nc > 0, nt / np.maximum(nc, 1e-12), 0.0)
    q[~np.isfinite(q)] = 0.0

    # The random-targeting reference is the straight line to the endpoint.
    depth = np.arange(1, n + 1, dtype=np.float64) / n
    q_end = q[-1]
    random_line = depth * q_end

    area_model = np.trapezoid(q, depth)
    area_random = np.trapezoid(random_line, depth)
    qini_coef = float((area_model - area_random) / abs(area_random)) if area_random != 0 else float("nan")
    auuc = float(area_model / n)

    idx = np.unique(np.linspace(0, n - 1, n_points).astype(np.int64))
    return QiniResult(x=depth[idx], y=q[idx], qini_coefficient=qini_coef, auuc=auuc, n=n)


def uplift_by_decile(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Observed RCT uplift within bins of predicted uplift.

    The single most diagnostic table in uplift modelling: if the ranking is
    real, observed uplift decreases monotonically from bin 0 to bin 9. Bins are
    formed on the score, but every number in them is measured, not predicted.
    """
    from . import stats

    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t)
    score = np.asarray(score, dtype=np.float64)

    # Rank-based binning: robust to the score's scale and to heavy ties.
    order = np.argsort(-score, kind="stable")
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(len(score))
    b = np.minimum((rank * n_bins) // len(score), n_bins - 1)

    rows = []
    for k in range(n_bins):
        m = b == k
        est = stats.diff_in_means(y[m], t[m])
        rows.append({
            "bin": k,
            "n": int(m.sum()),
            "n_treated": est.n_treated,
            "n_control": est.n_control,
            "mean_predicted_uplift": float(score[m].mean()),
            "observed_uplift": est.estimate,
            "se": est.se,
            "ci_low": est.ci[0],
            "ci_high": est.ci[1],
            "control_rate": est.mean_control,
            "treated_rate": est.mean_treated,
        })
    return pd.DataFrame(rows)


def precision_at_k(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, ks: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
) -> pd.DataFrame:
    """Incremental performance from targeting only the top K of the ranking.

    "Precision" for uplift cannot mean the classification quantity: there is no
    label saying whether an individual was persuadable, only arm-level
    averages. What is reported instead, per depth K:

      * `uplift_at_k`      -- measured ATE inside the top K (the RCT contrast)
      * `lift_vs_overall`  -- that ATE divided by the ATE over everyone, i.e.
                              how much better than untargeted spend
      * `captured_share`   -- fraction of the campaign's total incremental
                              conversions that land inside the top K
    """
    from . import stats

    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t)
    n = y.size
    overall = stats.diff_in_means(y, t)
    total_incremental = overall.estimate * n

    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    rows = []
    for k in ks:
        top = order[: max(1, int(round(k * n)))]
        est = stats.diff_in_means(y[top], t[top])
        incremental = est.estimate * top.size
        rows.append({
            "k": k,
            "n_targeted": int(top.size),
            "uplift_at_k": est.estimate,
            "se": est.se,
            "lift_vs_overall": est.estimate / overall.estimate if overall.estimate else np.nan,
            "captured_share": incremental / total_incremental if total_incremental else np.nan,
            "control_rate": est.mean_control,
            "treated_rate": est.mean_treated,
        })
    return pd.DataFrame(rows)


def response_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Ranking and calibration quality for a response model on rare events."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y)
    base = float(y.mean())
    ap = float(average_precision_score(y, p))
    return {
        "base_rate": base,
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": ap,
        # AUC-PR of a random model equals the base rate, so the ratio is the
        # only version of the number that means anything at 0.29% prevalence.
        "auc_pr_lift_over_base": ap / base if base else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": float(np.mean(p)),
        "observed": base,
        "calibration_ratio": float(np.mean(p) / base) if base else float("nan"),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate by predicted-probability decile."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    rank = np.empty(len(p), dtype=np.int64)
    rank[order] = np.arange(len(p))
    b = np.minimum((rank * n_bins) // len(p), n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "observed_rate": float(y[m].mean())})
    return pd.DataFrame(rows)
