"""Step 2 - break randomisation on purpose.

The Criteo sample is randomised, so a naive treated-vs-control comparison is
already correct. To have anything to recover in Step 3 we have to destroy that
property deliberately, in a way whose exact mechanism we know.

The construction selects rows into an observational subsample with a
probability that depends on both the arm and a *prognostic score* -- a model of
the outcome fitted on control units only. Treated units with a low score are
dropped and control units with a high score are dropped, so the surviving
treated look systematically more promising than the surviving controls, and the
naive contrast is biased upward.

Two design points that are easy to get wrong:

1. Selection must depend on covariates that PREDICT THE OUTCOME. Selecting on a
   covariate unrelated to Y produces imbalance but no bias, which would make
   Step 3 a demonstration of nothing.

2. Selecting on X changes the covariate distribution, so the true ATE *in the
   subsample* is no longer the full-sample ATE whenever effects are
   heterogeneous. `rct_benchmark` recomputes the target by re-standardising
   stratum-level RCT effects onto the subsample's covariate mix. Scoring Step 3
   against the raw full-sample number instead would charge every estimator for
   a shift that is not bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import stats


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def rank_pct(x: np.ndarray) -> np.ndarray:
    """Fractional ranks in [0, 1). Ties broken arbitrarily but deterministically."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks / len(x)


@dataclass
class PrognosticScore:
    score: np.ndarray          # P(Y=1 | X), one per row
    auc_holdout: float
    ap_holdout: float
    outcome: str
    n_folds: int


def fit_prognostic_score(
    df: pd.DataFrame,
    outcome: str = "visit",
    seed: int = C.SEED,
    n_folds: int = 2,
    n_estimators: int = 200,
) -> PrognosticScore:
    """Cross-fitted P(Y=1 | X) estimated on CONTROL units only.

    Control-only because the prognostic score must describe the outcome under
    no treatment; fitting it on treated rows would fold the treatment effect
    into the thing we then select on.

    Cross-fitted so that no control row is scored by a model that saw it. An
    in-sample score would be sharper for the training half than the holdout
    half, and the selection would then partly reflect fold membership rather
    than covariates.

    `visit` is the default anchor rather than `conversion`: with 40k positives
    against 2k it is far better estimated, and it empirically ranks conversion
    *better* (AUC 0.948) than a conversion-trained model does (0.880).
    """
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    is_ctrl = df[C.TREATMENT].to_numpy() == 0
    ctrl_idx = np.flatnonzero(is_ctrl)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=ctrl_idx.size)

    X_all = df[C.FEATURES]
    score = np.zeros(len(df), dtype=np.float64)
    treated_preds = []
    aucs, aps = [], []

    for k in range(n_folds):
        tr = ctrl_idx[fold != k]
        te = ctrl_idx[fold == k]
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, num_leaves=63, learning_rate=0.08,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=seed + k,
        )
        model.fit(X_all.iloc[tr], df[outcome].to_numpy()[tr])
        p_te = model.predict_proba(X_all.iloc[te])[:, 1]
        score[te] = p_te
        y_te = df[outcome].to_numpy()[te]
        aucs.append(roc_auc_score(y_te, p_te))
        aps.append(average_precision_score(y_te, p_te))
        treated_preds.append(model.predict_proba(X_all.iloc[np.flatnonzero(~is_ctrl)])[:, 1])

    # Treated rows are never in any training fold, so averaging the fold models
    # is honest and lower-variance than picking one.
    score[~is_ctrl] = np.mean(treated_preds, axis=0)

    return PrognosticScore(
        score=score, auc_holdout=float(np.mean(aucs)), ap_holdout=float(np.mean(aps)),
        outcome=outcome, n_folds=n_folds,
    )


def selection_probability(
    score: np.ndarray, t: np.ndarray, strength: float, direction: str = "inflate"
) -> np.ndarray:
    """Per-row probability of surviving into the observational subsample.

    With z the score's rank rescaled to [-1, 1]:

        P(keep | treated) = expit(+k z)      P(keep | control) = expit(-k z)

    so treated units are retained preferentially at the top of the prognostic
    distribution and controls at the bottom. `strength` (k) is the single knob:
    k = 0 reproduces a random subsample and recovers the RCT, larger k buys more
    bias at the cost of overlap.

    direction="deflate" mirrors the rule, biasing the naive contrast downward --
    used as a robustness scenario so Step 3 cannot look good by luck of sign.
    """
    if direction not in {"inflate", "deflate"}:
        raise ValueError(f"direction must be 'inflate' or 'deflate', got {direction!r}")
    z = 2.0 * rank_pct(score) - 1.0
    sign = 1.0 if direction == "inflate" else -1.0
    p = np.where(t == 1, _expit(sign * strength * z), _expit(-sign * strength * z))
    return p


def select(
    score: np.ndarray, t: np.ndarray, strength: float, direction: str = "inflate",
    seed: int = C.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the observational subsample. Returns (boolean mask, keep probabilities).

    The keep probabilities are returned because they are the *true* propensity
    ingredients: any estimator in Step 3 that had access to them would be using
    an oracle. They are reported for diagnostics and never fed to an estimator.
    """
    p_keep = selection_probability(score, t, strength, direction)
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(len(p_keep)) < p_keep
    return mask, p_keep


def rct_benchmark(
    df: pd.DataFrame,
    mask: np.ndarray,
    score: np.ndarray,
    outcome: str,
    n_bins: int = 100,
) -> dict:
    """The true ATE for the SUBSAMPLE's covariate mix, computed from the RCT.

    Randomisation holds inside any stratum defined by covariates, so the
    full-sample difference in means within a score bin is an unbiased estimate
    of that stratum's effect. Re-weighting those stratum effects by the bin
    shares of the selected subsample gives the ATE the subsample's population
    actually has:

        tau_target = sum_b  w_b^selected * tau_b^RCT

    Approximation: effects are treated as constant within a bin. With 100
    quantile bins over 14M rows (~140k rows each) the residual heterogeneity
    inside a bin is small, and the gap between this and the full-sample ATE is
    reported so a reader can see how much it mattered.
    """
    t = df[C.TREATMENT].to_numpy()
    y = df[outcome].to_numpy(dtype=np.float64)

    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(score, edges[1:-1])

    # Per-bin arm moments via bincount rather than a Python loop over boolean
    # masks: one pass instead of n_bins passes over 14M rows.
    def moments(sel):
        idx, ys = b[sel], y[sel]
        n = np.bincount(idx, minlength=n_bins).astype(np.float64)
        s1 = np.bincount(idx, weights=ys, minlength=n_bins)
        s2 = np.bincount(idx, weights=ys * ys, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s1 / n
            var = (s2 - n * mean**2) / (n - 1)      # ddof=1
        return n, mean, var

    n1, m1, v1 = moments(t == 1)
    n0, m0, v0 = moments(t == 0)
    w = np.bincount(b[mask], minlength=n_bins).astype(np.float64)
    n_sel = w.sum()

    valid = (n1 >= 2) & (n0 >= 2) & (w > 0)
    skipped = int(((w > 0) & ~valid).sum())

    tau = m1[valid] - m0[valid]
    var_tau = v1[valid] / n1[valid] + v0[valid] / n0[valid]
    wv = w[valid] / w[valid].sum()          # renormalise if any bin dropped

    target = float(np.sum(wv * tau))
    se = float(np.sqrt(np.sum(wv**2 * var_tau)))
    return {
        "target_ate": target,
        "se": se,
        "ci_low": target - stats.Z95 * se,
        "ci_high": target + stats.Z95 * se,
        "n_bins": n_bins,
        "n_bins_used": int(valid.sum()),
        "n_bins_skipped": skipped,
        "weight_dropped": float(1.0 - w[valid].sum() / n_sel),
        "outcome": outcome,
    }


def overlap_diagnostics(p_keep: np.ndarray, mask: np.ndarray, t: np.ndarray) -> dict:
    """How much common support survives selection.

    Reported on the *realised* subsample: the share of treated units whose
    covariate neighbourhood still contains controls is what decides whether
    Step 3's estimators have anything to match or weight against.
    """
    ts = t[mask]
    return {
        "n_selected": int(mask.sum()),
        "selection_rate": float(mask.mean()),
        "n_treated": int((ts == 1).sum()),
        "n_control": int((ts == 0).sum()),
        "treated_share": float((ts == 1).mean()),
        "keep_prob_treated_mean": float(p_keep[t == 1].mean()),
        "keep_prob_control_mean": float(p_keep[t == 0].mean()),
    }
