"""The servable uplift model: train, persist, load, score.

Deliberately small. At this level "production" means something else depends on
the model, not that the model owns a platform -- so this is one artefact, one
schema, one set of decile thresholds, and a metadata block that records what
the artefact was trained on and how well it scored.

The transformed-outcome learner is what gets served. Step 4 measured it at Qini
+0.659 against +0.003 for a two-model T-learner on the same split: with a 0.29%
conversion rate, modelling the effect directly beats differencing two outcome
models that are each larger than the effect they bracket.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C

ARTIFACT_VERSION = 1


@dataclass
class ModelMetadata:
    """Everything needed to answer "what is this thing and should I trust it?"."""

    version: int
    model_id: str
    trained_at: str
    n_train: int
    n_eval: int
    outcome: str
    features: list[str]
    treatment_share: float
    metrics: dict = field(default_factory=dict)
    decile_thresholds: list[float] = field(default_factory=list)
    # Share of TRAINING rows landing in each decile. Not 10% each: the score has
    # heavy ties (identical covariate vectors give identical predictions), so
    # quantile thresholds collide and some deciles absorb their neighbours.
    # Drift must be measured against what training actually produced.
    decile_train_shares: list[float] = field(default_factory=list)
    feature_reference: dict = field(default_factory=dict)
    # Binned empirical distribution per feature, for PSI. Storing edges plus the
    # expected mass makes drift exact; a mean/sd-only reference forces the
    # monitor to assume normality, and these features are far enough from normal
    # that it reported PSI > 11 against its own training data.
    reference_bins: dict = field(default_factory=dict)
    training_data_sha: str = ""
    library_versions: dict = field(default_factory=dict)
    artifact_version: int = ARTIFACT_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


class UpliftModel:
    """Wraps the booster with the schema and thresholds a caller needs."""

    def __init__(self, booster, meta: ModelMetadata):
        self.booster = booster
        self.meta = meta

    # -- inference --------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted incremental conversion probability per row."""
        missing = [f for f in self.meta.features if f not in X.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        return self.booster.predict(X[self.meta.features])

    def decile(self, scores: np.ndarray) -> np.ndarray:
        """Map scores to 0-9 using thresholds frozen at training time.

        Frozen rather than recomputed per request: a batch's own quantiles
        would make one user's decile depend on who else happened to be scored
        with them, which is not something a downstream bidder can reason about.
        """
        th = np.asarray(self.meta.decile_thresholds)
        return 9 - np.searchsorted(th, np.asarray(scores), side="right")

    def recommend(self, scores: np.ndarray, target_top_deciles: int = 3) -> np.ndarray:
        d = self.decile(scores)
        return np.where(d < target_top_deciles, "target",
                        np.where(d < 6, "test", "hold_out"))

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.booster_.save_model(str(path / "model.txt"))
        (path / "metadata.json").write_text(json.dumps(self.meta.as_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "UpliftModel":
        import lightgbm as lgb

        path = Path(path)
        meta = ModelMetadata(**json.loads((path / "metadata.json").read_text()))
        if meta.artifact_version != ARTIFACT_VERSION:
            raise ValueError(
                f"artifact version {meta.artifact_version} != expected {ARTIFACT_VERSION}"
            )
        booster = lgb.Booster(model_file=str(path / "model.txt"))

        class _Wrap:
            def __init__(self, b):
                self.booster_ = b

            def predict(self, X):
                return self.booster_.predict(X)

        return cls(_Wrap(booster), meta)


def train(
    X: pd.DataFrame,
    t: np.ndarray,
    y: np.ndarray,
    outcome: str,
    seed: int = C.SEED,
    n_estimators: int = 300,
) -> UpliftModel:
    """Fit the transformed-outcome regressor and freeze its decile thresholds."""
    import lightgbm as lgb

    from ..uplift import transformed_outcome

    p_treat = float(np.mean(t))
    z = transformed_outcome(y, t, p_treat)
    booster = lgb.LGBMRegressor(
        n_estimators=n_estimators, num_leaves=63, learning_rate=0.08,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=seed,
    )
    booster.fit(X[C.FEATURES], z)

    train_scores = booster.predict(X[C.FEATURES])
    # Ascending thresholds at the 10th..90th percentile; `decile()` inverts.
    thresholds = np.quantile(train_scores, np.arange(1, 10) / 10.0).tolist()
    train_deciles = 9 - np.searchsorted(np.asarray(thresholds), train_scores, side="right")
    decile_shares = (np.bincount(train_deciles, minlength=10) / len(train_scores)).tolist()

    n_bins = 20
    reference_bins = {}
    for f in C.FEATURES:
        col = X[f].to_numpy(dtype=np.float64)
        edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
        inner = edges[1:-1] if edges.size >= 3 else np.array([col.mean()])
        counts = np.bincount(np.digitize(col, inner), minlength=inner.size + 1).astype(float)
        reference_bins[f] = {
            "edges": inner.tolist(),
            "expected": (counts / counts.sum()).tolist(),
        }

    import lightgbm

    meta = ModelMetadata(
        version=int(time.time()),
        model_id=f"uplift-{outcome}-{time.strftime('%Y%m%d-%H%M%S')}",
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        n_train=len(X), n_eval=0, outcome=outcome,
        features=list(C.FEATURES), treatment_share=p_treat,
        decile_thresholds=thresholds,
        decile_train_shares=decile_shares,
        reference_bins=reference_bins,
        feature_reference={
            f: {"mean": float(X[f].mean()), "std": float(X[f].std()),
                "q01": float(X[f].quantile(0.01)), "q99": float(X[f].quantile(0.99))}
            for f in C.FEATURES
        },
        library_versions={"lightgbm": lightgbm.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__,
                          "python": platform.python_version()},
    )
    return UpliftModel(booster, meta)


def data_fingerprint(X: pd.DataFrame, y: np.ndarray) -> str:
    """Cheap, order-sensitive fingerprint of the training data."""
    h = hashlib.sha256()
    h.update(str(X.shape).encode())
    h.update(np.ascontiguousarray(X.head(1000).to_numpy(np.float32)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y[:1000])).tobytes())
    return h.hexdigest()[:16]
