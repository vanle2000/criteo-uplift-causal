"""Tests for the scoring service and the model artefact.

Covers the contract a caller depends on: the schema is enforced, bad input is
rejected with 4xx rather than a 500, deciles come from the frozen thresholds
rather than the batch, and a saved model round-trips to identical predictions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift import config as C  # noqa: E402
from uplift.serving import model as M  # noqa: E402

MODEL_DIR = ROOT / "models" / "current"
needs_model = pytest.mark.skipif(
    not (MODEL_DIR / "metadata.json").exists(),
    reason="no trained model; run scripts/retrain.py",
)


@pytest.fixture(scope="module")
def toy_model(tmp_path_factory):
    """A tiny model trained on synthetic data, for tests that need no artefact."""
    rng = np.random.default_rng(0)
    n = 20_000
    X = pd.DataFrame(rng.normal(size=(n, len(C.FEATURES))), columns=C.FEATURES)
    t = rng.binomial(1, 0.85, n)
    tau = 0.02 * (X["f0"] > 0)
    y = rng.binomial(1, np.clip(0.01 + tau * t, 0, 1))
    return M.train(X, t, y, outcome="conversion", n_estimators=30)


def test_train_produces_usable_metadata(toy_model):
    meta = toy_model.meta
    assert meta.features == list(C.FEATURES)
    assert len(meta.decile_thresholds) == 9
    assert meta.decile_thresholds == sorted(meta.decile_thresholds)
    assert 0 < meta.treatment_share < 1
    assert set(meta.feature_reference) == set(C.FEATURES)


def test_decile_uses_frozen_thresholds(toy_model):
    """A user's decile must not depend on who else is in the batch."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(500, len(C.FEATURES))), columns=C.FEATURES)
    s = toy_model.predict(X)
    alone = toy_model.decile(s[:1])
    together = toy_model.decile(s)[:1]
    assert alone == together
    assert set(np.unique(toy_model.decile(s))) <= set(range(10))


def test_recommend_maps_deciles_to_actions(toy_model):
    s = np.array(toy_model.meta.decile_thresholds + [1e9, -1e9])
    acts = set(toy_model.recommend(s).tolist())
    assert acts <= {"target", "test", "hold_out"}


def test_predict_rejects_missing_features(toy_model):
    X = pd.DataFrame(np.zeros((3, 2)), columns=["f0", "f1"])
    with pytest.raises(ValueError, match="missing features"):
        toy_model.predict(X)


def test_model_roundtrip_is_exact(toy_model, tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(200, len(C.FEATURES))), columns=C.FEATURES)
    before = toy_model.predict(X)
    toy_model.save(tmp_path / "m")
    after = M.UpliftModel.load(tmp_path / "m").predict(X)
    np.testing.assert_allclose(before, after, rtol=1e-10)


def test_load_rejects_wrong_artifact_version(toy_model, tmp_path):
    toy_model.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    meta["artifact_version"] = 999
    (tmp_path / "m" / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="artifact version"):
        M.UpliftModel.load(tmp_path / "m")


# --------------------------------------------------------------------------
# HTTP contract
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from uplift.serving import app as appmod

    with TestClient(appmod.app) as c:
        yield c


@needs_model
def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_id"]


@needs_model
def test_model_endpoint_exposes_metadata(client):
    body = client.get("/model").json()
    assert body["features"] == list(C.FEATURES)
    assert "qini_coefficient" in body["metrics"]


@needs_model
def test_score_returns_one_result_per_row(client):
    rng = np.random.default_rng(3)
    rows = [{f: float(v) for f, v in zip(C.FEATURES, rng.normal(size=12))}
            for _ in range(5)]
    r = client.post("/score", json={"rows": rows})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5 and len(body["results"]) == 5
    for res in body["results"]:
        assert 0 <= res["decile"] <= 9
        assert res["action"] in {"target", "test", "hold_out"}


@needs_model
def test_score_rejects_missing_features_with_422(client):
    r = client.post("/score", json={"rows": [{"f0": 1.0}]})
    assert r.status_code == 422


@needs_model
def test_score_rejects_non_finite_with_422(client):
    rows = [{f: 0.0 for f in C.FEATURES}]
    rows[0]["f3"] = "not a number"
    assert client.post("/score", json={"rows": rows}).status_code == 422


@needs_model
def test_score_rejects_empty_batch(client):
    assert client.post("/score", json={"rows": []}).status_code == 422
