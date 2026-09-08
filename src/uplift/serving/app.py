"""FastAPI service exposing the uplift model.

Scope is deliberate. Something else depends on this model, so it needs a stable
contract, a health check that actually reflects readiness, and enough
observability to debug a bad response. It does not need an authn stack, a
feature store, or an autoscaler -- those belong to whoever runs the platform.

Endpoints
    GET  /health   liveness + whether a model is loaded
    GET  /model    the metadata block: what was trained, when, on what, how well
    POST /score    batch scoring -> uplift, decile, recommended action
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .. import config as C
from .model import UpliftModel

MODEL_DIR = Path(os.environ.get("UPLIFT_MODEL_DIR", C.PROJECT_ROOT / "models" / "current"))
MAX_BATCH = int(os.environ.get("UPLIFT_MAX_BATCH", "10000"))

_state: dict = {"model": None, "loaded_at": None, "error": None, "n_scored": 0}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once at startup; a per-request load would dominate latency.

    A load failure is captured rather than raised so the process still starts
    and /health can report *why* it is unhealthy. A container that exits on a
    missing artefact just crash-loops with the reason buried in the logs.
    """
    try:
        _state["model"] = UpliftModel.load(MODEL_DIR)
        _state["loaded_at"] = time.time()
        _state["error"] = None
    except Exception as exc:  # noqa: BLE001 - surfaced via /health
        _state["model"] = None
        _state["error"] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="Criteo uplift scoring",
    version="1.0",
    description="Predicted incremental conversion probability per user.",
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    rows: list[dict] = Field(..., description="One object per user, keyed by feature name")

    @field_validator("rows")
    @classmethod
    def _check(cls, v):
        if not v:
            raise ValueError("rows must not be empty")
        if len(v) > MAX_BATCH:
            raise ValueError(f"batch of {len(v)} exceeds limit {MAX_BATCH}")
        return v


class ScoredRow(BaseModel):
    uplift: float
    decile: int
    action: str


class ScoreResponse(BaseModel):
    model_id: str
    n: int
    latency_ms: float
    results: list[ScoredRow]


@app.get("/health")
def health() -> dict:
    ok = _state["model"] is not None
    return {
        "status": "ok" if ok else "unavailable",
        "model_loaded": ok,
        "model_id": _state["model"].meta.model_id if ok else None,
        "loaded_at": _state["loaded_at"],
        "n_scored": _state["n_scored"],
        "error": _state["error"],
    }


@app.get("/model")
def model_info() -> dict:
    if _state["model"] is None:
        raise HTTPException(503, detail=f"no model loaded: {_state['error']}")
    return _state["model"].meta.as_dict()


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(503, detail=f"no model loaded: {_state['error']}")

    t0 = time.perf_counter()
    df = pd.DataFrame(req.rows)
    missing = [f for f in model.meta.features if f not in df.columns]
    if missing:
        # 422, not 500: the caller sent the wrong shape and can fix it.
        raise HTTPException(422, detail=f"missing features: {missing}")
    try:
        X = df[model.meta.features].astype(np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail=f"non-numeric feature values: {exc}") from exc
    if not np.isfinite(X.to_numpy()).all():
        raise HTTPException(422, detail="feature values must be finite")

    s = model.predict(X)
    d = model.decile(s)
    a = model.recommend(s)
    _state["n_scored"] += len(df)

    return ScoreResponse(
        model_id=model.meta.model_id,
        n=len(df),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        results=[ScoredRow(uplift=float(u), decile=int(k), action=str(act))
                 for u, k, act in zip(s, d, a)],
    )
