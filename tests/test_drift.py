"""Tests for the drift monitor.

A monitor that never fires is as useless as one that always fires, so both
directions are checked: no alarm on the reference distribution, a loud one on a
shifted distribution.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import monitor_drift as MD  # noqa: E402


def _bins(values, n_bins=20):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))[1:-1]
    counts = np.bincount(np.digitize(values, edges), minlength=edges.size + 1).astype(float)
    return {"edges": edges.tolist(), "expected": (counts / counts.sum()).tolist()}


def test_psi_is_zero_for_identical_distributions():
    e = np.full(10, 0.1)
    assert MD.psi(e, e) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_positive_and_symmetric_ish():
    a = np.array([0.5, 0.3, 0.2])
    b = np.array([0.2, 0.3, 0.5])
    assert MD.psi(a, b) > 0
    assert MD.psi(a, b) == pytest.approx(MD.psi(b, a), rel=1e-9)


def test_feature_psi_stable_on_the_reference_sample():
    rng = np.random.default_rng(0)
    ref = rng.lognormal(size=200_000)          # deliberately non-normal
    held_out = rng.lognormal(size=200_000)
    assert MD.feature_psi(_bins(ref), held_out) < MD.PSI_MODERATE


def test_feature_psi_detects_a_mean_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=100_000)
    shifted = rng.normal(loc=1.0, size=100_000)
    assert MD.feature_psi(_bins(ref), shifted) > MD.PSI_SIGNIFICANT


def test_feature_psi_detects_a_variance_change():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=100_000)
    wider = rng.normal(scale=3.0, size=100_000)
    assert MD.feature_psi(_bins(ref), wider) > MD.PSI_SIGNIFICANT


def test_psi_is_finite_when_a_bin_is_empty():
    """An unseen value range must produce a large-but-finite index, not inf."""
    rng = np.random.default_rng(3)
    ref = rng.normal(size=50_000)
    truncated = rng.normal(size=50_000)
    truncated = truncated[truncated > 0]
    v = MD.feature_psi(_bins(ref), truncated)
    assert np.isfinite(v) and v > MD.PSI_SIGNIFICANT


def test_band_thresholds():
    assert MD.band(0.05) == "stable"
    assert MD.band(0.15) == "moderate"
    assert MD.band(0.40) == "significant"


@pytest.mark.skipif(
    not (ROOT / "models" / "current" / "metadata.json").exists(),
    reason="no trained model; run scripts/retrain.py",
)
@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "criteo_uplift.parquet").exists(),
    reason="no ingested dataset; run scripts/00_ingest.py",
)
def test_monitor_runs_clean_against_its_own_training_data():
    """End-to-end: the reference batch must not trip a feature alert.

    The exit code and the presence of output are asserted BEFORE the substring
    check, and that ordering is the whole point. Checking only
    `"significant" not in stdout` passes vacuously when the script dies, since
    empty output contains no such substring. That is not hypothetical: with no
    dataset present this test reported a pass in CI while the script it
    exercises was crashing on the missing parquet.

    The dataset skip above is what should have happened there instead.
    """
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "monitor_drift.py"), "--sample", "50000"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, (
        f"monitor_drift.py exited {r.returncode} (non-zero means it raised an alert "
        f"against its own training reference, or failed outright)\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    assert r.stdout.strip(), "monitor_drift.py produced no output"
    assert "significant" not in r.stdout.split("prediction")[0], r.stdout
