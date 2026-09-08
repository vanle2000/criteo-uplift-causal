"""Drift monitor.

Compares a batch of live scoring traffic against the reference distribution
frozen into the model artefact at training time, and exits non-zero when
anything crosses a threshold -- so a scheduler or CI job can page on it.

Two things are watched, because they fail differently:

* **Covariate drift** (population stability index per feature). The input
  distribution moved. The model may still be fine, but its training data no
  longer describes the traffic.

* **Prediction drift** (PSI on the score, plus the share landing in the top
  deciles). This is the one that changes spend. A ranking model whose score
  distribution shifts will silently retarget the budget even if every input
  looks normal.

PSI convention, which is a rule of thumb and not a law: < 0.10 stable,
0.10-0.25 moderate, > 0.25 significant.

Usage:
    python scripts/monitor_drift.py --batch data/processed/criteo_uplift.parquet --sample 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config as C  # noqa: E402
from uplift.serving.model import UpliftModel  # noqa: E402

PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index from expected bin masses and actual counts.

    Epsilon-flooring each share keeps an empty bin from producing an infinite
    index, which would turn one unseen value range into an unactionable alert.
    """
    e = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    a = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(meta_bins: dict, values: np.ndarray) -> float:
    """PSI for one feature against the binned reference stored in the artefact."""
    inner = np.asarray(meta_bins["edges"], dtype=float)
    expected = np.asarray(meta_bins["expected"], dtype=float)
    counts = np.bincount(np.digitize(values, inner), minlength=expected.size).astype(float)
    return psi(expected, counts)


def band(v: float) -> str:
    return "significant" if v > PSI_SIGNIFICANT else ("moderate" if v > PSI_MODERATE else "stable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(C.PROJECT_ROOT / "models" / "current"))
    ap.add_argument("--batch", default=str(C.PARQUET), help="parquet of live traffic")
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--reference", default="", help="optional parquet for the reference sample")
    ap.add_argument("--out", default=str(C.REPORTS / "drift_report.json"))
    args = ap.parse_args()

    model = UpliftModel.load(Path(args.model_dir))
    batch = pd.read_parquet(args.batch, columns=C.FEATURES)
    if args.sample and len(batch) > args.sample:
        batch = batch.sample(args.sample, random_state=C.SEED)

    # Covariate drift against the binned empirical distribution frozen into the
    # artefact at training time. An explicit reference parquet overrides it.
    if args.reference:
        ref = pd.read_parquet(args.reference, columns=C.FEATURES)
        if args.sample and len(ref) > args.sample:
            ref = ref.sample(args.sample, random_state=C.SEED)
        ref_kind = "empirical sample from --reference"
    else:
        ref = None
        ref_kind = "binned training distribution stored in the artefact"

    if not model.meta.reference_bins:
        raise SystemExit(
            "this artefact predates reference_bins; retrain with scripts/retrain.py"
        )

    features = {}
    for f in C.FEATURES:
        vals = batch[f].to_numpy(dtype=np.float64)
        if ref is not None:
            inner = np.unique(np.quantile(ref[f].to_numpy(), np.linspace(0, 1, 21)))[1:-1]
            exp = np.bincount(np.digitize(ref[f].to_numpy(), inner),
                              minlength=inner.size + 1).astype(float)
            act = np.bincount(np.digitize(vals, inner), minlength=inner.size + 1).astype(float)
            v = psi(exp, act)
        else:
            v = feature_psi(model.meta.reference_bins[f], vals)
        features[f] = {"psi": v, "band": band(v),
                       "batch_mean": float(batch[f].mean()),
                       "reference_mean": model.meta.feature_reference[f]["mean"]}

    # Prediction drift: the reference here is exact, since the thresholds were
    # frozen from the training score distribution.
    scores = model.predict(batch)
    deciles = model.decile(scores)
    share = np.bincount(deciles, minlength=10) / len(deciles)
    # Compared against the shares training actually produced, which are not
    # 10% each because the score is heavily tied.
    expected_share = np.asarray(model.meta.decile_train_shares or [0.1] * 10)
    pred = {
        "score_mean": float(scores.mean()),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p99": float(np.percentile(scores, 99)),
        "decile_shares": share.tolist(),
        "expected_decile_shares": expected_share.tolist(),
        "decile_psi": psi(expected_share, share),
        "max_decile_share_deviation": float(np.abs(share - expected_share).max()),
        "top3_share": float(share[:3].sum()),
        "expected_top3_share": float(expected_share[:3].sum()),
    }
    # Banded on PSI, like the features, rather than on a raw share deviation:
    # with tied scores the decile shares are uneven to begin with, and a fixed
    # percentage-point threshold flags that structure as drift every run.
    pred["band"] = band(pred["decile_psi"])

    alerts = [f"{f}: PSI {d['psi']:.3f} ({d['band']})"
              for f, d in features.items() if d["psi"] > PSI_MODERATE]
    if pred["band"] != "stable":
        alerts.append(f"prediction distribution {pred['band']}: top-3 decile share "
                      f"{pred['top3_share']:.1%} against an expected "
                      f"{pred['expected_top3_share']:.1%}")

    report = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_id": model.meta.model_id,
        "n_batch": len(batch),
        "reference_kind": ref_kind,
        "features": features,
        "prediction": pred,
        "alerts": alerts,
        "status": "ALERT" if alerts else "OK",
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"[drift] model {model.meta.model_id}  n={len(batch):,}  ref={ref_kind}")
    worst = sorted(features.items(), key=lambda kv: -kv[1]["psi"])[:3]
    for f, d in worst:
        print(f"[drift]   {f}: PSI {d['psi']:.4f} ({d['band']})")
    print(f"[drift]   prediction: top-3 decile share {pred['top3_share']:.1%} "
          f"vs expected {pred['expected_top3_share']:.1%}, "
          f"PSI {pred['decile_psi']:.4f} ({pred['band']})")
    print(f"[drift] {report['status']}" + (f" - {len(alerts)} alert(s)" if alerts else ""))
    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
