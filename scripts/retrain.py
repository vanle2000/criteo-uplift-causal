"""Scheduled retrain job.

Trains the servable uplift model, evaluates it on a held-out slice, and
promotes it to `models/current` ONLY if it clears a quality gate. A retrain
that overwrites a good model with a worse one is worse than no retrain at all,
so promotion is conditional and every attempt is appended to a registry.

Serving the transformed-outcome learner rather than the response baseline is a
considered choice, not an oversight. Step 4 measured the response baseline as
the better *ranker* on this data (Qini +0.813 vs +0.659), because uplift here
is close to proportional to baseline conversion propensity. But a response
model outputs P(convert), not an increment, so it cannot answer "how much extra
conversion does one more impression buy" -- which is the question a bidder
actually asks. `--learner response` is available for anyone who wants rank
only.

Usage:
    python scripts/retrain.py
    python scripts/retrain.py --min-qini 0.4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config as C  # noqa: E402
from uplift import data, uplift as U  # noqa: E402
from uplift.serving import model as M  # noqa: E402

MODELS = C.PROJECT_ROOT / "models"
CURRENT = MODELS / "current"
REGISTRY = MODELS / "registry.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outcome", default=C.PRIMARY_OUTCOME)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--min-qini", type=float, default=0.30,
                    help="promotion gate on holdout Qini coefficient")
    ap.add_argument("--max-regression", type=float, default=0.10,
                    help="reject if Qini falls more than this fraction below the incumbent")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = data.load()
    cluster = data.cluster_ids(df)

    # Split by cluster: the file is impression-level, and a row-wise split
    # would leak the same user across train and holdout.
    rng = np.random.default_rng(C.SEED)
    is_test = (rng.random(int(cluster.max()) + 1) < args.test_frac)[cluster]

    X, t, y = df[C.FEATURES], df[C.TREATMENT].to_numpy(), df[args.outcome].to_numpy()
    print(f"[retrain] train {(~is_test).sum():,} / holdout {is_test.sum():,}")

    mdl = M.train(X[~is_test], t[~is_test], y[~is_test], outcome=args.outcome)
    scores = mdl.predict(X[is_test])
    q = U.qini_curve(y[is_test], t[is_test], scores)
    pk = U.precision_at_k(y[is_test], t[is_test], scores, ks=(0.10, 0.20))

    mdl.meta.n_eval = int(is_test.sum())
    mdl.meta.training_data_sha = M.data_fingerprint(X[~is_test], y[~is_test])
    mdl.meta.metrics = {
        "qini_coefficient": q.qini_coefficient,
        "auuc": q.auuc,
        "uplift_at_10pct": float(pk.iloc[0]["uplift_at_k"]),
        "lift_vs_overall_at_10pct": float(pk.iloc[0]["lift_vs_overall"]),
        "captured_share_at_10pct": float(pk.iloc[0]["captured_share"]),
        "holdout_base_rate": float(y[is_test].mean()),
    }
    print(f"[retrain] holdout Qini {q.qini_coefficient:+.4f}  "
          f"uplift@10% {mdl.meta.metrics['uplift_at_10pct']:+.6f}")

    # --- promotion gate ---------------------------------------------------
    incumbent = None
    if (CURRENT / "metadata.json").exists():
        incumbent = json.loads((CURRENT / "metadata.json").read_text())
    reasons = []
    if q.qini_coefficient < args.min_qini:
        reasons.append(f"Qini {q.qini_coefficient:.4f} below gate {args.min_qini}")
    if incumbent:
        prev = incumbent.get("metrics", {}).get("qini_coefficient")
        if prev and q.qini_coefficient < prev * (1 - args.max_regression):
            reasons.append(
                f"Qini {q.qini_coefficient:.4f} regresses >{args.max_regression:.0%} "
                f"below incumbent {prev:.4f}")
    promote = not reasons

    entry = {
        "model_id": mdl.meta.model_id,
        "trained_at": mdl.meta.trained_at,
        "metrics": mdl.meta.metrics,
        "n_train": mdl.meta.n_train,
        "n_eval": mdl.meta.n_eval,
        "training_data_sha": mdl.meta.training_data_sha,
        "promoted": promote and not args.dry_run,
        "rejection_reasons": reasons,
        "seconds": round(time.perf_counter() - t0, 1),
    }

    if args.dry_run:
        print(f"[retrain] DRY RUN - would {'promote' if promote else 'reject'}")
    elif promote:
        staged = MODELS / mdl.meta.model_id
        mdl.save(staged)
        if CURRENT.exists():
            shutil.rmtree(CURRENT)
        shutil.copytree(staged, CURRENT)
        print(f"[retrain] PROMOTED -> {CURRENT.relative_to(C.PROJECT_ROOT)}")
    else:
        print("[retrain] REJECTED: " + "; ".join(reasons))

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return 0 if (promote or args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
