"""Step 4 - uplift / heterogeneous treatment effects.

Trains two CATE estimators on the randomised sample and evaluates their
ranking on a held-out slice: a two-model (T-learner) approach and the
Athey-Imbens transformed outcome. A plain response model -- P(convert | X),
with no causal content at all -- is carried through the whole evaluation as a
baseline, because "target the people most likely to convert" is what most
production targeting actually does and it is the thing uplift has to beat.

Two choices worth flagging:

* The train/test split is by CLUSTER, not by row. The file is impression-level
  (Step 1), so a row-wise split would put the same user's impressions on both
  sides and inflate every holdout metric.

* Evaluation is on randomised data. Qini and uplift-decile curves compare
  treated and control outcomes inside a score bucket; only randomisation makes
  that comparison causal.

Usage:
    python scripts/04_uplift.py [--sample 0] [--test-frac 0.3]
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
from uplift import data, stats, uplift as U  # noqa: E402

LGB_PARAMS = dict(
    n_estimators=300, num_leaves=63, learning_rate=0.08, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8,
)


def cluster_split(cluster: np.ndarray, test_frac: float, seed: int = C.SEED) -> np.ndarray:
    """Boolean test mask that keeps every impression of a unit on one side."""
    rng = np.random.default_rng(seed)
    n_clusters = int(cluster.max()) + 1
    is_test_cluster = rng.random(n_clusters) < test_frac
    return is_test_cluster[cluster]


def fit_models(Xtr, ttr, ytr, seed: int = C.SEED) -> dict:
    """Fit the three scorers. Returns name -> callable(X) -> score."""
    import lightgbm as lgb

    models: dict = {}
    p_treat = float(ttr.mean())

    # --- two-model / T-learner -------------------------------------------
    # Via EconML to use the standard implementation where it is available.
    # (EconML's causal-forest and DR modules rely on compiled extensions that
    # are blocked by this machine's Application Control policy; the metalearner
    # module is pure Python and imports fine.)
    t0 = time.perf_counter()
    from econml.metalearners import TLearner

    tl = TLearner(models=[lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed),
                          lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed + 1)])
    tl.fit(ytr, ttr, X=Xtr.to_numpy(dtype=np.float32))
    models["two_model_tlearner"] = (
        lambda X: tl.effect(X.to_numpy(dtype=np.float32)),
        time.perf_counter() - t0,
    )

    # --- transformed outcome ---------------------------------------------
    t0 = time.perf_counter()
    z = U.transformed_outcome(ytr, ttr, p_treat)
    to = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed + 2)
    to.fit(Xtr, z)
    models["transformed_outcome"] = (lambda X: to.predict(X), time.perf_counter() - t0)

    # --- response-model baseline (no causal content) ----------------------
    t0 = time.perf_counter()
    rm = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed + 3)
    rm.fit(Xtr, ytr)
    models["response_baseline"] = (
        lambda X: rm.predict_proba(X)[:, 1],
        time.perf_counter() - t0,
    )
    models["_response_model"] = rm
    return models


def segment_profiles(Xte, score, y, t, n_bins: int = 10) -> list[dict]:
    """Feature profile of each predicted-uplift decile, for the Step 5 prompts.

    Reports each decile's mean covariates as a z-score against the population,
    so a downstream reader (or a language model) sees "unusually high on f9"
    rather than an uninterpretable raw magnitude.
    """
    order = U.rank_desc(score)
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(len(score))
    b = np.minimum((rank * n_bins) // len(score), n_bins - 1)

    mu, sd = Xte.mean(axis=0), Xte.std(axis=0).replace(0, 1.0)
    out = []
    for k in range(n_bins):
        m = b == k
        est = stats.diff_in_means(y[m], t[m])
        z = ((Xte[m].mean(axis=0) - mu) / sd).round(3)
        out.append({
            "decile": k,
            "n": int(m.sum()),
            "mean_predicted_uplift": float(score[m].mean()),
            "observed_uplift": est.estimate,
            "observed_uplift_se": est.se,
            "control_conversion_rate": est.mean_control,
            "treated_conversion_rate": est.mean_treated,
            "feature_z_scores": z.to_dict(),
            "top_features": z.abs().sort_values(ascending=False).head(4).index.tolist(),
        })
    return out


def figures(evald: dict, curves: dict, deciles: dict, outcome: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, q in curves.items():
        ax.plot(q["x"], q["y"], lw=2,
                label=f"{name} (Qini {evald[name]['qini_coefficient']:+.3f})")
    end = max(q["y"][-1] for q in curves.values())
    ax.plot([0, 1], [0, end], "k--", lw=1.2, label="random targeting")
    ax.set_xlabel("share of population targeted, ranked by predicted uplift")
    ax.set_ylabel(f"incremental {outcome}s")
    ax.set_title(f"Qini curves ({outcome}, held-out randomised sample)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"step4_qini_{outcome}.png", dpi=150)
    plt.close(fig)

    n = len(deciles)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, tbl) in zip(axes, deciles.items()):
        d = pd.DataFrame(tbl)
        ax.bar(d["bin"], d["observed_uplift"],
               yerr=stats.Z95 * d["se"], capsize=3, color="tab:blue")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("predicted-uplift decile (0 = highest)")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"observed RCT uplift on {outcome}")
    fig.suptitle(f"Observed uplift by predicted-uplift decile ({outcome})")
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"step4_deciles_{outcome}.png", dpi=150)
    plt.close(fig)


def write_markdown(res: dict, path: Path) -> None:
    o = res["outcome"]
    L = [f"# Step 4 - uplift modelling (`{o}`)\n"]
    L.append(f"Trained on {res['n_train']:,} rows, evaluated on {res['n_test']:,} "
             f"held-out rows, split by cluster so no unit appears on both sides. "
             f"Holdout base rate {res['base_rate']:.5f}.\n")
    L.append("## Ranking quality\n")
    L.append("| model | Qini coefficient | uplift @ top 10% | lift vs untargeted | "
             "captured share @ 10% | fit seconds |")
    L.append("|---|---|---|---|---|---|")
    for name, m in res["models"].items():
        k10 = next(r for r in m["precision_at_k"] if abs(r["k"] - 0.10) < 1e-9)
        L.append(
            f"| `{name}` | **{m['qini_coefficient']:+.4f}** | {k10['uplift_at_k']:+.6f} | "
            f"{k10['lift_vs_overall']:.2f}x | {k10['captured_share']:.1%} | "
            f"{m['fit_seconds']:.0f} |"
        )
    L.append("\n## Precision@K\n")
    for name, m in res["models"].items():
        L.append(f"\n**`{name}`**\n")
        L.append("| K | n targeted | uplift @ K | lift vs untargeted | captured share |")
        L.append("|---|---|---|---|---|")
        for r in m["precision_at_k"]:
            L.append(f"| {r['k']:.0%} | {r['n_targeted']:,} | {r['uplift_at_k']:+.6f} | "
                     f"{r['lift_vs_overall']:.2f}x | {r['captured_share']:.1%} |")
    rm = res["response_model"]
    L.append("\n## Response model quality (class imbalance)\n")
    L.append(
        f"Base rate {rm['base_rate']:.5f}. AUC-ROC {rm['auc_roc']:.4f}; "
        f"AUC-PR {rm['auc_pr']:.4f}, which is **{rm['auc_pr_lift_over_base']:.1f}x** the "
        f"base rate a random ranker would achieve. Brier {rm['brier']:.6f}; mean predicted "
        f"{rm['mean_predicted']:.5f} against observed {rm['observed']:.5f} "
        f"(calibration ratio {rm['calibration_ratio']:.3f}).\n"
    )
    L.append("Accuracy is not reported: predicting \"never converts\" for everyone scores "
             f"{1 - rm['base_rate']:.3%} and is worthless.\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outcome", default=C.PRIMARY_OUTCOME)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    t_start = time.perf_counter()
    df = data.load()
    cluster = data.cluster_ids(df)

    if args.sample:
        rng = np.random.default_rng(C.SEED)
        keep = np.sort(rng.choice(len(df), size=args.sample, replace=False))
        df, cluster = df.iloc[keep].reset_index(drop=True), cluster[keep]
        print(f"[step4] SMOKE TEST: {args.sample:,} rows")

    is_test = cluster_split(cluster, args.test_frac)
    X, t, y = df[C.FEATURES], df[C.TREATMENT].to_numpy(), df[args.outcome].to_numpy()
    Xtr, Xte = X[~is_test], X[is_test]
    print(f"[step4] train {(~is_test).sum():,} / test {is_test.sum():,} rows, "
          f"split by cluster")

    models = fit_models(Xtr, t[~is_test], y[~is_test])
    yte, tte = y[is_test], t[is_test]

    evald, curves, deciles, profiles = {}, {}, {}, {}
    for name, entry in models.items():
        if name.startswith("_"):
            continue
        fn, secs = entry
        s = np.asarray(fn(Xte), dtype=np.float64).ravel()
        q = U.qini_curve(yte, tte, s)
        pk = U.precision_at_k(yte, tte, s)
        dec = U.uplift_by_decile(yte, tte, s)
        evald[name] = {**q.as_dict(), "fit_seconds": secs,
                       "precision_at_k": pk.to_dict(orient="records")}
        curves[name] = {"x": q.x.tolist(), "y": q.y.tolist()}
        deciles[name] = dec.to_dict(orient="records")
        profiles[name] = segment_profiles(Xte, s, yte, tte)
        k10 = pk[np.isclose(pk["k"], 0.10)].iloc[0]
        print(f"[step4] {name:<22} Qini {q.qini_coefficient:+.4f}  "
              f"uplift@10% {k10['uplift_at_k']:+.6f}  "
              f"({k10['lift_vs_overall']:.2f}x untargeted)")

    p_resp = models["_response_model"].predict_proba(Xte)[:, 1]
    rm = U.response_metrics(yte, p_resp)
    print(f"[step4] response model AUC-PR {rm['auc_pr']:.4f} "
          f"({rm['auc_pr_lift_over_base']:.1f}x base rate {rm['base_rate']:.5f})")

    res = {
        "step": 4, "outcome": args.outcome,
        "n_train": int((~is_test).sum()), "n_test": int(is_test.sum()),
        "base_rate": float(yte.mean()),
        "models": evald, "deciles": deciles,
        "response_model": rm,
        "calibration": U.calibration_table(yte, p_resp).to_dict(orient="records"),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (C.REPORTS / f"step4_uplift_{args.outcome}.json").write_text(
        json.dumps(res, indent=2, default=float))
    (C.REPORTS / f"step4_segments_{args.outcome}.json").write_text(
        json.dumps(profiles, indent=2, default=float))
    write_markdown(res, C.REPORTS / f"step4_uplift_{args.outcome}.md")
    figures(evald, curves, deciles, args.outcome)
    print(f"[step4] done in {res['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
