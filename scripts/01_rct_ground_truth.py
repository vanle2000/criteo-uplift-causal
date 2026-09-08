"""Step 1 - establish the ground truth ATE on the full randomised sample.

This is the number every later step is scored against. Because assignment is
randomised, the unadjusted difference in means is already an unbiased estimator
of the ATE; everything else in this script exists to *demonstrate* that, and to
pin down exactly which estimand we are claiming to recover.

Outputs:
    reports/step1_rct_ground_truth.json   machine-readable, consumed by Step 3
    reports/step1_rct_ground_truth.md     the table that goes in the README
    reports/figures/step1_rct_balance.png love plot of baseline balance

Usage:
    python scripts/01_rct_ground_truth.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config as C  # noqa: E402
from uplift import data, stats  # noqa: E402


def describe_sample(df) -> dict:
    """Sample composition: arm sizes, outcome base rates, exposure structure."""
    t = df[C.TREATMENT].to_numpy()
    n, n1, n0 = len(df), int((t == 1).sum()), int((t == 0).sum())
    d = {
        "n_rows": n,
        "n_treated": n1,
        "n_control": n0,
        "treated_share": n1 / n,
        "outcome_base_rates": {},
    }
    for y in C.OUTCOMES:
        yv = df[y].to_numpy()
        d["outcome_base_rates"][y] = {
            "overall": float(yv.mean()),
            "treated": float(yv[t == 1].mean()),
            "control": float(yv[t == 0].mean()),
            "n_positive": int(yv.sum()),
        }
    e = df[C.EXPOSURE].to_numpy()
    d["exposure"] = {
        "overall": float(e.mean()),
        "given_treated": float(e[t == 1].mean()),
        "given_control": float(e[t == 0].mean()),
        "one_sided_compliance": bool(e[t == 0].sum() == 0),
    }
    return d


def randomisation_check(df) -> tuple["object", dict]:
    """Covariate balance across arms *before* any adjustment.

    In a valid RCT every |SMD| should sit near zero. This table is the
    reference that Step 2's deliberately-confounded sample gets compared
    against, and it is what makes the Step 3 balance claims interpretable.
    """
    X = df[C.FEATURES]
    t = df[C.TREATMENT].to_numpy()
    tbl = stats.balance_table(X, t)
    summary = {
        "max_abs_smd": float(tbl["abs_smd"].max()),
        "mean_abs_smd": float(tbl["abs_smd"].mean()),
        "n_covariates_above_0.10": int((tbl["abs_smd"] > 0.10).sum()),
        "worst_covariate": str(tbl["abs_smd"].idxmax()),
        "variance_ratio_min": float(tbl["variance_ratio"].min()),
        "variance_ratio_max": float(tbl["variance_ratio"].max()),
    }
    return tbl, summary


def dependence_check(df, cluster: np.ndarray) -> dict:
    """Quantify how far the rows are from the independent-units assumption.

    Every classical SE in this project assumes one independent unit per row.
    The Criteo file breaks that: it is impression-level, and treated units
    recur more often than control ones. This measures the damage.
    """
    t = df[C.TREATMENT].to_numpy()
    sizes = np.bincount(cluster)
    n_clusters = sizes.size
    # Treatment share within each cluster, to find clusters spanning both arms.
    sums = np.bincount(cluster, weights=t)
    pure = (sums == 0) | (sums == sizes)
    return {
        "n_rows": int(len(df)),
        "n_clusters": int(n_clusters),
        "rows_per_cluster_mean": float(sizes.mean()),
        "rows_per_cluster_max": int(sizes.max()),
        "impressions_per_treated_unit": float(sizes[sums == sizes].mean()),
        "impressions_per_control_unit": float(sizes[sums == 0].mean()),
        "share_clusters_mixed_arms": float(1.0 - pure.mean()),
        "treated_share_row_level": float(t.mean()),
        "treated_share_cluster_level": float((sums == sizes).sum() / n_clusters),
    }


def collapsed_ate(df, cluster: np.ndarray, outcome: str) -> dict:
    """User-level ATE: collapse each pure cluster to one unit, then compare.

    A different estimand from the impression-level ITT -- it weights every
    user equally instead of weighting by how often they were shown an ad --
    and the one a marketer usually means by "effect of the campaign".
    """
    t = df[C.TREATMENT].to_numpy()
    y = df[outcome].to_numpy(dtype=np.float64)
    sizes = np.bincount(cluster)
    sums = np.bincount(cluster, weights=t)
    pure = (sums == 0) | (sums == sizes)
    ymean = np.bincount(cluster, weights=y) / sizes
    tc = (sums == sizes).astype(np.int8)
    est = stats.diff_in_means(ymean[pure], tc[pure], name=f"user-level ATE ({outcome})")
    d = est.as_dict()
    d["n_clusters_used"] = int(pure.sum())
    d["n_clusters_dropped_mixed"] = int((~pure).sum())
    return d


def estimate_ate(df, outcome: str, cluster: np.ndarray) -> dict:
    """Several views of the same estimand, as a self-consistency check.

    1. Difference in means  - unbiased under randomisation, no modelling.
    2. Lin-adjusted         - same estimand, uses covariates only for
                              precision; a large gap here would be evidence
                              that randomisation is not what it claims.
    3. CACE                 - the effect on those actually served an ad,
                              rescaling the ITT by the first stage.
    """
    t = df[C.TREATMENT].to_numpy()
    y = df[outcome].to_numpy()

    dim = stats.diff_in_means(y, t, name=f"ITT / diff-in-means ({outcome})")
    crd = stats.cluster_robust_diff_in_means(
        y, t, cluster, name=f"ITT, cluster-robust ({outcome})"
    )

    t0 = time.perf_counter()
    lin, lin_se = stats.lin_ate(df[C.FEATURES], t, y)
    lin_secs = time.perf_counter() - t0

    # First stage: effect of assignment on actually being served an ad.
    first = stats.diff_in_means(df[C.EXPOSURE].to_numpy(), t, name="first stage (exposure)")
    cace, cace_se = stats.wald_ratio(dim.estimate, dim.se, first.estimate, first.se)

    out = dim.as_dict()
    out["outcome"] = outcome
    out["cluster_robust"] = {
        "se": crd.se,
        "ci_low": crd.ci[0],
        "ci_high": crd.ci[1],
        "design_effect": crd.se / dim.se,
    }
    out["lin_adjusted"] = {
        "estimate": lin,
        "se": lin_se,
        "ci_low": lin - stats.Z95 * lin_se,
        "ci_high": lin + stats.Z95 * lin_se,
        "se_ratio_vs_unadjusted": lin_se / dim.se,
        "seconds": round(lin_secs, 1),
    }
    out["cace"] = {
        "estimate": cace,
        "se": cace_se,
        "ci_low": cace - stats.Z95 * cace_se,
        "ci_high": cace + stats.Z95 * cace_se,
        "first_stage": first.estimate,
        "first_stage_se": first.se,
    }
    return out


def love_plot(tbl, path: Path) -> None:
    """Love plot of baseline SMDs - the visual the README leads Step 3 with."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = tbl["abs_smd"].sort_values(ascending=True).index
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tbl.loc[order, "smd"], range(len(order)), s=45, zorder=3)
    ax.axvline(0, color="0.3", lw=1)
    for thresh in (-0.10, 0.10):
        ax.axvline(thresh, color="crimson", ls="--", lw=1, zorder=1)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("standardised mean difference (treated - control)")
    ax.set_title("Step 1 - baseline covariate balance, full randomised sample")
    lim = max(0.12, float(tbl["abs_smd"].max()) * 1.3)
    ax.set_xlim(-lim, lim)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(res: dict, tbl, path: Path) -> None:
    L = []
    L.append("# Step 1 - RCT ground truth\n")
    s = res["sample"]
    L.append(
        f"Full randomised sample: **{s['n_rows']:,} rows**, "
        f"{s['n_treated']:,} treated ({s['treated_share']:.1%}) / "
        f"{s['n_control']:,} control.\n"
    )
    L.append("## Ground-truth ATE\n")
    L.append("| outcome | control rate | treated rate | ATE (ITT) | 95% CI | relative lift |")
    L.append("|---|---|---|---|---|---|")
    for y in C.OUTCOMES:
        r = res["ate"][y]
        L.append(
            f"| `{y}` | {r['mean_control']:.5f} | {r['mean_treated']:.5f} | "
            f"**{r['estimate']:+.6f}** | [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | "
            f"{r['relative_lift']:+.1%} |"
        )
    L.append("\n## Cross-checks (same estimand, different estimator)\n")
    L.append("| outcome | diff-in-means | Lin-adjusted | SE ratio | CACE (on exposed) |")
    L.append("|---|---|---|---|---|")
    for y in C.OUTCOMES:
        r = res["ate"][y]
        L.append(
            f"| `{y}` | {r['estimate']:+.6f} | {r['lin_adjusted']['estimate']:+.6f} | "
            f"{r['lin_adjusted']['se_ratio_vs_unadjusted']:.3f} | "
            f"{r['cace']['estimate']:+.6f} |"
        )
    b = res["balance"]
    L.append("\n## Randomisation check\n")
    L.append(
        f"Max |SMD| across the 12 covariates: **{b['max_abs_smd']:.4f}** "
        f"(mean {b['mean_abs_smd']:.4f}); {b['n_covariates_above_0.10']} of 12 exceed 0.10. "
        f"Variance ratios span [{b['variance_ratio_min']:.3f}, {b['variance_ratio_max']:.3f}].\n"
    )
    L.append(tbl.round(5).to_markdown())
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    t_start = time.perf_counter()
    print("[step1] loading parquet ...")
    df = data.load()
    print(f"[step1] {len(df):,} rows loaded ({df.memory_usage(deep=True).sum() / 1e9:.2f} GB)")

    sample = describe_sample(df)
    print(f"[step1] treated share {sample['treated_share']:.4f}; "
          f"one-sided compliance = {sample['exposure']['one_sided_compliance']}")

    print("[step1] building proxy cluster ids ...")
    cluster = data.cluster_ids(df)
    dep = dependence_check(df, cluster)
    print(f"[step1] {dep['n_clusters']:,} clusters; "
          f"{dep['impressions_per_treated_unit']:.3f} impressions/treated vs "
          f"{dep['impressions_per_control_unit']:.3f}/control")

    tbl, bal = randomisation_check(df)
    print(f"[step1] randomisation check: max |SMD| = {bal['max_abs_smd']:.4f}")

    ate, user_level = {}, {}
    for y in C.OUTCOMES:
        ate[y] = estimate_ate(df, y, cluster)
        user_level[y] = collapsed_ate(df, cluster, y)
        r = ate[y]
        print(f"[step1] {y:<11} ITT {r['estimate']:+.6f}  "
              f"welch SE {r['se']:.6f}  cluster SE {r['cluster_robust']['se']:.6f}  "
              f"(deff {r['cluster_robust']['design_effect']:.2f}x)  "
              f"| user-level {user_level[y]['estimate']:+.6f}")

    res = {
        "step": 1,
        "description": "Ground-truth ATE on the full randomised Criteo sample",
        "source_manifest": json.loads(C.INGEST_MANIFEST.read_text()),
        "sample": sample,
        "balance": bal,
        "dependence": dep,
        "ate": ate,
        "user_level_ate": user_level,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }

    C.REPORTS.mkdir(parents=True, exist_ok=True)
    (C.REPORTS / "step1_rct_ground_truth.json").write_text(json.dumps(res, indent=2))
    tbl.round(6).to_csv(C.REPORTS / "step1_rct_balance.csv")
    write_markdown(res, tbl, C.REPORTS / "step1_rct_ground_truth.md")
    love_plot(tbl, C.FIGURES / "step1_rct_balance.png")

    print(f"\n[step1] done in {res['runtime_seconds']}s -> reports/step1_rct_ground_truth.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
