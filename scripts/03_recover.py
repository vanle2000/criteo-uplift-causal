"""Step 3 - recover the truth from the confounded sample.

Runs PSM, IPW, AIPW and g-computation against the Step 2 subsample, reports
covariate balance before and after adjustment, adds a DiD on a synthetic
pre-period (in both a parallel-trends and a violated-trends version), and two
placebo tests. Every estimate is scored against a target re-standardised onto
exactly the rows that estimator actually used.

Usage:
    python scripts/03_recover.py [--direction inflate] [--outcomes conversion visit]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config as C  # noqa: E402
from uplift import confound, data, did, estimators as E, stats  # noqa: E402


def get_prognostic(df, outcome: str) -> np.ndarray:
    """Control-arm P(Y=1|X) fitted on the FULL randomised sample, cached.

    Used only to build the synthetic pre-period, which is part of the
    simulation design rather than part of any estimate, so fitting it on the
    full RCT is legitimate. No estimator ever sees it.
    """
    path = C.DATA_PROCESSED / f"prognostic_{outcome}.npy"
    if path.exists():
        s = np.load(path)
        if len(s) == len(df):
            return s
    print(f"[step3] fitting prognostic model for {outcome} ...")
    ps = confound.fit_prognostic_score(df, outcome=outcome)
    np.save(path, ps.score)
    print(f"[step3]   holdout AUC {ps.auc_holdout:.4f}")
    return ps.score


def targets(df, mask_full, score, outcome) -> dict:
    """RCT-derived targets for the ATE and the ATT on this exact row set."""
    t = df[C.TREATMENT].to_numpy()
    ate = confound.rct_benchmark(df, mask_full, score, outcome)
    att = confound.rct_benchmark(df, mask_full & (t == 1), score, outcome)
    return {"ate": ate, "att": att}


def run_outcome(df, mask, cluster_sub, score_sel, outcome, args) -> dict:
    """Full estimator suite for one outcome on the confounded subsample."""
    t = df[C.TREATMENT].to_numpy()[mask]
    y = df[outcome].to_numpy()[mask]
    X = df[C.FEATURES][mask]
    n = len(y)
    out: dict = {"outcome": outcome, "n": n}

    tgt = targets(df, mask, score_sel, outcome)
    out["target"] = tgt
    print(f"\n[step3] === {outcome} ===  target ATE {tgt['ate']['target_ate']:+.6f} "
          f"| ATT {tgt['att']['target_ate']:+.6f}")

    results: list[dict] = []

    # ---- naive ----------------------------------------------------------
    naive = stats.cluster_robust_diff_in_means(y, t, cluster_sub, name="naive")
    results.append({"method": "naive (no adjustment)", "estimate": naive.estimate,
                    "se": naive.se, "ci_low": naive.ci[0], "ci_high": naive.ci[1],
                    "n_used": n, "target_kind": "ate"})

    # ---- propensity models ---------------------------------------------
    props = {}
    for pm in ("lgbm", "logit"):
        t0 = time.perf_counter()
        p = E.fit_propensity(X, t, method=pm, n_folds=args.folds)
        props[pm] = p
        print(f"[step3] propensity {pm:<6} AUC {p.auc:.4f}  brier {p.brier:.5f}  "
              f"calib slope {p.calibration_slope:.3f}  ({time.perf_counter()-t0:.0f}s)")
    out["propensity"] = {
        k: {"method": v.method, "auc": v.auc, "brier": v.brier,
            "calibration_slope": v.calibration_slope} for k, v in props.items()
    }

    # ---- outcome models (for AIPW / g-computation) ----------------------
    t0 = time.perf_counter()
    mu0, mu1 = E.fit_outcome_models(X, t, y, n_folds=args.folds)
    print(f"[step3] outcome models fitted ({time.perf_counter()-t0:.0f}s)")

    # ---- estimators, per propensity model ------------------------------
    for pm, p in props.items():
        e = p.e
        results.append({**E.ipw_ate(y, t, e, cluster_sub, outcome=outcome).as_dict(),
                        "method": f"IPW (stabilized, {pm} PS)", "target_kind": "ate"})
        results.append({**E.aipw_ate(y, t, e, mu0, mu1, cluster_sub, outcome=outcome).as_dict(),
                        "method": f"AIPW doubly-robust ({pm} PS)", "target_kind": "ate"})
        psm, diag = E.psm_ate(y, t, e, outcome=outcome)
        results.append({**psm.as_dict(), "method": f"PSM 1:1 NN ({pm} PS)",
                        "target_kind": "ate"})
        if pm == "lgbm":
            out["psm_diagnostics"] = {k: v for k, v in diag.items()
                                      if not k.startswith("matched_")}
            out["balance"] = {
                "confounded": stats.balance_table(X, t)["abs_smd"].to_dict(),
                "ipw_weighted": E.weighted_balance(X, t, e)["abs_smd"].to_dict(),
                "matched": stats.balance_table(
                    X.iloc[np.r_[diag["matched_treated_idx"], diag["matched_control_idx"]]],
                    np.r_[np.ones(diag["n_treated_matched"], dtype=int),
                          np.zeros(diag["n_treated_matched"], dtype=int)],
                )["abs_smd"].to_dict(),
            }

    results.append({**E.gcomputation_ate(mu0, mu1, cluster_sub, outcome=outcome).as_dict(),
                    "method": "g-computation (outcome regression)", "target_kind": "ate"})

    # ---- trimmed to common support -------------------------------------
    e = props["lgbm"].e
    keep = E.overlap_mask(e, args.trim_lo, args.trim_hi)
    full_idx = np.flatnonzero(mask)
    mask_trim = np.zeros(len(df), dtype=bool)
    mask_trim[full_idx[keep]] = True
    tgt_trim = targets(df, mask_trim, score_sel, outcome)
    out["trimmed"] = {
        "trim_range": [args.trim_lo, args.trim_hi],
        "n_kept": int(keep.sum()),
        "share_kept": float(keep.mean()),
        "target": tgt_trim,
    }
    print(f"[step3] trimming to e in [{args.trim_lo}, {args.trim_hi}] keeps "
          f"{keep.mean():.1%}; target moves to {tgt_trim['ate']['target_ate']:+.6f}")
    results.append({
        **E.aipw_ate(y[keep], t[keep], e[keep], mu0[keep], mu1[keep],
                     cluster_sub[keep], outcome=outcome).as_dict(),
        "method": "AIPW, trimmed to common support", "target_kind": "ate_trimmed"})
    results.append({
        **E.ipw_ate(y[keep], t[keep], e[keep], cluster_sub[keep], outcome=outcome).as_dict(),
        "method": "IPW, trimmed to common support", "target_kind": "ate_trimmed"})

    # ---- DiD on a synthetic pre-period ---------------------------------
    prog = get_prognostic(df, outcome)[mask]
    for mode in ("parallel", "violated"):
        y_pre = did.make_pre_period(prog, t, mode=mode, trend_gap=args.trend_gap,
                                    seed=C.SEED)
        d = did.did_estimate(y, y_pre, t, cluster_sub, name=f"DiD ({mode} trends)")
        results.append({"method": f"DiD, {mode} trends", "estimate": d.estimate,
                        "se": d.se, "ci_low": d.ci[0], "ci_high": d.ci[1],
                        "n_used": n, "target_kind": "att",
                        "pre_rate_treated": float(y_pre[t == 1].mean()),
                        "pre_rate_control": float(y_pre[t == 0].mean())})

    # ---- placebo tests --------------------------------------------------
    y_pre = did.make_pre_period(prog, t, mode="parallel", seed=C.SEED + 99)
    placebo: dict = {"negative_control_outcome": {}, "permutation": {}}
    pl_naive = stats.cluster_robust_diff_in_means(y_pre.astype(float), t, cluster_sub)
    placebo["negative_control_outcome"]["naive"] = {
        "estimate": pl_naive.estimate, "se": pl_naive.se,
        "ci_low": pl_naive.ci[0], "ci_high": pl_naive.ci[1]}
    placebo["negative_control_outcome"]["ipw"] = E.ipw_ate(
        y_pre.astype(float), t, e, cluster_sub).as_dict()
    if outcome == C.PRIMARY_OUTCOME:
        t0 = time.perf_counter()
        p0, p1 = E.fit_outcome_models(X, t, y_pre, n_folds=args.folds)
        placebo["negative_control_outcome"]["aipw"] = E.aipw_ate(
            y_pre.astype(float), t, e, p0, p1, cluster_sub).as_dict()
        print(f"[step3] placebo AIPW fitted ({time.perf_counter()-t0:.0f}s)")
    placebo["permutation"] = did.permutation_placebo(y, t, n_draws=args.perm_draws,
                                                     seed=C.SEED)
    out["placebo"] = placebo

    # ---- score every estimate against its own target --------------------
    tk = {"ate": tgt["ate"]["target_ate"], "att": tgt["att"]["target_ate"],
          "ate_trimmed": tgt_trim["ate"]["target_ate"]}
    for r in results:
        target = tk[r["target_kind"]]
        r["target"] = target
        r["bias"] = r["estimate"] - target
        r["bias_pct"] = r["bias"] / abs(target)
        r["covers_target"] = bool(r["ci_low"] <= target <= r["ci_high"])
        print(f"[step3] {r['method']:<40} {r['estimate']:+.6f}  "
              f"bias {r['bias_pct']:+7.1%}  covers={r['covers_target']}")
    out["results"] = results
    return out


def figures(res: dict, direction: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for outcome, o in res["outcomes"].items():
        # Forest plot: every estimator against its target.
        rs = [r for r in o["results"] if r["target_kind"] == "ate"]
        rs = sorted(rs, key=lambda r: abs(r["bias_pct"]), reverse=True)
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(rs) + 2.2))
        ypos = np.arange(len(rs))
        for i, r in enumerate(rs):
            ok = r["covers_target"]
            ax.plot([r["ci_low"], r["ci_high"]], [i, i],
                    color="tab:green" if ok else "tab:red", lw=2, zorder=2)
            ax.plot(r["estimate"], i, "o", color="tab:green" if ok else "tab:red",
                    ms=7, zorder=3)
        tgt = o["target"]["ate"]["target_ate"]
        ax.axvline(tgt, color="k", ls="--", lw=1.4, zorder=1,
                   label=f"RCT target {tgt:+.6f}")
        ax.set_yticks(ypos, [r["method"] for r in rs], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"estimated ATE on {outcome}")
        ax.set_title(f"Step 3 - recovering the RCT target ({outcome}, {direction})\n"
                     "green = 95% CI covers the target")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(C.FIGURES / f"step3_forest_{outcome}_{direction}.png", dpi=150)
        plt.close(fig)

        # Love plot: balance before, after IPW, after matching.
        b = o["balance"]
        covs = list(b["confounded"].keys())
        order = sorted(covs, key=lambda c: b["confounded"][c])
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for key, lab, mk in [("confounded", "confounded (Step 2)", "D"),
                             ("ipw_weighted", "after IPW weighting", "o"),
                             ("matched", "after 1:1 matching", "^")]:
            ax.scatter([b[key][c] for c in order], range(len(order)), s=42,
                       marker=mk, label=lab, zorder=3)
        for th in (0.10, 0.25):
            ax.axvline(th, color="crimson", ls="--", lw=0.9, zorder=1)
        ax.set_yticks(range(len(order)), order)
        ax.set_xlabel("|standardised mean difference|")
        ax.set_xscale("log")
        ax.set_title(f"Step 3 - balance before and after adjustment ({outcome})")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(C.FIGURES / f"step3_balance_{outcome}_{direction}.png", dpi=150)
        plt.close(fig)


def write_markdown(res: dict, path: Path) -> None:
    L = [f"# Step 3 - recovering the truth (`{res['direction']}` scenario)\n"]
    for outcome, o in res["outcomes"].items():
        t_ate = o["target"]["ate"]["target_ate"]
        t_att = o["target"]["att"]["target_ate"]
        L.append(f"\n## `{outcome}`\n")
        L.append(f"Subsample: {o['n']:,} rows. RCT target ATE **{t_ate:+.6f}**, "
                 f"ATT {t_att:+.6f}.\n")
        L.append("| estimator | estimate | 95% CI | target | bias | covers target |")
        L.append("|---|---|---|---|---|---|")
        for r in o["results"]:
            L.append(
                f"| {r['method']} | **{r['estimate']:+.6f}** | "
                f"[{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {r['target']:+.6f} | "
                f"{r['bias_pct']:+.1%} | {'yes' if r['covers_target'] else 'NO'} |"
            )
        p = o["propensity"]
        L.append("\n**Propensity models.** " + "; ".join(
            f"`{k}` AUC {v['auc']:.4f}, calibration slope {v['calibration_slope']:.3f}"
            for k, v in p.items()) + ".\n")
        b = o["balance"]
        L.append("| covariate | \\|SMD\\| confounded | after IPW | after matching |")
        L.append("|---|---|---|---|")
        for c in b["confounded"]:
            L.append(f"| `{c}` | {b['confounded'][c]:.4f} | "
                     f"{b['ipw_weighted'][c]:.4f} | {b['matched'][c]:.4f} |")
        pl = o["placebo"]
        L.append("\n**Placebo tests.** Negative-control outcome (a pre-period that no "
                 "treatment could have affected; true effect exactly 0):\n")
        L.append("| estimator | placebo estimate | 95% CI | passes |")
        L.append("|---|---|---|---|")
        for k, v in pl["negative_control_outcome"].items():
            ok = v["ci_low"] <= 0 <= v["ci_high"]
            L.append(f"| {k} | {v['estimate']:+.6f} | [{v['ci_low']:+.6f}, "
                     f"{v['ci_high']:+.6f}] | {'yes' if ok else 'NO'} |")
        pm = pl["permutation"]
        L.append(f"\nPermuted-treatment placebo over {pm['n_draws']} draws: "
                 f"mean {pm['mean']:+.2e}, sd {pm['sd']:.2e}, "
                 f"range [{pm['min']:+.2e}, {pm['max']:+.2e}].\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", default="inflate", choices=["inflate", "deflate"])
    ap.add_argument("--outcomes", nargs="+", default=["conversion", "visit"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--trim-lo", type=float, default=0.02)
    ap.add_argument("--trim-hi", type=float, default=0.98)
    ap.add_argument("--trend-gap", type=float, default=0.30)
    ap.add_argument("--perm-draws", type=int, default=20)
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly downsample the subsample; for smoke tests only")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = data.load()
    mask = np.load(C.DATA_PROCESSED / f"confounded_{args.direction}.npy")
    if args.sample:
        idx = np.flatnonzero(mask)
        keep = np.random.default_rng(C.SEED).choice(idx, size=args.sample, replace=False)
        mask = np.zeros(len(df), dtype=bool)
        mask[np.sort(keep)] = True
        print(f"[step3] SMOKE TEST: downsampled to {args.sample:,} rows")
    cluster = data.cluster_ids(df)[mask]
    score_sel = np.load(C.DATA_PROCESSED / "prognostic_score.npy")
    print(f"[step3] {args.direction}: {mask.sum():,} rows, "
          f"{len(np.unique(cluster)):,} clusters")

    res = {"step": 3, "direction": args.direction, "outcomes": {}}
    for outcome in args.outcomes:
        res["outcomes"][outcome] = run_outcome(df, mask, cluster, score_sel, outcome, args)

    res["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    (C.REPORTS / f"step3_recovery_{args.direction}.json").write_text(
        json.dumps(res, indent=2, default=float))
    write_markdown(res, C.REPORTS / f"step3_recovery_{args.direction}.md")
    figures(res, args.direction)
    print(f"\n[step3] done in {res['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
