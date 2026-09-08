"""Step 2 - construct a deliberately confounded observational subsample.

Selects rows into an observational sample with an arm-dependent, covariate-
dependent probability, so that the naive treated-vs-control contrast is badly
wrong relative to the Step 1 ground truth -- and wrong for a reason we control.

Reports a strength sweep first, because the single knob `k` trades bias against
overlap and the right value is the one that makes the problem hard without
making it unidentifiable.

Outputs:
    data/processed/prognostic_score.npy      cross-fitted control-arm P(Y=1|X)
    data/processed/confounded_<dir>.npy      boolean selection masks
    reports/step2_confounding.{json,md}
    reports/figures/step2_*.png

Usage:
    python scripts/02_confound.py [--strength 3.0] [--refit]
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
from uplift import confound, data, stats  # noqa: E402

SCORE_CACHE = C.DATA_PROCESSED / "prognostic_score.npy"
SCORE_META = C.DATA_PROCESSED / "prognostic_score.json"
SWEEP_STRENGTHS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]


def get_score(df, refit: bool) -> tuple[np.ndarray, dict]:
    """Fit the prognostic score, or reuse the cached one (it costs ~2 min)."""
    if SCORE_CACHE.exists() and SCORE_META.exists() and not refit:
        meta = json.loads(SCORE_META.read_text())
        score = np.load(SCORE_CACHE)
        if len(score) == len(df):
            print(f"[step2] cached prognostic score (holdout AUC {meta['auc_holdout']:.4f})")
            return score, meta
    print("[step2] fitting cross-fitted prognostic score on control arm ...")
    t0 = time.perf_counter()
    ps = confound.fit_prognostic_score(df)
    meta = {
        "outcome": ps.outcome, "auc_holdout": ps.auc_holdout,
        "ap_holdout": ps.ap_holdout, "n_folds": ps.n_folds,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    np.save(SCORE_CACHE, ps.score)
    SCORE_META.write_text(json.dumps(meta, indent=2))
    print(f"[step2] holdout AUC {ps.auc_holdout:.4f}  AP {ps.ap_holdout:.4f}  "
          f"({meta['seconds']}s)")
    return ps.score, meta


def sweep(df, score, truth) -> list[dict]:
    """Bias vs overlap across confounding strengths.

    k = 0 is the control condition: a purely random subsample, whose naive
    estimate must reproduce the RAW full-sample difference in means. That is
    `harness_ratio`, and it must sit at ~1.00 or the harness is broken.

    `bias_ratio` compares instead against the re-standardised target, and is
    ~1.16 at k = 0 rather than 1.00. That gap is not induced bias -- it is the
    residual impression-level imbalance already documented in Step 1, which the
    stratified target removes and the raw contrast does not.
    """
    t = df[C.TREATMENT].to_numpy()
    X = df[C.FEATURES]
    rows = []
    for k in SWEEP_STRENGTHS:
        mask, _ = confound.select(score, t, strength=k)
        sub_t = t[mask]
        bal = stats.balance_table(X[mask], sub_t)
        row = {
            "strength": k,
            "n_selected": int(mask.sum()),
            "treated_share": float(sub_t.mean()),
            "max_abs_smd": float(bal["abs_smd"].max()),
            "mean_abs_smd": float(bal["abs_smd"].mean()),
        }
        for outcome in C.OUTCOMES:
            naive = stats.diff_in_means(df[outcome].to_numpy()[mask], sub_t)
            tgt = confound.rct_benchmark(df, mask, score, outcome)
            row[outcome] = {
                "naive_ate": naive.estimate,
                "naive_se": naive.se,
                "target_ate": tgt["target_ate"],
                "full_sample_ate": truth[outcome],
                "bias": naive.estimate - tgt["target_ate"],
                "bias_ratio": naive.estimate / tgt["target_ate"],
                # Harness check: at k=0 a random subsample must reproduce the
                # raw full-sample number. This ratio, not bias_ratio, is the
                # one that must be ~1.00 at k=0.
                "harness_ratio": naive.estimate / truth[outcome],
            }
        rows.append(row)
        print(f"[step2] k={k:<4} n={row['n_selected']:>9,}  maxSMD={row['max_abs_smd']:.3f}  "
              f"conv naive={row['conversion']['naive_ate']:+.6f} "
              f"({row['conversion']['bias_ratio']:.2f}x target)")
    return rows


def benchmark_validation(df, score, step1: dict) -> dict:
    """Check the re-standardisation machinery against an independent estimator.

    Run over the *whole* sample, `rct_benchmark` reduces to a 100-stratum
    standardised effect. That should agree with the Step 1 Lin-adjusted ATE --
    a linear interaction regression is a completely different way of removing
    the same covariate imbalance. Agreement is what licenses using the
    re-standardised number as the Step 3 target.

    It should NOT reproduce the raw difference in means: the raw contrast still
    carries the residual impression-level imbalance documented in Step 1.
    """
    allmask = np.ones(len(df), dtype=bool)
    out = {}
    for y in C.OUTCOMES:
        tgt = confound.rct_benchmark(df, allmask, score, y)
        raw = step1["ate"][y]["estimate"]
        lin = step1["ate"][y]["lin_adjusted"]["estimate"]
        out[y] = {
            "stratified_full_sample": tgt["target_ate"],
            "lin_adjusted": lin,
            "raw_diff_in_means": raw,
            "stratified_vs_lin_pct": abs(tgt["target_ate"] - lin) / abs(lin),
            "stratified_vs_raw_pct": abs(tgt["target_ate"] - raw) / abs(raw),
        }
    return out


def heterogeneity_by_decile(df, score, outcome: str, n: int = 10) -> list[dict]:
    """RCT uplift within prognostic deciles.

    This is why Step 2 needs re-standardising at all: if the effect were
    constant in X, selecting on X could not move the subsample's true ATE and
    the full-sample number would serve as the target unchanged.
    """
    t = df[C.TREATMENT].to_numpy()
    y = df[outcome].to_numpy(dtype=np.float64)
    q = np.quantile(score, np.linspace(0, 1, n + 1))
    q[0], q[-1] = -np.inf, np.inf
    b = np.digitize(score, q[1:-1])
    rows = []
    for k in range(n):
        m = b == k
        e = stats.diff_in_means(y[m], t[m])
        rows.append({
            "decile": k, "n": int(m.sum()),
            "base_rate_control": e.mean_control, "uplift": e.estimate, "se": e.se,
        })
    return rows


def build_scenario(df, score, direction, strength, truth) -> dict:
    """Materialise one confounded sample and fully characterise it."""
    t = df[C.TREATMENT].to_numpy()
    X = df[C.FEATURES]
    mask, p_keep = confound.select(score, t, strength=strength, direction=direction)
    np.save(C.DATA_PROCESSED / f"confounded_{direction}.npy", mask)

    bal = stats.balance_table(X[mask], t[mask])
    out = {
        "direction": direction,
        "strength": strength,
        "overlap": confound.overlap_diagnostics(p_keep, mask, t),
        "balance": {
            "max_abs_smd": float(bal["abs_smd"].max()),
            "mean_abs_smd": float(bal["abs_smd"].mean()),
            "n_above_0.10": int((bal["abs_smd"] > 0.10).sum()),
            "n_above_0.25": int((bal["abs_smd"] > 0.25).sum()),
            "worst_covariate": str(bal["abs_smd"].idxmax()),
        },
        "balance_table": bal.round(6).to_dict(orient="index"),
        "estimates": {},
    }
    for outcome in C.OUTCOMES:
        y = df[outcome].to_numpy()
        naive = stats.diff_in_means(y[mask], t[mask], name=f"naive ({outcome})")
        tgt = confound.rct_benchmark(df, mask, score, outcome)
        out["estimates"][outcome] = {
            "naive": naive.as_dict(),
            "target": tgt,
            "full_sample_truth": truth[outcome],
            "bias": naive.estimate - tgt["target_ate"],
            "bias_ratio": naive.estimate / tgt["target_ate"],
            "bias_in_target_ses": (naive.estimate - tgt["target_ate"]) / tgt["se"],
            "target_vs_full_sample_shift": tgt["target_ate"] - truth[outcome],
        }
    return out


def figures(df, score, mask_inflate) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df[C.TREATMENT].to_numpy()
    X = df[C.FEATURES]
    rct = stats.balance_table(X, t)
    conf = stats.balance_table(X[mask_inflate], t[mask_inflate])
    order = conf["abs_smd"].sort_values().index

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(rct.loc[order, "smd"], range(len(order)), s=45, label="RCT (Step 1)", zorder=3)
    ax.scatter(conf.loc[order, "smd"], range(len(order)), s=45, marker="D",
               label="confounded (Step 2)", zorder=3)
    ax.axvline(0, color="0.3", lw=1)
    for th in (-0.25, -0.10, 0.10, 0.25):
        ax.axvline(th, color="crimson", ls="--", lw=0.8, alpha=0.7, zorder=1)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("standardised mean difference (treated - control)")
    ax.set_title("Confounding destroys balance\n(dashed: Rubin thresholds 0.10 / 0.25)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "step2_balance_rct_vs_confounded.png", dpi=150)
    plt.close(fig)

    # Selection acts on the RANK of the prognostic score, and the raw score is
    # extremely right-skewed (most users never visit), so the rank scale is
    # both the honest and the legible one: uniform under randomisation.
    r = confound.rank_pct(score)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.linspace(0, 1, 51)
    for m, lab, sty in [
        (t == 1, "RCT treated", dict(histtype="step", lw=1.3, ls="--", color="C0")),
        (t == 0, "RCT control", dict(histtype="step", lw=1.3, ls=":", color="C1")),
        (mask_inflate & (t == 1), "confounded treated", dict(histtype="step", lw=2.2, color="C0")),
        (mask_inflate & (t == 0), "confounded control", dict(histtype="step", lw=2.2, color="C3")),
    ]:
        ax.hist(r[m], bins=bins, density=True, label=lab, **sty)
    ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
    ax.set_xlabel("percentile of prognostic score  P(visit | X)  (control-arm model)")
    ax.set_ylabel("density")
    ax.set_ylim(0, 2.2)
    ax.set_title("Selection pushes treated up and control down the prognostic scale\n"
                 "(flat at 1.0 = the uniform RCT baseline)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "step2_prognostic_overlap.png", dpi=150)
    plt.close(fig)


def write_markdown(res: dict, path: Path) -> None:
    L = ["# Step 2 - deliberate confounding\n"]
    m = res["prognostic_score"]
    L.append(
        f"Selection runs along a cross-fitted prognostic score -- "
        f"`P({m['outcome']}=1 | X)` fitted on control units only, holdout "
        f"AUC **{m['auc_holdout']:.4f}**. Confounding along a direction that did not "
        f"predict the outcome would create imbalance but no bias.\n"
    )
    L.append("## Strength sweep\n")
    L.append("| k | n selected | treated share | max \\|SMD\\| | naive conversion ATE | vs target |")
    L.append("|---|---|---|---|---|---|")
    for r in res["sweep"]:
        c = r["conversion"]
        L.append(
            f"| {r['strength']:.0f} | {r['n_selected']:,} | {r['treated_share']:.3f} | "
            f"{r['max_abs_smd']:.3f} | {c['naive_ate']:+.6f} | {c['bias_ratio']:.2f}x |"
        )
    L.append("\n`k = 0` is the control condition: a random subsample, which must reproduce "
             "the RCT answer. It does, which is what licenses reading the rest of the table "
             "as induced bias rather than a bug.\n")

    for direction in ("inflate", "deflate"):
        sc = res["scenarios"][direction]
        L.append(f"\n## Scenario: `{direction}` (k = {sc['strength']})\n")
        o = sc["overlap"]
        L.append(
            f"{o['n_selected']:,} rows retained ({o['selection_rate']:.1%}), "
            f"{o['treated_share']:.1%} treated. "
            f"Max |SMD| **{sc['balance']['max_abs_smd']:.3f}** "
            f"({sc['balance']['n_above_0.10']} of 12 covariates above 0.10, "
            f"{sc['balance']['n_above_0.25']} above 0.25).\n"
        )
        L.append("| outcome | naive ATE | true ATE for this subsample | bias | ratio |")
        L.append("|---|---|---|---|---|")
        for y in C.OUTCOMES:
            e = sc["estimates"][y]
            L.append(
                f"| `{y}` | **{e['naive']['estimate']:+.6f}** | {e['target']['target_ate']:+.6f} | "
                f"{e['bias']:+.6f} | {e['bias_ratio']:.2f}x |"
            )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strength", type=float, default=3.0)
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = data.load()
    truth = {
        y: json.loads((C.REPORTS / "step1_rct_ground_truth.json").read_text())["ate"][y]["estimate"]
        for y in C.OUTCOMES
    }
    print(f"[step2] ground truth: " + "  ".join(f"{k}={v:+.6f}" for k, v in truth.items()))

    score, meta = get_score(df, args.refit)

    step1 = json.loads((C.REPORTS / "step1_rct_ground_truth.json").read_text())
    validation = benchmark_validation(df, score, step1)
    for y, v in validation.items():
        print(f"[step2] benchmark check {y:<11} stratified {v['stratified_full_sample']:+.6f} "
              f"vs Lin {v['lin_adjusted']:+.6f} ({v['stratified_vs_lin_pct']:.2%} apart)")

    heterogeneity = {y: heterogeneity_by_decile(df, score, y) for y in C.OUTCOMES}
    sweep_rows = sweep(df, score, truth)

    scenarios = {
        d: build_scenario(df, score, d, args.strength, truth) for d in ("inflate", "deflate")
    }
    for d, sc in scenarios.items():
        e = sc["estimates"]["conversion"]
        print(f"[step2] {d:<8} conversion naive {e['naive']['estimate']:+.6f} vs target "
              f"{e['target']['target_ate']:+.6f}  ({e['bias_ratio']:.2f}x, "
              f"{e['bias_in_target_ses']:+.0f} target SEs)")

    res = {
        "step": 2,
        "prognostic_score": meta,
        "benchmark_validation": validation,
        "heterogeneity_by_prognostic_decile": heterogeneity,
        "sweep": sweep_rows,
        "chosen_strength": args.strength,
        "scenarios": scenarios,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (C.REPORTS / "step2_confounding.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, C.REPORTS / "step2_confounding.md")

    mask = np.load(C.DATA_PROCESSED / "confounded_inflate.npy")
    figures(df, score, mask)
    print(f"[step2] done in {res['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
