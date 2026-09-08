"""Step 3, part two - how much hidden bias would overturn the Step 3 results?

Steps 1-3 ask which estimators recover a known answer. That question exists
only because randomisation supplied the answer. This script asks the question
you are actually left with on observational data, where it does not:

    an unmeasured confounder of what strength would be required to explain
    this effect away?

Rosenbaum bounds run on the Step 3 matched pairs; E-values are computed for
every Step 3 estimate on the risk-ratio scale. Neither detects confounding --
both state what would be needed -- and the reader still has to judge whether a
confounder that strong is plausible for display advertising.

Usage:
    python scripts/03_sensitivity.py [--direction inflate] [--sample 0]
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
from uplift import data, estimators as E, sensitivity as S  # noqa: E402

GAMMAS = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", default="inflate", choices=["inflate", "deflate"])
    ap.add_argument("--outcomes", nargs="+", default=["conversion", "visit"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    df = data.load()
    mask = np.load(C.DATA_PROCESSED / f"confounded_{args.direction}.npy")
    if args.sample:
        idx = np.flatnonzero(mask)
        keep = np.random.default_rng(C.SEED).choice(idx, size=args.sample, replace=False)
        mask = np.zeros(len(df), dtype=bool)
        mask[np.sort(keep)] = True

    X = df[C.FEATURES][mask]
    t = df[C.TREATMENT].to_numpy()[mask]
    step3 = json.loads((C.REPORTS / f"step3_recovery_{args.direction}.json").read_text())

    # Same seed and fold assignment as Step 3, so this is the identical score.
    print("[step3] refitting propensity (deterministic, matches Step 3) ...")
    prop = E.fit_propensity(X, t, method="lgbm", n_folds=args.folds)
    print(f"[step3] propensity AUC {prop.auc:.4f}, calibration slope "
          f"{prop.calibration_slope:.3f}")

    out: dict = {"step": 3, "part": "sensitivity", "direction": args.direction, "outcomes": {}}

    for outcome in args.outcomes:
        y = df[outcome].to_numpy()[mask]
        o3 = step3["outcomes"][outcome]
        target = o3["target"]["ate"]["target_ate"]

        _, diag = E.psm_ate(y, t, prop.e, outcome=outcome)
        yt = y[diag["matched_treated_idx"]]
        yc = y[diag["matched_control_idx"]]

        bounds = S.rosenbaum_bounds(yt, yc, gammas=GAMMAS)
        gcrit = S.gamma_critical(yt, yc)

        # Baseline risk for the ratio scale. NOT the observed control rate in
        # the confounded sample: selection deliberately kept low-prognostic
        # controls, so that rate is depressed and would inflate every risk
        # ratio. The right denominator is the counterfactual untreated risk for
        # the subsample's covariate mix, which the cross-fitted outcome model
        # already estimates as mean(mu_0).
        observed_control_rate = float(y[t == 0].mean())
        baseline = next(
            (r["mean_mu0"] for r in o3["results"]
             if r["method"].startswith("AIPW doubly-robust (lgbm")),
            None,
        )
        if baseline is None:
            baseline = observed_control_rate
            print(f"[step3] WARNING: no mean_mu0 in Step 3 output for {outcome}; "
                  "falling back to the observed control rate")
        control_rate = float(baseline)
        evalues = []
        for r in o3["results"]:
            if r["target_kind"] != "ate":
                continue
            try:
                rr = S.risk_ratio_from_ate(r["estimate"], control_rate)
                rr_lo = S.risk_ratio_from_ate(r["ci_low"], control_rate)
                rr_hi = S.risk_ratio_from_ate(r["ci_high"], control_rate)
                if min(rr, rr_lo, rr_hi) <= 0:
                    raise ValueError("non-positive implied risk")
                evalues.append({
                    "method": r["method"], "estimate": r["estimate"],
                    "risk_ratio": rr,
                    "e_value_point": S.e_value(rr),
                    "e_value_ci": S.e_value_ci(rr_lo, rr_hi),
                    "bias_pct_vs_target": r["bias_pct"],
                })
            except ValueError:
                # A point estimate below -control_rate implies a negative
                # absolute risk, which has no risk-ratio representation.
                evalues.append({"method": r["method"], "estimate": r["estimate"],
                                "risk_ratio": None, "e_value_point": None,
                                "e_value_ci": None,
                                "bias_pct_vs_target": r["bias_pct"]})

        rr_target = S.risk_ratio_from_ate(target, control_rate)
        out["outcomes"][outcome] = {
            "target_ate": target,
            "baseline_risk": control_rate,
            "baseline_source": "mean(mu_0), cross-fitted outcome model",
            "observed_control_rate_in_subsample": observed_control_rate,
            "target_risk_ratio": rr_target,
            "target_e_value": S.e_value(rr_target),
            "matched_pairs": int(len(yt)),
            "n_discordant": int(bounds.attrs["n_discordant"]),
            "n_plus": int(bounds.attrs["n_plus"]),
            "n_minus": int(bounds.attrs["n_minus"]),
            "gamma_critical": gcrit,
            "rosenbaum": bounds.to_dict(orient="records"),
            "e_values": evalues,
        }
        print(f"[step3] {outcome}: {len(yt):,} matched pairs, "
              f"{bounds.attrs['n_discordant']:,} discordant "
              f"({bounds.attrs['n_plus']:,}+ / {bounds.attrs['n_minus']:,}-); "
              f"Gamma* = {gcrit:.2f}")
        print(f"[step3]   target RR {rr_target:.3f} -> E-value "
              f"{S.e_value(rr_target):.2f}")

    out["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    (C.REPORTS / f"step3_sensitivity_{args.direction}.json").write_text(
        json.dumps(out, indent=2, default=float))
    write_markdown(out, C.REPORTS / f"step3_sensitivity_{args.direction}.md")
    print(f"[step3] done in {out['runtime_seconds']}s")
    return 0


def write_markdown(res: dict, path: Path) -> None:
    L = [f"# Step 3, part two - sensitivity to unmeasured confounding "
         f"(`{res['direction']}` scenario)\n"]
    L.append(
        "Rosenbaum's Gamma is an odds ratio on **treatment assignment**: how "
        "differently two units identical on X could have been selected before the "
        "matched-pair inference stops being significant. The E-value is the minimum "
        "association, on the risk-ratio scale, that a confounder would need with "
        "**both** treatment and outcome to explain the estimate away.\n"
    )
    for outcome, o in res["outcomes"].items():
        L.append(f"\n## `{outcome}`\n")
        L.append(
            f"{o['matched_pairs']:,} matched pairs, {o['n_discordant']:,} discordant "
            f"({o['n_plus']:,} favouring treated, {o['n_minus']:,} favouring control). "
            f"**Gamma\\* = {o['gamma_critical']:.2f}**.\n"
        )
        L.append("| Gamma | upper-bound p | lower-bound p | still significant |")
        L.append("|---|---|---|---|")
        for r in o["rosenbaum"]:
            L.append(f"| {r['gamma']:.2f} | {r['p_upper']:.3g} | {r['p_lower']:.3g} | "
                     f"{'yes' if r['significant_at_alpha'] else 'NO'} |")
        L.append(
            f"\nThe RCT-derived target for this subsample is a risk ratio of "
            f"{o['target_risk_ratio']:.3f} against a counterfactual untreated risk "
            f"of {o['baseline_risk']:.5f}, an E-value of "
            f"**{o['target_e_value']:.2f}**. The *observed* control rate in the "
            f"confounded subsample is only "
            f"{o['observed_control_rate_in_subsample']:.5f}; using that as the "
            f"denominator would inflate every ratio below, because selection kept "
            f"low-prognostic controls on purpose.\n"
        )
        L.append("| estimator | estimate | implied RR | E-value (point) | E-value (CI) | bias vs target |")
        L.append("|---|---|---|---|---|---|")
        for e in o["e_values"]:
            if e["risk_ratio"] is None:
                L.append(f"| {e['method']} | {e['estimate']:+.6f} | n/a | n/a | n/a | "
                         f"{e['bias_pct_vs_target']:+.1%} |")
            else:
                L.append(
                    f"| {e['method']} | {e['estimate']:+.6f} | {e['risk_ratio']:.3f} | "
                    f"{e['e_value_point']:.2f} | {e['e_value_ci']:.2f} | "
                    f"{e['bias_pct_vs_target']:+.1%} |"
                )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
