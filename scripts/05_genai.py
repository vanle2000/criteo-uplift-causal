"""Step 5 - generate per-segment targeting guidance and evaluate it.

Feeds each predicted-uplift decile's measured profile to Claude, asks for a
targeting decision as structured JSON, then scores every output three ways:
deterministic rules, an LLM judge against the written rubric, and (via a
generated sheet) a human spot-check.

The harness includes NEGATIVE CONTROLS: a handful of outputs corrupted on
purpose -- invented demographics, fabricated numbers, a confident call on a
null result. A judge that scores those highly is not a judge, and without them
"the LLM judge gave us 4.6/5" is an unfalsifiable claim.

Backends: --backend anthropic | bedrock | replay. `replay` serves the committed
response cache and makes the whole step reproducible with no credentials.

Usage:
    python scripts/05_genai.py --backend anthropic
    python scripts/05_genai.py --backend replay        # offline, from cache
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config as C  # noqa: E402
from uplift.genai import evaluate as EV, prompts as P, provider as PV  # noqa: E402

CACHE_DIR = C.PROJECT_ROOT / "data" / "genai_cache"


def corrupt(parsed: dict, kind: str) -> dict:
    """Damage a good output in a specific, rubric-relevant way."""
    bad = copy.deepcopy(parsed)
    if kind == "invented_semantics":
        bad["segment_label"] = "Affluent urban millennial travel shoppers"
        bad["rationale"] = (
            "These are high-income urban millennials with strong travel and luxury "
            "apparel intent, so the creative should lean into aspirational imagery."
        )
    elif kind == "fabricated_numbers":
        bad["rationale"] = (
            "This segment delivers a 4.7x return on ad spend and an incremental "
            "$312,000 in revenue, well above the 2.1% industry benchmark."
        )
    elif kind == "base_rate_confusion":
        bad["action"] = "target"
        bad["confidence"] = "high"
        bad["rationale"] = (
            "This is the highest converting segment in the campaign, so we should "
            "target it because they convert most often."
        )
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="replay", choices=["anthropic", "bedrock", "replay"])
    ap.add_argument("--model", default=PV.DEFAULT_MODEL)
    ap.add_argument("--judge-model", default=PV.DEFAULT_MODEL)
    ap.add_argument("--outcome", default=C.PRIMARY_OUTCOME)
    ap.add_argument("--scorers", nargs="+",
                    default=["two_model_tlearner", "transformed_outcome", "response_baseline"])
    args = ap.parse_args()

    seg_path = C.REPORTS / f"step4_segments_{args.outcome}.json"
    if not seg_path.exists():
        raise SystemExit(f"{seg_path} missing - run scripts/04_uplift.py first")
    segments = json.loads(seg_path.read_text())
    step4 = json.loads((C.REPORTS / f"step4_uplift_{args.outcome}.json").read_text())
    n_total = step4["n_test"]

    gen = PV.Provider(backend=args.backend, model=args.model, cache_dir=CACHE_DIR)
    judge = PV.Provider(backend=args.backend, model=args.judge_model,
                        cache_dir=CACHE_DIR, max_tokens=900)

    records: list[dict] = []
    calls: list[PV.Call] = []
    t_start = time.perf_counter()

    for scorer in args.scorers:
        if scorer not in segments:
            print(f"[step5] no segments for {scorer!r}, skipping")
            continue
        segs = segments[scorer]
        pop_uplift = float(np.average([s["observed_uplift"] for s in segs],
                                      weights=[s["n"] for s in segs]))
        for seg in segs:
            prompt = P.build_generation_prompt(seg, pop_uplift, n_total)
            call = gen.complete(prompt, system=P.GENERATION_SYSTEM)
            calls.append(call)
            try:
                parsed = PV.extract_json(call.text)
            except Exception:
                parsed = None
            records.append({
                "id": f"{scorer}-d{seg['decile']}",
                "scorer": scorer, "decile": seg["decile"], "kind": "real",
                "segment": seg, "population_uplift": pop_uplift,
                "prompt": prompt, "raw": call.text, "parsed": parsed,
                "gen_call": call.as_dict(),
            })
            print(f"[step5] generated {records[-1]['id']:<32} "
                  f"{'cache' if call.cached else f'{call.latency_s:.1f}s'}  "
                  f"${call.cost_usd:.5f}")

    # --- negative controls -------------------------------------------------
    donors = [r for r in records if r["parsed"]][:2]
    for donor in donors:
        for kind in ("invented_semantics", "fabricated_numbers", "base_rate_confusion"):
            bad = corrupt(donor["parsed"], kind)
            records.append({
                "id": f"control-{kind}-{donor['decile']}",
                "scorer": "negative_control", "decile": donor["decile"], "kind": kind,
                "segment": donor["segment"], "population_uplift": donor["population_uplift"],
                "prompt": donor["prompt"], "raw": json.dumps(bad, indent=2),
                "parsed": bad, "gen_call": None,
            })
    print(f"[step5] added {len(records) - len(calls)} negative controls")

    # --- score: rules then judge ------------------------------------------
    for r in records:
        r["rules"] = EV.rule_scores(r["parsed"], r["raw"], r["segment"],
                                    r["population_uplift"])
        jp = P.build_judge_prompt(r["prompt"], r["raw"])
        jc = judge.complete(jp, system=P.JUDGE_SYSTEM)
        calls.append(jc)
        try:
            jd = PV.extract_json(jc.text)
            r["judge"] = {"scores": {k: int(v) for k, v in jd["scores"].items()},
                          "reasons": jd.get("reasons", {}),
                          "comment": jd.get("overall_comment", "")}
        except Exception as exc:  # noqa: BLE001
            r["judge"] = None
            print(f"[step5] judge parse failed for {r['id']}: {exc}")
        r["rules_weighted"] = P.weighted_score(r["rules"]["scores"])
        r["judge_weighted"] = (P.weighted_score(r["judge"]["scores"])
                               if r["judge"] else float("nan"))

    # --- did the judge catch the planted failures? -------------------------
    real = [r for r in records if r["kind"] == "real" and r["judge"]]
    ctrl = [r for r in records if r["kind"] != "real" and r["judge"]]
    control_check = {
        "n_real": len(real), "n_controls": len(ctrl),
        "mean_judge_real": float(np.mean([r["judge_weighted"] for r in real])) if real else None,
        "mean_judge_control": float(np.mean([r["judge_weighted"] for r in ctrl])) if ctrl else None,
        "mean_rules_real": float(np.mean([r["rules_weighted"] for r in real])) if real else None,
        "mean_rules_control": float(np.mean([r["rules_weighted"] for r in ctrl])) if ctrl else None,
    }
    if real and ctrl:
        control_check["judge_separates_controls"] = bool(
            control_check["mean_judge_control"] < control_check["mean_judge_real"] - 0.5)

    # --- cost and latency --------------------------------------------------
    live = [c for c in calls if not c.cached]
    lat = np.array([c.latency_s for c in live]) if live else np.array([])
    cost = {
        "n_calls_total": len(calls),
        "n_calls_live": len(live),
        "n_calls_cached": len(calls) - len(live),
        "total_cost_usd": float(sum(c.cost_usd for c in calls)),
        "mean_cost_per_call_usd": float(np.mean([c.cost_usd for c in calls])) if calls else 0.0,
        "total_input_tokens": int(sum(c.input_tokens for c in calls)),
        "total_output_tokens": int(sum(c.output_tokens for c in calls)),
        "latency_mean_s": float(lat.mean()) if lat.size else None,
        "latency_p50_s": float(np.percentile(lat, 50)) if lat.size else None,
        "latency_p95_s": float(np.percentile(lat, 95)) if lat.size else None,
        "wall_clock_s": round(time.perf_counter() - t_start, 1),
        "model": args.model, "judge_model": args.judge_model, "backend": args.backend,
        "price_table_note": "PRICE_PER_MTOK in provider.py is operator-supplied; verify before quoting",
    }

    agree_rules_judge = EV.compare_raters(records, "rules", "judge")

    out = {
        "step": 5, "outcome": args.outcome,
        "n_outputs": len(records),
        "control_check": control_check,
        "cost_and_latency": cost,
        "agreement_rules_vs_judge": agree_rules_judge,
        "records": [
            {k: v for k, v in r.items() if k not in ("prompt",)} for r in records
        ],
    }
    (C.REPORTS / "step5_genai.json").write_text(json.dumps(out, indent=2, default=float))

    # --- human spot-check sheet -------------------------------------------
    sheet = C.REPORTS / "step5_spotcheck_blank.csv"
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "kind", "segment_summary", "output"]
                   + [f"human_{c}" for c in P.RUBRIC])
        for r in records:
            s = r["segment"]
            w.writerow([
                r["id"], r["kind"],
                f"decile {s['decile']}, n={s['n']}, uplift={s['observed_uplift']:+.6f}",
                r["raw"].replace("\n", " ")[:1500],
            ] + [""] * len(P.RUBRIC))

    print(f"\n[step5] {len(records)} outputs; judge mean (real) "
          f"{control_check['mean_judge_real']}, (controls) "
          f"{control_check['mean_judge_control']}")
    print(f"[step5] rules-vs-judge pooled QWK "
          f"{agree_rules_judge['_pooled'].get('qwk', float('nan')):.3f}")
    print(f"[step5] cost ${cost['total_cost_usd']:.4f} over {cost['n_calls_total']} calls "
          f"({cost['n_calls_live']} live); latency p50 {cost['latency_p50_s']}s")
    print(f"[step5] spot-check sheet -> {sheet.relative_to(C.PROJECT_ROOT)} "
          f"(fill human_* columns, then run scripts/05b_agreement.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
