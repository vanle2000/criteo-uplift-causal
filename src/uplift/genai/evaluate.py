"""Scoring and agreement measurement for Step 5.

Three raters score every generated output on the same 1-5 rubric:

  `rules`  deterministic checks -- schema, forbidden semantic vocabulary,
           numeric groundedness, and consistency between the confidence claimed
           and the confidence interval supplied
  `judge`  an LLM scoring against the written rubric
  `human`  a spot-check sheet, filled in by hand

An LLM judge that nobody has checked is just a second opinion of unknown
quality. The point of carrying all three is that the harness reports how often
the judge agrees with rules it cannot game, and how often it agrees with a
person -- so the judge's own scores come with an error bar.

Agreement is reported as quadratic-weighted Cohen's kappa. On a 1-5 ordinal
scale, unweighted kappa treats a 4-vs-5 disagreement as badly as 1-vs-5, which
would understate agreement between raters who mostly differ by one notch.
"""

from __future__ import annotations

import math
import re

import numpy as np

from .prompts import RUBRIC

REQUIRED_KEYS = {
    "segment_label", "action", "rationale", "budget_share",
    "distinguishing_features", "risks", "confidence",
}
VALID_ACTIONS = {"target", "test", "hold_out"}
VALID_CONFIDENCE = {"high", "medium", "low"}

# Vocabulary that can only appear if the model invented meaning for an
# anonymised feature. Deliberately narrow: these are words with no legitimate
# reading given the input, so a hit is a genuine hallucination rather than a
# stylistic quibble.
FORBIDDEN_SEMANTICS = [
    "demographic", "age group", "income", "affluent", "wealthy", "urban", "rural",
    "male", "female", "gender", "millennial", "gen z", "boomer", "student",
    "luxury", "budget-conscious", "fashion", "electronics", "apparel", "grocery",
    "travel", "automotive", "b2b", "enterprise", "region", "country", "city",
    "mobile user", "desktop user", "ios", "android", "browser",
]
CAUSAL_TERMS = ["uplift", "incremental", "lift", "causal", "treatment effect"]
BASE_RATE_CONFUSION = [
    "because they convert", "high conversion rate means", "most likely to convert",
    "converts at the highest", "highest converting",
]


def _nums(text: str) -> list[float]:
    out = []
    for tok in re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?%?", text):
        pct = tok.endswith("%")
        try:
            v = float(tok.rstrip("%"))
        except ValueError:
            continue
        out.append(v / 100 if pct else v)
    return out


def rule_scores(parsed: dict | None, raw: str, segment: dict, population_uplift: float) -> dict:
    """Deterministic 1/3/5 scores per rubric criterion.

    Coarse on purpose. These checks are meant to be things a rule can be right
    about; graded judgement is what the LLM judge and the human are for.
    """
    ci_low = segment["observed_uplift"] - 1.96 * segment["observed_uplift_se"]
    ci_high = segment["observed_uplift"] + 1.96 * segment["observed_uplift_se"]
    crosses_zero = ci_low <= 0 <= ci_high
    scores, notes = {}, {}

    # -- format ----------------------------------------------------------
    if parsed is None:
        scores["format_compliance"] = 1
        notes["format_compliance"] = "response was not valid JSON"
    else:
        missing = REQUIRED_KEYS - set(parsed)
        bad = (
            parsed.get("action") not in VALID_ACTIONS
            or parsed.get("confidence") not in VALID_CONFIDENCE
            or not isinstance(parsed.get("budget_share"), (int, float))
            or not 0 <= float(parsed.get("budget_share", -1)) <= 1
        )
        scores["format_compliance"] = 1 if missing else (3 if bad else 5)
        notes["format_compliance"] = (
            f"missing {sorted(missing)}" if missing else ("invalid enum/range" if bad else "ok")
        )

    body = "" if parsed is None else " ".join(
        str(parsed.get(k, "")) for k in ("segment_label", "rationale", "risks")
    )
    low = body.lower()

    # -- groundedness -----------------------------------------------------
    hits = [w for w in FORBIDDEN_SEMANTICS if w in low]
    allowed = set()
    for v in (segment["observed_uplift"], ci_low, ci_high,
              segment["control_conversion_rate"], segment["treated_conversion_rate"],
              population_uplift, segment["n"], segment["decile"]):
        allowed.add(round(float(v), 6))
    for v in segment["feature_z_scores"].values():
        allowed.add(round(float(v), 6))
    unsupported = []
    for v in _nums(body):
        if any(math.isclose(v, a, rel_tol=0.02, abs_tol=1e-6) for a in allowed):
            continue
        # Small integers are almost always feature ids or counts of items.
        if float(v).is_integer() and abs(v) <= 12:
            continue
        unsupported.append(v)
    if hits:
        scores["groundedness"] = 1
    elif unsupported:
        scores["groundedness"] = 3
    else:
        scores["groundedness"] = 5
    notes["groundedness"] = (
        f"invented semantics {hits}" if hits
        else (f"unsupported numbers {unsupported}" if unsupported else "ok")
    )

    # -- causal correctness ----------------------------------------------
    uses_causal = any(w in low for w in CAUSAL_TERMS)
    confuses = any(w in low for w in BASE_RATE_CONFUSION)
    treats_null_as_effect = (
        crosses_zero and parsed is not None
        and parsed.get("action") == "target" and parsed.get("confidence") == "high"
    )
    if confuses or treats_null_as_effect:
        scores["causal_correctness"] = 1
    elif uses_causal and ("rate" in low or "base" in low):
        scores["causal_correctness"] = 5
    elif uses_causal:
        scores["causal_correctness"] = 3
    else:
        scores["causal_correctness"] = 1
    notes["causal_correctness"] = (
        "confuses base rate with uplift" if confuses
        else ("treats a null interval as a positive effect" if treats_null_as_effect
              else ("cites uplift" if uses_causal else "no causal language"))
    )

    # -- uncertainty ------------------------------------------------------
    mentions_ci = any(w in low for w in
                      ("confidence interval", "interval", "crosses zero", "not significant",
                       "uncertain", "noisy", "imprecise", "spans zero", "includes zero"))
    if crosses_zero:
        scores["uncertainty_handling"] = 5 if mentions_ci else 1
        notes["uncertainty_handling"] = (
            "CI crosses zero and is acknowledged" if mentions_ci
            else "CI crosses zero and is NOT acknowledged")
    else:
        scores["uncertainty_handling"] = 5 if mentions_ci else 3
        notes["uncertainty_handling"] = "CI excludes zero"

    # -- usefulness -------------------------------------------------------
    has_action = parsed is not None and parsed.get("action") in VALID_ACTIONS
    has_risk = parsed is not None and len(str(parsed.get("risks", ""))) > 25
    scores["decision_usefulness"] = 5 if (has_action and has_risk) else (3 if has_action else 1)
    notes["decision_usefulness"] = "action + stated risk" if (has_action and has_risk) else "thin"

    return {"scores": scores, "notes": notes, "ci_crosses_zero": crosses_zero}


def quadratic_weighted_kappa(a, b, min_rating: int = 1, max_rating: int = 5) -> float:
    """Cohen's kappa with quadratic weights on an ordinal scale.

    1.0 is perfect agreement, 0.0 is chance. Quadratic weights mean a
    one-notch disagreement costs a sixteenth of a four-notch one, which is the
    right shape for rubric scores.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.size == 0:
        return float("nan")
    n = max_rating - min_rating + 1
    O = np.zeros((n, n))
    for i, j in zip(a - min_rating, b - min_rating):
        O[i, j] += 1
    w = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], dtype=float)
    w /= (n - 1) ** 2
    ha = np.bincount(a - min_rating, minlength=n).astype(float)
    hb = np.bincount(b - min_rating, minlength=n).astype(float)
    Ex = np.outer(ha, hb)
    Ex *= O.sum() / Ex.sum()
    denom = (w * Ex).sum()
    return float(1.0 - (w * O).sum() / denom) if denom > 0 else float("nan")


def agreement(pairs: list[tuple[int, int]]) -> dict:
    """Agreement summary for one rater pair on one criterion."""
    if not pairs:
        return {"n": 0}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    d = np.abs(a - b)
    out = {
        "n": int(a.size),
        "exact_agreement": float((d == 0).mean()),
        "within_1": float((d <= 1).mean()),
        "mean_abs_diff": float(d.mean()),
        "qwk": quadratic_weighted_kappa(a, b),
        "mean_rater_a": float(a.mean()),
        "mean_rater_b": float(b.mean()),
    }
    if a.std() > 0 and b.std() > 0:
        from scipy.stats import spearmanr

        out["spearman"] = float(spearmanr(a, b).statistic)
    return out


def compare_raters(records: list[dict], rater_a: str, rater_b: str) -> dict:
    """Per-criterion and pooled agreement between two rater columns."""
    out = {}
    for crit in RUBRIC:
        pairs = [
            (r[rater_a]["scores"][crit], r[rater_b]["scores"][crit])
            for r in records
            if r.get(rater_a) and r.get(rater_b)
            and crit in r[rater_a]["scores"] and crit in r[rater_b]["scores"]
        ]
        out[crit] = agreement(pairs)
    pooled = [
        (r[rater_a]["scores"][c], r[rater_b]["scores"][c])
        for r in records if r.get(rater_a) and r.get(rater_b)
        for c in RUBRIC
        if c in r[rater_a]["scores"] and c in r[rater_b]["scores"]
    ]
    out["_pooled"] = agreement(pooled)
    return out
