"""Prompts and the scoring rubric for Step 5.

The rubric is the actual deliverable here. Anyone can call a model and get
fluent marketing copy; what makes the layer engineered rather than a wrapper is
a written standard for what a good output is, an automatic scorer against that
standard, and a measured agreement rate between the automatic scorer and a
human.

Two rubric criteria are specific to this dataset and carry most of the weight:

* **Groundedness.** The features are anonymised (`f0`..`f11`). Nobody knows what
  they mean. Any output that says "these are high-income urban shoppers" has
  invented a fact, and that is the single most likely failure mode when a
  language model is handed a segment profile.

* **Causal correctness.** Base conversion rate and uplift are different
  quantities, and the top-uplift decile is not automatically the top-response
  decile. An output that recommends targeting a segment "because they convert
  at 1.8%" has confused the two, which is exactly the error the preceding four
  steps exist to prevent.
"""

from __future__ import annotations

import json

GENERATION_SYSTEM = """\
You are a marketing data scientist writing targeting guidance for a display \
advertising team. You are given the measured results of a randomised \
experiment, already analysed. Your job is to turn one segment's numbers into a \
decision.

Hard constraints:
- The features are anonymised and named f0..f11. Their real-world meaning is \
UNKNOWN. Never guess what a feature represents, never describe the segment in \
demographic, geographic, psychographic or product terms, and never name an \
industry. Refer to features only by their identifiers and their relative level.
- Use only numbers present in the input. Do not compute new statistics, do not \
extrapolate revenue, and do not invent benchmarks.
- Incremental uplift, not conversion rate, is what justifies spend. A segment \
that converts often but is barely moved by advertising is a poor target.
- Where a confidence interval crosses zero, say so and soften the \
recommendation accordingly.

Reply with a single JSON object and nothing else."""

GENERATION_TEMPLATE = """\
Segment profile from a randomised advertising experiment on {n_total:,} \
held-out users.

Segment: predicted-uplift decile {decile} (0 = highest predicted uplift)
Users in segment: {n:,}
Measured incremental conversion uplift: {observed_uplift:+.6f}
95% confidence interval: [{ci_low:+.6f}, {ci_high:+.6f}]
Conversion rate if NOT advertised to: {control_rate:.5f}
Conversion rate if advertised to: {treated_rate:.5f}
Population-average uplift across all deciles: {population_uplift:+.6f}

Anonymised feature profile, as standard deviations from the population mean:
{feature_lines}

Return JSON with exactly these keys:
{{
  "segment_label": "short neutral name, <= 6 words, no invented semantics",
  "action": one of "target", "test", "hold_out",
  "rationale": "2-3 sentences citing the specific numbers above",
  "budget_share": a number 0-1, share of campaign budget you would allocate here,
  "distinguishing_features": ["at most 3 feature ids that characterise the segment"],
  "risks": "1-2 sentences on what would make this recommendation wrong",
  "confidence": one of "high", "medium", "low"
}}"""

RUBRIC = {
    "groundedness": {
        "weight": 0.30,
        "question": "Does the output avoid inventing facts?",
        "anchors": {
            5: "Every quantitative claim traces to the input. No semantic meaning is "
               "attributed to any anonymised feature.",
            3: "Broadly grounded, but one vague or unsupported quantitative claim.",
            1: "Invents feature meaning (demographics, product categories, intent) or "
               "cites numbers not in the input.",
        },
    },
    "causal_correctness": {
        "weight": 0.30,
        "question": "Does it reason from incremental uplift rather than conversion level?",
        "anchors": {
            5: "Justifies the action by uplift, and explicitly distinguishes it from the "
               "base conversion rate where the two diverge.",
            3: "Uses uplift but does not distinguish it from base rate.",
            1: "Recommends targeting because the segment converts often, or treats a "
               "CI crossing zero as a positive effect.",
        },
    },
    "decision_usefulness": {
        "weight": 0.20,
        "question": "Could a media buyer act on this tomorrow?",
        "anchors": {
            5: "Unambiguous action, a budget share consistent with the evidence, and a "
               "concrete stated risk.",
            3: "Action is clear but the rationale is generic.",
            1: "Hedged to the point of carrying no decision.",
        },
    },
    "uncertainty_handling": {
        "weight": 0.10,
        "question": "Is statistical uncertainty represented honestly?",
        "anchors": {
            5: "Confidence matches the interval; a CI crossing zero is called out.",
            3: "Uncertainty mentioned but not tied to the interval.",
            1: "States a confident effect where the interval crosses zero.",
        },
    },
    "format_compliance": {
        "weight": 0.10,
        "question": "Valid JSON, all keys present, constraints respected?",
        "anchors": {5: "Exact schema.", 3: "Valid JSON, minor deviation.",
                    1: "Invalid JSON or missing keys."},
    },
}

JUDGE_SYSTEM = """\
You are a strict evaluator of marketing analytics writing. You score against a \
fixed rubric and you justify each score in one clause. You are not generous: a \
5 means the criterion is fully met with no reservation. Reply with a single \
JSON object and nothing else."""

JUDGE_TEMPLATE = """\
Score the CANDIDATE OUTPUT against the rubric. Judge only what the rubric asks.

=== INPUT GIVEN TO THE WRITER ===
{source}

=== CANDIDATE OUTPUT ===
{candidate}

=== RUBRIC ===
{rubric}

Score each criterion on an integer 1-5 scale using the anchors. Return JSON:
{{
  "scores": {{{score_keys}}},
  "reasons": {{{reason_keys}}},
  "overall_comment": "one sentence"
}}"""


def rubric_text() -> str:
    lines = []
    for name, r in RUBRIC.items():
        lines.append(f"\n[{name}] (weight {r['weight']:.2f}) {r['question']}")
        for score in (5, 3, 1):
            lines.append(f"  {score} = {r['anchors'][score]}")
    return "\n".join(lines)


def build_generation_prompt(segment: dict, population_uplift: float, n_total: int) -> str:
    z = segment["feature_z_scores"]
    feature_lines = "\n".join(
        f"  {k}: {v:+.2f} sd" for k, v in sorted(z.items(), key=lambda kv: -abs(kv[1]))
    )
    return GENERATION_TEMPLATE.format(
        n_total=n_total,
        decile=segment["decile"],
        n=segment["n"],
        observed_uplift=segment["observed_uplift"],
        ci_low=segment["observed_uplift"] - 1.96 * segment["observed_uplift_se"],
        ci_high=segment["observed_uplift"] + 1.96 * segment["observed_uplift_se"],
        control_rate=segment["control_conversion_rate"],
        treated_rate=segment["treated_conversion_rate"],
        population_uplift=population_uplift,
        feature_lines=feature_lines,
    )


def build_judge_prompt(source: str, candidate: str) -> str:
    keys = list(RUBRIC)
    return JUDGE_TEMPLATE.format(
        source=source,
        candidate=candidate,
        rubric=rubric_text(),
        score_keys=", ".join(f'"{k}": <1-5>' for k in keys),
        reason_keys=", ".join(f'"{k}": "<one clause>"' for k in keys),
    )


def weighted_score(scores: dict) -> float:
    """Rubric-weighted mean on the 1-5 scale."""
    total = sum(RUBRIC[k]["weight"] for k in scores if k in RUBRIC)
    if total == 0:
        return float("nan")
    return sum(RUBRIC[k]["weight"] * float(v) for k, v in scores.items() if k in RUBRIC) / total


def as_json(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
