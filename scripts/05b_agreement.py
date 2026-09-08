"""Step 5b - fold the human spot-check back in and report three-way agreement.

Run `scripts/05_genai.py` first; it writes `reports/step5_spotcheck_blank.csv`.
Fill the `human_*` columns with integer 1-5 scores for as many rows as you are
willing to rate (30-50 is the useful range; partial sheets are fine, rows left
blank are skipped), save it as `reports/step5_spotcheck_filled.csv`, then run
this.

Reports quadratic-weighted kappa for each rater pair. The number that decides
whether the LLM judge is usable is human-vs-judge; rules-vs-judge only says
whether the judge agrees with things a regex already knew.

Usage:
    python scripts/05b_agreement.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift import config as C  # noqa: E402
from uplift.genai import evaluate as EV, prompts as P  # noqa: E402

FILLED = C.REPORTS / "step5_spotcheck_filled.csv"


def load_human() -> dict[str, dict]:
    if not FILLED.exists():
        raise SystemExit(
            f"{FILLED} not found.\n"
            "Fill the human_* columns in reports/step5_spotcheck_blank.csv and save it "
            "under that name."
        )
    out: dict[str, dict] = {}
    with FILLED.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            scores = {}
            for crit in P.RUBRIC:
                v = (row.get(f"human_{crit}") or "").strip()
                if v:
                    try:
                        s = int(round(float(v)))
                    except ValueError:
                        continue
                    if 1 <= s <= 5:
                        scores[crit] = s
            if scores:
                out[row["id"]] = {"scores": scores}
    return out


def summarise(name: str, agree: dict) -> list[str]:
    L = [f"\n### {name}\n",
         "| criterion | n | exact | within 1 | QWK | mean A | mean B |",
         "|---|---|---|---|---|---|---|"]
    for crit, a in agree.items():
        if not a.get("n"):
            continue
        label = "**pooled**" if crit == "_pooled" else f"`{crit}`"
        L.append(f"| {label} | {a['n']} | {a['exact_agreement']:.0%} | "
                 f"{a['within_1']:.0%} | {a['qwk']:.3f} | "
                 f"{a['mean_rater_a']:.2f} | {a['mean_rater_b']:.2f} |")
    return L


def main() -> int:
    res = json.loads((C.REPORTS / "step5_genai.json").read_text())
    records = res["records"]
    human = load_human()
    for r in records:
        if r["id"] in human:
            r["human"] = human[r["id"]]
    rated = [r for r in records if r.get("human")]
    if not rated:
        raise SystemExit("no usable human ratings found in the filled sheet")

    pairs = {
        "human vs LLM judge": EV.compare_raters(rated, "human", "judge"),
        "human vs deterministic rules": EV.compare_raters(rated, "human", "rules"),
        "deterministic rules vs LLM judge": EV.compare_raters(records, "rules", "judge"),
    }

    L = ["# Step 5b - inter-rater agreement\n",
         f"{len(rated)} of {len(records)} outputs carry human ratings.\n",
         "Quadratic-weighted kappa (QWK): 1.0 is perfect, 0.0 is chance. On a 1-5 "
         "ordinal rubric, unweighted kappa would treat a 4-vs-5 disagreement as badly "
         "as 1-vs-5, so the weighted form is the honest one.\n"]
    for name, a in pairs.items():
        L += summarise(name, a)

    ctrl = [r for r in rated if r["kind"] != "real"]
    if ctrl:
        L.append(f"\n### Negative controls\n")
        L.append(f"{len(ctrl)} deliberately corrupted outputs were rated by hand. "
                 "If the judge did not score these well below the real outputs, its "
                 "scores on the real ones mean nothing.\n")

    path = C.REPORTS / "step5b_agreement.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (C.REPORTS / "step5b_agreement.json").write_text(
        json.dumps({"n_rated": len(rated), "pairs": pairs}, indent=2, default=float))

    hj = pairs["human vs LLM judge"]["_pooled"]
    print(f"[step5b] {len(rated)} rated; human-vs-judge pooled QWK {hj.get('qwk', float('nan')):.3f} "
          f"(exact {hj.get('exact_agreement', 0):.0%}, within-1 {hj.get('within_1', 0):.0%})")
    print(f"[step5b] -> {path.relative_to(C.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
