"""Behavioral shape bias and the full composition of answers per condition (Table behavior).

    python -m rbb.analysis.behavior    -> results/answer_breakdown.csv
"""

from collections import Counter

from rbb import config
from rbb.analysis.common import CATEGORIES, main_conditions, read_answers, read_shape_bias, write_csv


def breakdown_row(stimulus_set: str, model: str, scale: int) -> dict:
    condition = config.condition(stimulus_set, scale)
    answers = read_answers(condition, model)
    counts = Counter(r["category"] for r in answers)
    n = len(answers)
    row = {"set": stimulus_set, "model": model, "scale": scale, "n_samples": n,
           "shape_bias": read_shape_bias(condition, model)}
    row.update({f"{c}_frac": round(counts[c] / n, 6) for c in CATEGORIES})
    row.update({f"{c}_count": counts[c] for c in CATEGORIES})
    return row


def main() -> None:
    rows = [breakdown_row("main", model, scale) for model, scale in main_conditions()]
    write_csv(config.RESULTS_DIR / "answer_breakdown.csv", rows)


if __name__ == "__main__":
    main()
