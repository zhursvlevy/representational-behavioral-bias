"""Control set with a cropped style source (LLaVA, lambda in {1000, 6000, 12000}).

Compares shape bias, final-layer probe F1, Delta F1 and within-condition AUROC with the
main set at the same lambda, and places the control points on the regression line of
shape bias on Delta F1 fitted to the main set.

    python -m rbb.analysis.crop_control
        -> results/crop_control_breakdown.csv, crop_control_within_condition.csv,
           crop_control_comparison.csv
"""

import csv

import numpy as np

from rbb import config
from rbb.analysis.behavior import breakdown_row
from rbb.analysis.common import last_layer, read_probe_metrics, read_shape_bias, write_csv
from rbb.analysis.correlation import main_regression_line
from rbb.analysis.within_condition import last_layer_rows

MODEL = "llava"


def main() -> None:
    write_csv(config.RESULTS_DIR / "crop_control_breakdown.csv",
              [breakdown_row("crop", MODEL, s) for s in config.CROP_SCALES])

    rng = np.random.default_rng(config.SEED)
    within = []
    for scale in config.CROP_SCALES:
        within += last_layer_rows(MODEL, "crop", scale, rng, variants=("probability",))
    write_csv(config.RESULTS_DIR / "crop_control_within_condition.csv", within)

    auroc = {("crop", r["scale"]): r for r in within}
    with (config.RESULTS_DIR / "within_condition_auroc.csv").open() as f:
        for r in csv.DictReader(f):
            if r["set"] == "main" and r["model"] == MODEL and r["variant"] == "probability":
                auroc[("main", int(r["scale"]))] = r
    main_line = main_regression_line()
    rows = []
    for scale in config.CROP_SCALES:
        for stimulus_set in ("main", "crop"):
            condition = config.condition(stimulus_set, scale)
            f1 = read_probe_metrics(condition, MODEL)
            shape_f1 = last_layer(f1["shape_f1_macro_mean"])
            texture_f1 = last_layer(f1["texture_f1_macro_mean"])
            delta = texture_f1 - shape_f1
            bias = read_shape_bias(condition, MODEL)
            predicted = main_line["intercept"] + main_line["slope"] * delta
            row = {
                "set": stimulus_set, "scale": scale, "shape_bias": bias,
                "shape_f1_last_layer": round(shape_f1, 6), "texture_f1_last_layer": round(texture_f1, 6),
                "delta_f1": round(delta, 6),
                "shape_bias_predicted_by_main_line": round(predicted, 6),
                "residual_from_main_line": round(bias - predicted, 6),
            }
            row["within_condition_auroc"] = float(auroc[(stimulus_set, scale)]["auroc"])
            row["excluded_frac"] = float(auroc[(stimulus_set, scale)]["excluded_frac"])
            rows.append(row)
    write_csv(config.RESULTS_DIR / "crop_control_comparison.csv", rows)


if __name__ == "__main__":
    main()
