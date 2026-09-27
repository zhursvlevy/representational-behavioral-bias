"""Relation between final-layer Delta F1 (texture - shape) and behavioral shape bias
across the main-set conditions, and whether the three models share one regression line.

The conditions are not independent (models share lambda values, and points of one model
are ordered by lambda), so the p-values are optimistic; the within-condition analysis
is the test that does not rely on the sweep.

    python -m rbb.analysis.correlation    -> results/correlation_stats.json
"""

import numpy as np
from scipy import stats as sps

from rbb import config
from rbb.analysis.common import last_layer, main_conditions, read_probe_metrics, read_shape_bias, write_json
from rbb.stats import ols


def load_points() -> list[dict]:
    points = []
    for model, scale in main_conditions():
        condition = config.condition("main", scale)
        f1 = read_probe_metrics(condition, model)
        shape_f1 = last_layer(f1["shape_f1_macro_mean"])
        texture_f1 = last_layer(f1["texture_f1_macro_mean"])
        points.append({
            "model": model, "scale": scale,
            "shape_f1_last_layer": shape_f1, "texture_f1_last_layer": texture_f1,
            "delta_f1": texture_f1 - shape_f1,
            "shape_bias": read_shape_bias(condition, model),
        })
    return points


def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    pearson_r, pearson_p = sps.pearsonr(x, y)
    spearman_r, spearman_p = sps.spearmanr(x, y)
    return {"n": len(x), "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r), "spearman_p": float(spearman_p)}


def common_vs_per_model_lines(points: list[dict]) -> dict:
    """Nested-model F-test: one line for all points vs. one line per model."""
    x = np.array([p["delta_f1"] for p in points])
    y = np.array([p["shape_bias"] for p in points])
    common = ols(x, y)
    per_model = {}
    for model in config.MODELS:
        xm = np.array([p["delta_f1"] for p in points if p["model"] == model])
        ym = np.array([p["shape_bias"] for p in points if p["model"] == model])
        per_model[model] = ols(xm, ym)
    rss_full = sum(m["rss"] for m in per_model.values())
    df_full = sum(m["dof"] for m in per_model.values())
    df_num = common["dof"] - df_full
    f_stat = ((common["rss"] - rss_full) / df_num) / (rss_full / df_full)
    return {
        "common_line": common,
        "per_model_lines": per_model,
        "f_test": {"rss_common_line": common["rss"], "rss_per_model_lines": rss_full,
                   "df_numerator": df_num, "df_denominator": df_full,
                   "f_statistic": float(f_stat), "p_value": float(1 - sps.f.cdf(f_stat, df_num, df_full))},
    }


def main_regression_line() -> dict:
    """Common regression line of shape bias on Delta F1 over the main set."""
    points = load_points()
    return ols(np.array([p["delta_f1"] for p in points]), np.array([p["shape_bias"] for p in points]))


def main() -> None:
    points = load_points()
    x = np.array([p["delta_f1"] for p in points])
    y = np.array([p["shape_bias"] for p in points])
    per_model = {}
    for model in config.MODELS:
        mask = np.array([p["model"] == model for p in points])
        per_model[model] = correlations(x[mask], y[mask])
    write_json(config.RESULTS_DIR / "correlation_stats.json", {
        "points": points,
        "correlations": {"overall": correlations(x, y), "per_model": per_model},
        "regression": common_vs_per_model_lines(points),
    })


if __name__ == "__main__":
    main()
