"""Final-layer probe macro-F1 against per-cue ceilings (Table probes).

Shape ceiling: unstylized shape sources (independent of lambda). Texture ceiling: a white
canvas stylized with the same style image at the same lambda. Same probe, layer and folds.

    python -m rbb.analysis.ceilings    -> results/ceilings.csv
"""

from rbb import config
from rbb.analysis.common import last_layer, read_probe_metrics, write_csv


def main() -> None:
    rows = []
    for model in config.MODELS:
        shape_ceiling = last_layer(read_probe_metrics("ceiling_shape", model)["shape_f1_macro_mean"])
        for scale in config.SCALES:
            f1 = read_probe_metrics(config.condition("main", scale), model)
            texture_ceiling = last_layer(
                read_probe_metrics(f"ceiling_texture_l{scale}", model)["texture_f1_macro_mean"]
            )
            for feature, ceiling in (("shape", shape_ceiling), ("texture", texture_ceiling)):
                value = last_layer(f1[f"{feature}_f1_macro_mean"])
                rows.append({
                    "model": model, "feature": feature, "scale": scale,
                    "f1": round(value, 6), "ceiling": round(ceiling, 6),
                    "fraction_of_ceiling": round(value / ceiling, 6),
                })
    write_csv(config.RESULTS_DIR / "ceilings.csv", rows)


if __name__ == "__main__":
    main()
