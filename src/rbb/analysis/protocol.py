"""Random vs. grouped train/validation split on the cue-conflict set of Geirhos et al.
("The measurement protocol"). That set has 10 shape and 3 texture exemplars per class,
so a random split lets the texture probe see its validation exemplars during training.

Needs grouped probes for the 'geirhos' condition (rbb.probing grouped --folds geirhos).

    python -m rbb.analysis.protocol    -> results/protocol_split_comparison.csv
"""

from rbb import config
from rbb.analysis.common import last_layer, read_probe_metrics, write_csv, write_json
from rbb.probing import load_hidden_states, random_split


def main() -> None:
    rows, per_layer = [], {}
    for model in config.MODELS:
        data = load_hidden_states(config.hidden_dir("geirhos", model))
        grouped = read_probe_metrics("geirhos", model)
        per_layer[model] = {}
        for cue, y in (("shape", data.y_shape), ("texture", data.y_texture)):
            random_f1 = random_split(data.X, y)
            per_layer[model][f"{cue}_random_split"] = random_f1
            per_layer[model][f"{cue}_grouped"] = grouped[f"{cue}_f1_macro_mean"]
            rows.append({"model": model, "cue": cue,
                         "f1_last_layer_random_split": last_layer(random_f1),
                         "f1_last_layer_grouped": last_layer(grouped[f"{cue}_f1_macro_mean"])})
            print(rows[-1])
    write_csv(config.RESULTS_DIR / "protocol_split_comparison.csv", rows)
    write_json(config.RESULTS_DIR / "protocol_split_by_layer.json", per_layer)


if __name__ == "__main__":
    main()
