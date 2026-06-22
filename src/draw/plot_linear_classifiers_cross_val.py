import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_STYLES = {
    "paligemma2mix": {"color": "red", "label": "paligemma"},
    "qwen3vl": {"color": "blue", "label": "qwen"},
    "llava": {"color": "green", "label": "llava"},
}

METRIC_STYLES = {
    "texture": {"linestyle": "-", "label_suffix": "texture"},
    "shape": {"linestyle": "--", "label_suffix": "shape"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read cross-validation F1 metrics from model subdirectories and plot "
            "shape/texture curves with std error bars on a single figure."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/linear_classifiers_cross_val"),
        help="Directory containing model subdirectories with f1_metrics.json files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/linear_classifiers_cross_val/f1_metrics_combined.png"),
        help="Path to save the resulting combined plot.",
    )
    return parser.parse_args()


def load_metrics(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_series(metric_dict: dict[str, float]) -> tuple[list[int], list[float]]:
    layer_indices = sorted(int(layer_idx) for layer_idx in metric_dict.keys())
    values = [metric_dict[str(layer_idx)] for layer_idx in layer_indices]
    return layer_indices, values


def plot_metric(
    layer_indices: list[int],
    mean_values: list[float],
    std_values: list[float],
    color: str,
    linestyle: str,
    label: str,
) -> None:
    plt.errorbar(
        layer_indices,
        mean_values,
        yerr=std_values,
        color=color,
        linestyle=linestyle,
        marker="o",
        markersize=4,
        linewidth=2,
        elinewidth=1.2,
        capsize=3,
        label=label,
    )


def main() -> None:
    args = parse_args()

    plt.figure(figsize=(10, 7))
    max_layer_idx = None

    for model_dir_name, model_style in MODEL_STYLES.items():
        json_path = args.input_dir / model_dir_name / "f1_metrics.json"
        if not json_path.exists():
            print(f"Skipping missing file: {json_path}")
            continue

        metrics = load_metrics(json_path)

        texture_layers, texture_mean = metric_series(metrics["texture_f1_macro_mean"])
        _, texture_std = metric_series(metrics["texture_f1_macro_std"])
        shape_layers, shape_mean = metric_series(metrics["shape_f1_macro_mean"])
        _, shape_std = metric_series(metrics["shape_f1_macro_std"])

        current_max_layer = max(texture_layers[-1], shape_layers[-1])
        if max_layer_idx is None or current_max_layer > max_layer_idx:
            max_layer_idx = current_max_layer

        plot_metric(
            layer_indices=texture_layers,
            mean_values=texture_mean,
            std_values=texture_std,
            color=model_style["color"],
            linestyle=METRIC_STYLES["texture"]["linestyle"],
            label=f"{model_style['label']} {METRIC_STYLES['texture']['label_suffix']}",
        )
        plot_metric(
            layer_indices=shape_layers,
            mean_values=shape_mean,
            std_values=shape_std,
            color=model_style["color"],
            linestyle=METRIC_STYLES["shape"]["linestyle"],
            label=f"{model_style['label']} {METRIC_STYLES['shape']['label_suffix']}",
        )

    plt.xlabel("Индекс слоя", fontsize=18)
    plt.ylabel("Macro F1-Score", fontsize=18)
    if max_layer_idx is not None:
        plt.xticks(range(0, max_layer_idx + 1, 5), fontsize=18)
    else:
        plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title("Cross-validated linear probe F1 across layers")
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved combined plot to: {args.output_path}")


if __name__ == "__main__":
    main()