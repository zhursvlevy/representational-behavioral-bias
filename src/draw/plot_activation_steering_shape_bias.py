import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODEL_COLORS = {
    "llava": "green",
    "paligemma2mix": "blue",
    "qwen3vl": "red",
}

MODEL_X_OFFSETS = {
    "llava": -0.08,
    "paligemma2mix": 0.0,
    "qwen3vl": 0.08,
}

SUBTRACTION_X_OFFSETS = {
    "texture": -0.02,
    "shape": 0.02,
}

MODEL_MARKERS = {
    "llava": "o",
    "paligemma2mix": "o",
    "qwen3vl": "o",
}

SUBTRACTION_LABELS = {
    "texture": "texture suppression",
    "shape": "shape suppression",
}

SUBTRACTION_LINESTYLES = {
    "texture": "-",
    "shape": "--",
}

PROMPT_TYPE_LABELS = {
    "default": "neutral prompts",
    "shape_biased": "shape-biased prompts",
    "texture_biased": "texture-biased prompts",
}

PROMPT_TYPE_ORDER = ("default", "shape_biased", "texture_biased")


def save_language_hidden_state_shape_bias_markdown(
    results: dict[str, dict[str, dict[str, float | int | str]]],
    output_path: Path,
) -> None:
    lines = [
        "| Model | Prompt type | Shape bias | Texture bias | Shape correct | Texture correct | N samples |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name in ("llava", "paligemma2mix", "qwen3vl"):
        prompt_results = results.get(model_name, {})
        for prompt_type in PROMPT_TYPE_ORDER:
            metrics = prompt_results.get(prompt_type)
            if metrics is None:
                continue

            lines.append(
                "| "
                f"{model_name} | "
                f"{PROMPT_TYPE_LABELS[prompt_type]} | "
                f"{metrics.get('shape_bias', 'N/A')} | "
                f"{metrics.get('texture_bias', 'N/A')} | "
                f"{metrics.get('shape_correct', 'N/A')} | "
                f"{metrics.get('texture_correct', 'N/A')} | "
                f"{metrics.get('n_samples', 'N/A')} |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_activation_steering_results(root_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    results = {}

    for json_path in root_dir.rglob("*.json"):
        model_name = json_path.parent.name
        if model_name not in MODEL_COLORS:
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            continue

        results[model_name] = metrics

    return results


def extract_metric_series(
    metrics_by_subtraction: dict[str, dict[str, dict[str, float]]],
    metric_name: str,
    alpha_min: float = 0.0,
    alpha_max: float = 10.0,
) -> dict[str, tuple[list[float], list[float]]]:
    series = {}

    for subtraction_target in ("texture", "shape"):
        alpha_to_metrics = metrics_by_subtraction.get(subtraction_target, {})
        filtered_items = []

        for alpha_key, metric_values in alpha_to_metrics.items():
            alpha = float(alpha_key)
            if alpha_min <= alpha <= alpha_max and metric_name in metric_values:
                filtered_items.append((alpha, metric_values[metric_name]))

        filtered_items.sort(key=lambda x: x[0])
        series[subtraction_target] = (
            [item[0] for item in filtered_items],
            [item[1] for item in filtered_items],
        )

    return series


def build_series_label(metric_name: str, model_name: str, subtraction_target: str) -> str:
    if metric_name == "shape_bias_patched":
        if subtraction_target == "texture":
            return f"{model_name}: texture suppression"
        return f"{model_name}: shape suppression"

    return f"{model_name}: {SUBTRACTION_LABELS[subtraction_target]}"


def plot_shape_bias_patched(
    root_dir: Path,
    output_path: Path,
    alpha_min: float = 0.0,
    alpha_max: float = 10.0,
) -> None:
    all_results = load_activation_steering_results(root_dir)

    metric_name = "shape_bias_patched"
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in ("llava", "paligemma2mix", "qwen3vl"):
        if model_name not in all_results:
            continue

        color = MODEL_COLORS[model_name]
        series_by_subtraction = extract_metric_series(
            all_results[model_name],
            metric_name=metric_name,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        for subtraction_target in ("texture", "shape"):
            x_values, y_values = series_by_subtraction[subtraction_target]
            if not x_values:
                continue

            x_values_shifted = [
                x + MODEL_X_OFFSETS[model_name] + SUBTRACTION_X_OFFSETS[subtraction_target]
                for x in x_values
            ]

            ax.plot(
                x_values_shifted,
                y_values,
                color=color,
                linestyle=SUBTRACTION_LINESTYLES[subtraction_target],
                linewidth=2,
                marker=MODEL_MARKERS[model_name],
                markersize=5,
                alpha=0.9,
                label=build_series_label(metric_name, model_name, subtraction_target),
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model_name],
            linestyle=SUBTRACTION_LINESTYLES[subtraction_target],
            linewidth=2,
            label=build_series_label(metric_name, model_name, subtraction_target),
        )
        for model_name in ("llava", "paligemma2mix", "qwen3vl")
        if model_name in all_results
        for subtraction_target in ("texture", "shape")
    ]

    ax.set_xlim(alpha_min - 0.15, alpha_max + 0.15)
    ax.set_xlabel(r"$\alpha$", fontsize=16)
    ax.set_ylabel("Shape bias", fontsize=16)
    ax.set_title("Shape bias after subtraction")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_language_hidden_state_shape_bias_results(
    root_dir: Path,
) -> dict[str, dict[str, dict[str, float | int | str]]]:
    results: dict[str, dict[str, dict[str, float | int | str]]] = {
        model_name: {} for model_name in MODEL_COLORS
    }

    for json_path in root_dir.rglob("shape_bias.json"):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model")
        prompt_type = data.get("prompt_type")

        if model_name not in MODEL_COLORS or prompt_type not in PROMPT_TYPE_LABELS:
            continue

        results[model_name][prompt_type] = data

    return {model_name: prompt_map for model_name, prompt_map in results.items() if prompt_map}


def plot_language_hidden_state_shape_bias(
    root_dir: Path,
    output_path: Path,
    markdown_output_path: Path | None = None,
) -> None:
    results = load_language_hidden_state_shape_bias_results(root_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = list(range(len(PROMPT_TYPE_ORDER)))

    for model_name in ("llava", "paligemma2mix", "qwen3vl"):
        prompt_results = results.get(model_name, {})
        if not prompt_results:
            continue

        xs = []
        ys = []

        for idx, prompt_type in enumerate(PROMPT_TYPE_ORDER):
            metrics = prompt_results.get(prompt_type)
            if metrics is None or "shape_bias" not in metrics:
                continue

            xs.append(idx + MODEL_X_OFFSETS[model_name])
            ys.append(metrics["shape_bias"])

        if not xs:
            continue

        ax.plot(
            xs,
            ys,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            markersize=7,
            label=model_name,
        )

        for idx, prompt_type in enumerate(PROMPT_TYPE_ORDER):
            metrics = prompt_results.get(prompt_type)
            if metrics is None or "shape_bias" not in metrics:
                continue

            x = idx + MODEL_X_OFFSETS[model_name]
            y = metrics["shape_bias"]
            label = f"{model_name}\n{PROMPT_TYPE_LABELS[prompt_type]}: {y:.3f}"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=MODEL_COLORS[model_name],
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            label=model_name,
        )
        for model_name in ("llava", "paligemma2mix", "qwen3vl")
        if model_name in results
    ]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([PROMPT_TYPE_LABELS[prompt_type] for prompt_type in PROMPT_TYPE_ORDER])
    ax.set_ylabel("Shape bias", fontsize=16)
    ax.set_xlabel("Prompt type", fontsize=16)
    ax.set_title("Shape bias by prompt type for language hidden states")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if markdown_output_path is not None:
        save_language_hidden_state_shape_bias_markdown(results, markdown_output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot shape bias metrics for activation steering or language hidden states."
    )
    parser.add_argument(
        "--mode",
        choices=("activation_steering", "language_hidden_states"),
        default="activation_steering",
        help="Which experiment results to plot.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/activation_steering"),
        help="Root directory with JSON results.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/activation_steering/shape_bias_patched_all_models.png"),
        help="Path to save the combined plot.",
    )
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=None,
        help="Optional path to save a Markdown table with language hidden state shape bias results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "activation_steering":
        plot_shape_bias_patched(
            root_dir=args.input_dir,
            output_path=args.output_path,
            alpha_min=0.0,
            alpha_max=10.0,
        )
        return

    markdown_output_path = args.markdown_output_path
    if markdown_output_path is None:
        markdown_output_path = args.output_path.with_suffix(".md")

    plot_language_hidden_state_shape_bias(
        root_dir=args.input_dir,
        output_path=args.output_path,
        markdown_output_path=markdown_output_path,
    )


if __name__ == "__main__":
    main()