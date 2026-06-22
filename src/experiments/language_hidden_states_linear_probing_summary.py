import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL_ORDER = ("llava", "paligemma2mix", "qwen3vl")
PROMPT_DIR_TO_LABEL = {
    "neutral_prompts": "neutral",
    "shape_biased_prompts": "shape_biased",
    "texture_biased_prompts": "texture_biased",
}
PROMPT_ORDER = ("neutral", "shape_biased", "texture_biased")


def extract_labels(filename: str) -> tuple[str | None, str | None]:
    stem = Path(filename).stem
    if "-" not in stem:
        return None, None

    shape_part, texture_part = stem.split("-", 1)
    shape_name = "".join(filter(str.isalpha, shape_part))
    texture_name = "".join(filter(str.isalpha, texture_part))

    if not shape_name or not texture_name:
        return None, None

    return shape_name, texture_name


def prepare_dataset(data_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    x_list = []
    y_shape_list = []
    y_texture_list = []

    for npy_path in tqdm(sorted(data_dir.glob("*.npy")), desc=f"Loading {data_dir}"):
        shape_name, texture_name = extract_labels(npy_path.name)
        if shape_name is None or texture_name is None:
            continue

        if shape_name == texture_name:
            continue

        arr = np.load(npy_path)
        x_list.append(arr)
        y_shape_list.append(shape_name)
        y_texture_list.append(texture_name)

    if not x_list:
        raise ValueError(f"No valid data found in directory: {data_dir}")

    x_full = np.stack(x_list, axis=0)
    return x_full, y_shape_list, y_texture_list


def make_splits(
    x_full: np.ndarray,
    y_shape_list: list[str],
    y_texture_list: list[str],
    random_state: int,
):
    n_samples = x_full.shape[0]
    x_reshaped = x_full.reshape(n_samples, -1)
    shape_y = np.array(y_shape_list)
    texture_y = np.array(y_texture_list)

    _, _, _, _, shape_train_idx, shape_test_idx = train_test_split(
        x_reshaped,
        shape_y,
        np.arange(n_samples),
        test_size=0.2,
        stratify=shape_y,
        random_state=random_state,
    )
    _, _, _, _, texture_train_idx, texture_test_idx = train_test_split(
        x_reshaped,
        texture_y,
        np.arange(n_samples),
        test_size=0.2,
        stratify=texture_y,
        random_state=random_state,
    )

    return shape_train_idx, shape_test_idx, texture_train_idx, texture_test_idx


def train_probe_for_layer(
    x_full: np.ndarray,
    labels: list[str],
    layer_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
) -> float:
    x_layer = x_full[:, layer_idx, :]
    y = np.array(labels)

    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        multi_class="ovr",
        solver="lbfgs",
    )
    clf.fit(x_layer[train_idx], y[train_idx])
    y_pred = clf.predict(x_layer[test_idx])
    return float(f1_score(y[test_idx], y_pred, average="macro"))


def compute_last_layer_delta_f1(
    data_dir: Path,
    last_n_layers: int,
    random_state: int,
) -> dict[str, float | list[int] | list[float]]:
    x_full, y_shape_list, y_texture_list = prepare_dataset(data_dir)

    if len(np.unique(y_shape_list)) < 2:
        raise ValueError(f"Less than 2 shape classes found in {data_dir}")
    if len(np.unique(y_texture_list)) < 2:
        raise ValueError(f"Less than 2 texture classes found in {data_dir}")

    shape_train_idx, shape_test_idx, texture_train_idx, texture_test_idx = make_splits(
        x_full,
        y_shape_list,
        y_texture_list,
        random_state=random_state,
    )

    num_layers = x_full.shape[1]
    start_layer = max(0, num_layers - last_n_layers)
    layer_indices = list(range(start_layer, num_layers))

    shape_f1_scores = []
    texture_f1_scores = []

    for layer_idx in layer_indices:
        shape_f1 = train_probe_for_layer(
            x_full=x_full,
            labels=y_shape_list,
            layer_idx=layer_idx,
            train_idx=shape_train_idx,
            test_idx=shape_test_idx,
            random_state=random_state,
        )
        texture_f1 = train_probe_for_layer(
            x_full=x_full,
            labels=y_texture_list,
            layer_idx=layer_idx,
            train_idx=texture_train_idx,
            test_idx=texture_test_idx,
            random_state=random_state,
        )
        shape_f1_scores.append(shape_f1)
        texture_f1_scores.append(texture_f1)

    avg_shape_f1 = float(np.mean(shape_f1_scores))
    avg_texture_f1 = float(np.mean(texture_f1_scores))
    delta_f1 = avg_texture_f1 - avg_shape_f1

    return {
        "avg_shape_f1": round(avg_shape_f1, 6),
        "avg_texture_f1": round(avg_texture_f1, 6),
        "delta_f1_texture_minus_shape": round(delta_f1, 6),
        "last_layers_used": layer_indices,
        "shape_f1_by_layer": [round(x, 6) for x in shape_f1_scores],
        "texture_f1_by_layer": [round(x, 6) for x in texture_f1_scores],
        "n_samples": int(x_full.shape[0]),
    }


def discover_language_hidden_state_dirs(root_dir: Path) -> dict[str, dict[str, Path]]:
    discovered: dict[str, dict[str, Path]] = {model: {} for model in MODEL_ORDER}

    for model_name in MODEL_ORDER:
        model_dir = root_dir / model_name
        if not model_dir.exists():
            continue

        if any(model_dir.glob("*.npy")):
            discovered[model_name]["neutral"] = model_dir

        for subdir in sorted(model_dir.iterdir()):
            if not subdir.is_dir():
                continue
            prompt_label = PROMPT_DIR_TO_LABEL.get(subdir.name)
            if prompt_label is None:
                continue
            if any(subdir.glob("*.npy")):
                discovered[model_name][prompt_label] = subdir

    return discovered


def build_markdown_table(results: dict[str, dict[str, dict[str, float | int | list[int] | list[float]]]]) -> str:
    lines = [
        "| model | neutral | shape_biased | texture_biased |",
        "| --- | ---: | ---: | ---: |",
    ]

    for model_name in MODEL_ORDER:
        row = [model_name]
        prompt_results = results.get(model_name, {})

        for prompt_type in PROMPT_ORDER:
            metrics = prompt_results.get(prompt_type)
            if metrics is None:
                row.append("N/A")
                continue

            row.append(str(metrics["delta_f1_texture_minus_shape"]))

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan language hidden states, train linear probes on the last layers, "
            "and save a Markdown summary table with delta F1 (texture - shape)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/language_hidden_states"),
        help="Root directory with model and prompt subdirectories containing hidden state .npy files.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/language_hidden_states/linear_probe_delta_f1_summary.md"),
        help="Path to save the Markdown summary table.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/language_hidden_states/linear_probe_delta_f1_summary.json"),
        help="Path to save detailed JSON metrics.",
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        default=4,
        help="How many last layers to use when averaging F1 scores.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for train/test split and LogisticRegression.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    discovered_dirs = discover_language_hidden_state_dirs(args.input_dir)
    results: dict[str, dict[str, dict[str, float | int | list[int] | list[float]]]] = {
        model_name: {} for model_name in MODEL_ORDER
    }

    for model_name in MODEL_ORDER:
        for prompt_type in PROMPT_ORDER:
            data_dir = discovered_dirs.get(model_name, {}).get(prompt_type)
            if data_dir is None:
                continue

            print(f"Processing model={model_name}, prompt_type={prompt_type}, dir={data_dir}")
            results[model_name][prompt_type] = compute_last_layer_delta_f1(
                data_dir=data_dir,
                last_n_layers=args.last_n_layers,
                random_state=args.random_state,
            )

    markdown = build_markdown_table(results)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")

    detailed_json = {
        "input_dir": args.input_dir.as_posix(),
        "last_n_layers": args.last_n_layers,
        "random_state": args.random_state,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(detailed_json, f, ensure_ascii=False, indent=2)

    print(f"Saved Markdown summary to: {args.output_md}")
    print(f"Saved detailed JSON to: {args.output_json}")


if __name__ == "__main__":
    main()