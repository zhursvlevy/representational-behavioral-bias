import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def extract_labels(filename: str) -> tuple[str | None, str | None]:
    """
    Извлекает shape_name и texture_name из имени файла.
    Формат: {shape_name}{числа}-{texture_name}{числа}.npy
    """
    stem = Path(filename).stem
    if "-" not in stem:
        return None, None

    shape_part, texture_part = stem.split("-", 1)
    shape_name = "".join(filter(str.isalpha, shape_part))
    texture_name = "".join(filter(str.isalpha, texture_part))

    if not shape_name or not texture_name:
        return None, None

    return shape_name, texture_name


def prepare_dataset(data_dir: Path):
    """
    Загружает данные и формирует датасет.

    Возвращает:
        X_full (np.ndarray): форма (n_samples, num_layers, hidden_dim)
        y_shape_list (list[str]): метки shape_name
        y_texture_list (list[str]): метки texture_name
    """
    X_list = []
    y_shape_list = []
    y_texture_list = []

    for npy_path in tqdm(sorted(data_dir.glob("*.npy")), desc="Loading hidden states"):
        shape_name, texture_name = extract_labels(npy_path.name)
        if shape_name is None or texture_name is None:
            continue

        if shape_name == texture_name:
            continue

        arr = np.load(npy_path)  # (num_layers, hidden_dim)
        X_list.append(arr)
        y_shape_list.append(shape_name)
        y_texture_list.append(texture_name)

    if not X_list:
        raise ValueError("No valid data found in the directory.")

    X_full = np.stack(X_list, axis=0)
    return X_full, y_shape_list, y_texture_list


def get_layer_probes(
    X_full,
    y_shape_list,
    y_texture_list,
    layer_idx,
    shape_train_idx,
    shape_test_idx,
    texture_train_idx,
    texture_test_idx,
    random_state=42,
):
    """
    Обучает классификаторы для конкретного слоя и возвращает веса и F1 на тестовом сплите.

    Возвращает:
        shape_probes (dict): { 'cat': np.array([hidden_dim]), 'dog': ... }
        texture_probes (dict): { 'clock': np.array([hidden_dim]), ... }
        shape_f1 (float): macro F1 для shape-классификатора на тесте
        texture_f1 (float): macro F1 для texture-классификатора на тесте
    """
    _, num_layers, _ = X_full.shape
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx must be in [0, {num_layers - 1}]")

    X_layer = X_full[:, layer_idx, :]

    def train_and_extract(labels, train_idx, test_idx):
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            raise ValueError("Less than 2 classes found for classification.")

        X_train = X_layer[train_idx]
        X_test = X_layer[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        clf = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            multi_class="ovr",
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")

        weights_dict = {}
        for idx, cls_name in enumerate(clf.classes_):
            weights_dict[cls_name] = clf.coef_[idx]

        return weights_dict, f1

    print(f"Training probes for layer {layer_idx}...")
    shape_weights, shape_f1 = train_and_extract(
        y_shape_list, shape_train_idx, shape_test_idx
    )
    texture_weights, texture_f1 = train_and_extract(
        y_texture_list, texture_train_idx, texture_test_idx
    )

    return shape_weights, texture_weights, shape_f1, texture_f1


def train_all_layers(X_full, y_shape_list, y_texture_list, random_state=42):
    """
    Обучает probes для всех слоев и возвращает:
        shape_ws_by_layer: {layer_idx: {class_name: weight_vector}}
        texture_ws_by_layer: {layer_idx: {class_name: weight_vector}}
        metrics_by_layer: {
            "shape_f1_macro": {layer_idx: score},
            "texture_f1_macro": {layer_idx: score},
        }

    Для каждой задачи (shape/texture) используется однократное
    стратифицированное разбиение train/test, общее для всех слоёв.
    """
    n_samples, num_layers, _ = X_full.shape

    shape_y = np.array(y_shape_list)
    texture_y = np.array(y_texture_list)

    if len(np.unique(shape_y)) < 2:
        raise ValueError("Less than 2 shape classes found for classification.")
    if len(np.unique(texture_y)) < 2:
        raise ValueError("Less than 2 texture classes found for classification.")

    X_reshaped = X_full.reshape(n_samples, -1)

    _, _, _, _, shape_train_idx, shape_test_idx = train_test_split(
        X_reshaped,
        shape_y,
        np.arange(n_samples),
        test_size=0.2,
        stratify=shape_y,
        random_state=random_state,
    )
    _, _, _, _, texture_train_idx, texture_test_idx = train_test_split(
        X_reshaped,
        texture_y,
        np.arange(n_samples),
        test_size=0.2,
        stratify=texture_y,
        random_state=random_state,
    )

    shape_ws_by_layer = {}
    texture_ws_by_layer = {}
    metrics_by_layer = {
        "shape_f1_macro": {},
        "texture_f1_macro": {},
    }

    for layer_idx in tqdm(range(num_layers), desc="Training linear probes"):
        shape_weights, texture_weights, shape_f1, texture_f1 = get_layer_probes(
            X_full,
            y_shape_list,
            y_texture_list,
            layer_idx=layer_idx,
            shape_train_idx=shape_train_idx,
            shape_test_idx=shape_test_idx,
            texture_train_idx=texture_train_idx,
            texture_test_idx=texture_test_idx,
            random_state=random_state,
        )
        shape_ws_by_layer[layer_idx] = shape_weights
        texture_ws_by_layer[layer_idx] = texture_weights
        metrics_by_layer["shape_f1_macro"][layer_idx] = shape_f1
        metrics_by_layer["texture_f1_macro"][layer_idx] = texture_f1

    return shape_ws_by_layer, texture_ws_by_layer, metrics_by_layer


def train_all_layers_multi_seed(X_full, y_shape_list, y_texture_list, random_states):
    """
    Запускает обучение для нескольких сидов и агрегирует метрики по слоям.

    Возвращает:
        runs_by_seed: {
            seed: {
                "shape_ws_by_layer": ...,
                "texture_ws_by_layer": ...,
                "metrics_by_layer": ...
            }
        }
        aggregated_metrics_by_layer: {
            "shape_f1_macro_mean": {layer_idx: score},
            "shape_f1_macro_std": {layer_idx: score},
            "texture_f1_macro_mean": {layer_idx: score},
            "texture_f1_macro_std": {layer_idx: score},
        }
    """
    if not random_states:
        raise ValueError("At least one random state must be provided.")

    runs_by_seed = {}

    for seed in random_states:
        print(f"Running linear probing with random_state={seed}")
        shape_ws_by_layer, texture_ws_by_layer, metrics_by_layer = train_all_layers(
            X_full,
            y_shape_list,
            y_texture_list,
            random_state=seed,
        )
        runs_by_seed[seed] = {
            "shape_ws_by_layer": shape_ws_by_layer,
            "texture_ws_by_layer": texture_ws_by_layer,
            "metrics_by_layer": metrics_by_layer,
        }

    layer_indices = sorted(
        next(iter(runs_by_seed.values()))["metrics_by_layer"]["shape_f1_macro"].keys()
    )

    aggregated_metrics_by_layer = {
        "shape_f1_macro_mean": {},
        "shape_f1_macro_std": {},
        "texture_f1_macro_mean": {},
        "texture_f1_macro_std": {},
    }

    for layer_idx in layer_indices:
        shape_scores = [
            runs_by_seed[seed]["metrics_by_layer"]["shape_f1_macro"][layer_idx]
            for seed in random_states
        ]
        texture_scores = [
            runs_by_seed[seed]["metrics_by_layer"]["texture_f1_macro"][layer_idx]
            for seed in random_states
        ]

        aggregated_metrics_by_layer["shape_f1_macro_mean"][layer_idx] = float(
            np.mean(shape_scores)
        )
        aggregated_metrics_by_layer["shape_f1_macro_std"][layer_idx] = float(
            np.std(shape_scores)
        )
        aggregated_metrics_by_layer["texture_f1_macro_mean"][layer_idx] = float(
            np.mean(texture_scores)
        )
        aggregated_metrics_by_layer["texture_f1_macro_std"][layer_idx] = float(
            np.std(texture_scores)
        )

    return runs_by_seed, aggregated_metrics_by_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear probes on hidden states and save class weight vectors."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with hidden state .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where probe weights will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Single random state for backward-compatible runs.",
    )
    parser.add_argument(
        "--random-states",
        type=int,
        nargs="+",
        help="One or more random states. If provided, overrides --random-state.",
    )
    return parser.parse_args()


def save_metrics_json(output_path: Path, metrics_json: dict) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)


def build_metrics_json_from_single_run(metrics_by_layer: dict) -> dict:
    return {
        "shape_f1_macro": {
            str(layer_idx): round(score, 6)
            for layer_idx, score in metrics_by_layer["shape_f1_macro"].items()
        },
        "texture_f1_macro": {
            str(layer_idx): round(score, 6)
            for layer_idx, score in metrics_by_layer["texture_f1_macro"].items()
        },
    }


def build_metrics_json_from_multi_seed(aggregated_metrics_by_layer: dict) -> dict:
    return {
        "shape_f1_macro_mean": {
            str(layer_idx): round(score, 6)
            for layer_idx, score in aggregated_metrics_by_layer[
                "shape_f1_macro_mean"
            ].items()
        },
        "shape_f1_macro_std": {
            str(layer_idx): round(score, 6)
            for layer_idx, score in aggregated_metrics_by_layer[
                "shape_f1_macro_std"
            ].items()
        },
        "texture_f1_macro_mean": {
            str(layer_idx): round(score, 6)
            for layer_idx, score in aggregated_metrics_by_layer[
                "texture_f1_macro_mean"
            ].items()
        },
        "texture_f1_macro_std": {
            str(layer_idx): round(score, 6)
            for layer_idx, score in aggregated_metrics_by_layer[
                "texture_f1_macro_std"
            ].items()
        },
    }


def save_f1_plot(output_path: Path, metrics_json: dict) -> None:
    if "shape_f1_macro" in metrics_json:
        layer_indices = sorted(int(idx) for idx in metrics_json["shape_f1_macro"].keys())
        shape_values = [metrics_json["shape_f1_macro"][str(idx)] for idx in layer_indices]
        texture_values = [
            metrics_json["texture_f1_macro"][str(idx)] for idx in layer_indices
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(layer_indices, shape_values, marker="o", label="shape F1")
        plt.plot(layer_indices, texture_values, marker="o", label="texture F1")
    else:
        layer_indices = sorted(
            int(idx) for idx in metrics_json["shape_f1_macro_mean"].keys()
        )
        shape_mean = [
            metrics_json["shape_f1_macro_mean"][str(idx)] for idx in layer_indices
        ]
        shape_std = [
            metrics_json["shape_f1_macro_std"][str(idx)] for idx in layer_indices
        ]
        texture_mean = [
            metrics_json["texture_f1_macro_mean"][str(idx)] for idx in layer_indices
        ]
        texture_std = [
            metrics_json["texture_f1_macro_std"][str(idx)] for idx in layer_indices
        ]

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            layer_indices,
            shape_mean,
            yerr=shape_std,
            marker="o",
            capsize=4,
            label="shape F1 mean±std",
        )
        plt.errorbar(
            layer_indices,
            texture_mean,
            yerr=texture_std,
            marker="o",
            capsize=4,
            label="texture F1 mean±std",
        )

    plt.xlabel("Layer")
    plt.ylabel("Macro F1")
    plt.title("Linear probe F1 across layers")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    random_states = args.random_states if args.random_states is not None else [args.random_state]

    X_full, y_shape_list, y_texture_list = prepare_dataset(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(random_states) == 1:
        random_state = random_states[0]
        shape_ws_by_layer, texture_ws_by_layer, metrics_by_layer = train_all_layers(
            X_full,
            y_shape_list,
            y_texture_list,
            random_state=random_state,
        )

        np.save(args.output_dir / "shape_ws.npy", shape_ws_by_layer, allow_pickle=True)
        np.save(args.output_dir / "texture_ws.npy", texture_ws_by_layer, allow_pickle=True)

        metrics_json = build_metrics_json_from_single_run(metrics_by_layer)
        save_metrics_json(args.output_dir / "f1_metrics.json", metrics_json)
        save_f1_plot(args.output_dir / "f1_metrics.png", metrics_json)

        print(f"Saved shape probes to: {args.output_dir / 'shape_ws.npy'}")
        print(f"Saved texture probes to: {args.output_dir / 'texture_ws.npy'}")
    else:
        runs_by_seed, aggregated_metrics_by_layer = train_all_layers_multi_seed(
            X_full,
            y_shape_list,
            y_texture_list,
            random_states=random_states,
        )

        shape_ws_by_seed = {
            seed: run_data["shape_ws_by_layer"] for seed, run_data in runs_by_seed.items()
        }
        texture_ws_by_seed = {
            seed: run_data["texture_ws_by_layer"]
            for seed, run_data in runs_by_seed.items()
        }

        np.save(args.output_dir / "shape_ws_by_seed.npy", shape_ws_by_seed, allow_pickle=True)
        np.save(
            args.output_dir / "texture_ws_by_seed.npy",
            texture_ws_by_seed,
            allow_pickle=True,
        )

        metrics_json = build_metrics_json_from_multi_seed(aggregated_metrics_by_layer)
        metrics_json["random_states"] = random_states
        save_metrics_json(args.output_dir / "f1_metrics.json", metrics_json)
        save_f1_plot(args.output_dir / "f1_metrics.png", metrics_json)

        print(f"Saved shape probes by seed to: {args.output_dir / 'shape_ws_by_seed.npy'}")
        print(
            f"Saved texture probes by seed to: "
            f"{args.output_dir / 'texture_ws_by_seed.npy'}"
        )

    print(f"Saved F1 metrics JSON to: {args.output_dir / 'f1_metrics.json'}")
    print(f"Saved F1 plot to: {args.output_dir / 'f1_metrics.png'}")


if __name__ == "__main__":
    main()