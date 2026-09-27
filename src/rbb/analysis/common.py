"""Readers for per-condition outputs and small writers shared by the analyses."""

import csv
import json
from pathlib import Path

from rbb import config

CATEGORIES = ("shape_match", "texture_match", "other_valid", "unparseable")


def main_conditions() -> list[tuple[str, int]]:
    """(model, lambda) for every main-set condition: 5 lambdas x 3 models + 2 dense for LLaVA."""
    return [(model, scale) for model in config.MODELS for scale in config.main_scales(model)]


def read_answers(condition: str, model: str) -> list[dict]:
    with (config.hidden_dir(condition, model) / "answers.csv").open() as f:
        return list(csv.DictReader(f))


def read_shape_bias(condition: str, model: str) -> float:
    return json.loads((config.hidden_dir(condition, model) / "shape_bias.json").read_text())["shape_bias"]


def read_probe_metrics(condition: str, model: str) -> dict:
    return json.loads((config.probe_dir(condition, model) / "f1_metrics.json").read_text())


def last_layer(metric: dict[str, float]) -> float:
    return metric[str(max(int(k) for k in metric))]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"saved {path}")
