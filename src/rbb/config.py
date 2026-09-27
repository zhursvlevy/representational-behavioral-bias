"""Paths and experiment constants shared by every pipeline stage.

Inputs are read from environment variables (see configs/paths.env.example);
every generated artifact lives under OUTPUT_DIR.
"""

import os
from pathlib import Path

RESOURCES = Path(__file__).parent / "resources"

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
MODELS_ROOT_PATH = os.environ.get("MODELS_ROOT_PATH", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or None


def input_dir(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        raise SystemExit(f"Set {env_name} (see configs/paths.env.example).")
    path = Path(value)
    if not path.is_dir():
        raise SystemExit(f"{env_name}={path} is not a directory.")
    return path


CLASSES = (
    "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
    "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck",
)
LETTERS = {name: chr(ord("A") + i) for i, name in enumerate(CLASSES)}
LETTER_TO_CLASS = {letter: name for name, letter in LETTERS.items()}

MODELS = ("llava", "paligemma2mix", "qwen3vl")
MODEL_LABELS = {"llava": "LLaVA", "paligemma2mix": "PaliGemma2", "qwen3vl": "Qwen3-VL"}

SEED = 0
IMG_SIZE = 256

# Style weight lambda of the Gatys generator. The dense points are run for LLaVA only.
SCALES = (1000, 3000, 6000, 9000, 12000)
DENSE_SCALES = (1500, 2000)
CROP_SCALES = (1000, 6000, 12000)
CROP_N_STIMULI = 300

N_SPLITS = 3
SPLIT_SEED = 42  # StratifiedGroupKFold folds and the random train/validation split
PROBE_C = 1.0
PROBE_MAX_ITER = 1000


def main_scales(model: str) -> tuple[int, ...]:
    if model == "llava":
        return tuple(sorted(SCALES + DENSE_SCALES))
    return SCALES


# ---- artifact locations -------------------------------------------------------

def imagenet16_dir() -> Path:
    return OUTPUT_DIR / "imagenet16"


def stimuli_dir(stimulus_set: str, scale: int) -> Path:
    """stimulus_set: 'main' (full-photo style source) or 'crop' (crop control)."""
    return OUTPUT_DIR / "stimuli" / f"{stimulus_set}_l{scale}"


def style_crops_dir() -> Path:
    return OUTPUT_DIR / "stimuli" / "style_crops"


def ceiling_stimuli_dir(kind: str, scale: int | None = None) -> Path:
    """kind: 'shape_photos', 'texture_photos' or 'white_canvas' (needs scale)."""
    name = f"{kind}_l{scale}" if scale is not None else kind
    return OUTPUT_DIR / "ceiling_stimuli" / name


def hidden_dir(condition: str, model: str) -> Path:
    return OUTPUT_DIR / "hidden_states" / condition / model


def probe_dir(condition: str, model: str) -> Path:
    return OUTPUT_DIR / "probes" / condition / model


def folds_dir() -> Path:
    return OUTPUT_DIR / "probes" / "folds"


RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"


def condition(stimulus_set: str, scale: int) -> str:
    return f"{stimulus_set}_l{scale}"


def condition_images(name: str) -> Path:
    """Image directory of an extraction condition.

    main_l<λ>             generated cue-conflict stimuli
    crop_l<λ>             crop-control stimuli
    ceiling_shape         unstylized shape sources
    ceiling_texture_l<λ>  white canvases stylized at λ
    geirhos               cue-conflict set of Geirhos et al. ($GEIRHOS_STIMULI_DIR)
    """
    if name == "geirhos":
        return input_dir("GEIRHOS_STIMULI_DIR")
    if name == "ceiling_shape":
        return ceiling_stimuli_dir("shape_photos")
    prefix, _, scale = name.rpartition("_l")
    if prefix in ("main", "crop"):
        return stimuli_dir(prefix, int(scale))
    if prefix == "ceiling_texture":
        return ceiling_stimuli_dir("white_canvas", int(scale))
    raise ValueError(f"Unknown condition: {name}")
