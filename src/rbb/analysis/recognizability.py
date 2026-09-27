"""How often LLaVA names the class of a style source shown on its own, for full photographs
and for the crops used by the control set (runs the model; needs a GPU).

    python -m rbb.analysis.recognizability    -> results/style_recognizability.csv
"""

import csv

import torch
from tqdm import tqdm

from rbb import config, vlm
from rbb.analysis.common import write_csv
from rbb.filenames import extract_labels
from rbb.stimuli.cue_conflict import PAIRS_REFERENCE

MODEL = "llava"


def answer(model, processor, img_path, dtype) -> str:
    inputs = vlm.prepare_inputs(MODEL, processor, model, img_path, dtype)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, return_dict=True)
    return vlm.decode_answer(processor, outputs.logits[:, -1, :])


def accuracy(model, processor, dtype, items, name: str) -> dict:
    correct = sum(answer(model, processor, p, dtype) == config.LETTERS[cls] for p, cls in tqdm(items, desc=name))
    return {"source_type": name, "n_samples": len(items), "n_correct": correct,
            "accuracy": round(correct / len(items), 6)}


def main() -> None:
    with PAIRS_REFERENCE.open() as f:
        texture_class = {r["texture_src"]: r["texture_class"] for r in csv.DictReader(f)}

    photos = [(p, extract_labels(p.stem)[1]) for p in sorted(config.ceiling_stimuli_dir("texture_photos").glob("*.png"))]
    crops = [(p, texture_class[p.stem]) for p in sorted(config.style_crops_dir().glob("*.png"))]

    device = vlm.get_device()
    model, processor = vlm.load_model_and_processor(MODEL, device)
    dtype = vlm.get_input_dtype(device)
    write_csv(config.RESULTS_DIR / "style_recognizability.csv", [
        accuracy(model, processor, dtype, photos, "whole_photo"),
        accuracy(model, processor, dtype, crops, "crop"),
    ])


if __name__ == "__main__":
    main()
