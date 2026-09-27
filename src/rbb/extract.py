"""Record decoder hidden states and the model's answer for every image of a condition.

For each image the hidden state at the final sequence position of every decoder layer
(embeddings + all layers, before the LM head) is saved as <stem>.npy with shape
(num_layers, hidden_dim). The answer is the argmax of the LM head applied to the last
layer's state; answers.csv holds it per image, shape_bias.json the aggregate.

    python -m rbb.extract --model llava --condition main_l6000
"""

import argparse
import csv
import json

import numpy as np
import torch
from tqdm import tqdm

from rbb import config, vlm
from rbb.filenames import extract_labels

ANSWER_FIELDS = ["file_name", "shape_class", "texture_class", "answer", "category"]


def shape_bias(rows: list[dict]) -> dict:
    """Shape Bias of Geirhos et al.: shape answers among answers matching either cue."""
    n_shape = sum(r["answer"] == config.LETTERS[r["shape_class"]] for r in rows)
    n_texture = sum(r["answer"] == config.LETTERS[r["texture_class"]] for r in rows)
    eps = 1e-8
    return {
        "shape_bias": round(n_shape / (n_shape + n_texture + eps), 6),
        "texture_bias": round(n_texture / (n_shape + n_texture + eps), 6),
        "shape_correct": n_shape,
        "texture_correct": n_texture,
        "n_samples": len(rows),
    }


def extract(model_name: str, condition: str, overwrite: bool = False) -> None:
    input_dir = config.condition_images(condition)
    output_dir = config.hidden_dir(condition, model_name)
    if (output_dir / "answers.csv").exists() and not overwrite:
        print(f"{model_name}/{condition}: already extracted, skipping")
        return

    device = vlm.get_device()
    dtype = vlm.get_input_dtype(device)
    model, processor = vlm.load_model_and_processor(model_name, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in tqdm(sorted(input_dir.rglob("*.png")), desc=f"{model_name}/{condition}"):
        shape_name, texture_name = extract_labels(img_path.stem)
        if shape_name is None or texture_name is None:
            continue

        with torch.no_grad():
            inputs = vlm.prepare_inputs(model_name, processor, model, img_path, dtype)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden = np.stack([s[0, -1, :].cpu().float().numpy() for s in outputs.hidden_states], axis=0)
            np.save(output_dir / f"{img_path.stem}.npy", hidden)
            answer = vlm.decode_answer(processor, model.lm_head(outputs.hidden_states[-1][:, -1, :]))

        rows.append({
            "file_name": img_path.stem,
            "shape_class": shape_name,
            "texture_class": texture_name,
            "answer": answer,
            "category": vlm.classify_answer(answer, shape_name, texture_name),
        })

    with (output_dir / "shape_bias.json").open("w") as f:
        json.dump({"model": model_name, "condition": condition, **shape_bias(rows)}, f, indent=2)
    with (output_dir / "answers.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANSWER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=config.MODELS, required=True)
    parser.add_argument("--condition", required=True, help="see rbb.config.condition_images")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract(args.model, args.condition, args.overwrite)


if __name__ == "__main__":
    main()
