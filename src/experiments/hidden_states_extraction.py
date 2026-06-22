import os

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)
from transformers.image_utils import load_image

MODELS_ROOT_PATH = os.getenv("MODELS_ROOT_PATH", "data/models")

COMMON_VQA_PROMPT = (
    "Which option best describes the image?\n"
    "A. airplane\n"
    "B. bear\n"
    "C. bicycle\n"
    "D. bird\n"
    "E. boat\n"
    "F. bottle\n"
    "G. car\n"
    "H. cat\n"
    "I. chair\n"
    "J. clock\n"
    "K. dog\n"
    "L. elephant\n"
    "M. keyboard\n"
    "N. knife\n"
    "O. oven\n"
    "P. truck\n"
    "Answer with the option's letter from the given choices directly."
)

SHAPE_BIASED_VQA_PROMPT = (
    "Focus on the OBJECT SHAPE and ignore surface patterns."
    "Which option best describes the image?\n"
    "A. airplane\n"
    "B. bear\n"
    "C. bicycle\n"
    "D. bird\n"
    "E. boat\n"
    "F. bottle\n"
    "G. car\n"
    "H. cat\n"
    "I. chair\n"
    "J. clock\n"
    "K. dog\n"
    "L. elephant\n"
    "M. keyboard\n"
    "N. knife\n"
    "O. oven\n"
    "P. truck\n"
    "Answer with the option's letter from the given choices directly."
)

TEXTURE_BIASED_VQA_PROMPT = (
    "Focus on the SURFACE TEXTURE and ignore the object outline."
    "Which option best describes the image?\n"
    "A. airplane\n"
    "B. bear\n"
    "C. bicycle\n"
    "D. bird\n"
    "E. boat\n"
    "F. bottle\n"
    "G. car\n"
    "H. cat\n"
    "I. chair\n"
    "J. clock\n"
    "K. dog\n"
    "L. elephant\n"
    "M. keyboard\n"
    "N. knife\n"
    "O. oven\n"
    "P. truck\n"
    "Answer with the option's letter from the given choices directly."
)

PROMPTS = {
    "default": COMMON_VQA_PROMPT,
    "shape_biased": SHAPE_BIASED_VQA_PROMPT,
    "texture_biased": TEXTURE_BIASED_VQA_PROMPT,
}

NAME_LETTER_MAPPER = {
    "airplane": "A",
    "bear": "B",
    "bicycle": "C",
    "bird": "D",
    "boat": "E",
    "bottle": "F",
    "car": "G",
    "cat": "H",
    "chair": "I",
    "clock": "J",
    "dog": "K",
    "elephant": "L",
    "keyboard": "M",
    "knife": "N",
    "oven": "O",
    "truck": "P",
}

MODEL_CONFIGS = {
    "llava": {
        "model_path": MODELS_ROOT_PATH / "llava-hf" / "llava-1.5-7b-hf",
        "input_dir": Path("data/stylized-imagenet/style-transfer-preprocessed-512"),
        "output_dir": Path("data/language_hidden_states/llava"),
        "prompt": COMMON_VQA_PROMPT,
        "model_class": LlavaForConditionalGeneration,
    },
    "paligemma2mix": {
        "model_path": MODELS_ROOT_PATH / "google" / "paligemma2-10b-mix-224",
        "input_dir": Path("data/stylized-imagenet/style-transfer-preprocessed-512"),
        "output_dir": Path("data/language_hidden_states/paligemma2mix"),
        "prompt": f"<image>{COMMON_VQA_PROMPT}",
        "model_class": PaliGemmaForConditionalGeneration,
    },
    "qwen3vl": {
        "model_path": MODELS_ROOT_PATH / "Qwen" / "Qwen3-VL-8B-Instruct",
        "input_dir": Path("data/stylized-imagenet/style-transfer-preprocessed-512"),
        "output_dir": Path("data/language_hidden_states/qwen3vl"),
        "prompt": COMMON_VQA_PROMPT,
        "model_class": Qwen3VLForConditionalGeneration,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract last-token hidden states from all decoder layers for multiple VLMs."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS.keys()),
        required=True,
        help="Model configuration to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for .npy files.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=sorted(PROMPTS.keys()),
        default="default",
        help="Which prompt to use for hidden state extraction.",
    )
    return parser.parse_args()


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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def load_model_and_processor(model_name: str, device: torch.device, dtype: torch.dtype):
    config = MODEL_CONFIGS[model_name]
    model_path = config["model_path"]
    model_class = config["model_class"]

    processor = AutoProcessor.from_pretrained(model_path)

    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["low_cpu_mem_usage"] = True
    else:
        model_kwargs["torch_dtype"] = dtype

    model = model_class.from_pretrained(model_path, **model_kwargs)

    if device.type == "cpu":
        model = model.to(device)

    model.eval()
    return model, processor


def prepare_inputs(
    model_name: str,
    processor,
    model,
    img_path: Path,
    prompt: str,
    dtype: torch.dtype,
):
    if model_name == "llava":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        rendered_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        image = Image.open(img_path).convert("RGB")
        return processor(
            images=image,
            text=rendered_prompt,
            return_tensors="pt",
        ).to(model.device, dtype)

    if model_name == "paligemma2mix":
        image = load_image(img_path.as_posix())
        return processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(model.device, dtype)

    if model_name == "qwen3vl":
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": img_path.as_posix(),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        return processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype)

    raise ValueError(f"Unsupported model: {model_name}")


def decode_single_token(processor, token_id: int) -> str:
    return processor.decode([token_id], skip_special_tokens=True).strip()


def compute_shape_bias(results: list[dict[str, str]], epsilon: float = 1e-8) -> dict[str, float]:
    gt_shapes = [r["gt_shape"] for r in results]
    gt_textures = [r["gt_texture"] for r in results]
    answers = [r["answer"] for r in results]

    shape_correct = sum(1 for pred, gt in zip(answers, gt_shapes) if pred == gt)
    texture_correct = sum(1 for pred, gt in zip(answers, gt_textures) if pred == gt)

    shape_bias = shape_correct / (shape_correct + texture_correct + epsilon)
    texture_bias = texture_correct / (shape_correct + texture_correct + epsilon)

    return {
        "shape_bias": round(shape_bias, 6),
        "texture_bias": round(texture_bias, 6),
        "shape_correct": shape_correct,
        "texture_correct": texture_correct,
        "n_samples": len(results),
    }


def extract_hidden_states(
    model_name: str,
    output_dir: Path,
    prompt_type: str,
) -> None:
    config = MODEL_CONFIGS[model_name]
    device = get_device()
    dtype = get_dtype(device)

    model, processor = load_model_and_processor(model_name, device, dtype)

    prompt = PROMPTS[prompt_type]
    if model_name == "paligemma2mix":
        prompt = f"<image>{prompt}"

    image_paths = sorted(config["input_dir"].rglob("*.png"))
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = []

    for img_path in tqdm(image_paths, desc=f"Extracting hidden states for {model_name}"):
        shape_name, texture_name = extract_labels(img_path.stem)
        if shape_name is None or texture_name is None:
            continue

        with torch.no_grad():
            inputs = prepare_inputs(
                model_name=model_name,
                processor=processor,
                model=model,
                img_path=img_path,
                prompt=prompt,
                dtype=dtype,
            )
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        layer_vectors = []
        for layer_state in outputs.hidden_states:
            h_last = layer_state[0, -1, :].cpu().float().numpy()
            layer_vectors.append(h_last)

        hidden_states = np.stack(layer_vectors, axis=0)
        np.save(output_dir / f"{img_path.stem}.npy", hidden_states)

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = model.lm_head(last_hidden)
        ans_token_id = torch.argmax(logits, dim=-1).item()

        answers.append(
            {
                "file_name": img_path.name,
                "file_path": img_path.as_posix(),
                "gt_shape": NAME_LETTER_MAPPER[shape_name],
                "gt_texture": NAME_LETTER_MAPPER[texture_name],
                "answer": decode_single_token(processor, ans_token_id),
            }
        )

    shape_bias_metrics = {
        "model": model_name,
        "prompt_type": prompt_type,
        **compute_shape_bias(answers),
    }
    with (output_dir / "shape_bias.json").open("w", encoding="utf-8") as f:
        json.dump(shape_bias_metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    output_dir = args.output_dir or config["output_dir"]
    extract_hidden_states(
        model_name=args.model,
        output_dir=output_dir,
        prompt_type=args.prompt_type,
    )


if __name__ == "__main__":
    main()