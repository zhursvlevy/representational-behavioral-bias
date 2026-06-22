import os

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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

MODEL_CONFIGS = {
    "llava": {
        "model_path": MODELS_ROOT_PATH / "llava-hf" / "llava-1.5-7b-hf",
        "input_dir": Path("data/stylized-imagenet/style-transfer-preprocessed-512"),
        "output_dir": Path("data/activation_steering/llava"),
        "linear_weights_path": Path("data/linear_classifiers/llava"),
        "prompt": COMMON_VQA_PROMPT,
        "model_class": LlavaForConditionalGeneration,
    },
    "paligemma2mix": {
        "model_path": MODELS_ROOT_PATH / "google" / "paligemma2-10b-mix-224",
        "input_dir": Path("data/stylized-imagenet/style-transfer-preprocessed-512"),
        "output_dir": Path("data/activation_steering/paligemma2mix"),
        "linear_weights_path": Path("data/linear_classifiers/paligemma2mix"),
        "prompt": f"<image>{COMMON_VQA_PROMPT}",
        "model_class": PaliGemmaForConditionalGeneration,
    },
    "qwen3vl": {
        "model_path": MODELS_ROOT_PATH / "Qwen" / "Qwen3-VL-8B-Instruct",
        "input_dir": Path("data/stylized-imagenet/style-transfer-preprocessed-512"),
        "output_dir": Path("data/activation_steering/qwen3vl"),
        "linear_weights_path": Path("data/linear_classifiers/qwen3vl"),
        "prompt": COMMON_VQA_PROMPT,
        "model_class": Qwen3VLForConditionalGeneration,
    },
}


def compute_bias_metrics(
    results: list[dict[str, str]],
    subtraction_target: str,
    epsilon: float = 1e-8,  # для численной стабильности
) -> dict[str, float]:
    """
    Вычисляет bias-метрики до и после активационного патчинга.

    subtraction_target:
    - "texture": вычитаем texture-вектор, ожидаем рост shape bias
    - "shape": вычитаем shape-вектор, ожидаем рост texture bias
    """

    def count_predictions(predictions: list[str], gt_shapes: list[str], gt_textures: list[str]):
        shape_correct = sum(1 for p, s in zip(predictions, gt_shapes) if p == s)
        texture_correct = sum(1 for p, t in zip(predictions, gt_textures) if p == t)
        return shape_correct, texture_correct

    gt_shapes = [r["gt_shape"] for r in results]
    gt_textures = [r["gt_texture"] for r in results]
    answers_orig = [r["answer"] for r in results]
    answers_patched = [r["answer_patched"] for r in results]

    shape_orig, tex_orig = count_predictions(answers_orig, gt_shapes, gt_textures)
    bias_shape_orig = shape_orig / (shape_orig + tex_orig + epsilon)
    bias_texture_orig = tex_orig / (shape_orig + tex_orig + epsilon)

    shape_patched, tex_patched = count_predictions(answers_patched, gt_shapes, gt_textures)
    bias_shape_patched = shape_patched / (shape_patched + tex_patched + epsilon)
    bias_texture_patched = tex_patched / (shape_patched + tex_patched + epsilon)

    shape_bias_gain = bias_shape_patched - bias_shape_orig
    texture_bias_gain = bias_texture_patched - bias_texture_orig
    expected_bias_gain = shape_bias_gain if subtraction_target == "texture" else texture_bias_gain

    flips = {
        "tex_to_shape": 0,
        "shape_to_tex": 0,
        "other": 0,
        "stable": 0,
    }

    for r in results:
        orig, patched = r["answer"], r["answer_patched"]
        gt_s, gt_t = r["gt_shape"], r["gt_texture"]

        if orig == patched:
            flips["stable"] += 1
        elif orig == gt_t and patched == gt_s:
            flips["tex_to_shape"] += 1
        elif orig == gt_s and patched == gt_t:
            flips["shape_to_tex"] += 1
        else:
            flips["other"] += 1

    n_total = len(results)
    expected_flip_rate_pct = (
        round(100 * flips["tex_to_shape"] / max(tex_orig, 1), 1)
        if subtraction_target == "texture"
        else round(100 * flips["shape_to_tex"] / max(shape_orig, 1), 1)
    )

    return {
        "subtraction_target": subtraction_target,
        "expected_bias_direction": "shape" if subtraction_target == "texture" else "texture",
        "shape_bias_orig": round(bias_shape_orig, 3),
        "shape_bias_patched": round(bias_shape_patched, 3),
        "texture_bias_orig": round(bias_texture_orig, 3),
        "texture_bias_patched": round(bias_texture_patched, 3),
        "shape_bias_gain": round(shape_bias_gain, 3),
        "texture_bias_gain": round(texture_bias_gain, 3),
        "expected_bias_gain": round(expected_bias_gain, 3),
        "shape_correct_orig": shape_orig,
        "texture_correct_orig": tex_orig,
        "shape_correct_patched": shape_patched,
        "texture_correct_patched": tex_patched,
        "flip_tex_to_shape": flips["tex_to_shape"],
        "flip_shape_to_tex": flips["shape_to_tex"],
        "flip_rate_tex_to_shape_pct": round(100 * flips["tex_to_shape"] / max(tex_orig, 1), 1),
        "flip_rate_shape_to_tex_pct": round(100 * flips["shape_to_tex"] / max(shape_orig, 1), 1),
        "expected_flip_rate_pct": expected_flip_rate_pct,
        "stable_predictions_pct": round(100 * flips["stable"] / max(n_total, 1), 1),
        "n_samples": n_total,
    }


def build_alpha_values(alpha_start: float, alpha_end: float, alpha_step: float) -> list[float]:
    if alpha_step <= 0:
        raise ValueError("alpha_step must be positive.")
    if alpha_end < alpha_start:
        raise ValueError("alpha_end must be greater than or equal to alpha_start.")

    alphas = np.arange(alpha_start, alpha_end + alpha_step / 2, alpha_step, dtype=float)
    return [round(float(alpha), 10) for alpha in alphas]


def alpha_to_key(alpha: float) -> str:
    return f"{alpha:.10g}"


def build_results_for_alpha(
    answers: list[dict[str, object]],
    steering_target: str,
    alpha_key: str,
) -> list[dict[str, str]]:
    return [
        {
            "gt_shape": answer["gt_shape"],
            "gt_texture": answer["gt_texture"],
            "answer": answer["answer"],
            "answer_patched": answer["patched_answers"][steering_target][alpha_key]["answer"],
        }
        for answer in answers
    ]


def extract_labels(filename: str) -> tuple[str, str]:
    """
    Извлекает shape_name и texture_name из имени файла.
    Формат: {shape_name}{числа}-{texture_name}{числа}
    """
    shape_part, texture_part = filename.split("-", 1)
    shape_name = "".join(filter(str.isalpha, shape_part))
    texture_name = "".join(filter(str.isalpha, texture_part))
    return shape_name, texture_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Activation steering experiment for multiple VLMs."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS.keys()),
        required=True,
        help="Model configuration to run.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit path to output JSON file.",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        required=True,
        help="Start of alpha range.",
    )
    parser.add_argument(
        "--alpha-end",
        type=float,
        required=True,
        help="End of alpha range.",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        required=True,
        help="Step for alpha range.",
    )
    return parser.parse_args()


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


def prepare_inputs(model_name: str, processor, model, img_path: Path, prompt: str, dtype: torch.dtype):
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


def plot_metrics_vs_alpha(metrics: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    alpha_keys = list(metrics["shape"].keys())
    alpha_values = [float(alpha) for alpha in alpha_keys]

    subtract_texture_expected_gain = [metrics["texture"][alpha]["expected_bias_gain"] for alpha in alpha_keys]
    subtract_shape_expected_gain = [metrics["shape"][alpha]["expected_bias_gain"] for alpha in alpha_keys]
    subtract_texture_flip = [metrics["texture"][alpha]["expected_flip_rate_pct"] for alpha in alpha_keys]
    subtract_shape_flip = [metrics["shape"][alpha]["expected_flip_rate_pct"] for alpha in alpha_keys]
    subtract_texture_stability = [metrics["texture"][alpha]["stable_predictions_pct"] for alpha in alpha_keys]
    subtract_shape_stability = [metrics["shape"][alpha]["stable_predictions_pct"] for alpha in alpha_keys]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

    axes[0, 0].plot(alpha_values, subtract_texture_expected_gain, marker="o")
    axes[0, 0].set_ylabel("shape_bias_gain")
    axes[0, 0].set_title("Subtract texture: shape bias gain")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(alpha_values, subtract_shape_expected_gain, marker="o")
    axes[0, 1].set_ylabel("texture_bias_gain")
    axes[0, 1].set_title("Subtract shape: texture bias gain")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(alpha_values, subtract_texture_flip, marker="o")
    axes[1, 0].set_ylabel("flip_rate_tex_to_shape_pct")
    axes[1, 0].set_title("Subtract texture: texture → shape flip rate")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(alpha_values, subtract_shape_flip, marker="o")
    axes[1, 1].set_ylabel("flip_rate_shape_to_tex_pct")
    axes[1, 1].set_title("Subtract shape: shape → texture flip rate")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(alpha_values, subtract_texture_stability, marker="o")
    axes[2, 0].set_xlabel("alpha")
    axes[2, 0].set_ylabel("stable_predictions_pct")
    axes[2, 0].set_title("Subtract texture: stability")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(alpha_values, subtract_shape_stability, marker="o")
    axes[2, 1].set_xlabel("alpha")
    axes[2, 1].set_ylabel("stable_predictions_pct")
    axes[2, 1].set_title("Subtract shape: stability")
    axes[2, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    model_name: str,
    output_json_path: Path,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
) -> None:
    config = MODEL_CONFIGS[model_name]
    device = get_device()
    dtype = get_dtype(device)
    alpha_values = build_alpha_values(alpha_start, alpha_end, alpha_step)

    model, processor = load_model_and_processor(model_name, device, dtype)

    shape_ws = np.load(
        config["linear_weights_path"] / "shape_ws.npy",
        allow_pickle=True,
    ).item()
    texture_ws = np.load(
        config["linear_weights_path"] / "texture_ws.npy",
        allow_pickle=True,
    ).item()

    answers = []
    image_paths = sorted(config["input_dir"].rglob("*.png"))

    for img_path in tqdm(image_paths, desc=f"Running {model_name}"):
        shape_name, texture_name = extract_labels(img_path.stem)

        with torch.no_grad():
            inputs = prepare_inputs(
                model_name=model_name,
                processor=processor,
                model=model,
                img_path=img_path,
                prompt=config["prompt"],
                dtype=dtype,
            )
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = model.lm_head(last_hidden)
        ans_token_id = torch.argmax(logits, dim=-1).item()

        w_shape = torch.tensor(shape_ws[shape_name], device=model.device, dtype=last_hidden.dtype)
        w_shape = w_shape / (w_shape.norm() + 1e-8)
        projection_shape = last_hidden @ w_shape

        w_texture = torch.tensor(texture_ws[texture_name], device=model.device, dtype=last_hidden.dtype)
        w_texture = w_texture / (w_texture.norm() + 1e-8)
        projection_texture = last_hidden @ w_texture

        patched_answers = {
            "shape": {},
            "texture": {},
        }

        for alpha in alpha_values:
            alpha_key = alpha_to_key(alpha)

            h_patched_shape = last_hidden - alpha * projection_shape.unsqueeze(-1) * w_shape
            patched_logits_shape = model.lm_head(h_patched_shape)
            ans_patched_shape_token_id = torch.argmax(patched_logits_shape, dim=-1).item()
            patched_answers["shape"][alpha_key] = {
                "answer": decode_single_token(processor, ans_patched_shape_token_id),
                "projection": projection_shape.item(),
            }

            h_patched_texture = last_hidden - alpha * projection_texture.unsqueeze(-1) * w_texture
            patched_logits_texture = model.lm_head(h_patched_texture)
            ans_patched_texture_token_id = torch.argmax(patched_logits_texture, dim=-1).item()
            patched_answers["texture"][alpha_key] = {
                "answer": decode_single_token(processor, ans_patched_texture_token_id),
                "projection": projection_texture.item(),
            }

        answers.append(
            {
                "file_name": img_path.name,
                "file_path": img_path.as_posix(),
                "gt_texture": NAME_LETTER_MAPPER[texture_name],
                "gt_shape": NAME_LETTER_MAPPER[shape_name],
                "answer": decode_single_token(processor, ans_token_id),
                "patched_answers": patched_answers,
            }
        )

    metrics = {
        "shape": {},
        "texture": {},
    }
    for steering_target in ("shape", "texture"):
        for alpha in alpha_values:
            alpha_key = alpha_to_key(alpha)
            metrics[steering_target][alpha_key] = compute_bias_metrics(
                build_results_for_alpha(answers, steering_target, alpha_key),
                subtraction_target=steering_target,
            )

    final_results = {
        "model": model_name,
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "alpha_step": alpha_step,
        "alphas": [alpha_to_key(alpha) for alpha in alpha_values],
        "metrics": metrics,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    plot_output_path = output_json_path.with_name(f"{output_json_path.stem}_metrics.png")
    plot_metrics_vs_alpha(metrics, plot_output_path)


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    output_json_path = args.output_json or (config["output_dir"] / "answers.json")
    run_experiment(
        model_name=args.model,
        output_json_path=output_json_path,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_step=args.alpha_step,
    )


if __name__ == "__main__":
    main()