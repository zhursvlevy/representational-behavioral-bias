"""Model loading, the multiple-choice prompt, and input preparation for the three VLMs.

Weights are loaded from $MODELS_ROOT_PATH/<hub id> if that directory exists, otherwise
from the Hugging Face Hub (PaliGemma2 is gated: set HF_TOKEN).
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)
from transformers.image_utils import load_image

from rbb import config

PROMPT = (
    "Which option best describes the image?\n"
    + "".join(f"{letter}. {name}\n" for name, letter in config.LETTERS.items())
    + "Answer with the option's letter from the given choices directly."
)

MODEL_CONFIGS = {
    "llava": {"hub_id": "llava-hf/llava-1.5-7b-hf", "model_class": LlavaForConditionalGeneration},
    "paligemma2mix": {"hub_id": "google/paligemma2-10b-mix-224", "model_class": PaliGemmaForConditionalGeneration},
    "qwen3vl": {"hub_id": "Qwen/Qwen3-VL-8B-Instruct", "model_class": Qwen3VLForConditionalGeneration},
}


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_input_dtype(device: torch.device) -> torch.dtype:
    """dtype the processor outputs are cast to (weights are loaded in bfloat16 on GPU)."""
    return torch.float16 if device.type == "cuda" else torch.float32


def model_path(model_name: str) -> Path:
    return Path(config.MODELS_ROOT_PATH, MODEL_CONFIGS[model_name]["hub_id"])


def load_processor(model_name: str):
    path = model_path(model_name)
    if model_name == "qwen3vl":
        return AutoProcessor.from_pretrained(path, min_pixels=224 * 224, max_pixels=224 * 224)
    return AutoProcessor.from_pretrained(path, token=config.HF_TOKEN)


def load_model_and_processor(model_name: str, device: torch.device):
    """bfloat16 weights, accelerate device_map with an 11 GiB GPU budget (fits a 16 GB card)."""
    model_class = MODEL_CONFIGS[model_name]["model_class"]
    kwargs = {"token": config.HF_TOKEN}
    if device.type == "cuda":
        kwargs.update(low_cpu_mem_usage=True, dtype=torch.bfloat16, device_map="auto",
                      max_memory={0: "11GiB", "cpu": "24GiB"})
    else:
        kwargs["torch_dtype"] = torch.float32

    model = model_class.from_pretrained(model_path(model_name), **kwargs)
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model, load_processor(model_name)


def prepare_inputs(model_name: str, processor, model, img_path: Path, dtype: torch.dtype):
    if model_name == "llava":
        conversation = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image"}]}]
        rendered = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.open(img_path).convert("RGB")
        return processor(images=image, text=rendered, return_tensors="pt").to(model.device, dtype)

    if model_name == "paligemma2mix":
        image = load_image(img_path.as_posix())
        return processor(text=f"<image>{PROMPT}", images=image, return_tensors="pt").to(model.device, dtype)

    if model_name == "qwen3vl":
        conversation = [{"role": "user", "content": [
            {"type": "image", "url": img_path.as_posix()},
            {"type": "text", "text": PROMPT},
        ]}]
        return processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device, dtype)

    raise ValueError(f"Unsupported model: {model_name}")


def decode_answer(processor, logits: torch.Tensor) -> str:
    """The answer is the single most likely next token, decoded and stripped."""
    token_id = torch.argmax(logits, dim=-1).item()
    return processor.decode([token_id], skip_special_tokens=True).strip()


def classify_answer(answer: str, shape_class: str, texture_class: str) -> str:
    if answer == config.LETTERS[shape_class]:
        return "shape_match"
    if answer == config.LETTERS[texture_class]:
        return "texture_match"
    if answer in config.LETTER_TO_CLASS:
        return "other_valid"
    return "unparseable"
