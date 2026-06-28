import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageFilter


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


TRANSFORM_DESCRIPTIONS = {
    "original": "No suppression; image is resized if --resize-size is set.",
    "gaussian_blur": "Texture suppression via Gaussian blur, following Burgert-style smoothing controls.",
    "bilateral": "Texture suppression via OpenCV bilateral filtering. Requires cv2 and numpy.",
    "patch_shuffle": "Shape suppression by shuffling image patches on a grid.",
    "patch_rotation": "Shape suppression by rotating patches independently on a grid.",
    "grayscale": "Color suppression by blending image with its grayscale version.",
    "channel_shuffle": "Color correlation suppression by permuting RGB channels.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Burgert-style feature-suppressed image variants for "
            "ImageNet-16/SIN diagnostics."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory with images. Images are discovered recursively.",
    )
    input_group.add_argument(
        "--manifest",
        type=Path,
        help=(
            "CSV manifest with image paths. The script uses output_path if it "
            "exists, otherwise source_path, otherwise file_path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for transformed images and metadata.",
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=sorted(TRANSFORM_DESCRIPTIONS),
        default=("original", "gaussian_blur", "patch_shuffle", "patch_rotation", "grayscale"),
        help="Which suppressed variants to generate.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=224,
        help="Resize images to a square size before suppression. Use 0 to preserve input size.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=6,
        help="Patch grid size for patch_shuffle and patch_rotation.",
    )
    parser.add_argument(
        "--gaussian-radius",
        type=float,
        default=2.0,
        help="Pillow GaussianBlur radius. Burgert reports Gaussian sigma=2.0 as a strong texture control.",
    )
    parser.add_argument(
        "--bilateral-d",
        type=int,
        default=12,
        help="OpenCV bilateral filter pixel-neighborhood diameter.",
    )
    parser.add_argument(
        "--bilateral-sigma-color",
        type=int,
        default=170,
        help="OpenCV bilateral filter sigmaColor.",
    )
    parser.add_argument(
        "--bilateral-sigma-space",
        type=int,
        default=75,
        help="OpenCV bilateral filter sigmaSpace.",
    )
    parser.add_argument(
        "--gray-alpha",
        type=float,
        default=1.0,
        help="Grayscale strength: 0 keeps original color, 1 is fully grayscale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for deterministic patch shuffle/rotation and channel shuffle.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How to materialize the original variant.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def stable_seed(base_seed: int, image_path: Path, transform_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{image_path.as_posix()}:{transform_name}".encode()).hexdigest()
    return int(digest[:16], 16)


def iter_input_records_from_dir(input_dir: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append(
                {
                    "source_path": path.as_posix(),
                    "relative_path": path.relative_to(input_dir).as_posix(),
                }
            )
    return records


def select_manifest_path(row: dict[str, str]) -> str:
    for key in ("output_path", "source_path", "file_path", "path"):
        value = row.get(key, "").strip()
        if value and Path(value).exists():
            return value

    for key in ("output_path", "source_path", "file_path", "path"):
        value = row.get(key, "").strip()
        if value:
            return value

    raise ValueError(f"Manifest row does not contain an image path: {row}")


def iter_input_records_from_manifest(manifest_path: Path) -> list[dict[str, str]]:
    records = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path(select_manifest_path(row))
            entry_class = row.get("entry_class", "").strip()
            if entry_class:
                relative_path = f"{entry_class}/{image_path.name}"
            else:
                relative_path = image_path.name

            record = dict(row)
            record["source_path"] = image_path.as_posix()
            record["relative_path"] = relative_path
            records.append(record)

    return records


def load_rgb_image(path: Path, resize_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if resize_size > 0:
        image = image.resize((resize_size, resize_size), resample=Image.Resampling.BICUBIC)
    return image


def save_image(image: Image.Image, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def materialize_original(source_path: Path, output_path: Path, image: Image.Image, copy_mode: str, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.size != Image.open(source_path).size:
        image.save(output_path)
        return

    if copy_mode == "copy":
        shutil.copy2(source_path, output_path)
        return

    if copy_mode == "symlink":
        output_path.symlink_to(source_path.resolve())
        return

    if copy_mode == "hardlink":
        output_path.hardlink_to(source_path)
        return

    raise ValueError(f"Unsupported copy mode: {copy_mode}")


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_grayscale(image: Image.Image, alpha: float) -> Image.Image:
    if not 0 <= alpha <= 1:
        raise ValueError("--gray-alpha must be in [0, 1].")
    gray = image.convert("L").convert("RGB")
    return Image.blend(image, gray, alpha)


def apply_channel_shuffle(image: Image.Image, rng: random.Random) -> Image.Image:
    channels = list(image.split())
    rng.shuffle(channels)
    return Image.merge("RGB", channels)


def grid_bounds(length: int, grid_size: int) -> list[tuple[int, int]]:
    bounds = []
    for idx in range(grid_size):
        start = round(idx * length / grid_size)
        end = round((idx + 1) * length / grid_size)
        bounds.append((start, end))
    return bounds


def apply_patch_shuffle(image: Image.Image, grid_size: int, rng: random.Random) -> Image.Image:
    width, height = image.size
    x_bounds = grid_bounds(width, grid_size)
    y_bounds = grid_bounds(height, grid_size)

    patches = []
    for y0, y1 in y_bounds:
        for x0, x1 in x_bounds:
            patches.append(image.crop((x0, y0, x1, y1)))

    shuffled = list(patches)
    rng.shuffle(shuffled)

    output = Image.new("RGB", image.size)
    patch_idx = 0
    for y0, y1 in y_bounds:
        for x0, x1 in x_bounds:
            patch = shuffled[patch_idx]
            patch = patch.resize((x1 - x0, y1 - y0), resample=Image.Resampling.BICUBIC)
            output.paste(patch, (x0, y0))
            patch_idx += 1

    return output


def resize_to_square_divisible(image: Image.Image, grid_size: int) -> tuple[Image.Image, tuple[int, int]]:
    original_size = image.size
    width, height = image.size
    if width == height and width % grid_size == 0:
        return image, original_size

    side = max(width, height)
    side = ((side + grid_size - 1) // grid_size) * grid_size
    return image.resize((side, side), resample=Image.Resampling.BICUBIC), original_size


def apply_patch_rotation(image: Image.Image, grid_size: int, rng: random.Random) -> Image.Image:
    working, original_size = resize_to_square_divisible(image, grid_size)
    width, height = working.size
    patch_width = width // grid_size
    patch_height = height // grid_size

    output = Image.new("RGB", working.size)
    for row in range(grid_size):
        for col in range(grid_size):
            x0 = col * patch_width
            y0 = row * patch_height
            patch = working.crop((x0, y0, x0 + patch_width, y0 + patch_height))
            patch = patch.rotate(rng.choice((0, 90, 180, 270)), expand=False)
            output.paste(patch, (x0, y0))

    if output.size != original_size:
        output = output.resize(original_size, resample=Image.Resampling.BICUBIC)
    return output


def apply_bilateral_filter(
    image: Image.Image,
    d: int,
    sigma_color: int,
    sigma_space: int,
) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "The bilateral transform requires opencv-python and numpy. Install them "
            "in the environment used to run this script, or use gaussian_blur for "
            "a dependency-light texture suppression control."
        ) from exc

    image_np = np.array(image)
    filtered = cv2.bilateralFilter(
        image_np,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
    return Image.fromarray(filtered)


def transform_image(image: Image.Image, source_path: Path, transform_name: str, args: argparse.Namespace) -> Image.Image:
    rng = random.Random(stable_seed(args.seed, source_path, transform_name))

    if transform_name == "original":
        return image
    if transform_name == "gaussian_blur":
        return apply_gaussian_blur(image, args.gaussian_radius)
    if transform_name == "bilateral":
        return apply_bilateral_filter(
            image,
            d=args.bilateral_d,
            sigma_color=args.bilateral_sigma_color,
            sigma_space=args.bilateral_sigma_space,
        )
    if transform_name == "patch_shuffle":
        return apply_patch_shuffle(image, args.grid_size, rng)
    if transform_name == "patch_rotation":
        return apply_patch_rotation(image, args.grid_size, rng)
    if transform_name == "grayscale":
        return apply_grayscale(image, args.gray_alpha)
    if transform_name == "channel_shuffle":
        return apply_channel_shuffle(image, rng)

    raise ValueError(f"Unsupported transform: {transform_name}")


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "sample_index",
        "transform",
        "source_path",
        "output_path",
        "relative_path",
        "entry_class",
        "synset",
        "source_filename",
    ]
    extra_keys = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extra_keys)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive.")
    if args.resize_size < 0:
        raise ValueError("--resize-size must be non-negative.")

    records = (
        iter_input_records_from_manifest(args.manifest)
        if args.manifest is not None
        else iter_input_records_from_dir(args.input_dir)
    )
    if not records:
        raise ValueError("No input images found.")

    manifest_rows = []
    for sample_index, record in enumerate(records):
        source_path = Path(record["source_path"])
        if not source_path.exists():
            raise FileNotFoundError(f"Missing input image: {source_path}")

        image = load_rgb_image(source_path, args.resize_size)
        relative_path = Path(record["relative_path"])

        for transform_name in args.transforms:
            output_path = args.output_dir / transform_name / relative_path
            if transform_name == "original":
                materialize_original(source_path, output_path, image, args.copy_mode, args.overwrite)
            else:
                transformed = transform_image(image, source_path, transform_name, args)
                save_image(transformed, output_path, args.overwrite)

            row = dict(record)
            row.update(
                {
                    "sample_index": str(sample_index),
                    "transform": transform_name,
                    "source_path": source_path.as_posix(),
                    "output_path": output_path.as_posix(),
                    "relative_path": relative_path.as_posix(),
                }
            )
            manifest_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_rows, args.output_dir / "manifest.csv")

    metadata = {
        "n_input_images": len(records),
        "n_output_images": len(manifest_rows),
        "transforms": list(args.transforms),
        "transform_descriptions": TRANSFORM_DESCRIPTIONS,
        "parameters": {
            "resize_size": args.resize_size,
            "grid_size": args.grid_size,
            "gaussian_radius": args.gaussian_radius,
            "bilateral_d": args.bilateral_d,
            "bilateral_sigma_color": args.bilateral_sigma_color,
            "bilateral_sigma_space": args.bilateral_sigma_space,
            "gray_alpha": args.gray_alpha,
            "seed": args.seed,
            "copy_mode": args.copy_mode,
        },
        "methodology_note": (
            "Transform choices follow Burgert et al.'s controlled feature "
            "suppression protocol: patch shuffle/rotation for shape, "
            "bilateral/Gaussian smoothing for texture, and grayscale/channel "
            "shuffle for color. This script materializes image variants for "
            "VLM diagnostics rather than evaluating CNN accuracy."
        ),
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(records)} input images")
    print(f"Saved {len(manifest_rows)} transformed image records to {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
