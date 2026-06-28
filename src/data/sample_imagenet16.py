import argparse
import csv
import json
import random
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path


GEIRHOS_IMAGE_NAMES_BASE_URL = (
    "https://raw.githubusercontent.com/rgeirhos/generalisation-humans-DNNs/"
    "master/16-class-ImageNet/image_names"
)

IMAGENET16_CLASSES = (
    "airplane",
    "bear",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "car",
    "cat",
    "chair",
    "clock",
    "dog",
    "elephant",
    "keyboard",
    "knife",
    "oven",
    "truck",
)

IMAGE_EXTENSIONS = (".JPEG", ".JPG", ".jpg", ".jpeg", ".png", ".PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a natural ImageNet-16 subset using the Geirhos et al. "
            "entry-level class filename lists."
        )
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        required=True,
        help=(
            "Path to ImageNet train directory with synset subdirectories "
            "(for example ILSVRC2012_img_train or .../train)."
        ),
    )
    parser.add_argument(
        "-n",
        "--samples-per-class",
        type=int,
        default=80,
        help="Number of images to sample per ImageNet-16 entry-level class.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where sampled images and metadata will be written.",
    )
    parser.add_argument(
        "--image-names-dir",
        type=Path,
        default=None,
        help=(
            "Optional local directory containing Geirhos image_names/*.txt files. "
            "If omitted, the files are downloaded into --cache-dir."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/metadata/geirhos_imagenet16_image_names"),
        help="Where to cache downloaded Geirhos image name lists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink", "hardlink", "manifest-only"),
        default="copy",
        help=(
            "How to materialize selected images. Use manifest-only to only write "
            "metadata without copying/linking image files."
        ),
    )
    parser.add_argument(
        "--balance-synsets",
        action="store_true",
        help=(
            "Sample as evenly as possible across ImageNet synsets within each "
            "entry-level class."
        ),
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help=(
            "Allow classes with fewer than N available images. By default this "
            "is an error."
        ),
    )
    return parser.parse_args()


def download_image_name_lists(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    for class_name in IMAGENET16_CLASSES:
        output_path = cache_dir / f"{class_name}.txt"
        if output_path.exists() and output_path.stat().st_size > 0:
            continue

        url = f"{GEIRHOS_IMAGE_NAMES_BASE_URL}/{class_name}.txt"
        print(f"Downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                output_path.write_bytes(response.read())
        except OSError as exc:
            raise RuntimeError(
                "Could not download Geirhos image name lists. Either enable "
                "network access or pass --image-names-dir pointing to a local "
                "copy of the 16 .txt files."
            ) from exc

    return cache_dir


def get_image_names_dir(args: argparse.Namespace) -> Path:
    image_names_dir = args.image_names_dir or download_image_name_lists(args.cache_dir)

    missing = [
        class_name
        for class_name in IMAGENET16_CLASSES
        if not (image_names_dir / f"{class_name}.txt").exists()
    ]
    if missing:
        missing_files = ", ".join(f"{class_name}.txt" for class_name in missing)
        raise FileNotFoundError(f"Missing Geirhos image name files: {missing_files}")

    return image_names_dir


def read_image_names(image_names_dir: Path, class_name: str) -> list[str]:
    path = image_names_dir / f"{class_name}.txt"
    names = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            image_name = line.strip()
            if not image_name:
                continue
            names.append(image_name)

    return names


def resolve_imagenet_train_root(imagenet_root: Path) -> Path:
    if any(imagenet_root.glob("n[0-9]*")):
        return imagenet_root

    train_dir = imagenet_root / "train"
    if train_dir.exists() and any(train_dir.glob("n[0-9]*")):
        return train_dir

    raise FileNotFoundError(
        "Could not find ImageNet synset directories under --imagenet-root or "
        "under --imagenet-root/train."
    )


def find_image_path(train_root: Path, image_name: str) -> Path | None:
    synset = image_name.split("_", 1)[0]
    candidates = [train_root / synset / image_name]

    stem = Path(image_name).stem
    for extension in IMAGE_EXTENSIONS:
        candidates.append(train_root / synset / f"{stem}{extension}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def build_available_records(
    train_root: Path,
    image_names_dir: Path,
) -> dict[str, list[dict[str, str]]]:
    records_by_class = {}

    for class_name in IMAGENET16_CLASSES:
        records = []
        missing = 0

        for image_name in read_image_names(image_names_dir, class_name):
            image_path = find_image_path(train_root, image_name)
            if image_path is None:
                missing += 1
                continue

            synset = image_name.split("_", 1)[0]
            records.append(
                {
                    "entry_class": class_name,
                    "synset": synset,
                    "source_filename": image_name,
                    "source_path": image_path.as_posix(),
                }
            )

        records_by_class[class_name] = records
        print(
            f"{class_name}: found {len(records)} available images "
            f"({missing} listed images missing locally)"
        )

    return records_by_class


def sample_flat(
    records: list[dict[str, str]],
    n_samples: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    records = list(records)
    rng.shuffle(records)
    return records[:n_samples]


def sample_balanced_synsets(
    records: list[dict[str, str]],
    n_samples: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    by_synset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_synset[record["synset"]].append(record)

    for synset_records in by_synset.values():
        rng.shuffle(synset_records)

    selected = []
    synsets = sorted(by_synset)
    while len(selected) < n_samples and synsets:
        rng.shuffle(synsets)
        next_synsets = []

        for synset in synsets:
            if len(selected) >= n_samples:
                break
            synset_records = by_synset[synset]
            if not synset_records:
                continue
            selected.append(synset_records.pop())
            if synset_records:
                next_synsets.append(synset)

        synsets = next_synsets

    return selected


def sample_records(
    records_by_class: dict[str, list[dict[str, str]]],
    n_samples: int,
    seed: int,
    balance_synsets: bool,
    allow_fewer: bool,
) -> list[dict[str, str]]:
    selected_records = []

    for class_index, class_name in enumerate(IMAGENET16_CLASSES):
        records = records_by_class[class_name]
        if len(records) < n_samples and not allow_fewer:
            raise ValueError(
                f"Class {class_name} has only {len(records)} available images, "
                f"but {n_samples} were requested. Use --allow-fewer to continue."
            )

        class_n = min(n_samples, len(records))
        rng = random.Random(seed + class_index)
        if balance_synsets:
            selected = sample_balanced_synsets(records, class_n, rng)
        else:
            selected = sample_flat(records, class_n, rng)

        selected_records.extend(selected)

    return selected_records


def materialize_image(
    source_path: Path,
    output_path: Path,
    copy_mode: str,
) -> None:
    if copy_mode == "manifest-only":
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
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


def write_outputs(
    selected_records: list[dict[str, str]],
    output_dir: Path,
    copy_mode: str,
    config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"

    manifest_rows = []
    counts_by_class: dict[str, int] = defaultdict(int)
    counts_by_synset: dict[str, int] = defaultdict(int)

    for sample_index, record in enumerate(selected_records):
        entry_class = record["entry_class"]
        source_path = Path(record["source_path"])
        output_path = images_dir / entry_class / source_path.name

        materialize_image(source_path, output_path, copy_mode)

        row = {
            "sample_index": sample_index,
            "entry_class": entry_class,
            "synset": record["synset"],
            "source_filename": record["source_filename"],
            "source_path": source_path.as_posix(),
            "output_path": output_path.as_posix() if copy_mode != "manifest-only" else "",
        }
        manifest_rows.append(row)

        counts_by_class[entry_class] += 1
        counts_by_synset[record["synset"]] += 1

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "sample_index",
                "entry_class",
                "synset",
                "source_filename",
                "source_path",
                "output_path",
            ),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "config": config,
        "n_total": len(selected_records),
        "counts_by_class": dict(sorted(counts_by_class.items())),
        "counts_by_synset": dict(sorted(counts_by_synset.items())),
        "source": {
            "geirhos_image_names_base_url": GEIRHOS_IMAGE_NAMES_BASE_URL,
            "class_files": [f"{class_name}.txt" for class_name in IMAGENET16_CLASSES],
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved manifest to {manifest_path}")
    print(f"Saved metadata to {output_dir / 'metadata.json'}")


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive.")

    train_root = resolve_imagenet_train_root(args.imagenet_root)
    image_names_dir = get_image_names_dir(args)
    records_by_class = build_available_records(train_root, image_names_dir)
    selected_records = sample_records(
        records_by_class=records_by_class,
        n_samples=args.samples_per_class,
        seed=args.seed,
        balance_synsets=args.balance_synsets,
        allow_fewer=args.allow_fewer,
    )

    config = {
        "imagenet_root": args.imagenet_root.as_posix(),
        "resolved_train_root": train_root.as_posix(),
        "image_names_dir": image_names_dir.as_posix(),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "balance_synsets": args.balance_synsets,
        "allow_fewer": args.allow_fewer,
    }
    write_outputs(
        selected_records=selected_records,
        output_dir=args.output_dir,
        copy_mode=args.copy_mode,
        config=config,
    )


if __name__ == "__main__":
    main()
