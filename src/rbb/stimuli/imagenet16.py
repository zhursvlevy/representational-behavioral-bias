"""Build the 16-class ImageNet validation subset the stimuli are drawn from.

The subset is fixed by resources/imagenet16_manifest.csv (1450 validation images).
It was originally sampled with `sample_manifest` below (seed 0, at most 100 images per
class). That sampling iterates directories in filesystem order, so re-running it on
another machine can select different images; the shipped manifest makes the subset,
and hence every stimulus, reproducible.

The WNID -> class table (resources/imagenet16_wnids.json) is the 16-class mapping of
Geirhos et al. (2019), https://github.com/rgeirhos/texture-vs-shape.

    python -m rbb.stimuli.imagenet16 --imagenet-val $IMAGENET_VAL_DIR
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from rbb import config

MANIFEST = config.RESOURCES / "imagenet16_manifest.csv"
WNIDS = config.RESOURCES / "imagenet16_wnids.json"
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def sample_manifest(imagenet_val: Path, n_per_class: int = 100, seed: int = 0) -> list[tuple[str, str, str]]:
    """The procedure that produced the shipped manifest (filesystem-order dependent)."""
    rng = random.Random(seed)
    wnids_by_class = json.loads(WNIDS.read_text())
    rows = []
    for category in sorted(wnids_by_class):
        paths = []
        for wnid in wnids_by_class[category]:
            wnid_dir = imagenet_val / wnid
            if wnid_dir.is_dir():
                paths.extend(p for p in wnid_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(paths)
        rows.extend((category, p.parent.name, p.name) for p in paths[:n_per_class])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagenet-val", type=Path, default=None,
                        help="ImageNet validation directory with one subfolder per WNID (default: $IMAGENET_VAL_DIR)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink")
    args = parser.parse_args()

    imagenet_val = args.imagenet_val or config.input_dir("IMAGENET_VAL_DIR")
    output_dir = args.output_dir or config.imagenet16_dir()

    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))

    missing = 0
    for row in rows:
        src = imagenet_val / row["wnid"] / row["filename"]
        dst = output_dir / row["class"] / row["filename"]
        if not src.exists():
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if args.mode == "copy":
            shutil.copyfile(src, dst)
        else:
            dst.symlink_to(src.resolve())

    if missing:
        raise SystemExit(f"{missing} manifest images not found under {imagenet_val}")
    print(f"{len(rows)} images -> {output_dir}")


if __name__ == "__main__":
    main()
