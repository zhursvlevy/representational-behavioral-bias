"""Stimuli for the per-cue ceilings and the style-recognizability check.

    photos        unstylized shape sources (shape ceiling) and texture sources
                  (recognizability of full style photographs), one per source image
    white-canvas  a white canvas stylized with each texture source at a given lambda
                  (texture ceiling: style statistics without any content layout)

File names follow the stimulus scheme so labels and groups parse the same way; the
field of the cue that is absent (texture for shape photos, shape for texture images)
is taken from the source's first pairing and is never used.

    python -m rbb.stimuli.ceilings photos
    python -m rbb.stimuli.ceilings white-canvas --scale 6000
"""

import argparse
import csv
import time

import torch
from PIL import Image
from torchvision import transforms

from rbb import config
from rbb.filenames import stimulus_name
from rbb.stimuli import gatys
from rbb.stimuli.cue_conflict import PAIRS_REFERENCE


def unique_sources(key: str) -> list[dict]:
    """First pairing of every distinct shape_src or texture_src, in pairing order."""
    seen, rows = set(), []
    with PAIRS_REFERENCE.open() as f:
        for row in csv.DictReader(f):
            if row[key] not in seen:
                seen.add(row[key])
                rows.append(row)
    return rows


def source_photo(cls: str, src: str):
    return config.imagenet16_dir() / cls / f"{src}.JPEG"


def make_photos() -> None:
    shape_dir = config.ceiling_stimuli_dir("shape_photos")
    texture_dir = config.ceiling_stimuli_dir("texture_photos")
    shape_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    for row in unique_sources("shape_src"):
        img = gatys.resize_crop(Image.open(source_photo(row["shape_class"], row["shape_src"])).convert("RGB"))
        name = stimulus_name(row["shape_class"], row["shape_src"], row["texture_class"], row["texture_src"])
        img.save(shape_dir / f"{name}.png")

    for row in unique_sources("texture_src"):
        img = gatys.resize_crop(Image.open(source_photo(row["texture_class"], row["texture_src"])).convert("RGB"))
        name = stimulus_name(row["shape_class"], row["shape_src"], row["texture_class"], row["texture_src"])
        img.save(texture_dir / f"{name}.png")

    print(f"shape photos -> {shape_dir}; texture photos -> {texture_dir}")


def make_white_canvas(scale: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.ceiling_stimuli_dir("white_canvas", scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    vgg = gatys.build_vgg(device)

    white = torch.ones(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    normalize = transforms.Normalize(mean=gatys.IMAGENET_MEAN, std=gatys.IMAGENET_STD)
    white_content = normalize(white[0]).unsqueeze(0).to(device)

    jobs = unique_sources("texture_src")
    t_start, n_done = time.time(), 0
    for i, row in enumerate(jobs, start=1):
        name = stimulus_name(row["shape_class"], row["shape_src"], row["texture_class"], row["texture_src"])
        out_path = output_dir / f"{name}.png"
        if out_path.exists():
            continue
        style = gatys.load_prepped(source_photo(row["texture_class"], row["texture_src"]), device)
        out = gatys.style_transfer(vgg, white_content, style, scale, gatys.MAX_ITER, device)
        gatys.postprocess(out[0].cpu()).save(out_path)
        n_done += 1
        print(f"[{i}/{len(jobs)}] {row['texture_class']} ({(time.time() - t_start) / n_done:.1f}s/img)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("photos")
    wc = sub.add_parser("white-canvas")
    wc.add_argument("--scale", type=int, required=True)
    args = parser.parse_args()

    if args.command == "photos":
        make_photos()
    else:
        make_white_canvas(args.scale)


if __name__ == "__main__":
    main()
