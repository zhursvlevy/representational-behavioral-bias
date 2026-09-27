"""Generate cue-conflict stimuli: shape (content) from one image, texture (style) from another.

Each of the 16 classes' images is split into two disjoint pools: 25 shape sources and
25 texture sources, so no photograph supplies both cues. Each shape image is paired
with 3 texture images from distinct other classes (1200 stimuli). Texture images are
drawn round-robin from each class's pool so every texture source is used equally often.
Pool assignment and pairings depend only on --seed and are identical for every lambda;
they are checked against resources/cue_conflict_pairs.csv.

--style-source crop builds the control set: the style image is a 128x128 crop of the
source photograph (fixed per source image), resized to 256x256.

    python -m rbb.stimuli.cue_conflict --scale 6000
    python -m rbb.stimuli.cue_conflict --scale 6000 --style-source crop --limit 300
"""

import argparse
import csv
import random
import time
from itertools import cycle
from pathlib import Path

import torch
from PIL import Image

from rbb import config
from rbb.filenames import stimulus_name
from rbb.stimuli import gatys

PAIRS_REFERENCE = config.RESOURCES / "cue_conflict_pairs.csv"
SHAPE_PER_CLASS = 25
TEXTURE_PER_CLASS = 25
TEXTURES_PER_SHAPE = 3
CROP_SIZE = config.IMG_SIZE // 2


def build_jobs(imagenet16: Path, seed: int = config.SEED) -> list[tuple[str, Path, str, Path]]:
    rng = random.Random(seed)
    classes = sorted(d.name for d in imagenet16.iterdir() if d.is_dir())

    shape_pools, texture_pools = {}, {}
    for cls in classes:
        images = sorted((imagenet16 / cls).glob("*.JPEG"))
        rng.shuffle(images)
        need = SHAPE_PER_CLASS + TEXTURE_PER_CLASS
        if len(images) < need:
            half = len(images) // 2
            shape_pools[cls], texture_pools[cls] = images[:half], images[half:]
        else:
            shape_pools[cls] = images[:SHAPE_PER_CLASS]
            texture_pools[cls] = images[SHAPE_PER_CLASS:need]

    tex_cyclers = {}
    for cls in classes:
        pool = list(texture_pools[cls])
        rng.shuffle(pool)
        tex_cyclers[cls] = cycle(pool)

    jobs = []
    for shape_cls in classes:
        other_classes = [c for c in classes if c != shape_cls]
        for shape_path in shape_pools[shape_cls]:
            for texture_cls in rng.sample(other_classes, min(TEXTURES_PER_SHAPE, len(other_classes))):
                jobs.append((shape_cls, shape_path, texture_cls, next(tex_cyclers[texture_cls])))
    rng.shuffle(jobs)
    return jobs


def check_against_reference(jobs) -> None:
    with PAIRS_REFERENCE.open() as f:
        reference = [tuple(r.values()) for r in csv.DictReader(f)]
    produced = [(s, sp.stem, t, tp.stem) for s, sp, t, tp in jobs]
    if produced != reference:
        n_diff = sum(a != b for a, b in zip(produced, reference)) + abs(len(produced) - len(reference))
        raise SystemExit(
            f"Stimulus pairings differ from {PAIRS_REFERENCE.name} in {n_diff} rows; "
            "the ImageNet subset or seed does not match the one used in the paper."
        )


def style_crop(texture_path: Path, cache_dir: Path) -> tuple[Image.Image, int, int]:
    """Random 128x128 crop of the 256x256 source, seeded by the source id, resized to 256."""
    rng = random.Random(f"{config.SEED}-{texture_path.stem}")
    max_offset = config.IMG_SIZE - CROP_SIZE
    left, top = rng.randint(0, max_offset), rng.randint(0, max_offset)
    cache_path = cache_dir / f"{texture_path.stem}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB"), left, top
    img256 = gatys.resize_crop(Image.open(texture_path).convert("RGB"))
    crop = img256.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))
    crop = crop.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BICUBIC)
    cache_dir.mkdir(parents=True, exist_ok=True)
    crop.save(cache_path)
    return crop, left, top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=int, required=True, help="style weight lambda")
    parser.add_argument("--style-source", choices=["full", "crop"], default="full")
    parser.add_argument("--limit", type=int, default=None, help="generate only the first N stimuli")
    parser.add_argument("--max-iter", type=int, default=gatys.MAX_ITER)
    args = parser.parse_args()

    stimulus_set = "main" if args.style_source == "full" else "crop"
    output_dir = config.stimuli_dir(stimulus_set, args.scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jobs = build_jobs(config.imagenet16_dir())
    check_against_reference(jobs)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"{stimulus_set} set, lambda={args.scale}: {len(jobs)} stimuli -> {output_dir}")

    vgg = gatys.build_vgg(device)
    metadata_rows = []
    t_start, n_done = time.time(), 0
    for shape_cls, shape_path, texture_cls, texture_path in jobs:
        name = stimulus_name(shape_cls, shape_path.stem, texture_cls, texture_path.stem)
        out_path = output_dir / shape_cls / f"{name}.png"
        row = [f"{shape_cls}/{name}.png", shape_cls, shape_path.stem, texture_cls, texture_path.stem]

        if args.style_source == "crop":
            crop_img, left, top = style_crop(texture_path, config.style_crops_dir())
            row += [left, top, CROP_SIZE]
        metadata_rows.append(row)
        if out_path.exists():
            continue

        content = gatys.load_prepped(shape_path, device)
        if args.style_source == "crop":
            style = gatys.normalize(crop_img).unsqueeze(0).to(device)
        else:
            style = gatys.load_prepped(texture_path, device)
        out = gatys.style_transfer(vgg, content, style, args.scale, args.max_iter, device)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gatys.postprocess(out[0].cpu()).save(out_path)

        n_done += 1
        rate = (time.time() - t_start) / n_done
        print(f"[{len(metadata_rows)}/{len(jobs)}] {shape_cls} <- {texture_cls} ({rate:.1f}s/img)", flush=True)

    header = ["file", "shape_class", "shape_src", "texture_class", "texture_src"]
    if args.style_source == "crop":
        header += ["crop_left", "crop_top", "crop_size"]
    with (output_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(metadata_rows)


if __name__ == "__main__":
    main()
