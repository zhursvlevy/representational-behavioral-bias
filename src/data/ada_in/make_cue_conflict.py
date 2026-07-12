import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from .function import adaptive_instance_normalization, coral


# ---------------------------------------------------------------------------
# Original transforms / style_transfer kept intact
# ---------------------------------------------------------------------------
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


IMG_EXTS = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}


def list_class_images(class_dir):
    return sorted(p for p in class_dir.iterdir() if p.suffix in IMG_EXTS)


def load_rgb(path, tf):
    """Open as RGB (ImageNet has grayscale/CMYK) and apply transform."""
    return tf(Image.open(str(path)).convert("RGB"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# --- MODIFIED: cue-conflict generation from a 16-class image root ---
parser.add_argument('--images_dir', type=str, required=True,
                    help='Root dir with <class_name>/<file> subfolders '
                         '(16 Geirhos classes of clean ImageNet images)')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the cue-conflict datasets')
parser.add_argument('--alphas', type=float, nargs='+', default=[1.0],
                    help='One or more AdaIN stylization strengths in [0,1]. '
                         'Each produces cue-conflict-alpha-<a>/ (VGG/decoder '
                         'are loaded once and reused across all alphas).')
parser.add_argument('--n_per_class', type=int, default=80,
                    help='Content images per shape class (default 80 -> 1280)')
parser.add_argument('--styles_per_texclass', type=int, default=5,
                    help='Distinct style images sampled per texture class')
parser.add_argument('--seed', type=int, default=42)

# --- kept from original ---
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, '
                         'keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, '
                         'keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = random.Random(args.seed)

images_root = Path(args.images_dir)
classes = sorted(p.name for p in images_root.iterdir() if p.is_dir())
if len(classes) != 16:
    print(f"[warn] expected 16 classes, found {len(classes)}: {classes}")

images_by_class = {c: list_class_images(images_root / c) for c in classes}

# 1-based instance index per source image (the Geirhos id component)
idx_of = {}
for c in classes:
    for i, p in enumerate(images_by_class[c], start=1):
        idx_of[p] = i

# ---------------------------------------------------------------------------
# Build ONE fixed pairing plan, reused for every alpha
#   * n_per_class contents per shape class
#   * texture class cycled over the other 15 classes (~balanced)
#   * shape != texture (diagonal excluded)
#   * small fixed style pool per class (styles_per_texclass), cycled globally
# ---------------------------------------------------------------------------
import itertools

style_cycle = {}
for c in classes:
    pool = images_by_class[c]
    k = min(args.styles_per_texclass, len(pool))
    style_cycle[c] = itertools.cycle(rng.sample(pool, k))

plan = []  # each: (shape_cls, tex_cls, content_path, style_path, shape_id, tex_id)
for c in classes:
    imgs = images_by_class[c]
    if len(imgs) < args.n_per_class:
        raise SystemExit(f"class '{c}': {len(imgs)} imgs < n_per_class={args.n_per_class}")
    contents = rng.sample(imgs, args.n_per_class)
    others = [x for x in classes if x != c]
    reps = -(-args.n_per_class // len(others))
    tex_seq = (others * reps)[:args.n_per_class]
    rng.shuffle(tex_seq)
    for content, tex in zip(contents, tex_seq):
        style = next(style_cycle[tex])
        plan.append((c, tex, content, style,
                     f"{c}{idx_of[content]}", f"{tex}{idx_of[style]}"))

print(f"[plan] {len(plan)} pairs across {len(classes)} classes")

# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------
decoder = net.decoder
vgg = net.vgg
decoder.eval()
vgg.eval()
decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

output_root = Path(args.output)

# ---------------------------------------------------------------------------
# Generate: outer loop over alpha, inner loop over the fixed plan
# Final layout:
#   <output>/cue-conflict-alpha-<a>/<shape_class>/<shape_id>-<tex_id>.jpg
# ---------------------------------------------------------------------------
for alpha in args.alphas:
    tag = f"{alpha:.2f}"
    ds_root = output_root / f"cue-conflict-alpha-{tag}"
    print(f"[alpha {tag}] -> {ds_root}")

    for i, (shape_cls, tex_cls, content_path, style_path,
            shape_id, tex_id) in enumerate(plan):
        content = load_rgb(content_path, content_tf)
        style = load_rgb(style_path, style_tf)
        if args.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, alpha)
        output = output.cpu()

        out_dir = ds_root / shape_cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / f"{shape_id}-{tex_id}{args.save_ext}"
        save_image(output, str(out_name))

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(plan)}]")

print("[done]")