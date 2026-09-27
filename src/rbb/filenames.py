"""Labels and source-exemplar groups encoded in stimulus file names.

Two naming schemes are supported:
  - generated stimuli: {shape}_shape-{shape_src}_texture-{texture}-{texture_src}
  - Geirhos et al. cue-conflict set: {shape}{i}-{texture}{j}  (e.g. airplane1-bear3)
"""

import re
from pathlib import Path

_GENERATED_RE = re.compile(
    r"^(?P<shape_class>.+?)_shape-(?P<shape_src>.+?)_texture-"
    r"(?P<texture_class>[^-]+)-(?P<texture_src>.+)$"
)


def stimulus_name(shape_class: str, shape_src: str, texture_class: str, texture_src: str) -> str:
    return f"{shape_class}_shape-{shape_src}_texture-{texture_class}-{texture_src}"


def extract_labels(filename: str) -> tuple[str | None, str | None]:
    stem = Path(filename).stem
    m = _GENERATED_RE.match(stem)
    if m:
        return m.group("shape_class"), m.group("texture_class")
    if "-" not in stem:
        return None, None
    shape_part, texture_part = stem.split("-", 1)
    shape_name = "".join(filter(str.isalpha, shape_part))
    texture_name = "".join(filter(str.isalpha, texture_part))
    if not shape_name or not texture_name:
        return None, None
    return shape_name, texture_name


def extract_groups(filename: str) -> tuple[str, str]:
    """Source-exemplar ids (shape, texture) used as cross-validation groups."""
    stem = Path(filename).stem
    m = _GENERATED_RE.match(stem)
    if m:
        return m.group("shape_src"), m.group("texture_src")
    shape_src, texture_src = stem.split("-", 1)
    return shape_src, texture_src
