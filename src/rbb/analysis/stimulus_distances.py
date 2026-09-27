"""Objective measure of what lambda changes in the stimuli (right panel of Fig. stimuli).

    style distance    Gram-matrix distance between a stimulus and its style source over
                      relu1_1..relu5_1, with the generator's per-layer weights but without
                      the lambda factor
    content distance  MSE between relu4_2 feature maps of a stimulus and its content source

Averaged over all 1200 main-set stimuli per lambda, 95% bootstrap CI, both normalized to
their value at lambda = 1000.

    python -m rbb.analysis.stimulus_distances    -> results/stimulus_distances.csv
"""

import csv

import numpy as np
import torch
import torch.nn.functional as F

from rbb import config
from rbb.analysis.common import write_csv
from rbb.filenames import stimulus_name
from rbb.stats import bootstrap_mean_ci
from rbb.stimuli import gatys
from rbb.stimuli.cue_conflict import PAIRS_REFERENCE

N_BOOTSTRAP = 2000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = gatys.build_vgg(device)
    gram = gatys.GramMatrix()
    rng = np.random.default_rng(config.SEED)
    with PAIRS_REFERENCE.open() as f:
        rows = list(csv.DictReader(f))

    style_targets, content_targets = {}, {}

    def source(cls, src):
        return gatys.load_prepped(config.imagenet16_dir() / cls / f"{src}.JPEG", device)

    style_dist, content_dist = {}, {}
    for scale in config.SCALES:
        sd, cd = [], []
        for row in rows:
            name = stimulus_name(row["shape_class"], row["shape_src"], row["texture_class"], row["texture_src"])
            stim = gatys.load_prepped(config.stimuli_dir("main", scale) / row["shape_class"] / f"{name}.png", device)
            with torch.no_grad():
                if row["texture_src"] not in style_targets:
                    feats = vgg(source(row["texture_class"], row["texture_src"]), gatys.STYLE_LAYERS)
                    style_targets[row["texture_src"]] = [gram(f).detach() for f in feats]
                if row["shape_src"] not in content_targets:
                    content_targets[row["shape_src"]] = vgg(
                        source(row["shape_class"], row["shape_src"]), gatys.CONTENT_LAYERS)[0].detach()

                feats = vgg(stim, gatys.STYLE_LAYERS + gatys.CONTENT_LAYERS)
                sd.append(sum(w * float(F.mse_loss(gram(f), t).item())
                              for w, f, t in zip(gatys.STYLE_LAYER_WEIGHTS, feats[:5], style_targets[row["texture_src"]])))
                cd.append(float(F.mse_loss(feats[5], content_targets[row["shape_src"]]).item()))
        style_dist[scale], content_dist[scale] = np.array(sd), np.array(cd)
        print(f"lambda={scale}: style={style_dist[scale].mean():.6f} content={content_dist[scale].mean():.6f}")

    style_ref = style_dist[config.SCALES[0]].mean()
    content_ref = content_dist[config.SCALES[0]].mean()
    out = []
    for scale in config.SCALES:
        s, c = style_dist[scale] / style_ref, content_dist[scale] / content_ref
        s_ci, c_ci = bootstrap_mean_ci(s, N_BOOTSTRAP, rng), bootstrap_mean_ci(c, N_BOOTSTRAP, rng)
        out.append({
            "scale": scale,
            "style_distance_norm_mean": round(float(s.mean()), 6),
            "style_distance_norm_ci_low": round(s_ci[0], 6), "style_distance_norm_ci_high": round(s_ci[1], 6),
            "content_distance_norm_mean": round(float(c.mean()), 6),
            "content_distance_norm_ci_low": round(c_ci[0], 6), "content_distance_norm_ci_high": round(c_ci[1], 6),
            "n_stimuli": len(s),
        })
    write_csv(config.RESULTS_DIR / "stimulus_distances.csv", out)


if __name__ == "__main__":
    main()
