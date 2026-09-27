"""Within a fixed condition, does cue availability predict the answered cue?

For every image, out-of-fold probes give m = p_shape(y_shape) - p_texture(y_texture)
(and the same on logits). Images answered with neither cue are excluded. We test whether
m separates shape-answered from texture-answered images: AUROC, bootstrap CI of the
difference in mean margin, Mann-Whitney U, Cliff's delta, Cohen's d.

    last-layer  final decoder layer, probability and logit margins, all main conditions
    depth       probability margin at 0/25/50/75/100% of decoder depth, all main conditions

    python -m rbb.analysis.within_condition last-layer
    python -m rbb.analysis.within_condition depth
"""

import argparse

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from rbb import config
from rbb.analysis.common import main_conditions, read_answers, write_csv
from rbb.probing import load_folds, load_hidden_states, out_of_fold_true_class
from rbb.stats import bootstrap_mean_diff_ci, cliffs_delta, cohens_d

N_BOOTSTRAP = 2000
DEPTH_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
MARGIN_EXAMPLE = ("llava", 6000)


def load_condition(model: str, condition: str):
    """Hidden states and the answer category of each image, in the same order."""
    data = load_hidden_states(config.hidden_dir(condition, model))
    category = {r["file_name"]: r["category"] for r in read_answers(condition, model)}
    return data, np.array([category[n] for n in data.names])


def margins(data, folds_set: str, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probability margin and logit margin per image at one layer."""
    p_shape, l_shape = out_of_fold_true_class(data.X, data.y_shape, data.shape_groups, load_folds(folds_set, "shape"), layer_idx)
    p_tex, l_tex = out_of_fold_true_class(data.X, data.y_texture, data.texture_groups, load_folds(folds_set, "texture"), layer_idx)
    return p_shape - p_tex, l_shape - l_tex


def counts(categories: np.ndarray) -> dict:
    n_total = len(categories)
    n_excluded = int((~np.isin(categories, ["shape_match", "texture_match"])).sum())
    return {"n_total": n_total,
            "n_shape": int((categories == "shape_match").sum()),
            "n_texture": int((categories == "texture_match").sum()),
            "n_excluded": n_excluded,
            "excluded_frac": round(n_excluded / n_total, 6)}


def separation(margin: np.ndarray, categories: np.ndarray, rng: np.random.Generator) -> dict:
    included = np.isin(categories, ["shape_match", "texture_match"])
    m, is_shape = margin[included], categories[included] == "shape_match"
    x, y = m[is_shape], m[~is_shape]
    ci_low, ci_high = bootstrap_mean_diff_ci(x, y, N_BOOTSTRAP, rng)
    mw_stat, mw_p = sps.mannwhitneyu(x, y, alternative="two-sided")
    return {
        "auroc": round(float(roc_auc_score(is_shape, m)), 6),
        "mean_diff": round(float(x.mean() - y.mean()), 6),
        "mean_diff_ci_low": round(ci_low, 6),
        "mean_diff_ci_high": round(ci_high, 6),
        "mannwhitney_u": float(mw_stat),
        "mannwhitney_p": float(mw_p),
        "cliffs_delta": round(cliffs_delta(x, y), 6),
        "cohens_d": round(cohens_d(x, y), 6),
    }


def bootstrap_auroc_ci(margin: np.ndarray, is_shape: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Resamples images within each answer group."""
    pos, neg = np.where(is_shape)[0], np.where(~is_shape)[0]
    aurocs = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        labels = np.concatenate([np.ones(len(bp)), np.zeros(len(bn))])
        aurocs[b] = roc_auc_score(labels, np.concatenate([margin[bp], margin[bn]]))
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def last_layer_rows(model: str, stimulus_set: str, scale: int, rng: np.random.Generator,
                    save_margins: bool = False, variants=("probability", "logit")) -> list[dict]:
    condition = config.condition(stimulus_set, scale)
    data, categories = load_condition(model, condition)
    prob_margin, logit_margin = margins(data, stimulus_set, data.X.shape[1] - 1)
    if save_margins:
        write_csv(config.RESULTS_DIR / f"margins_{model}_{condition}.csv",
                  [{"margin_probability": round(float(p), 6), "margin_logit": round(float(l), 6), "category": c}
                   for p, l, c in zip(prob_margin, logit_margin, categories)])
    rows = []
    for variant, margin in (("probability", prob_margin), ("logit", logit_margin)):
        if variant not in variants:
            continue
        rows.append({"set": stimulus_set, "model": model, "scale": scale, "variant": variant,
                     **counts(categories), **separation(margin, categories, rng)})
        print(f"{model} {condition} [{variant}]: AUROC={rows[-1]['auroc']:.4f}")
    return rows


def depth_rows(model: str, scale: int, rng: np.random.Generator) -> list[dict]:
    condition = config.condition("main", scale)
    data, categories = load_condition(model, condition)
    n_layers = data.X.shape[1]
    rows = []
    for frac in DEPTH_FRACTIONS:
        layer_idx = int(np.floor(frac * (n_layers - 1)))
        margin, _ = margins(data, "main", layer_idx)
        included = np.isin(categories, ["shape_match", "texture_match"])
        m, is_shape = margin[included], categories[included] == "shape_match"
        ci_low, ci_high = bootstrap_auroc_ci(m, is_shape, rng)
        rows.append({
            "model": model, "scale": scale, "depth_fraction": frac, "layer_idx": layer_idx,
            "num_saved_layers": n_layers, **counts(categories),
            "auroc": round(float(roc_auc_score(is_shape, m)), 6),
            "auroc_ci_low": round(ci_low, 6), "auroc_ci_high": round(ci_high, 6),
            "cliffs_delta": round(cliffs_delta(m[is_shape], m[~is_shape]), 6),
        })
        print(f"{model} {condition} depth={frac:.2f} (layer {layer_idx}): AUROC={rows[-1]['auroc']:.4f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=["last-layer", "depth"])
    args = parser.parse_args()
    rng = np.random.default_rng(config.SEED)

    if args.analysis == "last-layer":
        rows = []
        for model, scale in main_conditions():
            rows += last_layer_rows(model, "main", scale, rng, save_margins=(model, scale) == MARGIN_EXAMPLE)
        write_csv(config.RESULTS_DIR / "within_condition_auroc.csv", rows)
    else:
        rows = []
        for model, scale in main_conditions():
            rows += depth_rows(model, scale, rng)
        write_csv(config.RESULTS_DIR / "within_condition_by_layer.csv", rows)


if __name__ == "__main__":
    main()
