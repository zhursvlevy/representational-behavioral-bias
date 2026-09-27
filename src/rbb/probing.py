"""Linear probes on decoder hidden states.

Probes are L2-regularized logistic regressions (C = 1.0, max_iter = 1000) fitted with
L-BFGS on features standardized with training-fold statistics. They are implemented in
PyTorch so that one probe per layer can be trained for all layers in a single batched
problem; `rbb.analysis.validate_probe` checks them against scikit-learn.

Grouped cross-validation holds out source exemplars: the group of a stimulus is its
content image for the shape probe and its style image for the texture probe. Fold
assignments depend only on stimulus names and are derived once per stimulus set, so
every condition, layer and model uses the same partitions.

    python -m rbb.probing folds --set main
    python -m rbb.probing grouped --model llava --condition main_l6000 --folds main
    python -m rbb.probing grouped --model llava --condition ceiling_shape --folds main --labels shape
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from tqdm import tqdm

from rbb import config
from rbb.filenames import extract_groups, extract_labels

# sklearn minimizes 0.5||w||^2 + C * sum(loss); dividing by n*C gives
# mean(loss) + (0.5 / C) ||w||^2 / n, the objective used below.
WEIGHT_DECAY = 0.5 / config.PROBE_C


def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- data ---------------------------------------------------------------------------

@dataclass
class HiddenStates:
    X: np.ndarray               # (n_samples, n_layers, hidden_dim)
    y_shape: np.ndarray
    y_texture: np.ndarray
    shape_groups: np.ndarray
    texture_groups: np.ndarray
    names: list[str]


def usable_names(names) -> list[str]:
    """Stimulus stems with parseable, conflicting labels, in the canonical (sorted) order."""
    keep = []
    for name in sorted(names, key=lambda n: f"{n}.npy"):
        shape_name, texture_name = extract_labels(name)
        if shape_name is None or texture_name is None or shape_name == texture_name:
            continue
        keep.append(name)
    return keep


def load_hidden_states(directory: Path) -> HiddenStates:
    names = usable_names(p.stem for p in directory.glob("*.npy"))
    if not names:
        raise ValueError(f"No hidden states in {directory}")
    X = np.stack([np.load(directory / f"{n}.npy") for n in tqdm(names, desc=f"load {directory.name}")])
    labels = [extract_labels(n) for n in names]
    groups = [extract_groups(n) for n in names]
    return HiddenStates(
        X=X,
        y_shape=np.array([s for s, _ in labels]),
        y_texture=np.array([t for _, t in labels]),
        shape_groups=np.array([s for s, _ in groups]),
        texture_groups=np.array([t for _, t in groups]),
        names=names,
    )


# ---- probe fitting --------------------------------------------------------------------

@dataclass
class ProbeFit:
    classes: np.ndarray
    test_logits: np.ndarray     # (n_layers, n_test, n_classes)
    W: np.ndarray               # (n_layers, hidden_dim, n_classes), standardized feature space
    b: np.ndarray               # (n_layers, n_classes)
    mean: np.ndarray            # (n_layers, hidden_dim) training-fold statistics
    std: np.ndarray

    def f1_by_layer(self, y_test_idx: np.ndarray) -> list[float]:
        preds = self.test_logits.argmax(axis=-1)
        return [float(f1_score(y_test_idx, p, average="macro")) for p in preds]


def fit_probes(X, y, train_idx, test_idx, multi_class: str = "multinomial",
               max_iter: int = config.PROBE_MAX_ITER, weight_decay: float = WEIGHT_DECAY) -> ProbeFit:
    """One independent probe per layer, all layers optimized jointly as one batched problem.

    Features are standardized per layer: the batched L-BFGS shares one line search across
    layers, and without standardization layers with large activations dominate it and
    the others stay near zero.

    multi_class="multinomial": softmax regression (grouped protocol);
    multi_class="ovr": independent binary problems per class (random-split protocol).
    """
    y = np.asarray(y)
    classes, y_idx = np.unique(y, return_inverse=True)
    num_classes = len(classes)
    device = torch_device()

    X_train = torch.as_tensor(np.ascontiguousarray(X[train_idx].transpose(1, 0, 2)), dtype=torch.float32, device=device)
    X_test = torch.as_tensor(np.ascontiguousarray(X[test_idx].transpose(1, 0, 2)), dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_idx[train_idx], dtype=torch.long, device=device)
    num_layers, n_train, hidden_dim = X_train.shape

    mean = X_train.mean(dim=1, keepdim=True)
    std = X_train.std(dim=1, keepdim=True).clamp_min(1e-6)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    W = torch.zeros(num_layers, hidden_dim, num_classes, device=device, requires_grad=True)
    b = torch.zeros(num_layers, num_classes, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    if multi_class == "ovr":
        y_onehot = F.one_hot(y_train, num_classes).float().unsqueeze(0).expand(num_layers, -1, -1)
    elif multi_class != "multinomial":
        raise ValueError(f"Unknown multi_class: {multi_class!r}")

    def closure():
        optimizer.zero_grad()
        logits = torch.einsum("lnd,ldc->lnc", X_train, W) + b.unsqueeze(1)
        if multi_class == "ovr":
            loss = F.binary_cross_entropy_with_logits(logits, y_onehot)
        else:
            loss = F.cross_entropy(logits.reshape(-1, num_classes), y_train.repeat(num_layers))
        loss = loss + weight_decay * W.pow(2).sum() / n_train
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        test_logits = torch.einsum("lnd,ldc->lnc", X_test, W) + b.unsqueeze(1)

    return ProbeFit(
        classes=classes,
        test_logits=test_logits.cpu().numpy(),
        W=W.detach().cpu().numpy(),
        b=b.detach().cpu().numpy(),
        mean=mean.squeeze(1).cpu().numpy(),
        std=std.squeeze(1).cpu().numpy(),
    )


# ---- folds ----------------------------------------------------------------------------

def derive_folds(names: list[str], cue: str, n_splits: int = config.N_SPLITS,
                 seed: int = config.SPLIT_SEED) -> dict[str, int]:
    """{source exemplar id: fold} from StratifiedGroupKFold over the stimulus names."""
    names = usable_names(names)
    idx = 0 if cue == "shape" else 1
    y = np.array([extract_labels(n)[idx] for n in names])
    groups = np.array([extract_groups(n)[idx] for n in names])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_to_fold = {}
    for fold_idx, (_, val_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        for g in np.unique(groups[val_idx]):
            group_to_fold[str(g)] = fold_idx
    return group_to_fold


def fold_path(stimulus_set: str, cue: str) -> Path:
    return config.folds_dir() / f"{stimulus_set}_{cue}.json"


def load_folds(stimulus_set: str, cue: str) -> dict[str, int]:
    return json.loads(fold_path(stimulus_set, cue).read_text())


def fold_indices(groups: np.ndarray, group_to_fold: dict[str, int]) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_of = np.array([group_to_fold[str(g)] for g in groups])
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in sorted(set(group_to_fold.values()))]


def stimulus_names(stimulus_set: str) -> list[str]:
    """Stimulus stems of a set, without needing the images or hidden states."""
    if stimulus_set == "geirhos":
        return [p.stem for p in config.input_dir("GEIRHOS_STIMULI_DIR").rglob("*.png")]
    from rbb.filenames import stimulus_name
    from rbb.stimuli.cue_conflict import PAIRS_REFERENCE
    with PAIRS_REFERENCE.open() as f:
        rows = list(csv.DictReader(f))
    if stimulus_set == "crop":
        rows = rows[: config.CROP_N_STIMULI]
    return [stimulus_name(r["shape_class"], r["shape_src"], r["texture_class"], r["texture_src"]) for r in rows]


# ---- evaluation protocols -------------------------------------------------------------

def grouped_cv(X, y, groups, group_to_fold) -> dict:
    """Per-layer macro-F1 over the folds (multinomial probes)."""
    classes, y_idx = np.unique(y, return_inverse=True)
    fold_scores = []
    for train_idx, val_idx in fold_indices(groups, group_to_fold):
        fit = fit_probes(X, y, train_idx, val_idx)
        fold_scores.append(fit.f1_by_layer(y_idx[val_idx]))
    fold_scores = np.array(fold_scores)  # (n_folds, n_layers)
    return {
        "mean": {str(l): round(float(v), 6) for l, v in enumerate(fold_scores.mean(axis=0))},
        "std": {str(l): round(float(v), 6) for l, v in enumerate(fold_scores.std(axis=0))},
        "folds": {str(l): [round(float(v), 6) for v in fold_scores[:, l]] for l in range(fold_scores.shape[1])},
    }


def random_split(X, y, seed: int = config.SPLIT_SEED, test_size: float = 0.2) -> dict:
    """Per-layer macro-F1 on one stratified random split (one-vs-rest probes)."""
    classes, y_idx = np.unique(y, return_inverse=True)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    fit = fit_probes(X, y, train_idx, test_idx, multi_class="ovr")
    return {str(l): round(v, 6) for l, v in enumerate(fit.f1_by_layer(y_idx[test_idx]))}


def out_of_fold_true_class(X, y, groups, group_to_fold, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Probability and logit of each image's true class, from the probe whose training
    folds excluded the image's source exemplar."""
    y = np.asarray(y)
    classes, y_idx = np.unique(y, return_inverse=True)
    X_layer = X[:, layer_idx:layer_idx + 1, :]
    prob = np.full(len(y), np.nan)
    logit = np.full(len(y), np.nan)
    for train_idx, val_idx in fold_indices(groups, group_to_fold):
        fit = fit_probes(X_layer, y, train_idx, val_idx)
        val_logits = fit.test_logits[0]
        val_probs = torch.softmax(torch.as_tensor(val_logits), dim=1).numpy()
        rows = np.arange(len(val_idx))
        prob[val_idx] = val_probs[rows, y_idx[val_idx]]
        logit[val_idx] = val_logits[rows, y_idx[val_idx]]
    return prob, logit


# ---- CLI ------------------------------------------------------------------------------

def last_layer(metric: dict[str, float]) -> float:
    return metric[str(max(int(k) for k in metric))]


def run_grouped(model: str, condition: str, folds_set: str, labels: str) -> None:
    data = load_hidden_states(config.hidden_dir(condition, model))
    result = {"n_splits": config.N_SPLITS, "folds": folds_set}
    for cue in (["shape", "texture"] if labels == "both" else [labels]):
        y = data.y_shape if cue == "shape" else data.y_texture
        groups = data.shape_groups if cue == "shape" else data.texture_groups
        scores = grouped_cv(data.X, y, groups, load_folds(folds_set, cue))
        result[f"{cue}_f1_macro_mean"] = scores["mean"]
        result[f"{cue}_f1_macro_std"] = scores["std"]
        result[f"{cue}_f1_macro_folds"] = scores["folds"]
        print(f"{model}/{condition} {cue}: last-layer F1 = {last_layer(scores['mean']):.4f}")
    out = config.probe_dir(condition, model)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f1_metrics.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("folds", help="derive and save the fold assignment of a stimulus set")
    f.add_argument("--set", choices=["main", "crop", "geirhos"], required=True)
    g = sub.add_parser("grouped", help="grouped cross-validation for one model and condition")
    g.add_argument("--model", choices=config.MODELS, required=True)
    g.add_argument("--condition", required=True)
    g.add_argument("--folds", choices=["main", "crop", "geirhos"], required=True)
    g.add_argument("--labels", choices=["both", "shape", "texture"], default="both")
    args = parser.parse_args()

    if args.command == "folds":
        names = stimulus_names(args.set)
        config.folds_dir().mkdir(parents=True, exist_ok=True)
        for cue in ("shape", "texture"):
            fold_path(args.set, cue).write_text(json.dumps(derive_folds(names, cue), indent=2))
        print(f"folds for {args.set} -> {config.folds_dir()}")
    else:
        run_grouped(args.model, args.condition, args.folds, args.labels)


if __name__ == "__main__":
    main()
