"""Check the PyTorch probes against scikit-learn LogisticRegression(C=1.0, max_iter=1000)
on real hidden states (LLaVA, lambda = 6000, final layer, first grouped fold).

The torch probes standardize features, so the fair comparison fits scikit-learn on the
same standardized features; weights are compared by per-class cosine similarity.

    python -m rbb.analysis.validate_probe    -> results/torch_vs_sklearn_validation.json
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from rbb import config
from rbb.analysis.common import write_json
from rbb.probing import fit_probes, fold_indices, load_folds, load_hidden_states

MODEL, SCALE = "llava", 6000


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    data = load_hidden_states(config.hidden_dir(config.condition("main", SCALE), MODEL))
    train_idx, val_idx = fold_indices(data.shape_groups, load_folds("main", "shape"))[0]
    layer = data.X.shape[1] - 1
    y = data.y_shape
    X_train, X_val = data.X[train_idx, layer], data.X[val_idx, layer]

    fit = fit_probes(data.X[:, layer:layer + 1], y, train_idx, val_idx)
    y_val_idx = np.searchsorted(fit.classes, y[val_idx])
    torch_f1 = fit.f1_by_layer(y_val_idx)[0]
    torch_w = fit.W[0] / fit.std[0][:, None]  # back to raw feature space

    mean = X_train.mean(axis=0, keepdims=True)
    std = np.clip(X_train.std(axis=0, keepdims=True), 1e-6, None)
    sk = LogisticRegression(C=config.PROBE_C, max_iter=config.PROBE_MAX_ITER).fit((X_train - mean) / std, y[train_idx])
    sk_f1 = f1_score(y[val_idx], sk.predict((X_val - mean) / std), average="macro")
    sk_w = sk.coef_ / std

    cos = {str(c): cosine(sk_w[i], torch_w[:, i]) for i, c in enumerate(sk.classes_)}
    write_json(config.RESULTS_DIR / "torch_vs_sklearn_validation.json", {
        "model": MODEL, "scale": SCALE, "layer_idx": layer,
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "sklearn_f1_macro": round(float(sk_f1), 6), "torch_f1_macro": round(float(torch_f1), 6),
        "f1_abs_diff": round(abs(float(sk_f1) - float(torch_f1)), 6),
        "weight_cosine_similarity_by_class": cos,
        "weight_cosine_similarity_min": round(min(cos.values()), 6),
    })


if __name__ == "__main__":
    main()
