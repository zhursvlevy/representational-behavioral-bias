"""Small statistics helpers shared by the analyses."""

import numpy as np
from scipy import stats


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_mean_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = rng.choice(x, size=len(x), replace=True).mean() - rng.choice(y, size=len(y), replace=True).mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y)))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    pooled = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")


def ols(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Simple linear regression y = a + b x with t-based confidence intervals."""
    n = len(x)
    xbar, ybar = x.mean(), y.mean()
    sxx = np.sum((x - xbar) ** 2)
    b = np.sum((x - xbar) * (y - ybar)) / sxx
    a = ybar - b * xbar
    rss = float(np.sum((y - (a + b * x)) ** 2))
    dof = n - 2
    s = np.sqrt(rss / dof)
    se_b = s / np.sqrt(sxx)
    se_a = s * np.sqrt(1 / n + xbar**2 / sxx)
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    return {
        "intercept": float(a),
        "slope": float(b),
        "intercept_ci95": [float(a - t_crit * se_a), float(a + t_crit * se_a)],
        "slope_ci95": [float(b - t_crit * se_b), float(b + t_crit * se_b)],
        "slope_p_value": float(2 * (1 - stats.t.cdf(abs(b / se_b), dof))),
        "r_squared": float(1 - rss / np.sum((y - ybar) ** 2)),
        "residual_std": float(s),
        "rss": rss,
        "dof": int(dof),
        "n": int(n),
        "x_mean": float(xbar),
        "sxx": float(sxx),
    }
