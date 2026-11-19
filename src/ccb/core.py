from dataclasses import dataclass
from math import log

import numpy as np

from .errors import validated_alpha, validated_variants


@dataclass(frozen=True)
class Grid:
    lams: np.ndarray
    weights: np.ndarray

    @classmethod
    def default(cls, n: int = 14) -> "Grid":
        return cls(2.0 ** (-np.arange(n) / 2.0), np.full(n, 1.0 / n))


def selection_correction(variants: int) -> float:
    """log V correction for selecting the best of V variants, Proposition 5.5."""
    return log(validated_variants(variants))


def threshold(alpha: float, selection_variants: int = 1) -> float:
    """Evidence threshold log(1/alpha) + log V, paper eq. (9)."""
    return float(
        log(1.0 / validated_alpha(alpha)) + selection_correction(selection_variants)
    )


def costs(lam: float) -> tuple[float, float]:
    """Insertion and replacement cost in nats, paper eq. (6)."""
    return (
        float(np.log1p(lam / 2.0)),
        float(np.log((2.0 + lam) / (2.0 - lam))),
    )


def log_wealth(outcomes: np.ndarray, lam: float) -> np.ndarray:
    """Capped-bet log-wealth path, paper eq. (5)."""
    return np.cumsum(np.log1p(lam * (np.asarray(outcomes, dtype=float) - 0.5)))


def log_wealth_final(outcomes, lam: float) -> float:
    """Final capped-bet log wealth without the path, paper eq. (5)."""
    values = np.asarray(outcomes, dtype=float)
    return float(np.log1p(lam * (values - 0.5)).sum())


def exp_costs(gamma: float) -> tuple[float, float]:
    """Insertion and replacement cost for the exponential wealth, paper eq. (10)."""
    return gamma / 2.0 - gamma**2 / 8.0, gamma


def exp_log_wealth(outcomes: np.ndarray, gamma: float) -> np.ndarray:
    """Exponential log-wealth path, paper eq. (10)."""
    x = np.asarray(outcomes, dtype=float) - 0.5
    return np.cumsum(gamma * x - gamma**2 / 8.0)


def grid_costs(grid: Grid, cost: str) -> np.ndarray:
    index = 0 if cost == "insertion" else 1
    return np.array([costs(float(lam))[index] for lam in grid.lams])


def log_phi(
    log_paths: np.ndarray,
    cost_vector: np.ndarray,
    weights: np.ndarray,
    budget,
) -> np.ndarray:
    """Deflated average wealth, paper eq. (8); budget may be a scalar or per-t."""
    offset = np.log(weights)[:, None] - budget * cost_vector[:, None]
    return np.logaddexp.reduce(log_paths + offset, axis=0)


def budget_path(
    log_paths: np.ndarray,
    cost_vector: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    cap: int | None = None,
) -> np.ndarray:
    """Largest integer budget with Phi >= threshold at each t, paper eq. (9)."""
    columns = log_paths.shape[1]
    if columns == 0:
        return np.zeros(0, dtype=np.int64)
    cap = columns if cap is None else cap
    lo = np.zeros(columns, dtype=np.int64)
    hi = np.full(columns, cap, dtype=np.int64)
    feasible = log_phi(log_paths, cost_vector, weights, 0.0) >= threshold
    while np.any(hi - lo > 1):
        mid = (lo + hi) // 2
        ok = log_phi(log_paths, cost_vector, weights, mid) >= threshold
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return np.where(feasible, lo, -1)


def budget_at(
    log_levels: np.ndarray,
    cost_vector: np.ndarray,
    weights: np.ndarray,
    threshold_value: float,
    cap: int,
) -> int:
    """Largest integer budget at one time point, paper eq. (9)."""
    levels = log_levels[:, None]

    def feasible(budget: int) -> bool:
        return (
            float(log_phi(levels, cost_vector, weights, budget)[0]) >= threshold_value
        )

    if not feasible(0):
        return -1
    low, high = 0, cap
    while high - low > 1:
        middle = (low + high) // 2
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low
