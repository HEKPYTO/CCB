import numpy as np

from .core import Grid, log_phi, log_wealth, validated_outcomes
from .errors import CCBError


def wald_violation(observed, every: int = 50) -> np.ndarray:
    """Checkpoints where a 95% Wald lower bound on the win rate exceeds 1/2."""
    values = validated_outcomes(observed)
    if every < 1:
        raise ValueError("every must be a positive integer")
    if values.size < every:
        return np.zeros(0, dtype=bool)
    checkpoints = np.arange(every, values.size + 1, every)
    mean = np.cumsum(values)[checkpoints - 1] / checkpoints
    half_width = 1.96 * np.sqrt(mean * (1.0 - mean) / checkpoints)
    return mean - half_width > 0.5


def naive_violation(
    observed, grid: Grid | None = None, alpha: float = 0.05
) -> np.ndarray:
    """Undeflated grid wealth above 1/alpha at each t, paper Section 6 baseline."""
    values = validated_outcomes(observed)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise CCBError("invalid_alpha", "alpha must be in (0, 1)")
    grid = grid if grid is not None else Grid.default()
    paths = np.vstack([log_wealth(values, float(lam)) for lam in grid.lams])
    live = log_phi(paths, np.zeros(len(grid.lams)), grid.weights, 0.0)
    return live >= np.log(1.0 / alpha)


def per_step_violation(
    observed, budget: int, records: int | None = None, alpha: float = 0.05
) -> np.ndarray:
    """Per-step robust test with eps = budget/records and a summable schedule."""
    values = validated_outcomes(observed)
    nominal = values.size if records is None else records
    if nominal < 1:
        raise ValueError("records must be positive")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise CCBError("invalid_alpha", "alpha must be in (0, 1)")
    eps = budget / nominal
    t = np.arange(1, values.size + 1)
    schedule = alpha * 6.0 / (np.pi**2 * t**2)
    width = np.sqrt(np.log(1.0 / schedule) / (2.0 * t))
    mean = np.cumsum(values) / t
    return mean - eps > 0.5 + width
