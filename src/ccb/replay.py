from dataclasses import dataclass

import numpy as np

from .attacks import (
    attack_burst,
    attack_burst_random,
    attack_flip,
    attack_spread,
)
from .baselines import per_step_violation, undeflated_violation, wald_violation
from .core import Grid, grid_costs, log_phi, log_wealth
from .errors import CCBError, validated_outcomes

METHODS = ("wald", "naive", "per_step", "insertion", "replacement")
DEFAULT_BUDGETS = (0, 100, 400)
ATTACKS = ("burst", "burst_random", "flip", "spread")


@dataclass(frozen=True)
class ReplayResult:
    attack: str
    budgets: list[int]
    replays: int
    seed: int
    records: int
    alpha: float
    rates: dict[str, dict[int, float]]
    se: dict[str, dict[int, float]]

    @property
    def methods(self) -> list[str]:
        return list(self.rates)

    def to_dict(self) -> dict:
        return {
            "attack": self.attack,
            "records": self.records,
            "replays": self.replays,
            "seed": self.seed,
            "alpha": self.alpha,
            "budgets": self.budgets,
            "methods": self.methods,
            "rates": {
                method: {str(budget): rate for budget, rate in values.items()}
                for method, values in self.rates.items()
            },
            "se": {
                method: {str(budget): error for budget, error in values.items()}
                for method, values in self.se.items()
            },
        }


def _int(value, name: str, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise CCBError("usage", f"{name} must be an integer >= {minimum}")
    return int(value)


def _attack(
    base: np.ndarray, name: str, budget: int, rng
) -> tuple[np.ndarray, np.ndarray]:
    if name == "burst":
        return attack_burst(base, budget)
    if name == "burst_random":
        return attack_burst_random(base, budget, rng)
    if name == "flip":
        return attack_flip(base, budget)
    return attack_spread(base, budget, rng)


def replay(
    stream,
    *,
    attack: str = "burst",
    budgets=DEFAULT_BUDGETS,
    replays: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> ReplayResult:
    """Resampled false-statement replay, paper Section 6.1."""
    values = validated_outcomes(stream)
    if values.size == 0:
        raise CCBError("empty_input", "no records")
    if attack not in ATTACKS:
        raise CCBError("usage", f"unknown attack: {attack}")
    budgets = list(dict.fromkeys(_int(budget, "budget", 0) for budget in budgets))
    if not budgets:
        raise CCBError("usage", "at least one budget is required")
    replays = _int(replays, "replays", 1)
    seed = _int(seed, "seed", 0)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise CCBError("invalid_alpha", "alpha must be in (0, 1)")

    rng = np.random.default_rng(seed)
    threshold = np.log(1.0 / alpha)
    grid = Grid.default()
    costs = {
        "insertion": grid_costs(grid, "insertion"),
        "replacement": grid_costs(grid, "replacement"),
    }
    hits = {method: dict.fromkeys(budgets, 0) for method in METHODS}

    for _ in range(replays):
        base = rng.choice(values, size=values.size, replace=True)
        for budget in budgets:
            observed, flags = _attack(base, attack, budget, rng)
            corruptions = np.cumsum(flags)
            paths = np.empty((len(grid.lams), observed.size))
            for index, lam in enumerate(grid.lams):
                paths[index] = log_wealth(observed, float(lam))
            verdicts = {
                "wald": bool(np.any(wald_violation(observed))),
                "naive": bool(np.any(undeflated_violation(paths, grid, alpha))),
                "per_step": bool(
                    np.any(per_step_violation(observed, budget, records=values.size))
                ),
            }
            for cost, vector in costs.items():
                verdicts[cost] = bool(
                    np.any(
                        log_phi(paths, vector, grid.weights, corruptions) >= threshold
                    )
                )
            for method, verdict in verdicts.items():
                hits[method][budget] += int(verdict)

    rates = {
        method: {budget: hits[method][budget] / replays for budget in budgets}
        for method in METHODS
    }
    se = {
        method: {
            budget: float(
                np.sqrt(
                    hits[method][budget] * (replays - hits[method][budget]) / replays
                )
                / replays
            )
            for budget in budgets
        }
        for method in METHODS
    }
    return ReplayResult(
        attack=attack,
        budgets=budgets,
        replays=replays,
        seed=seed,
        records=int(values.size),
        alpha=alpha,
        rates=rates,
        se=se,
    )
