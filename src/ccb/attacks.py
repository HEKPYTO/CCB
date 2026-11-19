import numpy as np

from .errors import validated_outcomes


def _budget(budget) -> int:
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError("budget must be a non-negative integer")
    if budget < 0:
        raise ValueError("budget must be a non-negative integer")
    return int(budget)


def attack_burst(stream, budget) -> tuple[np.ndarray, np.ndarray]:
    """Prefix burst of forged wins, paper Section 6.1 panel (a)."""
    values = validated_outcomes(stream)
    budget = _budget(budget)
    observed = np.concatenate([np.ones(budget), values])
    flags = np.concatenate(
        [np.ones(budget, dtype=bool), np.zeros(values.size, dtype=bool)]
    )
    return observed, flags


def attack_burst_random(stream, budget, rng) -> tuple[np.ndarray, np.ndarray]:
    """One burst of forged wins at a uniformly random time, paper Table 1."""
    values = validated_outcomes(stream)
    budget = _budget(budget)
    position = int(rng.integers(0, values.size + 1))
    observed = np.concatenate([values[:position], np.ones(budget), values[position:]])
    flags = np.concatenate(
        [
            np.zeros(position, dtype=bool),
            np.ones(budget, dtype=bool),
            np.zeros(values.size - position, dtype=bool),
        ]
    )
    return observed, flags


def attack_flip(stream, budget) -> tuple[np.ndarray, np.ndarray]:
    """Flip the first losses to wins, paper Section 6.1 panel (b)."""
    values = validated_outcomes(stream)
    budget = _budget(budget)
    observed = values.copy()
    flags = np.zeros(values.size, dtype=bool)
    losses = np.flatnonzero(values < 0.5)[:budget]
    observed[losses] = 1.0
    flags[losses] = True
    return observed, flags


def attack_spread(stream, budget, rng) -> tuple[np.ndarray, np.ndarray]:
    """One forged win before each record with probability budget/n, paper panel (c)."""
    values = validated_outcomes(stream)
    budget = _budget(budget)
    n = values.size
    if n == 0:
        return values.copy(), np.zeros(0, dtype=bool)
    draws = rng.random(n) < min(budget / n, 1.0)
    draws &= np.cumsum(draws) <= budget
    preceding = np.concatenate([[0], np.cumsum(draws)[:-1]])
    observed = np.empty(n + int(draws.sum()))
    observed[np.arange(n) + np.cumsum(draws)] = values
    inserted = np.flatnonzero(draws)
    slots = inserted + preceding[inserted]
    observed[slots] = 1.0
    flags = np.zeros(observed.size, dtype=bool)
    flags[slots] = True
    return observed, flags


def attack_suppress(stream, budget) -> tuple[np.ndarray, np.ndarray]:
    """Drop uncorrupted wins. Experimental; withholding is outside the guarantee."""
    values = validated_outcomes(stream)
    budget = _budget(budget)
    dropped = np.zeros(values.size, dtype=bool)
    wins = np.flatnonzero(values > 0.5)[:budget]
    dropped[wins] = True
    return values[~dropped], dropped
