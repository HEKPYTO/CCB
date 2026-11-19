from dataclasses import dataclass

import numpy as np

from .core import Grid, budget_path, grid_costs, log_phi, log_wealth
from .errors import CCBError, validated_alpha, validated_cost, validated_outcomes


@dataclass(frozen=True)
class PairResult:
    model_a: str
    model_b: str
    n: int
    tolerance: int
    certified_at: int
    claim: str
    cost: str
    alpha: float
    method: str
    status: str


@dataclass(frozen=True)
class _Family:
    claims: list
    horizon: int
    grid: Grid
    cost: str
    alpha: float
    cost_vector: np.ndarray
    log_e_values: np.ndarray | None


def _claims(streams, both_directions: bool) -> list:
    if not streams:
        raise CCBError("empty_input", "no pairs")
    claims = []
    for (model_a, model_b), values in streams.items():
        outcomes = validated_outcomes(values)
        claims.append((model_a, model_b, outcomes))
        if both_directions:
            claims.append((model_b, model_a, 1.0 - outcomes))
    return claims


def _prepare(streams, alpha, cost, grid, both_directions) -> _Family:
    grid = grid if grid is not None else Grid.default()
    cost = validated_cost(cost)
    claims = _claims(streams, both_directions)
    horizon = min(values.size for _, _, values in claims)
    if horizon == 0:
        log_e_values = None
    else:
        log_e_values = np.array(
            [
                [log_wealth(values[:horizon], float(lam))[-1] for lam in grid.lams]
                for _, _, values in claims
            ]
        ).T
    return _Family(
        claims=claims,
        horizon=horizon,
        grid=grid,
        cost=cost,
        alpha=validated_alpha(alpha),
        cost_vector=grid_costs(grid, cost),
        log_e_values=log_e_values,
    )


def _closed_log_average(levels: np.ndarray, index: int) -> float:
    others = np.sort(np.delete(levels, index))
    prefix = np.concatenate([[-np.inf], np.logaddexp.accumulate(others)])
    k = np.arange(prefix.size)
    return float(np.min(np.logaddexp(levels[index], prefix) - np.log(k + 1)))


def _budget(family: _Family, index: int, log_threshold: float) -> int:
    def feasible(budget: int) -> bool:
        levels = log_phi(
            family.log_e_values,
            family.cost_vector,
            family.grid.weights,
            budget,
        )
        return _closed_log_average(levels, index) >= log_threshold

    if not feasible(0):
        return -1
    low, high = 0, family.horizon
    while high - low > 1:
        middle = (low + high) // 2
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low


def _results(family: _Family, tolerances, method: str) -> list[PairResult]:
    results = [
        PairResult(
            model_a=model_a,
            model_b=model_b,
            n=family.horizon,
            tolerance=int(tolerance),
            certified_at=family.horizon if tolerance >= 0 else -1,
            claim="edge",
            cost=family.cost,
            alpha=family.alpha,
            method=method,
            status="certified" if tolerance >= 0 else "not_certified",
        )
        for (model_a, model_b, _), tolerance in zip(family.claims, tolerances)
    ]
    return sorted(results, key=lambda result: (result.model_a, result.model_b))


def closed_test(
    streams,
    *,
    alpha: float = 0.05,
    cost: str = "replacement",
    grid: Grid | None = None,
    both_directions: bool = False,
) -> list[PairResult]:
    """Closed testing on deflated wealths, Proposition 5.6."""
    family = _prepare(streams, alpha, cost, grid, both_directions)
    if family.log_e_values is None:
        return _results(family, np.full(len(family.claims), -1), "closed")
    log_threshold = np.log(1.0 / family.alpha)
    tolerances = [
        _budget(family, index, log_threshold) for index in range(len(family.claims))
    ]
    return _results(family, tolerances, "closed")


def bonferroni(
    streams,
    *,
    alpha: float = 0.05,
    cost: str = "replacement",
    grid: Grid | None = None,
    both_directions: bool = False,
) -> list[PairResult]:
    """Bonferroni correction across the claim family, paper Section 5.2."""
    family = _prepare(streams, alpha, cost, grid, both_directions)
    if family.log_e_values is None:
        return _results(family, np.full(len(family.claims), -1), "bonferroni")
    log_threshold = np.log(len(family.claims) / family.alpha)
    tolerances = budget_path(
        family.log_e_values,
        family.cost_vector,
        family.grid.weights,
        log_threshold,
        cap=family.horizon,
    )
    return _results(family, tolerances, "bonferroni")
