from dataclasses import dataclass

import numpy as np

from .core import Grid, budget_path, grid_costs, log_phi, log_wealth_final
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
                [log_wealth_final(values[:horizon], float(lam)) for lam in grid.lams]
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


def _closed_min(levels: np.ndarray) -> np.ndarray:
    """Minimum closed-test average per claim at one budget, Proposition 5.6."""
    count = levels.size
    order = np.argsort(levels)
    ranked = levels[order]
    ranks = np.empty(count, dtype=np.int64)
    ranks[order] = np.arange(count)
    prefix = np.concatenate([[-np.inf], np.logaddexp.accumulate(ranked)])
    k = np.arange(count)
    with_own = np.logaddexp(levels[:, None], prefix[None, :count]) - np.log(k + 1)
    without = prefix[None, 1:] - np.log(k + 1)
    return np.where(k[None, :] < ranks[:, None], with_own, without).min(axis=1)


def _levels(family: _Family, budget: int) -> np.ndarray:
    return log_phi(family.log_e_values, family.cost_vector, family.grid.weights, budget)


def _closed_budgets(family: _Family, log_threshold: float) -> np.ndarray:
    count = len(family.claims)
    lo = np.zeros(count, dtype=np.int64)
    hi = np.full(count, family.horizon, dtype=np.int64)
    feasible = _closed_min(_levels(family, 0)) >= log_threshold
    while np.any(hi - lo > 1):
        middle = (lo + hi) // 2
        verdict = np.zeros(count, dtype=bool)
        for budget in np.unique(middle):
            chosen = middle == budget
            verdict[chosen] = (
                _closed_min(_levels(family, int(budget)))[chosen] >= log_threshold
            )
        lo = np.where(verdict, middle, lo)
        hi = np.where(verdict, hi, middle)
    return np.where(feasible, lo, -1)


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
    return _results(family, _closed_budgets(family, log_threshold), "closed")


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
