from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .core import (
    Grid,
    budget_at,
    exp_costs,
    exp_log_wealth,
    grid_costs,
    log_phi,
    log_wealth,
    threshold,
)
from .errors import (
    validated_alpha,
    validated_cost,
    validated_outcomes,
    validated_variants,
)

Cost = Literal["insertion", "replacement"]
Claim = Literal["edge", "lead"]


@dataclass(frozen=True)
class Certificate:
    claim: Claim
    cost: Cost
    alpha: float
    n: int
    tolerance: int
    certified_at: int
    log_e_value: float
    selection_variants: int


def _paths(outcomes, lams, *, exponential: bool) -> np.ndarray:
    builder = exp_log_wealth if exponential else log_wealth
    paths = np.empty((lams.size, outcomes.size))
    for index, lam in enumerate(lams):
        paths[index] = builder(outcomes, float(lam))
    return paths


def _certificate(
    claim: Claim,
    cost: Cost,
    alpha: float,
    weights: np.ndarray,
    cost_vector: np.ndarray,
    log_paths: np.ndarray,
    selection_variants: int,
    horizon: int | None = None,
) -> Certificate:
    threshold_value = threshold(alpha, selection_variants)
    live = log_phi(log_paths, cost_vector, weights, 0.0) >= threshold_value
    hits = np.flatnonzero(live)
    n = int(log_paths.shape[1]) if horizon is None else horizon
    tolerance = -1
    if live.size and live[-1]:
        tolerance = budget_at(
            log_paths[:, -1], cost_vector, weights, threshold_value, n
        )
    log_e_value = (
        float(log_phi(log_paths[:, -1:], cost_vector, weights, max(tolerance, 0))[0])
        if log_paths.shape[1]
        else 0.0
    )
    return Certificate(
        claim=claim,
        cost=cost,
        alpha=alpha,
        n=n,
        tolerance=tolerance,
        certified_at=int(hits[0]) + 1 if hits.size else -1,
        log_e_value=log_e_value,
        selection_variants=selection_variants,
    )


def _normalized(
    grid: Grid | None, alpha: float, cost: Cost, selection_variants: int
) -> tuple[Grid, float, Cost, int]:
    return (
        grid if grid is not None else Grid.default(),
        validated_alpha(alpha),
        validated_cost(cost),
        validated_variants(selection_variants),
    )


def certify_edge(
    outcomes: np.ndarray,
    *,
    alpha: float = 0.05,
    cost: Cost = "replacement",
    grid: Grid | None = None,
    selection_variants: int = 1,
) -> Certificate:
    """Deflated capped betting, paper eq. (8)-(9), Theorem 5.1."""
    x = validated_outcomes(outcomes)
    grid, alpha, cost, variants = _normalized(grid, alpha, cost, selection_variants)
    cost_vector = grid_costs(grid, cost)
    log_paths = np.vstack([log_wealth(x, float(lam)) for lam in grid.lams])
    return _certificate(
        "edge", cost, alpha, grid.weights, cost_vector, log_paths, variants
    )


def certify_lead(
    outcomes: np.ndarray,
    *,
    alpha: float = 0.05,
    cost: Cost = "replacement",
    grid: Grid | None = None,
    selection_variants: int = 1,
) -> Certificate:
    """Deflated exponential wealth, paper eq. (10), Theorem 5.7."""
    x = validated_outcomes(outcomes)
    grid, alpha, cost, variants = _normalized(grid, alpha, cost, selection_variants)
    index = 0 if cost == "insertion" else 1
    cost_vector = np.array([exp_costs(float(lam))[index] for lam in grid.lams])
    log_paths = _paths(x, grid.lams, exponential=True)
    return _certificate(
        "lead", cost, alpha, grid.weights, cost_vector, log_paths, variants
    )


class _StreamingCertificate:
    _claim: Claim

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        cost: Cost = "replacement",
        grid: Grid | None = None,
        selection_variants: int = 1,
    ):
        (
            self._grid,
            self._alpha,
            self._cost,
            self._variants,
        ) = _normalized(grid, alpha, cost, selection_variants)
        self._log_wealth = np.zeros(len(self._grid.lams))
        self._t = 0
        self._certified_at = -1

    def update(self, outcome: float) -> None:
        x = float(validated_outcomes(outcome))
        self._log_wealth = self._log_wealth + self._log_step(x)
        self._t += 1
        if self._certified_at < 0:
            live = log_phi(
                self._log_wealth[:, None],
                self._cost_vector(),
                self._grid.weights,
                0.0,
            )
            if live[0] >= threshold(self._alpha, self._variants):
                self._certified_at = self._t

    def certificate(self) -> Certificate:
        certificate = _certificate(
            self._claim,
            self._cost,
            self._alpha,
            self._grid.weights,
            self._cost_vector(),
            self._log_wealth[:, None],
            self._variants,
            horizon=self._t,
        )
        return replace(certificate, certified_at=self._certified_at)


class EdgeCertificate(_StreamingCertificate):
    _claim = "edge"

    def _log_step(self, x: float) -> np.ndarray:
        return np.log1p(self._grid.lams * (x - 0.5))

    def _cost_vector(self) -> np.ndarray:
        return grid_costs(self._grid, self._cost)


class LeadCertificate(_StreamingCertificate):
    _claim = "lead"

    def _log_step(self, x: float) -> np.ndarray:
        return self._grid.lams * (x - 0.5) - self._grid.lams**2 / 8.0

    def _cost_vector(self) -> np.ndarray:
        index = 0 if self._cost == "insertion" else 1
        return np.array([exp_costs(float(lam))[index] for lam in self._grid.lams])
