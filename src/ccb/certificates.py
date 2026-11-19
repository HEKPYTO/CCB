from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .core import (
    Grid,
    budget_path,
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
    path = budget_path(log_paths, cost_vector, weights, threshold_value, cap=horizon)
    hits = np.flatnonzero(path >= 0)
    tolerance = int(path[-1]) if path.size else -1
    probe = max(tolerance, 0)
    log_e_value = (
        float(log_phi(log_paths, cost_vector, weights, probe)[-1]) if path.size else 0.0
    )
    return Certificate(
        claim=claim,
        cost=cost,
        alpha=alpha,
        n=int(log_paths.shape[1]) if horizon is None else horizon,
        tolerance=tolerance,
        certified_at=int(hits[0]) + 1 if hits.size else -1,
        log_e_value=log_e_value,
        selection_variants=selection_variants,
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
    grid = grid if grid is not None else Grid.default()
    alpha = validated_alpha(alpha)
    cost = validated_cost(cost)
    variants = validated_variants(selection_variants)
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
    grid = grid if grid is not None else Grid.default()
    alpha = validated_alpha(alpha)
    cost = validated_cost(cost)
    variants = validated_variants(selection_variants)
    index = 0 if cost == "insertion" else 1
    cost_vector = np.array([exp_costs(float(lam))[index] for lam in grid.lams])
    log_paths = np.vstack([exp_log_wealth(x, float(lam)) for lam in grid.lams])
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
        self._grid = grid if grid is not None else Grid.default()
        self._alpha = validated_alpha(alpha)
        self._cost = validated_cost(cost)
        self._variants = validated_variants(selection_variants)
        self._log_wealth = np.zeros(len(self._grid.lams))
        self._t = 0
        self._certified_at = -1

    def update(self, outcome: float) -> None:
        x = float(validated_outcomes(outcome))
        self._log_wealth = self._log_wealth + self._log_step(x)
        self._t += 1
        if self._certified_at < 0 and self.certificate().tolerance >= 0:
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
