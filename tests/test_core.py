import numpy as np
import pytest

from ccb.core import Grid, budget_path, costs, grid_costs, log_phi, log_wealth


def test_costs_match_closed_form():
    insertion, replacement = costs(1.0)
    assert insertion == pytest.approx(np.log(1.5), abs=1e-12)
    assert replacement == pytest.approx(np.log(3.0), abs=1e-12)


def test_costs_match_formula_at_small_lambda():
    insertion, replacement = costs(0.2)
    assert insertion == pytest.approx(np.log1p(0.1), abs=1e-15)
    assert replacement == pytest.approx(np.log(2.2 / 1.8), abs=1e-15)


def test_log_wealth_ties_are_flat():
    assert np.array_equal(log_wealth(np.full(5, 0.5), 1.0), np.zeros(5))


def test_log_wealth_known_values():
    path = log_wealth(np.array([1.0, 0.0, 0.5]), 1.0)
    assert path == pytest.approx([np.log(1.5), np.log(0.75), np.log(0.75)])


def test_grid_shape():
    grid = Grid.default()
    assert grid.lams.shape == (14,)
    assert grid.weights.shape == (14,)
    assert grid.weights.sum() == pytest.approx(1.0)


def test_grid_endpoints():
    grid = Grid.default()
    assert grid.lams[0] == 1.0
    assert grid.lams[-1] == pytest.approx(2**-6.5)


def test_budget_path_is_largest_feasible():
    outcomes = (np.random.default_rng(1).random(200) < 0.7).astype(float)
    grid = Grid.default()
    cost_vector = grid_costs(grid, "replacement")
    paths = np.vstack([log_wealth(outcomes, lam) for lam in grid.lams])
    threshold = np.log(20.0)
    path = budget_path(paths, cost_vector, grid.weights, threshold)
    budget = int(path[-1])
    assert log_phi(paths, cost_vector, grid.weights, budget)[-1] >= threshold
    assert log_phi(paths, cost_vector, grid.weights, budget + 1)[-1] < threshold


def test_budget_path_never_exceeds_t():
    outcomes = np.ones(1000)
    grid = Grid.default()
    path = budget_path(
        np.vstack([log_wealth(outcomes, lam) for lam in grid.lams]),
        grid_costs(grid, "insertion"),
        grid.weights,
        np.log(20.0),
    )
    assert path[-1] <= 1000


def test_empty_stream_has_no_budget():
    grid = Grid.default()
    path = budget_path(
        np.zeros((14, 0)),
        grid_costs(grid, "replacement"),
        grid.weights,
        np.log(20.0),
    )
    assert path.size == 0
