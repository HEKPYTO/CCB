import json
from pathlib import Path

import numpy as np
import pytest

from ccb.certificates import certify_edge
from ccb.core import selection_correction
from ccb.leaderboard import bonferroni, closed_test

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "leaderboard_matrix.json").read_text()
)
STREAMS = {
    tuple(key.split(">")): np.array(values, dtype=float)
    for key, values in GOLDEN["streams"].items()
}
EXPECTED = {("a", "b"): 0, ("b", "c"): 1, ("a", "c"): 2}


def _by_pair(results):
    return {(result.model_a, result.model_b): result.tolerance for result in results}


def test_closed_matches_reference_small():
    for cost in ("insertion", "replacement"):
        reference = GOLDEN["closed"][cost]
        expected = {
            pair: reference[EXPECTED[pair]]
            for pair in (("a", "b"), ("b", "c"), ("a", "c"))
        }
        assert _by_pair(closed_test(STREAMS, cost=cost)) == expected


def test_bonferroni_matches_reference_small():
    for cost in ("insertion", "replacement"):
        reference = GOLDEN["bonferroni"][cost]
        expected = {
            pair: reference[EXPECTED[pair]]
            for pair in (("a", "b"), ("b", "c"), ("a", "c"))
        }
        assert _by_pair(bonferroni(STREAMS, cost=cost)) == expected


def test_bonferroni_is_conservative():
    for pair, values in STREAMS.items():
        single = certify_edge(values).tolerance
        family = _by_pair(bonferroni(STREAMS))[pair]
        assert family <= single


def test_closed_beats_bonferroni():
    closed = _by_pair(closed_test(STREAMS))
    family = _by_pair(bonferroni(STREAMS))
    for pair in closed:
        assert closed[pair] >= family[pair]


def test_pair_order_is_sorted():
    results = closed_test(STREAMS)
    assert [(result.model_a, result.model_b) for result in results] == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def test_all_ties_are_uncertified():
    ties = {
        ("a", "b"): np.full(10, 0.5),
        ("b", "c"): np.full(10, 0.5),
    }
    assert all(result.tolerance == -1 for result in closed_test(ties))
    assert all(result.tolerance == -1 for result in bonferroni(ties))


def test_unequal_streams_use_common_horizon():
    streams = {
        ("a", "b"): np.ones(2000),
        ("b", "c"): np.ones(1000),
    }
    assert {result.n for result in closed_test(streams)} == {1000}


def test_both_directions_doubles_claims():
    results = closed_test(STREAMS, both_directions=True)
    assert len(results) == 2 * len(STREAMS)
    reverse = _by_pair(bonferroni(STREAMS, both_directions=True))
    assert reverse[("b", "a")] <= reverse[("a", "b")]


def test_selection_log_v():
    assert selection_correction(1) == 0.0
    assert selection_correction(27) == pytest.approx(np.log(27.0))


def test_selection_matches_alpha_division():
    values = STREAMS[("a", "b")]
    corrected = certify_edge(values, selection_variants=27)
    divided = certify_edge(values, alpha=0.05 / 27)
    assert corrected.tolerance == divided.tolerance


def test_selection_lowers_budget():
    values = STREAMS[("a", "b")]
    assert (
        certify_edge(values, selection_variants=27).tolerance
        < certify_edge(values).tolerance
    )
