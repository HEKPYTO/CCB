import numpy as np
import pytest

from ccb.attacks import (
    attack_burst,
    attack_burst_random,
    attack_flip,
    attack_spread,
    attack_suppress,
)

STREAM = np.array([1, 0, 1, 0.5, 0, 1, 0, 1, 1, 0], dtype=float)


def test_burst_adds_budget_wins():
    observed, flags = attack_burst(STREAM, 3)
    assert observed.size == STREAM.size + 3
    assert flags.sum() == 3
    assert flags[:3].all()
    assert np.array_equal(observed[3:], STREAM)


def test_burst_random_is_one_block():
    observed, flags = attack_burst_random(STREAM, 4, np.random.default_rng(0))
    assert observed.size == STREAM.size + 4
    block = np.flatnonzero(flags)
    assert block.size == 4
    assert block[-1] - block[0] == 3


def test_flip_only_touches_losses():
    observed, flags = attack_flip(STREAM, 2)
    assert flags.sum() == 2
    assert np.all(STREAM[flags] < 0.5)
    assert np.all(observed[flags] == 1.0)
    assert np.array_equal(observed[~flags], STREAM[~flags])


def test_flip_caps_at_available_losses():
    observed, flags = attack_flip(STREAM, 100)
    assert flags.sum() == int((STREAM < 0.5).sum())
    assert np.array_equal(observed, np.where(STREAM < 0.5, 1.0, STREAM))


def test_spread_respects_cap_and_matches_shape():
    observed, flags = attack_spread(STREAM, 3, np.random.default_rng(0))
    assert flags.sum() <= 3
    assert flags.sum() > 0
    assert observed.size == STREAM.size + flags.sum()


def test_suppress_drops_wins():
    observed, dropped = attack_suppress(STREAM, 2)
    assert dropped.sum() == 2
    assert np.all(STREAM[dropped] > 0.5)
    assert observed.size == STREAM.size - 2


def test_random_attacks_are_deterministic():
    first = attack_burst_random(STREAM, 3, np.random.default_rng(7))
    second = attack_burst_random(STREAM, 3, np.random.default_rng(7))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    spread_a = attack_spread(STREAM, 3, np.random.default_rng(7))
    spread_b = attack_spread(STREAM, 3, np.random.default_rng(7))
    assert np.array_equal(spread_a[0], spread_b[0])
    assert np.array_equal(spread_a[1], spread_b[1])


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        attack_burst(STREAM, -1)
