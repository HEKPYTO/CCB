import json
from pathlib import Path

import numpy as np
import pytest

from ccb.certificates import (
    EdgeCertificate,
    LeadCertificate,
    certify_edge,
    certify_lead,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "p70_1000_seed0.json").read_text()
)
P70 = np.array(GOLDEN["stream"], dtype=float)


def test_all_win_1000():
    outcomes = np.ones(1000)
    assert certify_edge(outcomes, cost="insertion").tolerance == 987
    assert certify_edge(outcomes).tolerance == 452
    assert certify_edge(outcomes).certified_at == 13


def test_all_win_100():
    outcomes = np.ones(100)
    assert certify_edge(outcomes, cost="insertion").tolerance == 87
    assert certify_edge(outcomes).tolerance == 34


def test_p70_1000_seed0():
    assert certify_edge(P70, cost="insertion").tolerance == GOLDEN["edge"]["insertion"]
    assert certify_edge(P70).tolerance == GOLDEN["edge"]["replacement"]
    assert certify_edge(P70).certified_at == GOLDEN["certified_at"]


def test_lead_matches_reference():
    assert certify_lead(np.ones(1000), cost="insertion").tolerance == 986
    assert certify_lead(np.ones(1000)).tolerance == 451
    assert certify_lead(np.ones(100), cost="insertion").tolerance == 86
    assert certify_lead(np.ones(100)).tolerance == 34
    assert certify_lead(P70, cost="insertion").tolerance == GOLDEN["lead"]["insertion"]
    assert certify_lead(P70).tolerance == GOLDEN["lead"]["replacement"]


def test_all_ties_is_uncertified():
    ties = np.full(10, 0.5)
    assert certify_edge(ties).tolerance == -1
    assert certify_edge(ties).certified_at == -1
    assert certify_lead(ties).tolerance == -1


def test_short_stream_is_uncertified():
    assert certify_edge(np.ones(5)).tolerance == -1


def test_empty_stream_is_uncertified():
    certificate = certify_edge(np.array([]))
    assert certificate.tolerance == -1
    assert certificate.n == 0


def test_monotone_in_wins():
    base = certify_edge(P70[:200]).tolerance
    win = certify_edge(np.append(P70[:200], 1.0)).tolerance
    assert win >= base
    assert win >= certify_edge(np.append(P70[:200], 0.0)).tolerance


def test_monotone_in_alpha():
    assert certify_edge(P70, alpha=0.01).tolerance <= certify_edge(P70).tolerance


def test_replacement_tolerance_le_insertion():
    assert certify_edge(P70).tolerance <= certify_edge(P70, cost="insertion").tolerance


def test_selection_correction_lowers_budget():
    assert (
        certify_edge(P70, selection_variants=27).tolerance
        < certify_edge(P70, selection_variants=1).tolerance
    )


def test_streaming_matches_batch():
    outcomes = (np.random.default_rng(7).random(10_000) < 0.65).astype(float)
    certificate = EdgeCertificate()
    for index, outcome in enumerate(outcomes, start=1):
        certificate.update(outcome)
        if index % 1_000 == 0:
            assert certificate.certificate() == certify_edge(outcomes[:index])


def test_lead_streaming_matches_batch():
    certificate = LeadCertificate()
    for index, outcome in enumerate(P70, start=1):
        certificate.update(outcome)
        if index in (500, 1000):
            assert certificate.certificate() == certify_lead(P70[:index])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        certify_edge(P70, alpha=alpha)


@pytest.mark.parametrize("outcome", [-0.1, 1.1, np.nan])
def test_invalid_outcome(outcome):
    with pytest.raises(ValueError):
        certify_edge(np.array([1.0, outcome]))


@pytest.mark.parametrize("variants", [0, -3])
def test_non_positive_selection_variants(variants):
    with pytest.raises(ValueError):
        certify_edge(P70, selection_variants=variants)


@pytest.mark.parametrize("variants", [2.5, True])
def test_non_integer_selection_variants(variants):
    with pytest.raises(TypeError):
        certify_edge(P70, selection_variants=variants)


@pytest.mark.slow
def test_lead_validity():
    rng = np.random.default_rng(0)
    null = (rng.random(5000) < 0.5).astype(float)
    hits = 0
    for _ in range(5000):
        sample = rng.choice(null, size=null.size, replace=True)
        hits += certify_lead(sample).tolerance >= 0
    assert hits / 5000 <= 0.06


@pytest.mark.slow
def test_alpha_001():
    rng = np.random.default_rng(0)
    null = (rng.random(1000) < 0.5).astype(float)
    hits = 0
    for _ in range(5000):
        sample = rng.choice(null, size=null.size, replace=True)
        hits += certify_edge(sample, alpha=0.01).tolerance >= 0
    assert hits / 5000 <= 0.016
