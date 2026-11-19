import time

import numpy as np
import pytest

from ccb.certificates import certify_edge
from ccb.leaderboard import closed_test


@pytest.mark.performance
def test_batch_certificate_1m_under_2s():
    outcomes = (np.random.default_rng(0).random(1_000_000) < 0.7).astype(np.float64)
    start = time.perf_counter()
    certificate = certify_edge(outcomes)
    elapsed = time.perf_counter() - start
    assert certificate.tolerance >= 0
    assert elapsed < 2.0


@pytest.mark.performance
def test_leaderboard_1000_pairs_under_5s():
    streams = {
        (f"m{index}", f"m{index + 1}"): (
            np.random.default_rng(index).random(1000) < 0.55
        ).astype(np.float64)
        for index in range(1000)
    }
    start = time.perf_counter()
    results = closed_test(streams)
    elapsed = time.perf_counter() - start
    assert len(results) == 1000
    assert elapsed < 5.0
