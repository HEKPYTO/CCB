import numpy as np
import pytest

from ccb.certificates import certify_edge
from ccb.errors import CCBError


def test_zero_records():
    certificate = certify_edge(np.array([]))
    assert certificate.tolerance == -1
    assert certificate.n == 0
    assert certificate.certified_at == -1


def test_one_record():
    assert certify_edge(np.array([1.0])).tolerance == -1


def test_very_long_stream_100k_completes():
    outcomes = (np.random.default_rng(3).random(100_000) < 0.55).astype(np.float64)
    certificate = certify_edge(outcomes)
    assert 0 <= certificate.tolerance <= certificate.n


def test_infinite_outcome_rejected():
    with pytest.raises(CCBError) as excinfo:
        certify_edge(np.array([1.0, np.inf]))
    assert excinfo.value.code == "invalid_outcome"
