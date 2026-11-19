import json

import numpy as np
import pytest

from ccb.cli import main
from ccb.replay import replay

NULL = (np.random.default_rng(1).random(5000) < 0.5).astype(float)


@pytest.fixture(scope="module")
def flip_replay():
    return replay(NULL, attack="flip", budgets=[100], replays=5000, seed=0)


@pytest.fixture(scope="module")
def burst_replay():
    return replay(NULL, attack="burst", budgets=[100], replays=5000, seed=0)


def test_replay_smoke():
    result = replay(NULL, attack="flip", budgets=[100], replays=100, seed=0)
    assert result.methods == ["wald", "naive", "per_step", "insertion", "replacement"]
    assert result.records == 5000
    assert result.replays == 100
    for values in result.rates.values():
        assert 0.0 <= values[100] <= 1.0
    for values in result.se.values():
        assert values[100] >= 0.0


def test_replay_deterministic():
    first = replay(NULL, attack="burst", budgets=[0, 100], replays=50, seed=3)
    second = replay(NULL, attack="burst", budgets=[0, 100], replays=50, seed=3)
    assert first == second


def test_replay_deduplicates_budgets():
    result = replay(NULL, attack="flip", budgets=[100, 100, 0], replays=10, seed=0)
    assert result.budgets == [100, 0]


def test_replay_budget_zero_is_the_null_rate():
    result = replay(NULL, attack="flip", budgets=[0], replays=200, seed=0)
    assert result.rates["replacement"][0] <= 0.1


@pytest.mark.slow
def test_wald_breaks_under_flip(flip_replay):
    assert flip_replay.rates["wald"][100] > 0.5


@pytest.mark.slow
def test_replacement_certificate_survives_flip(flip_replay):
    assert flip_replay.rates["replacement"][100] <= 0.06


@pytest.mark.slow
def test_insertion_certificate_survives_burst(burst_replay):
    assert burst_replay.rates["insertion"][100] <= 0.06


@pytest.mark.slow
def test_insertion_certificate_fails_under_flip(flip_replay):
    assert flip_replay.rates["insertion"][100] > 0.5


def test_replay_json_envelope(tmp_path, capsys):
    path = tmp_path / "null.jsonl"
    path.write_text(
        "".join(
            f'{{"model_a":"a","model_b":"b","outcome":{outcome},"t":{index}}}\n'
            for index, outcome in enumerate([1, 0] * 250)
        )
    )
    argv = [
        "replay",
        "--input",
        str(path),
        "--attack",
        "flip",
        "--budget",
        "10",
        "--replays",
        "50",
        "--json",
    ]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "ccb/v1"
    assert payload["command"] == "replay"
    assert payload["error"] is None
    data = payload["data"]
    for key in ("methods", "budgets", "rates", "se"):
        assert key in data
    assert data["budgets"] == [10]
    assert data["seed"] == 0
    assert data["replays"] == 50


def test_replay_rejects_multiple_pairs(tmp_path, capsys):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"model_a":"a","model_b":"b","outcome":1}\n'
        '{"model_a":"c","model_b":"d","outcome":0}\n'
    )
    assert main(["replay", "--input", str(path), "--replays", "10"]) == 1
    assert "multiple_pairs" in capsys.readouterr().err


@pytest.mark.slow
def test_edge_validity_burst(burst_replay):
    assert burst_replay.rates["replacement"][100] <= 0.06
