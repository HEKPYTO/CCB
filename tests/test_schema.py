import json
from pathlib import Path

import jsonschema
import pytest

from ccb.cli import main

SCHEMA_DIR = Path(__file__).parent.parent / "schema"
ENVELOPE = json.loads((SCHEMA_DIR / "envelope-v1.schema.json").read_text())
CERTIFICATE = json.loads((SCHEMA_DIR / "certificate-v1.schema.json").read_text())
FIXTURE = str(Path(__file__).parent / "fixtures" / "records.jsonl")


def _envelope(argv, capsys):
    assert main(argv + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(ENVELOPE).validate(payload)
    return payload


def test_certify_envelope_schema(capsys):
    payload = _envelope(["certify", "--input", FIXTURE], capsys)
    jsonschema.Draft202012Validator(CERTIFICATE).validate(
        payload["data"]["certificate"]
    )


def test_leaderboard_envelope_schema(capsys):
    payload = _envelope(["leaderboard", "--input", FIXTURE], capsys)
    assert payload["data"]["claims"]


def test_replay_envelope_schema(capsys):
    payload = _envelope(
        ["replay", "--input", FIXTURE, "--replays", "10", "--budget", "1"], capsys
    )
    assert set(payload["data"]["rates"]) == {
        "wald",
        "naive",
        "per_step",
        "insertion",
        "replacement",
    }


def test_error_envelope_schema(capsys):
    assert main(["certify", "--input", FIXTURE, "--alpha", "nan", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    jsonschema.Draft202012Validator(ENVELOPE).validate(payload)
    assert payload["error"]["code"] == "invalid_alpha"
    assert payload["data"] is None


def test_certificate_schema_rejects_bad_tolerance():
    certificate = {
        "claim": "edge",
        "cost": "replacement",
        "alpha": 0.05,
        "n": 10,
        "tolerance": -2,
        "certified_at": -1,
        "selection_variants": 1,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(CERTIFICATE).validate(certificate)
