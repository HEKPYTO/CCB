import json
from pathlib import Path

import numpy as np
import pytest

from ccb.errors import CCBError
from ccb.records import Record, load_records, pair_streams

FIXTURES = Path(__file__).parent / "fixtures"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def _code(call):
    with pytest.raises(CCBError) as excinfo:
        call()
    return excinfo.value.code


def test_jsonl_and_csv_parse_identically():
    from_jsonl = load_records(FIXTURES / "records.jsonl")
    from_csv = load_records(FIXTURES / "records.csv")
    assert from_jsonl == from_csv
    assert from_jsonl[0] == Record("a", "b", 1.0, 0, None)


def test_reverse_orientation_flips_outcome():
    streams = pair_streams(load_records(FIXTURES / "records.jsonl"))
    assert list(streams) == [("a", "b")]
    assert np.array_equal(streams[("a", "b")], np.array([1.0, 0.0, 1.0]))


def test_mixed_time_rejected(tmp_path):
    path = _write(
        tmp_path,
        "mixed.jsonl",
        '{"model_a":"a","model_b":"b","outcome":1}\n'
        '{"model_a":"a","model_b":"b","outcome":1,"t":1}\n',
    )
    assert _code(lambda: load_records(path)) == "mixed_time"


def test_same_model_rejected(tmp_path):
    path = _write(tmp_path, "same.jsonl", '{"model_a":"a","model_b":"a","outcome":1}\n')
    assert _code(lambda: load_records(path)) == "invalid_pair"


def test_empty_model_rejected(tmp_path):
    path = _write(
        tmp_path, "blank.jsonl", '{"model_a":" ","model_b":"b","outcome":1}\n'
    )
    assert _code(lambda: load_records(path)) == "invalid_pair"


@pytest.mark.parametrize("outcome", ["1.5", "-0.1", "NaN"])
def test_invalid_outcome_rejected(tmp_path, outcome):
    path = _write(
        tmp_path,
        "bad.jsonl",
        f'{{"model_a":"a","model_b":"b","outcome":{outcome}}}\n',
    )
    assert _code(lambda: load_records(path)) == "invalid_outcome"


def test_empty_input_rejected(tmp_path):
    path = _write(tmp_path, "empty.jsonl", "\n")
    assert _code(lambda: load_records(path)) == "empty_input"


def test_unknown_format_rejected(tmp_path):
    path = _write(tmp_path, "records.txt", "")
    assert _code(lambda: load_records(path)) == "unknown_format"


def test_parse_error_on_bad_json(tmp_path):
    path = _write(tmp_path, "bad.jsonl", "not json\n")
    assert _code(lambda: load_records(path)) == "parse_error"


def test_mixed_variants_rejected(tmp_path):
    path = _write(
        tmp_path,
        "variants.jsonl",
        '{"model_a":"a","model_b":"b","outcome":1,"variant":"v1"}\n'
        '{"model_a":"a","model_b":"b","outcome":0,"variant":"v2"}\n',
    )
    records = load_records(path)
    assert _code(lambda: pair_streams(records)) == "mixed_variants"


def test_variant_filter_keeps_one_variant(tmp_path):
    path = _write(
        tmp_path,
        "variants.jsonl",
        '{"model_a":"a","model_b":"b","outcome":1,"variant":"v1"}\n'
        '{"model_a":"a","model_b":"b","outcome":0,"variant":"v2"}\n',
    )
    streams = pair_streams(load_records(path), variant="v2")
    assert np.array_equal(streams[("a", "b")], np.array([0.0]))


def test_stable_order_for_equal_time(tmp_path):
    path = _write(
        tmp_path,
        "same_time.jsonl",
        '{"model_a":"a","model_b":"b","outcome":1,"t":0}\n'
        '{"model_a":"a","model_b":"b","outcome":0,"t":0}\n',
    )
    streams = pair_streams(load_records(path))
    assert np.array_equal(streams[("a", "b")], np.array([1.0, 0.0]))


def test_unknown_field_rejected(tmp_path):
    path = _write(
        tmp_path,
        "extra.jsonl",
        json.dumps({"model_a": "a", "model_b": "b", "outcome": 1, "timestamp": 0})
        + "\n",
    )
    assert _code(lambda: load_records(path)) == "parse_error"
