import json

import pytest

from ccb.cli import main
from ccb.errors import CCBError
from ccb.records import load_records


def test_random_bytes_never_crash(tmp_path):
    path = tmp_path / "hostile.jsonl"
    path.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(CCBError) as excinfo:
        load_records(path)
    assert excinfo.value.code == "parse_error"


def test_one_megabyte_model_name(tmp_path):
    name = "m" * 1_000_000
    path = tmp_path / "long.jsonl"
    path.write_text(json.dumps({"model_a": name, "model_b": "b", "outcome": 1}) + "\n")
    records = load_records(path)
    assert len(records[0].model_a) == 1_000_000


def test_csv_quoted_comma_and_newline(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('model_a,model_b,outcome,t\n"a,inc",b,1,0\n"multi\nline",b,0,1\n')
    records = load_records(path)
    assert records[0].model_a == "a,inc"
    assert records[1].model_a == "multi\nline"


def test_out_path_is_a_directory(tmp_path, capsys):
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"model_a": "a", "model_b": "b", "outcome": 1}) + "\n")
    target = tmp_path / "outdir"
    target.mkdir()
    argv = [
        "certify",
        "--input",
        str(source),
        "--out",
        str(target),
        "--force",
    ]
    assert main(argv) == 1
    assert "io_error" in capsys.readouterr().err


def test_csv_oversized_field_is_typed(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text('model_a,model_b,outcome\n"' + "x" * 200_000 + '",b,1\n')
    with pytest.raises(CCBError) as excinfo:
        load_records(path)
    assert excinfo.value.code == "parse_error"
