import json

import pytest

from ccb.certificates import Certificate
from ccb.cli import main
from ccb.report import statement


def _wins(tmp_path, count=20, name="wins.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(
            f'{{"model_a":"a","model_b":"b","outcome":1,"t":{index}}}\n'
            for index in range(count)
        )
    )
    return path


def test_certify_json_envelope(tmp_path, capsys):
    path = _wins(tmp_path)
    assert main(["certify", "--input", str(path), "--pair", "a,b", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "ccb/v1"
    assert payload["command"] == "certify"
    assert payload["error"] is None
    certificate = payload["data"]["certificate"]
    assert certificate["claim"] == "edge"
    assert certificate["cost"] == "replacement"
    assert certificate["alpha"] == 0.05
    assert certificate["tolerance"] == 2


def test_certify_human_output(tmp_path, capsys):
    path = _wins(tmp_path)
    assert main(["certify", "--input", str(path), "--pair", "a,b"]) == 0
    out = capsys.readouterr().out
    assert "a > b" in out
    assert "B_rep" in out
    assert "alpha         0.05" in out


def test_lead_claim(tmp_path, capsys):
    path = _wins(tmp_path, count=1000)
    assert main(["certify", "--input", str(path), "--claim", "lead", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["certificate"]["tolerance"] == 451


def test_reverse_pair_flag(tmp_path, capsys):
    path = _wins(tmp_path)
    assert main(["certify", "--input", str(path), "--reverse-pair", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["certificate"]["tolerance"] == -1
    assert payload["data"]["statement"].startswith("At alpha=0.05, no edge claim")


def test_reversed_pair_order_flips_claim(tmp_path, capsys):
    path = _wins(tmp_path)
    argv = ["certify", "--input", str(path), "--pair", "b,a", "--json"]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["certificate"]["tolerance"] == -1
    assert payload["data"]["pair"] == ["b", "a"]
    assert "b over a" in payload["data"]["statement"]


def test_reversed_pair_order_with_reverse_flag(tmp_path, capsys):
    path = _wins(tmp_path)
    argv = [
        "certify",
        "--input",
        str(path),
        "--pair",
        "b,a",
        "--reverse-pair",
        "--json",
    ]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["certificate"]["tolerance"] == 2
    assert payload["data"]["pair"] == ["a", "b"]


def test_multiple_pairs_requires_pair(tmp_path, capsys):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"model_a":"a","model_b":"b","outcome":1}\n'
        '{"model_a":"c","model_b":"d","outcome":1}\n'
    )
    assert main(["certify", "--input", str(path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "multiple_pairs"
    assert payload["data"] is None


def test_missing_pair(tmp_path, capsys):
    path = _wins(tmp_path)
    assert main(["certify", "--input", str(path), "--pair", "x,y"]) == 1
    assert "invalid_pair" in capsys.readouterr().err


def test_invalid_alpha_error_envelope(tmp_path, capsys):
    path = _wins(tmp_path)
    argv = ["certify", "--input", str(path), "--alpha", "nan", "--json"]
    assert main(argv) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "invalid_alpha"


def test_output_exists_refuses(tmp_path, capsys):
    path = _wins(tmp_path)
    out = tmp_path / "report.txt"
    out.write_text("keep")
    argv = ["certify", "--input", str(path), "--pair", "a,b", "--out", str(out)]
    assert main(argv) == 1
    assert "output_exists" in capsys.readouterr().err
    assert out.read_text() == "keep"


def test_output_force_writes(tmp_path, capsys):
    path = _wins(tmp_path)
    out = tmp_path / "report.txt"
    argv = [
        "certify",
        "--input",
        str(path),
        "--pair",
        "a,b",
        "--out",
        str(out),
        "--force",
    ]
    assert main(argv) == 0
    assert "B_rep" in out.read_text()
    assert "holds unless more than" in out.read_text()


def test_selection_requires_variant(tmp_path):
    path = _wins(tmp_path)
    argv = ["certify", "--input", str(path), "--selection-variants", "27"]
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["certify"])
    assert excinfo.value.code == 2


def _certificate(claim, cost, tolerance):
    return Certificate(
        claim=claim,
        cost=cost,
        alpha=0.05,
        n=14858,
        tolerance=tolerance,
        certified_at=100,
        log_e_value=0.0,
        selection_variants=1,
    )


def test_statement_exact():
    edge_insertion = (
        "At alpha=0.05, the edge claim that A has had an advantage over B "
        "holds unless more than 2056 of the first 14858 records were forged "
        "(insertion cost)."
    )
    edge_replacement = (
        "At alpha=0.05, the edge claim that A has had an advantage over B "
        "holds unless more than 1009 of the first 14858 records were forged "
        "or altered (replacement cost)."
    )
    lead_insertion = (
        "At alpha=0.05, the lead claim that A leads B on the uncorrupted "
        "margin over the first 14858 records holds unless more than 1940 of "
        "those records were forged (insertion cost)."
    )
    lead_replacement = (
        "At alpha=0.05, the lead claim that A leads B on the uncorrupted "
        "margin over the first 14858 records holds unless more than 957 of "
        "those records were forged or altered (replacement cost)."
    )
    assert (
        statement(_certificate("edge", "insertion", 2056), "A", "B") == edge_insertion
    )
    assert (
        statement(_certificate("edge", "replacement", 1009), "A", "B")
        == edge_replacement
    )
    assert (
        statement(_certificate("lead", "insertion", 1940), "A", "B") == lead_insertion
    )
    assert (
        statement(_certificate("lead", "replacement", 957), "A", "B")
        == lead_replacement
    )


def test_statement_uncertified_exact():
    assert statement(_certificate("edge", "replacement", -1), "A", "B") == (
        "At alpha=0.05, no edge claim is certified for A over B on the first "
        "14858 records."
    )
    assert statement(_certificate("lead", "replacement", -1), "A", "B") == (
        "At alpha=0.05, no lead claim is certified for A over B on the first "
        "14858 records."
    )
