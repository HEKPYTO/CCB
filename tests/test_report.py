import json

import pytest

from ccb.cli import main


def _records(tmp_path, model_a="a", count=20, name="records.jsonl"):
    path = tmp_path / name
    lines = [
        json.dumps(
            {
                "model_a": model_a,
                "model_b": "b",
                "outcome": 1,
                "t": index,
            }
        )
        for index in range(count)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _demo(tmp_path, capsys, model_a="a", seed=0, name="demo.html"):
    source = _records(tmp_path, model_a=model_a)
    output = tmp_path / name
    argv = ["demo", "--input", str(source), "--out", str(output), "--seed", str(seed)]
    assert main(argv) == 0
    return output


def test_demo_writes_one_file(tmp_path, capsys):
    output = _demo(tmp_path, capsys)
    assert output.exists()
    assert "wrote" in capsys.readouterr().out


def test_demo_escapes_model_names(tmp_path, capsys):
    output = _demo(tmp_path, capsys, model_a="<script>alert(1)</script>")
    content = output.read_text()
    assert "<script>alert" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content


def test_demo_is_self_contained(tmp_path, capsys):
    content = _demo(tmp_path, capsys).read_text()
    assert "http://" not in content
    assert "https://" not in content
    assert "src=" not in content


def test_demo_contains_replay_panel(tmp_path, capsys):
    content = _demo(tmp_path, capsys).read_text()
    for label in ("Wald", "Per-step", "Insertion", "Replacement"):
        assert label in content
    for budget in ("B=0", "B=100", "B=400"):
        assert budget in content


def test_demo_is_deterministic(tmp_path, capsys):
    first = _demo(tmp_path, capsys, name="first.html").read_text()
    second = _demo(tmp_path, capsys, name="second.html").read_text()
    assert first == second


def test_demo_requires_input(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["demo", "--out", "unused.html"])
    assert excinfo.value.code == 2
