import pytest

from ccb.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out == "ccb 0.0.0\n"


def test_unknown_command_exits_2():
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert excinfo.value.code == 2
