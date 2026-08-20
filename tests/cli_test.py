# MIT License
#
# Copyright (c) 2026 René-Jean Corneille
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import click
import pytest
from click.testing import CliRunner

from amb.cli.commands import COMMANDS
from amb.cli.commands.run import PARAM, ParamType, run
from amb.cli.main import cli, main

# --- ParamType.coerce -------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("none", None),
        ("None", None),
        ("NONE", None),
        ("null", None),
        ("Null", None),
        ("42", 42),
        ("-7", -7),
        ("0", 0),
        ("1", 1),  # int, never bool: only the literals true/false coerce
        ("3.5", 3.5),
        ("-0.25", -0.25),
        ("1e3", 1000.0),
        ("gpt-5-mini", "gpt-5-mini"),
        ("truthy", "truthy"),  # prefix of a literal is still a plain string
        ("", ""),
    ],
)
def test_coerce(raw, expected):
    coerced = ParamType.coerce(raw)
    assert coerced == expected
    # 1 == True and 1000 == 1000.0, so the type must match too
    assert type(coerced) is type(expected)


# --- ParamType.convert ------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected_key, expected_value",
    [
        ("model=gpt-5-mini", "model", "gpt-5-mini"),
        ("reasoning=false", "reasoning", False),
        ("reasoning=true", "reasoning", True),
        ("timeout=none", "timeout", None),
        ("k=10", "k", 10),
        ("temperature=0.5", "temperature", 0.5),
        # only the first `=` splits, so URLs and expressions survive intact
        ("url=http://host:8080/db?user=amb", "url", "http://host:8080/db?user=amb"),
        ("expr=a=b=c", "expr", "a=b=c"),
        ("empty=", "empty", ""),
    ],
)
def test_convert(value, expected_key, expected_value):
    key, coerced = PARAM.convert(value, None, None)
    assert key == expected_key
    assert coerced == expected_value
    assert type(coerced) is type(expected_value)


def test_convert_without_equals_fails():
    with pytest.raises(click.UsageError, match="'noequals' is not KEY=VALUE"):
        ParamType().convert("noequals", None, None)


def test_run_rejects_param_without_equals():
    result = CliRunner().invoke(
        cli,
        ["run", "--system", "s", "--dataset", "d", "--param", "noequals"],
    )
    assert result.exit_code == 2
    assert "'noequals' is not KEY=VALUE" in result.output


def test_run_params_reach_callback_coerced(monkeypatch):
    received = {}

    def recorder(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(run, "callback", recorder)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "--system",
            "s",
            "--dataset",
            "d",
            "--param",
            "reasoning=false",
            "--param",
            "url=http://host:8080/db?user=amb",
            "--param",
            "k=5",
        ],
    )
    assert result.exit_code == 0
    assert received["system"] == "s"
    assert received["params"] == (
        ("reasoning", False),
        ("url", "http://host:8080/db?user=amb"),
        ("k", 5),
    )
    # --param k=5 configures the system; the run's own --k keeps its default
    assert received["k"] == 10
    assert received["mode"] == "direct"


def test_run_requires_system_and_dataset():
    result = CliRunner().invoke(cli, ["run"])
    assert result.exit_code == 2
    assert "Missing option" in result.output


# --- command registration ---------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: str(c.name))
def test_command_attached_to_group(command):
    assert cli.commands[str(command.name)] is command


def test_group_has_no_commands_beyond_the_registry():
    assert set(cli.commands) == {str(command.name) for command in COMMANDS}


def test_main_is_the_group():
    assert main is cli


# --- help smokes ------------------------------------------------------------


def _help_paths() -> list[tuple[str, ...]]:
    """Every command path: each registered command, plus group subcommands."""
    paths: list[tuple[str, ...]] = []
    for command in COMMANDS:
        paths.append((str(command.name),))
        if isinstance(command, click.Group):
            paths.extend((str(command.name), sub) for sub in command.commands)
    return paths


def test_group_help_lists_every_command():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert str(command.name) in result.output


def test_group_short_help_alias():
    result = CliRunner().invoke(cli, ["-h"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: cli")


@pytest.mark.parametrize("path", _help_paths(), ids=" ".join)
def test_subcommand_help(path):
    result = CliRunner().invoke(cli, [*path, "--help"])
    assert result.exit_code == 0
    assert result.output.startswith(f"Usage: cli {' '.join(path)}")


def test_version_option():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("amb, version ")


def test_rejects_unknown_log_level():
    result = CliRunner().invoke(cli, ["--log-level", "verbose", "systems"])
    assert result.exit_code == 2
    assert "--log-level" in result.output
