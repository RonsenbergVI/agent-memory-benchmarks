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

"""Unit tests for amb.logs.configure().

The suite runs with --capture=no, so loguru output is intercepted by swapping
sys.stderr for a StringIO *before* configure() binds its sink to it.
"""

import io
import sys

import pytest
from loguru import logger

from amb import logs


def test_levels_constant():
    assert logs.LEVELS == ("debug", "info", "warning", "error")


def test_sink_writes_to_stderr(stderr_stream):
    logs.configure()
    logger.warning("sink-target-probe")
    assert "sink-target-probe" in stderr_stream.getvalue()


def test_format_includes_level_name_and_amb_scope(stderr_stream):
    logs.configure()
    logger.warning("format-probe")
    out = stderr_stream.getvalue()
    assert "WARNING" in out
    # extra["scope"] defaults to "amb", padded to 9 characters by the format
    assert "amb      " in out


def test_default_level_is_warning(stderr_stream):
    logs.configure()
    logger.debug("debug-hidden")
    logger.info("info-hidden")
    logger.warning("warning-shown")
    logger.error("error-shown")
    out = stderr_stream.getvalue()
    assert "debug-hidden" not in out
    assert "info-hidden" not in out
    assert "warning-shown" in out
    assert "error-shown" in out


@pytest.mark.parametrize("configured", logs.LEVELS)
def test_level_filtering(stderr_stream, configured):
    logs.configure(configured)
    for name in logs.LEVELS:
        getattr(logger, name)(f"probe-{name}")
    out = stderr_stream.getvalue()
    threshold = logs.LEVELS.index(configured)
    for position, name in enumerate(logs.LEVELS):
        if position >= threshold:
            assert f"probe-{name}" in out
        else:
            assert f"probe-{name}" not in out


def test_level_argument_is_case_insensitive(stderr_stream):
    logs.configure("Info")
    logger.info("case-probe")
    assert "case-probe" in stderr_stream.getvalue()


def test_reconfigure_does_not_duplicate_sinks(stderr_stream):
    logs.configure()
    logs.configure()
    logs.configure()
    logger.warning("dedupe-probe")
    assert stderr_stream.getvalue().count("dedupe-probe") == 1


def test_configure_removes_preexisting_sinks(stderr_stream):
    records = []
    logger.add(records.append, level="DEBUG", format="{message}")
    logs.configure()
    logger.warning("orphan-probe")
    assert records == []
    assert "orphan-probe" in stderr_stream.getvalue()


def test_reconfigure_rebinds_to_current_stderr(stderr_stream, monkeypatch):
    logs.configure()
    replacement = io.StringIO()
    monkeypatch.setattr(sys, "stderr", replacement)
    logs.configure()
    logger.warning("rebind-probe")
    assert "rebind-probe" not in stderr_stream.getvalue()
    assert "rebind-probe" in replacement.getvalue()


def test_unknown_level_raises_value_error(stderr_stream):
    with pytest.raises(ValueError, match="Level 'VERBOSE' does not exist"):
        logs.configure("verbose")
    # the failed call already removed the old sink, so nothing is logged
    logger.warning("after-failure-probe")
    assert "after-failure-probe" not in stderr_stream.getvalue()
