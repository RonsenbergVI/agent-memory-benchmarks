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


def test_untracked_usage_reports_na_not_zero():
    """A system whose spend is invisible must not read as free.

    hindsight extracts inside its own server against a provider this
    harness never touches, so a 0 in the cost column would sit beside
    naive's genuine zero and mean something entirely different.
    """
    from amb.reporting.run import ComparisonReport

    tracked = {"system": "fraise", "retrieval_f1": 0.8, "memory_tokens_total": 130112.0}
    untracked = {"system": "hindsight", "retrieval_f1": 0.9, "tracks_usage": False}

    assert (
        ComparisonReport._cell(tracked, "memory_tokens_total", "{:,.0f}") == "130,112"
    )
    assert ComparisonReport._cell(untracked, "memory_tokens_total", "{:,.0f}") == "n/a"
    # a tracked system with no spend recorded yet is still blank, not "n/a"
    assert (
        ComparisonReport._cell({"system": "x"}, "memory_tokens_total", "{:,.0f}") == ""
    )
