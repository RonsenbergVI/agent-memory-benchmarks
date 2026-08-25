def test_untracked_usage_reports_na_not_zero():
    """A system whose spend is invisible must not read as free.

    hindsight extracts inside its own server against a provider this
    harness never touches, so a 0 in the cost column would sit beside
    naive's genuine zero and mean something entirely different.
    """
    from amb.reporting.run import ComparisonReport

    tracked = {"system": "fraise", "retrieval_f1": 0.8, "memory_tokens_total": 130112.0}
    untracked = {"system": "hindsight", "retrieval_f1": 0.9, "tracks_usage": False}

    assert ComparisonReport._cell(tracked, "memory_tokens_total", "{:,.0f}") == "130,112"
    assert ComparisonReport._cell(untracked, "memory_tokens_total", "{:,.0f}") == "n/a"
    # a tracked system with no spend recorded yet is still blank, not "n/a"
    assert ComparisonReport._cell({"system": "x"}, "memory_tokens_total", "{:,.0f}") == ""
