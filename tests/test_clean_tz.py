#!/usr/bin/env python3
"""Regression tests for #111: clean() must handle tz-aware, tz-naive,
AND mixed date columns (yahooquery's own fetch mixes them; run
43807031623 crashed in clean() with
ValueError: Cannot mix tz-aware with tz-naive values).

Runnable with pytest or plain `python3 tests/test_clean_tz.py`.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from kronos_forecast import clean  # noqa: E402


def _frame(dates):
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0] * len(dates),
            "high": [10.5] * len(dates),
            "low": [9.9] * len(dates),
            "close": [10.2] * len(dates),
            "volume": [100] * len(dates),
        }
    )


def _assert_normalized(out, expected):
    assert len(out) == len(expected), (len(out), expected)
    assert str(out["date"].dtype) == "datetime64[ns]", out["date"].dtype
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == expected, list(out["date"])


def test_mixed_tz_aware_and_naive():
    # The exact #111 crash: tz-aware and tz-naive values in one column.
    out = clean(
        _frame(
            [
                pd.Timestamp("2024-01-02", tz="UTC"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-04", tz="UTC"),
                pd.Timestamp("2024-01-07"),
            ]
        )
    )
    _assert_normalized(out, ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-07"])


def test_all_tz_naive():
    out = clean(_frame([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]))
    _assert_normalized(out, ["2024-01-02", "2024-01-03"])


def test_all_tz_aware():
    out = clean(
        _frame(
            [
                pd.Timestamp("2024-01-02", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
            ]
        )
    )
    _assert_normalized(out, ["2024-01-02", "2024-01-03"])


def test_string_dates_with_offsets_normalized():
    # Intraday timestamps / offsets collapse to midnight UTC date.
    out = clean(
        _frame(["2024-01-02 15:30:00+02:00", "2024-01-03 09:00:00", "2024-01-04T00:00:00Z"])
    )
    _assert_normalized(out, ["2024-01-02", "2024-01-03", "2024-01-04"])


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("all clean() tz regression tests passed")
