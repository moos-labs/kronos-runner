#!/usr/bin/env python3
"""Tests for --as-of (borsagator #436, walk-forward backtest).

--as-of YYYY-MM-DD: fetch full history, drop every row with date >
as-of, fail loudly if fewer than --min-context rows remain. The
truncation must happen inside the runner, before any model call, so
rows after the cutoff never reach Kronos.

Run: python3 tests/test_asof_436.py   (stdlib + pandas only, no
network, no model -- fetch_data and run_forecast are stubbed for the
main() tests).

Failing-before probe: point KF_UNDER_TEST at the pre-change script
and every test below fails (the helpers do not exist there).
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
KF_PATH = os.environ.get("KF_UNDER_TEST", REPO / "scripts" / "kronos_forecast.py")

spec = importlib.util.spec_from_file_location("kronos_forecast_asof", KF_PATH)
kf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kf)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}: {name}"
          + (f" -- {detail}" if detail and not ok else ""))


def _frame(n=100, start="2024-01-01"):
    """n business-day sessions; close rises 10.0, 10.1, ... so the
    as-of close is distinguishable from any later close."""
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0 + 0.1 * i for i in range(n)],
            "high": [10.5 + 0.1 * i for i in range(n)],
            "low": [9.9 + 0.1 * i for i in range(n)],
            "close": [10.2 + 0.1 * i for i in range(n)],
            "volume": [100 + i for i in range(n)],
        }
    )


# 1. truncate_asof drops every row after the cutoff, keeps the cutoff row.
df = _frame()
out = kf.truncate_asof(df, "2024-03-01", min_rows=20)
expected_last = pd.Timestamp("2024-03-01")
check("truncate drops rows after as-of",
      out["date"].max() == expected_last and len(out) == 45,
      f"max={out['date'].max()} len={len(out)}")
check("truncate keeps the as-of row itself",
      out["date"].iloc[-1] == expected_last
      and out["close"].iloc[-1] == 10.2 + 0.1 * 44)

# 2. A cutoff on a non-session date keeps the last session on/before it.
out2 = kf.truncate_asof(df, "2024-03-02", min_rows=20)  # a Saturday
check("non-session as-of keeps last session on/before it",
      out2["date"].iloc[-1] == expected_last)

# 3. Too-short remainder fails loudly instead of forecasting on a sliver.
try:
    kf.truncate_asof(_frame(30), "2024-01-15", min_rows=20)
    check("too-short remainder raises RuntimeError", False)
except RuntimeError as e:
    check("too-short remainder raises RuntimeError",
          "only 11 sessions" in str(e) and "2024-01-15" in str(e), str(e))

# 4. select_context: preferred window ends at as-of, not at today.
trunc = kf.truncate_asof(df, "2024-03-01", min_rows=20)
ctx, fb, _ = kf.select_context(trunc, "2024-02-01", fallback_len=400,
                               min_context=20)
check("preferred context ends at as-of",
      ctx["date"].iloc[-1] == expected_last
      and ctx["date"].iloc[0] == pd.Timestamp("2024-02-01")
      and not fb,
      f"ctx {ctx['date'].iloc[0]}..{ctx['date'].iloc[-1]} fb={fb}")

# 5. select_context: as-of before context-start falls back to the last
# fallback_len sessions ENDING at as-of (same rule as live, as-of
# substituted for "latest session").
ctx5, fb5, reason5 = kf.select_context(trunc, "2024-04-01", fallback_len=10,
                                       min_context=20)
check("fallback ends at as-of when preferred window is empty",
      fb5 and len(ctx5) == 10
      and ctx5["date"].iloc[-1] == expected_last,
      f"len={len(ctx5)} last={ctx5['date'].iloc[-1]} fb={fb5} reason={reason5}")

# 6. main() wiring: stubbed fetch + stubbed model; the context handed
# to run_forecast must contain no row after as-of, and the report must
# record the cutoff.
seen = {}


def fake_fetch(ticker):
    return _frame(), "stub"


def fake_forecast(context_df, pred_len, samples=None):
    seen["context"] = context_df.copy()
    dates = pd.bdate_range("2024-03-04", periods=pred_len)
    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": [11.0] * pred_len,
            "high": [11.5] * pred_len,
            "low": [10.9] * pred_len,
            "close": [11.2] * pred_len,
        }
    )


tmp = Path(tempfile.mkdtemp(prefix="k436-main-"))
os.chdir(tmp)
kf.fetch_data = fake_fetch
kf.run_forecast = fake_forecast
old_argv = sys.argv
sys.argv = ["kronos_forecast.py", "--ticker", "TEST",
            "--as-of", "2024-03-01", "--pred-len", "3"]
try:
    kf.main()
    ctx = seen["context"]
    check("main(): no row after as-of reaches the model",
          ctx["date"].max() == expected_last,
          f"context max={ctx['date'].max()}")
    report = (tmp / "results" / "report.md").read_text()
    check("main(): report records the as-of cutoff",
          "As-of cutoff: 2024-03-01" in report, report[:400])
    check("main(): last actual close is the as-of close",
          "Last actual close: 14.6 on 2024-03-01" in report,
          [l for l in report.splitlines() if "Last actual" in l])
finally:
    sys.argv = old_argv

# 7. main() fails loudly when the truncated series is too short.
tmp2 = Path(tempfile.mkdtemp(prefix="k436-short-"))
os.chdir(tmp2)


def fake_fetch_short(ticker):
    return _frame(30), "stub"


kf.fetch_data = fake_fetch_short
sys.argv = ["kronos_forecast.py", "--ticker", "TEST",
            "--as-of", "2024-01-15"]
try:
    kf.main()
    check("main(): too-short remainder raises", False)
except RuntimeError as e:
    check("main(): too-short remainder raises",
          "only 11 sessions" in str(e), str(e))
finally:
    sys.argv = old_argv

failed = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
