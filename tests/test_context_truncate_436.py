#!/usr/bin/env python3
"""Tests for the max_context supply cap + reproducibility record
(borsagator #436, step-4 evidence).

Upstream ``KronosPredictor.predict()`` normalises over ALL supplied
bars but attends over only the last ``max_context`` (512) tokens.
Supplying 1,231 bars for COMI.CA @ 2023-12-29 anchored the forecast
to the 2019-2022 regime (month error 57.3 points); truncating the
supplied context to the last 512 cut the error to 17.8 (day error
52.6 -> 1.3). The runner must therefore truncate the supplied frame
to the last MAX_CONTEXT sessions before predict(), and the report
must record the supplied/used lengths, the pinned upstream commit,
the HF revisions and the dependency versions so future cells are
reproducible evidence.

Run: python3 tests/test_context_truncate_436.py   (stdlib + pandas
only, no network, no model -- git calls are injected fakes and
run_forecast is stubbed for the main() test).
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "kronos_forecast_ctx", REPO / "scripts" / "kronos_forecast.py")
kf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kf)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}: {name}"
          + (f" -- {detail}" if detail and not ok else ""))


def _frame(n, start="2019-01-02"):
    """n business-day sessions; close = 10.0 + 0.01*i so every row is
    distinguishable and the tail is recognisable."""
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0 + 0.01 * i for i in range(n)],
            "high": [10.5 + 0.01 * i for i in range(n)],
            "low": [9.9 + 0.01 * i for i in range(n)],
            "close": [10.2 + 0.01 * i for i in range(n)],
            "volume": [100 + i for i in range(n)],
        }
    )


# 1. A >512-bar input is cut to exactly the last 512, order preserved.
df = _frame(1231)
out = kf.truncate_context(df)
check("1231 bars truncated to exactly 512",
      len(out) == 512, f"len={len(out)}")
check("truncation keeps the LAST 512 sessions",
      out["date"].iloc[0] == df["date"].iloc[1231 - 512]
      and out["date"].iloc[-1] == df["date"].iloc[-1]
      and out["close"].tolist() == df["close"].tail(512).tolist())
check("truncated frame stays in ascending date order",
      out["date"].is_monotonic_increasing)

# 2. Exactly 512 bars are untouched.
df512 = _frame(512)
out512 = kf.truncate_context(df512)
check("512 bars pass through untouched",
      len(out512) == 512
      and out512["close"].tolist() == df512["close"].tolist()
      and out512["date"].iloc[0] == df512["date"].iloc[0])

# 3. A short frame (60 bars) is untouched.
df60 = _frame(60)
out60 = kf.truncate_context(df60)
check("60 bars pass through untouched",
      len(out60) == 60
      and out60["close"].tolist() == df60["close"].tolist())

# 4. A custom cap is honoured.
out10 = kf.truncate_context(_frame(100), max_context=10)
check("custom max_context=10 keeps the last 10",
      len(out10) == 10
      and out10["close"].iloc[0] == 10.2 + 0.01 * 90)

# 5. The cap constant is the predictor's attention window.
check("MAX_CONTEXT is 512", kf.MAX_CONTEXT == 512,
      f"MAX_CONTEXT={kf.MAX_CONTEXT}")

# 6. Wiring: run_forecast truncates before predict() and passes the
# same constant to the predictor.
src = (REPO / "scripts" / "kronos_forecast.py").read_text()
body = src[src.index("def run_forecast"):]
body = body[:body.index("def main")]
check("run_forecast truncates the context before predict()",
      0 <= body.index("truncate_context(") < body.index("predictor.predict("))
check("run_forecast passes max_context=MAX_CONTEXT to the predictor",
      "max_context=MAX_CONTEXT" in body)

# 7. The upstream pin: a full 40-hex commit, fetched + checked out by
# ensure_kronos_src right after a fresh clone (fake git, as in the
# #198 tests).
import re

check("UPSTREAM_KRONOS_COMMIT is a full 40-hex SHA",
      re.fullmatch(r"[0-9a-f]{40}", kf.UPSTREAM_KRONOS_COMMIT) is not None,
      kf.UPSTREAM_KRONOS_COMMIT)


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRun:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            (dest / "model").mkdir(parents=True, exist_ok=True)
            (dest / "requirements.txt").write_text("torch>=2.0.0\n")
        return FakeResult(0)


tmp = Path(tempfile.mkdtemp(prefix="k436-pin-"))
os.chdir(tmp)
fake = FakeRun()
kf.ensure_kronos_src("kronos_src", run=fake)
pin = kf.UPSTREAM_KRONOS_COMMIT
check("ensure fetches the pinned upstream commit after cloning",
      any(c[:4] == ["git", "-C", "kronos_src", "fetch"] and pin in c
          for c in fake.calls),
      str(fake.calls))
check("ensure checks out the pinned upstream commit",
      any(c[:4] == ["git", "-C", "kronos_src", "checkout"] and pin in c
          for c in fake.calls),
      str(fake.calls))

# 8. main() wiring: the report records supplied vs used context, the
# upstream commit, HF revisions and dependency versions (run_forecast
# stubbed; it populates LAST_RUN_META exactly as the real one does).
seen = {}


def fake_fetch(ticker):
    return _frame(1231), "stub"


def fake_forecast(context_df, pred_len):
    seen["supplied"] = len(context_df)
    kf.LAST_RUN_META.clear()
    kf.LAST_RUN_META.update(
        context_supplied=len(context_df),
        context_used=min(len(context_df), kf.MAX_CONTEXT),
        max_context=kf.MAX_CONTEXT,
        upstream_commit=kf.UPSTREAM_KRONOS_COMMIT,
        model_revision="modelrev",
        tokenizer_revision="tokrev",
        dependency_versions={"torch": "0.0", "pandas": "0.0"},
    )
    dates = pd.bdate_range(context_df["date"].iloc[-1], periods=pred_len + 1)[1:]
    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": [11.0] * pred_len,
            "high": [11.5] * pred_len,
            "low": [10.9] * pred_len,
            "close": [11.2] * pred_len,
        }
    )


tmp2 = Path(tempfile.mkdtemp(prefix="k436-report-"))
os.chdir(tmp2)
kf.fetch_data = fake_fetch
kf.run_forecast = fake_forecast
old_argv = sys.argv
sys.argv = ["kronos_forecast.py", "--ticker", "TEST",
            "--context-start", "2019-01-01", "--pred-len", "3"]
try:
    kf.main()
    report = (tmp2 / "results" / "report.md").read_text()
    check("report records supplied vs used context",
          "Context supplied to Kronos: 1231 sessions, truncated to "
          "the last 512 before predict()" in report,
          [l for l in report.splitlines() if "supplied" in l])
    check("report records the upstream Kronos commit",
          f"Upstream Kronos commit: `{pin}`" in report,
          [l for l in report.splitlines() if "Upstream" in l])
    check("report records HF revisions and dependency versions",
          "Hugging Face revisions:" in report
          and "modelrev" in report and "tokrev" in report
          and "Dependency versions:" in report,
          report[:600])
finally:
    sys.argv = old_argv

failed = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
