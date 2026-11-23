#!/usr/bin/env python3
"""Tests for Kronos multi-sampling (borsagator #436, panel evaluation
2026-11-23).

The runner used to ship ONE stochastic draw (sample_count=1, T=1.0,
seed 42) as the forecast. It now draws N samples (default 32,
--samples / KRONOS_SAMPLES) and aggregates them per horizon step:
median = point forecast, close_p10/close_p90 = the per-step band.
Upstream predict(sample_count=N) averages its draws internally and
returns no quantiles, so the runner loops predict(sample_count=1);
the seed is set once and the draws consume the RNG stream in order.

These tests cover the pure aggregation/resolution helpers and the
main() wiring with run_forecast stubbed -- no torch, no network, no
model (stdlib + pandas + numpy only).

Run: python3 tests/test_multisample_436.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "kronos_forecast_ms", REPO / "scripts" / "kronos_forecast.py")
kf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kf)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}: {name}"
          + (f" -- {detail}" if detail and not ok else ""))


def _draw(closes, base=100.0):
    """One fake draw frame with the upstream predict() columns."""
    n = len(closes)
    return pd.DataFrame({
        "open": [base] * n,
        "high": [base] * n,
        "low": [base] * n,
        "close": list(closes),
        "volume": [1000.0] * n,
        "amount": [100000.0] * n,
    })


# 1. resolve_samples: default, env, CLI precedence, validation.
check("resolve_samples default is DEFAULT_SAMPLES",
      kf.resolve_samples(None, env={}) == kf.DEFAULT_SAMPLES)
check("DEFAULT_SAMPLES is 32", kf.DEFAULT_SAMPLES == 32)
check("resolve_samples reads KRONOS_SAMPLES",
      kf.resolve_samples(None, env={"KRONOS_SAMPLES": "7"}) == 7)
check("resolve_samples CLI beats env",
      kf.resolve_samples(3, env={"KRONOS_SAMPLES": "7"}) == 3)
check("resolve_samples accepts 1 (old behaviour)",
      kf.resolve_samples(1, env={}) == 1)
for bad in ("abc", "0", "-2", 0, -5):
    try:
        kf.resolve_samples(bad, env={})
        check(f"resolve_samples rejects {bad!r}", False, "no ValueError")
    except ValueError:
        check(f"resolve_samples rejects {bad!r}", True)

# 2. aggregate_samples: median is the point forecast; p10/p90 bracket it.
draws = [_draw([10.0, 20.0]), _draw([12.0, 22.0]), _draw([11.0, 21.0]),
         _draw([13.0, 23.0]), _draw([100.0, 24.0])]
agg = kf.aggregate_samples(draws)
check("aggregate keeps existing columns in order",
      list(agg.columns[:6]) == ["open", "high", "low", "close",
                                "volume", "amount"],
      f"got {list(agg.columns)}")
check("aggregate adds close_p10/close_p90",
      "close_p10" in agg.columns and "close_p90" in agg.columns)
check("median close step 1 is the middle draw, not the mean",
      agg["close"].iloc[0] == 12.0 and agg["close"].iloc[1] == 22.0,
      f"got {list(agg['close'])} (mean of step 1 would be 29.2)")
check("p10 <= median <= p90 per step",
      bool((agg["close_p10"] <= agg["close"]).all()
           and (agg["close"] <= agg["close_p90"]).all()))
check("p10/p90 match numpy quantiles of the draws",
      np.isclose(agg["close_p10"].iloc[0],
                 np.quantile([10.0, 12.0, 11.0, 13.0, 100.0], 0.10))
      and np.isclose(agg["close_p90"].iloc[0],
                     np.quantile([10.0, 12.0, 11.0, 13.0, 100.0], 0.90)))

# 3. Single draw degenerates to the old behaviour exactly.
one = _draw([5.0, 6.0, 7.0])
agg1 = kf.aggregate_samples([one])
check("single-draw aggregate equals the draw",
      bool((agg1["close"] == one["close"]).all())
      and bool((agg1["close_p10"] == one["close"]).all())
      and bool((agg1["close_p90"] == one["close"]).all()))

# 4. collect_sample_paths calls predict_once exactly N times, in order.
calls = []

def fake_once():
    calls.append(len(calls))
    return _draw([float(len(calls)), float(len(calls))])

agg2 = kf.collect_sample_paths(fake_once, 4)
check("collect_sample_paths makes exactly N calls",
      calls == [0, 1, 2, 3], f"got {calls}")
check("collect_sample_paths aggregates the draws",
      agg2["close"].iloc[0] == 2.5, f"got {agg2['close'].iloc[0]}")

# 5. aggregate_samples refuses an empty draw list.
try:
    kf.aggregate_samples([])
    check("aggregate_samples rejects empty list", False, "no ValueError")
except ValueError:
    check("aggregate_samples rejects empty list", True)

# 6. Source wiring: run_forecast loops sample_count=1 draws through
# collect_sample_paths, and the truncation/pins are untouched.
src = (REPO / "scripts" / "kronos_forecast.py").read_text()
body = src[src.index("def run_forecast"):src.index("def main")]
check("run_forecast aggregates via collect_sample_paths",
      "collect_sample_paths(predict_once, n_samples)" in body)
check("run_forecast still draws with sample_count=1 per call",
      "sample_count=1" in body)
check("run_forecast does NOT use upstream sample_count=N averaging",
      "sample_count=n_samples" not in body
      and "sample_count=samples" not in body)
check("512 context cap still applied before sampling",
      "truncate_context(context_df, MAX_CONTEXT)" in body)
check("HF revision pins still passed to from_pretrained",
      "revision=MODEL_REVISION" in body
      and "revision=TOKENIZER_REVISION" in body)
check("seed discipline unchanged (manual_seed(42) once, in run_forecast)",
      body.count("torch.manual_seed(42)") == 1)

# 7. main() wiring: --samples reaches run_forecast; forecast.csv gains
# close_p10/close_p90; the report states the draw count.
def _frame(n, start="2024-01-02"):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": [10.0 + 0.01 * i for i in range(n)],
        "high": [10.5 + 0.01 * i for i in range(n)],
        "low": [9.9 + 0.01 * i for i in range(n)],
        "close": [10.2 + 0.01 * i for i in range(n)],
        "volume": [100 + i for i in range(n)],
    })


seen = {}

def fake_fetch(ticker):
    return _frame(200), "stub"

def fake_forecast(context_df, pred_len, samples=None):
    seen["samples"] = samples
    kf.LAST_RUN_META.clear()
    kf.LAST_RUN_META.update(
        context_supplied=len(context_df),
        context_used=min(len(context_df), kf.MAX_CONTEXT),
        max_context=kf.MAX_CONTEXT,
        upstream_commit=kf.UPSTREAM_KRONOS_COMMIT,
        model_revision=kf.MODEL_REVISION,
        tokenizer_revision=kf.TOKENIZER_REVISION,
        dependency_versions={"torch": "0.0", "pandas": "0.0"},
        samples=kf.resolve_samples(samples),
    )
    dates = pd.bdate_range(context_df["date"].iloc[-1],
                           periods=pred_len + 1)[1:]
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": [11.0] * pred_len,
        "high": [11.5] * pred_len,
        "low": [10.9] * pred_len,
        "close": [11.2] * pred_len,
        "close_p10": [10.8] * pred_len,
        "close_p90": [11.9] * pred_len,
        "volume": [100.0] * pred_len,
        "amount": [1120.0] * pred_len,
    })


orig_fetch, orig_forecast = kf.fetch_data, kf.run_forecast
kf.fetch_data, kf.run_forecast = fake_fetch, fake_forecast
old_argv, old_cwd = sys.argv, os.getcwd()
tmp = Path(tempfile.mkdtemp(prefix="k436-ms-main-"))
try:
    os.chdir(tmp)
    sys.argv = ["kronos_forecast.py", "--ticker", "TEST",
                "--pred-len", "5", "--samples", "9"]
    kf.main()
    out = pd.read_csv(tmp / "results" / "forecast.csv")
    report = (tmp / "results" / "report.md").read_text()
finally:
    sys.argv, kf.fetch_data, kf.run_forecast = old_argv, orig_fetch, orig_forecast
    os.chdir(old_cwd)

check("main passes --samples through to run_forecast",
      seen.get("samples") == 9, f"got {seen.get('samples')}")
check("forecast.csv carries close_p10/close_p90 after close",
      list(out.columns) == ["date", "open", "high", "low", "close",
                            "close_p10", "close_p90", "volume", "amount"],
      f"got {list(out.columns)}")
check("report states the aggregated draw count",
      "samples=9 draws" in report)

failed = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
