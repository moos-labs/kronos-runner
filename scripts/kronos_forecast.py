#!/usr/bin/env python3
"""Kronos-small candlestick forecast for a ticker (generic, market-agnostic).

Data rules (hard-won, see README):
- Never use yfinance for Cairo tickers (Africa/Cairo DST crash).
- Primary source: yahooquery (proven for POUL.CA). Fallback: the direct
  Yahoo chart API via requests (may intermittently 429 from GitHub IPs).
- Always drop rows before 2019-01-01 (corrupted segment: stale flat
  prices at a wrong scale).

Preferred context: all sessions from --context-start (default
2026-09-28, POUL's post-merger resumption) to the latest session.
If Kronos refuses that short context (or it is too short to attempt),
fall back to the last --fallback-len sessions ending at the latest
session and FLAG the merger discontinuity in the report.
"""
import argparse
import os
import sys
import time

import pandas as pd

RESULTS_DIR = "results"
DROP_BEFORE = pd.Timestamp("2019-01-01")
MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"


def fetch_yahooquery(ticker):
    from yahooquery import Ticker

    hist = Ticker(ticker).history(period="max")
    if hist is None or len(hist) == 0:
        raise RuntimeError("yahooquery returned no rows")
    df = hist.reset_index()
    # reset_index gives columns: symbol, date, open, high, low, close, ...
    df.columns = [str(c).lower() for c in df.columns]
    if "date" not in df.columns:
        raise RuntimeError(f"yahooquery: no date column in {list(df.columns)}")
    return df


def fetch_chart_api(ticker):
    import requests

    period1 = int(pd.Timestamp("2019-01-01").timestamp())
    period2 = int(time.time()) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) kronos-runner"},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    return df


def fetch_data(ticker):
    errors = []
    for name, fn in (("yahooquery", fetch_yahooquery), ("yahoo-chart-api", fetch_chart_api)):
        try:
            df = fn(ticker)
            print(f"[data] source={name} raw_rows={len(df)}")
            return df, name
        except Exception as exc:  # noqa: BLE001 - report and try next source
            print(f"[data] {name} failed: {exc}")
            errors.append(f"{name}: {exc}")
    raise RuntimeError("All data sources failed: " + " | ".join(errors))


def clean(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0 if col == "volume" else float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["date"] >= DROP_BEFORE]
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def next_sessions(last_date, n):
    """Next n EGX sessions (Sun-Thu; Fri/Sat weekend). Holidays not modelled."""
    out = []
    d = pd.Timestamp(last_date)
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() not in (4, 5):  # 4=Fri, 5=Sat
            out.append(d)
    return pd.Series(out)


def run_forecast(context_df, pred_len):
    # Kronos is not a PyPI package: the workflow clones the upstream repo
    # into kronos_src/ and we import its `model` package from there.
    kronos_src = os.environ.get("KRONOS_SRC", "kronos_src")
    if os.path.isdir(os.path.join(kronos_src, "model")):
        sys.path.insert(0, os.path.abspath(kronos_src))
    else:
        sys.path.insert(0, os.path.abspath("."))
    import torch

    torch.manual_seed(42)
    try:
        torch.set_num_threads(4)
    except Exception:
        pass
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    x_df = context_df[["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = pd.Series(pd.to_datetime(context_df["date"]))
    last_date = pd.Timestamp(context_df["date"].iloc[-1])
    y_timestamp = next_sessions(last_date, pred_len)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )
    pred_df = pred_df.reset_index(drop=True)
    pred_df.insert(0, "date", [d.strftime("%Y-%m-%d") for d in y_timestamp])
    return pred_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="POUL.CA")
    ap.add_argument("--pred-len", type=int, default=10)
    ap.add_argument("--context-start", default="2026-09-28",
                    help="Preferred context start (post-merger resumption for POUL)")
    ap.add_argument("--fallback-len", type=int, default=400)
    ap.add_argument("--min-context", type=int, default=20,
                    help="Below this many preferred sessions, go straight to fallback")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, source = fetch_data(args.ticker)
    df = clean(df)
    if df.empty:
        raise RuntimeError("No usable rows after cleaning")
    print(f"[data] cleaned_rows={len(df)} range={df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}")

    last_actual_close = float(df["close"].iloc[-1])
    last_actual_date = pd.Timestamp(df["date"].iloc[-1])

    preferred = df[df["date"] >= pd.Timestamp(args.context_start)].reset_index(drop=True)
    fallback_used = False
    fallback_reason = ""
    context = preferred
    if len(preferred) < args.min_context:
        fallback_used = True
        fallback_reason = (
            f"preferred window ({args.context_start} onwards) has only "
            f"{len(preferred)} sessions (< {args.min_context} minimum)"
        )
        context = df.tail(args.fallback_len).reset_index(drop=True)
    print(f"[context] preferred_sessions={len(preferred)} using={len(context)} fallback={fallback_used}")

    try:
        pred_df = run_forecast(context, args.pred_len)
    except Exception as exc:  # Kronos refused the short context -> fallback
        if fallback_used:
            raise
        print(f"[context] Kronos refused preferred context: {exc}")
        fallback_used = True
        fallback_reason = f"Kronos refused the preferred short context: {exc}"
        context = df.tail(args.fallback_len).reset_index(drop=True)
        print(f"[context] falling back to last {len(context)} sessions")
        pred_df = run_forecast(context, args.pred_len)

    out_cols = ["date", "open", "high", "low", "close"]
    for extra in ("volume", "amount"):
        if extra in pred_df.columns:
            out_cols.append(extra)
    forecast_csv = pred_df[out_cols].copy()
    for col in out_cols[1:]:
        forecast_csv[col] = forecast_csv[col].astype(float).round(4)
    forecast_path = os.path.join(RESULTS_DIR, "forecast.csv")
    forecast_csv.to_csv(forecast_path, index=False)
    print(f"[out] wrote {forecast_path}")

    closes = forecast_csv["close"].tolist()
    pct_change = (closes[-1] / last_actual_close - 1.0) * 100.0
    ctx_start = pd.Timestamp(context["date"].iloc[0])
    ctx_end = pd.Timestamp(context["date"].iloc[-1])

    lines = []
    lines.append(f"# Kronos-small forecast — {args.ticker}")
    lines.append("")
    lines.append(f"- Model: `{MODEL_ID}` with tokenizer `{TOKENIZER_ID}` (CPU inference, seed 42, T=1.0, top_p=0.9, sample_count=1)")
    lines.append(f"- Data source: {source} (yfinance is never used for Cairo tickers — Africa/Cairo DST crash; rows before 2019-01-01 dropped as corrupted)")
    lines.append(f"- Context window used: {ctx_start.date()} to {ctx_end.date()} — {len(context)} sessions")
    if fallback_used:
        lines.append(f"- **FALLBACK USED**: {fallback_reason}. The fallback window spans the POUL merger discontinuity, so pre-merger and post-merger prices are mixed in one context — treat this forecast with extra caution.")
    else:
        lines.append(f"- Preferred post-merger context (from {args.context_start}) was accepted; no fallback needed.")
    lines.append(f"- Last actual close: {last_actual_close} on {last_actual_date.date()}")
    lines.append(f"- Forecast horizon: next {args.pred_len} sessions (Sun–Thu calendar; exchange holidays not modelled)")
    lines.append("")
    lines.append("## Forecasted close path")
    lines.append("")
    for d, c in zip(forecast_csv["date"], closes):
        lines.append(f"- {d}: {c}")
    lines.append("")
    lines.append(f"**Change over the {args.pred_len} sessions (last forecasted close vs last actual close): {pct_change:+.2f}%**")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append(
        "Independent evaluations show Kronos direction accuracy near coin-flip — "
        "this forecast is a research lens, not a trading signal. Kronos is a "
        "sampling model: reruns can differ. Do not trade on this output alone."
    )
    lines.append("")
    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[out] wrote {report_path}")
    print(f"[result] last_actual_close={last_actual_close} date={last_actual_date.date()} pct_change={pct_change:+.2f}%")
    print(f"[result] forecasted_closes={closes}")


if __name__ == "__main__":
    main()
