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

Context cap (borsagator #436, step-4 evidence): upstream
``KronosPredictor.predict()`` computes its normalisation statistics
over ALL supplied bars but attends over only the last ``max_context``
tokens. Supplying more bars than the predictor attends to therefore
anchors the forecast to the old regime (COMI.CA @ 2023-12-29: 1,231
bars supplied -> month error 57.3 points; truncating the supplied
context to the last 512 cut it to 17.8). The runner truncates the
supplied frame to the last MAX_CONTEXT sessions before predict(), so
normalisation and attention always cover the same window.
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd

RESULTS_DIR = "results"
DROP_BEFORE = pd.Timestamp("2019-01-01")
MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
UPSTREAM_KRONOS_REPO = "https://github.com/shiyu-coder/Kronos.git"
# Pinned upstream commit (borsagator #436 reproducibility): the clone
# is checked out at this commit instead of tracking upstream HEAD, so
# a backtest cell runs the same model code every time. This is the
# commit the #436 step 1-4 investigation actually measured.
UPSTREAM_KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
# The predictor's attention window; also the cap on supplied context
# (see the module docstring). Must match the max_context passed to
# KronosPredictor in run_forecast.
MAX_CONTEXT = 512
# Metadata of the most recent run_forecast call, for the report:
# supplied/used context lengths, upstream commit, HF revisions,
# dependency versions. Empty until a forecast has run.
LAST_RUN_META = {}


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
    # Normalise per value: yahooquery can return a frame mixing tz-aware
    # and tz-naive dates, and .dt.tz_localize(None) raises ValueError on
    # such a mix (#111, run 43807031623). Parsing with utc=True coerces
    # every value to UTC first, then the tz is dropped — same pattern as
    # fetch_chart_api above. format="mixed" also tolerates mixed string
    # formats (offsets vs plain). .normalize() behaviour is unchanged.
    df["date"] = (
        pd.to_datetime(df["date"], utc=True, format="mixed").dt.tz_convert(None).dt.normalize()
    )
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0 if col == "volume" else float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["date"] >= DROP_BEFORE]
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def truncate_asof(df, as_of, min_rows):
    """Walk-forward cutoff (borsagator #436): keep only sessions on or
    before ``as_of`` (YYYY-MM-DD), then fail loudly if fewer than
    ``min_rows`` sessions remain.

    Truncation happens here, inside the runner after fetch and before
    any model call, so no row dated after the cutoff can reach Kronos.
    ``as_of`` itself is kept. A cutoff on a non-session date keeps the
    last session on/before it.
    """
    cutoff = pd.Timestamp(as_of)
    out = df[df["date"] <= cutoff].reset_index(drop=True)
    if len(out) < min_rows:
        raise RuntimeError(
            f"--as-of {as_of}: only {len(out)} sessions on/before the "
            f"cutoff (< {min_rows} minimum rows); refusing to forecast"
        )
    return out


def truncate_context(df, max_context=MAX_CONTEXT):
    """Keep only the last ``max_context`` sessions (borsagator #436).

    Upstream ``predict()`` normalises over ALL supplied bars but
    attends over only the last ``max_context`` tokens; supplying more
    than that anchors the forecast to the old regime (step-4 evidence:
    COMI.CA month error 57.3 points at 1,231 supplied bars vs 17.8 at
    512). Truncating here -- inside the runner, before the model call
    -- makes normalisation statistics and attention cover the same
    window. A frame at or under the cap is returned untouched; order
    is preserved.
    """
    if len(df) <= max_context:
        return df
    return df.tail(max_context).reset_index(drop=True)


def select_context(df, context_start, fallback_len, min_context):
    """Preferred window ``context_start``..latest session; if that window
    is below ``min_context`` sessions, fall back to the last
    ``fallback_len`` sessions. Returns (context, fallback_used, reason).

    "Latest session" is the last row of ``df`` -- after --as-of
    truncation that is the as-of session, so the same rule serves the
    live run and the backtest, with as-of substituted for "today".
    """
    preferred = df[df["date"] >= pd.Timestamp(context_start)].reset_index(drop=True)
    if len(preferred) >= min_context:
        return preferred, False, ""
    reason = (
        f"preferred window ({context_start} onwards) has only "
        f"{len(preferred)} sessions (< {min_context} minimum)"
    )
    return df.tail(fallback_len).reset_index(drop=True), True, reason


def next_sessions(last_date, n):
    """Next n EGX sessions (Sun-Thu; Fri/Sat weekend). Holidays not modelled."""
    out = []
    d = pd.Timestamp(last_date)
    while len(out) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() not in (4, 5):  # 4=Fri, 5=Sat
            out.append(d)
    return pd.Series(out)


def ensure_kronos_src(kronos_src="kronos_src", run=None):
    """Clone the upstream Kronos model package when it is missing.

    Kronos is not a PyPI package: run_forecast imports its `model`
    package from kronos_src/. The runner's own CI does this clone in
    .github/workflows/run.yml, but the worker path never did -- so on
    worker-5 `from model import Kronos` would fail right after the
    torch import was fixed (issue #198: the torch error only masked
    the missing-src error). Returns True when it cloned, False when
    kronos_src/model already existed. Raises RuntimeError if the
    clone or the upstream requirements install fails, so the tool
    fails loudly instead of running against a half-present model.
    CPU torch is installed first (this repo's requirements.txt, the
    PyTorch CPU wheel index), so the upstream `torch>=2.0.0` line is
    already satisfied and pip does not pull the CUDA wheel.
    """
    if os.path.isdir(os.path.join(kronos_src, "model")):
        return False
    run = run or subprocess.run
    r = run(["git", "clone", "--depth", "1",
             UPSTREAM_KRONOS_REPO, kronos_src],
            capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            "cloning upstream Kronos failed: %s"
            % ((r.stderr or r.stdout or "")[-500:]))
    # Pin the fresh clone to UPSTREAM_KRONOS_COMMIT (borsagator #436):
    # a --depth 1 clone tracks upstream HEAD, so fetch the pinned
    # commit explicitly and detach there. Fail loudly if either step
    # fails -- running against an unpinned upstream would silently
    # make backtest cells non-reproducible.
    for cmd in (
        ["git", "-C", kronos_src, "fetch", "--depth", "1", "origin",
         UPSTREAM_KRONOS_COMMIT],
        ["git", "-C", kronos_src, "checkout", "--detach",
         UPSTREAM_KRONOS_COMMIT],
    ):
        r = run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(
                "pinning upstream Kronos to %s failed: %s"
                % (UPSTREAM_KRONOS_COMMIT,
                   (r.stderr or r.stdout or "")[-500:]))
    req = os.path.join(kronos_src, "requirements.txt")
    if os.path.isfile(req):
        r = run([sys.executable, "-m", "pip", "install", "-q",
                 "-r", req], capture_output=True, text=True,
                timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(
                "installing upstream Kronos requirements failed: %s"
                % ((r.stderr or r.stdout or "")[-500:]))
    return True


def kronos_src_commit(kronos_src):
    """Full HEAD commit of the local upstream clone, or "unknown".

    Read-only (``git rev-parse``); used only for the report, so any
    failure degrades to "unknown" instead of failing the run.
    """
    try:
        r = subprocess.run(
            ["git", "-C", kronos_src, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:  # noqa: BLE001 - report metadata only
        pass
    return "unknown"


def hf_revision(repo_id, obj=None):
    """Best-effort HF revision (commit hash) for a model/tokenizer.

    Tries the loaded object's config first, then the Hub API. Report
    metadata only: any failure degrades to "unknown" and never fails
    the run (borsagator #436 reproducibility record).
    """
    try:
        rev = getattr(getattr(obj, "config", None), "_commit_hash", None)
        if rev:
            return str(rev)
    except Exception:  # noqa: BLE001 - report metadata only
        pass
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id)
        if getattr(info, "sha", None):
            return str(info.sha)
    except Exception:  # noqa: BLE001 - report metadata only
        pass
    return "unknown"


def dependency_versions():
    """Installed versions of the packages that shape a forecast."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("torch", "pandas", "numpy", "transformers", "yahooquery"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
        except Exception:  # noqa: BLE001 - report metadata only
            out[pkg] = "unknown"
    return out


def run_forecast(context_df, pred_len):
    # Kronos is not a PyPI package: the workflow clones the upstream repo
    # into kronos_src/ and we import its `model` package from there.
    kronos_src = os.environ.get("KRONOS_SRC", "kronos_src")
    # Issue #198: provision kronos_src on the worker path too (the
    # runner's CI does this in run.yml; the worker never did).
    ensure_kronos_src(kronos_src)
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
    predictor = KronosPredictor(model, tokenizer, device="cpu",
                                max_context=MAX_CONTEXT)

    # Borsagator #436: cap the supplied context at the predictor's
    # attention window BEFORE predict() -- upstream normalises over
    # all supplied bars but attends over only the last MAX_CONTEXT,
    # so a longer supply anchors the forecast to the old regime.
    supplied_len = len(context_df)
    context_df = truncate_context(context_df, MAX_CONTEXT)
    LAST_RUN_META.clear()
    LAST_RUN_META.update(
        context_supplied=supplied_len,
        context_used=len(context_df),
        max_context=MAX_CONTEXT,
        upstream_commit=kronos_src_commit(kronos_src),
        model_revision=hf_revision(MODEL_ID, model),
        tokenizer_revision=hf_revision(TOKENIZER_ID, tokenizer),
        dependency_versions=dependency_versions(),
    )
    print(f"[context] supplied={supplied_len} used={len(context_df)} "
          f"max_context={MAX_CONTEXT}")

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
    ap.add_argument("--as-of", default=None,
                    help="Walk-forward cutoff (YYYY-MM-DD, borsagator #436): "
                         "drop every session after this date before any "
                         "model call, so the forecast sees only data "
                         "available at the close of that date.")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, source = fetch_data(args.ticker)
    df = clean(df)
    if df.empty:
        raise RuntimeError("No usable rows after cleaning")
    if args.as_of:
        df = truncate_asof(df, args.as_of, args.min_context)
        print(f"[as-of] cutoff={args.as_of} rows_on_or_before={len(df)} "
              f"last_session={df['date'].iloc[-1].date()}")
    print(f"[data] cleaned_rows={len(df)} range={df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}")

    last_actual_close = float(df["close"].iloc[-1])
    last_actual_date = pd.Timestamp(df["date"].iloc[-1])

    context, fallback_used, fallback_reason = select_context(
        df, args.context_start, args.fallback_len, args.min_context)
    print(f"[context] preferred_sessions={len(df[df['date'] >= pd.Timestamp(args.context_start)])} using={len(context)} fallback={fallback_used}")

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
    if args.as_of:
        lines.append(f"- As-of cutoff: {args.as_of} (walk-forward backtest, borsagator #436 — every session after the cutoff was dropped before the model ran; context ends at the cutoff)")
    meta = LAST_RUN_META
    if meta:
        if meta["context_supplied"] > meta["context_used"]:
            cap_note = (f"truncated to the last {meta['context_used']} "
                        f"before predict()")
        else:
            cap_note = "no truncation needed"
        lines.append(
            f"- Context supplied to Kronos: {meta['context_supplied']} sessions, "
            f"{cap_note} (max_context={meta['max_context']}; upstream predict() "
            "normalises over all supplied bars but attends over only the last "
            "max_context — borsagator #436)")
        lines.append(
            f"- Upstream Kronos commit: `{meta['upstream_commit']}` "
            f"(pinned target: `{UPSTREAM_KRONOS_COMMIT}`)")
        lines.append(
            f"- Hugging Face revisions: model `{MODEL_ID}` @ "
            f"{meta['model_revision']}; tokenizer `{TOKENIZER_ID}` @ "
            f"{meta['tokenizer_revision']}")
        deps = ", ".join(f"{k} {v}"
                         for k, v in meta["dependency_versions"].items())
        lines.append(f"- Dependency versions: {deps}")
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
