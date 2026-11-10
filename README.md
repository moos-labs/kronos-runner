# kronos-runner

Runs **Kronos** (github.com/shiyu-coder/Kronos) — a foundation model for
candlestick (K-line) forecasting — via GitHub Actions, and returns the
forecast as a downloadable artifact. Part of the moos-labs stock-tools
fleet: one public repo per tool, commands in a repository variable,
manual dispatch, results as artifacts. Scripts are generic — ticker and
parameters are CLI arguments, nothing is EGX-hardcoded.

## The pattern

1. Edit the repo **Actions variable `COMMANDS`** (Settings → Secrets and
   variables → Actions → Variables, or the variables API).
2. Dispatch the workflow: Actions → *Run analysis* → *Run workflow*
   (or `POST /repos/moos-labs/kronos-runner/actions/workflows/run.yml/dispatches`).
3. Download the **`results`** artifact from the finished run.

Current `COMMANDS`:

```
mkdir -p results
set -o pipefail; python scripts/kronos_forecast.py --ticker POUL.CA --pred-len 10 2>&1 | tee results/run.log
```

(`set -o pipefail;` matters: without it, piping through `tee` would mask
a script failure and the run would wrongly show as success.)

## What the pilot does

`scripts/kronos_forecast.py` (args: `--ticker`, `--pred-len`,
`--context-start`, `--fallback-len`):

- Fetches daily OHLC via **yahooquery** (proven for POUL.CA), falling
  back to the direct Yahoo chart API. **yfinance is never used for
  Cairo tickers** — it crashes on Africa/Cairo DST timestamps. Rows
  before 2019-01-01 are always dropped (corrupted segment).
- Context: all sessions from **2026-09-28** (POUL's post-merger
  resumption) to the latest session. If Kronos refuses that short
  context, it falls back to the last 400 sessions and **flags the
  merger discontinuity** in the report.
- Runs **Kronos-small** (`NeoQuasar/Kronos-small` +
  `NeoQuasar/Kronos-Tokenizer-base`) on CPU. Kronos is not a PyPI
  package, so the workflow clones the upstream repo into `kronos_src/`
  and installs its `requirements.txt`; CPU-only torch is installed
  first from the PyTorch CPU wheel index. The Hugging Face cache
  (`~/.cache/huggingface`) is cached with `actions/cache` under a
  `kronos-hf-v1-` key so reruns skip the model download.

## Outputs (in the `results` artifact)

- `forecast.csv` — dates + forecasted OHLC (plus volume/amount) for
  the next N sessions (Sun–Thu calendar; holidays not modelled).
- `report.md` — context window used, last actual close with date,
  forecasted close path, % change over the horizon, and the caveat
  below.
- `run.log` — full run log (via the `tee` in `COMMANDS`).

## Honest notes on the tool

- Kronos-small is a 24.7M-parameter model with a 512-token max context;
  it is small enough for a free CPU GitHub runner, but inference is
  sampling-based (T=1.0, top_p=0.9) — reruns can differ even with a
  fixed seed.
- **Independent evaluations show Kronos direction accuracy near
  coin-flip.** This runner is a research lens, not a trading signal.
  The upstream authors likewise describe their pipelines as
  demonstrations, not production trading systems.
- Forecast dates are the next Sun–Thu sessions; EGX holidays are not
  modelled, so dates can drift around holidays.
