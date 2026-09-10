# Signal Desk

A quantitative scanner (Mag 7 + Semis + Energy + Crypto), backtester, and
live signal tracker. See **[CONTEXT.md](./CONTEXT.md)** for the full
design history, the reasoning behind every parameter, and known
unresolved issues — read that first if you're picking this project up.

## Quick start

**Locally / in a scheduled job:**
```bash
pip install -r requirements.txt
python signal_tracker.py
```

**In Google Colab:** paste the contents of `signal_tracker.py` into a
cell, after running `!pip install -r requirements.txt` (or
`!pip install yfinance numpy pandas scipy`) in the cell before it.

## Modes

- `SIGNAL_DESK_MODE=full` (default) — full historical backtest + deep
  dive analysis + live signal tracking.
- `SIGNAL_DESK_MODE=tracker` — skips the backtest, just scores today's
  signals and updates the live tracking log. This is what the scheduled
  GitHub Actions job uses.

## Automation

`.github/workflows/signal-tracker.yml` runs the tracker every 15 minutes
during US market hours and commits the updated `signal_desk_log.csv`
back to this repo. **Before it can push:** go to this repo's
Settings → Actions → General → Workflow permissions, and select
"Read and write permissions" — otherwise the commit-back step will fail.

No API keys or secrets are needed (Yahoo Finance doesn't require auth).

## The live signal log

`signal_desk_log.csv` (created on first run) is the actual record of
every LONG signal the system has issued and what happened to it —
this is the honest, forward-tested answer to "is this any good," as
opposed to another backtest. Browse the file's git history for a
timestamped trail of every entry ever logged.
