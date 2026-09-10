# Signal Desk — Project Context

This file exists so a fresh Claude Code session (or a human) can pick up
this project's full history and reasoning without re-reading the chat
that produced it. It reflects the state as of the last update below —
if you make changes, update this file to match.

## What this is

A quantitative signal-generation + backtesting + live-tracking system for
a fixed universe: Mag 7, a semiconductor basket, energy majors, and
BTC/ETH. It scores each ticker daily on a composite of three signals
(curvature, momentum, statistical anomaly), decides LONG/NEUTRAL/SKIP,
and — the newer half of the project — logs every real LONG signal it
issues and tracks what actually happens to it going forward, so the
system's quality can be judged on real outcomes instead of another
backtest.

## Files

- `signal_tracker.py` — the whole system in one file: data loading,
  feature engineering, the scoring pipeline, the backtester, the
  historical "deep dive" analysis, and the live signal log/scorecard.
  Runs standalone (`python signal_tracker.py`) or pasted into a Colab
  cell (run `pip install -r requirements.txt` first in Colab).
- `requirements.txt` — yfinance, numpy, pandas, scipy.
- `.github/workflows/signal-tracker.yml` — runs `signal_tracker.py`
  every 15 minutes, all day every day, in `tracker` mode (see below), and
  commits the updated log back to the repo. The cron is deliberately
  unfiltered — `is_extended_hours_now()` inside the script is the real
  gate (see step 8 below).

## Key environment variables

- `SIGNAL_DESK_MODE` — `"full"` (default) runs everything including the
  historical backtest and the deep-dive analysis section; `"tracker"`
  skips straight to load → score → regime-gate → live-signal-log/
  scorecard. The GitHub Actions workflow sets this to `tracker` so it
  isn't re-running a full historical backtest every 15 minutes.
- `SIGNAL_LOG_PATH` — where the live signal log CSV lives. The workflow
  sets this to `signal_desk_log.csv` (repo root) so it can be committed.
  Falls back to Google Drive if running in Colab, or a local path
  otherwise.

## How the persistence model works

GitHub Actions runners are ephemeral — nothing survives between runs
except what's in the repo. So the workflow's last step commits
`signal_desk_log.csv` back to the repo after every run (`git add` /
`git commit` / `git push`). This also means the repo's commit history is
a full audit trail of every signal ever logged.

**Setup gotcha:** the default `GITHUB_TOKEN` needs write access for the
push step to succeed. If pushes fail with a permission error, go to repo
Settings → Actions → General → Workflow permissions → select "Read and
write permissions."

## Design history (why things are the way they are)

The system went through several iterations, each in response to a
specific, empirically-observed problem — not speculative hardening.
Worth knowing this order, because some fixes only make sense in light of
what came before:

1. **Original backtest.** Composite score (curvature + momentum +
   anomaly), fixed stop/target/time exits, `max_pos=4`,
   `risk_frac=0.0075`. On the real universe: **-8.99% total return,
   46.4% max drawdown, profit factor 0.97.** Also found: the conviction
   score's quartiles were NOT cleanly monotonic — Q4 (highest score)
   barely beat Q1, and Q3 sometimes underperformed Q1. That specific
   problem (score quality) was never fixed — see "Known unresolved
   issues" below.

2. **Option B (structural risk controls)** — added on user request after
   comparing three restructuring options (a Grok recommendation):
   - Hard sector caps: max 2 concurrent Semis positions, max 1 Crypto.
   - Regime filter: only take new longs when QQQ is above its 50-day MA.
   - Stop moves to breakeven at +1R, then trails by `stop_k × ATR`.
   Result on real data: **+31.77% return, but max drawdown barely moved
   (47.09%)** — worse than the original. Diagnosis: the regime filter
   only ever blocked *new* entries; it never touched positions already
   open when the regime flipped. That gap motivated the next change.

3. **Portfolio circuit breaker + aggregate risk cap.** When regime flips
   ON→OFF while positions are open, every open position (any sector) is
   force-closed or halved (`breaker_action`, default `"halve"`). Also
   added `max_total_risk_frac` — a hard ceiling on total $ risk across
   ALL open positions combined (not just per-sector counts), with new
   positions sized down (not skipped) to fit the remaining budget.
   Result: **max drawdown finally moved (47.1% → 39.3%), but total
   return collapsed to 9.4%** — return/drawdown ratio actually got
   *worse* (0.68 → 0.24). Diagnosis: the breaker fired 22 times in ~2.3
   years — too often to be catching real crises, mostly whipsawing on
   ordinary noise around the bare 50-day MA cross. `regime_breaker_half`
   exits averaged **-1.32%**, i.e. realizing small losses, not
   protecting gains.

4. **Hysteresis + cooldown.** Fixed the trigger, not the response size
   (user had no strong preference between softening the haircut vs.
   fixing the signal; this was the more evidence-backed choice). Regime
   signal now uses a 2% band around the MA (`regime_buffer_pct`) instead
   of a bare cross — must close back above/below the band, not just tick
   across the average. Added `breaker_cooldown_days=10` as a second,
   independent guard against re-firing too often even within the band.
   Synthetic validation: bare-cross flips 18 → 8 with hysteresis;
   breaker activations 22 → 3 with the cooldown added.
   **This version has NOT yet been run against real market data** — that
   real-data run is the natural next step if picking this project back
   up.

5. **Live signal tracking (forward-test, not backtest).** Added
   `record_signals` / `mark_to_market` / `signal_scorecard`. Logs every
   real LONG signal the system issues, then checks it against REAL
   future price action to resolve it (stop/target/time-out) — the
   explicit point being to answer "are these signals good" without
   hindsight bias, since a backtest can't self-certify. Initially
   close-price-only for resolution.

6. **Intraday High/Low resolution.** Necessary once GitHub Actions
   scheduling entered the picture: checking only the daily Close means
   running the checker more than once a day changes nothing. Updated
   `mark_to_market` to use each day's intraday High/Low (when supplied)
   to catch a stop/target touch that happened mid-day, not just at
   close — conservative fill assumption: filled AT the stop/target
   level, not the day's actual best/worst print. Validated with a
   synthetic case where Close-only missed an intraday stop touch
   entirely and High/Low correctly caught it 2 days earlier.
   Entry day itself is deliberately excluded from resolution (using that
   day's full range could flag a stop hit from *before* the signal
   actually fired).

7. **GitHub Actions automation** (this stage). Added `SIGNAL_DESK_MODE`
   (full vs tracker) so the scheduled job doesn't rerun the full
   backtest every 15 minutes, `is_market_open_now()` as a real
   America/New_York market-hours gate (the cron schedule itself is
   deliberately coarse — a wide UTC band covering both EST/EDT — with
   the Python-side check doing the actual precision), and made the log
   path configurable via `SIGNAL_LOG_PATH` for repo-based persistence.

8. **Extended-hours + weekend crypto tracking.** The original gate
   (`is_market_open_now()`) was scoped to the 9:30am-4:00pm ET regular
   session only, and hard-blocked weekends entirely — which meant the
   tracker never ran during pre-market, after-hours, or on Saturday/
   Sunday, even though BTC-USD/ETH-USD trade 24/7 and have real signals
   to log and mark-to-market at any of those times. Renamed to
   `is_extended_hours_now()` and widened to 4:00am-8:00pm ET on
   weekdays (pre-market + regular + after-hours), and made it
   unconditionally `True` on weekends rather than `False` — an equity
   ticker just re-shows its last close with nothing new on a Saturday,
   which is a harmless no-op (same tradeoff the holiday-calendar gap
   already accepted), but crypto keeps moving. Also simplified the
   workflow's cron to `*/15 * * * *` (every 15 min, unconditionally)
   instead of trying to hand-encode the widened window in UTC — the
   Python-side check was already the precise gate by design (see step
   7), so there's no reason for the cron itself to also carry that
   precision, and this avoids the DST/day-of-week wraparound edge cases
   a hand-tuned cron window would introduce.

## Known unresolved issues (don't assume these are fixed)

- **The conviction score itself is still not well-calibrated.** This was
  flagged in the very first backtest and never addressed — none of the
  Option B/breaker/hysteresis work touched the scoring mechanism. If
  score quartiles are still non-monotonic on a fresh run, that's
  expected, not a regression.
- **No walk-forward or out-of-sample validation has been done.** Every
  parameter (sector caps, breaker thresholds, hysteresis band, cooldown
  days) was tuned by looking at results on the same historical window.
  That's normal early development, but none of the reported numbers
  demonstrate the system generalizes to unseen data.
- **No transaction costs or slippage modeled anywhere** — at ~400-500
  backtested trades, that's not a rounding error.
- **Not tested against a real crisis period** (e.g. 2022 rate-hike bear
  market, 2020 COVID crash) — the historical window used so far is a
  comparatively calm few years.
- **`is_extended_hours_now()` has no holiday calendar.** It'll still
  "run" on market holidays; Yahoo Finance just won't have a new bar, so
  it's a harmless but non-free no-op.
- **The workflow now fires every 15 minutes 24/7 instead of only during
  a market-hours-shaped window.** More GitHub Actions minutes are spent
  on no-op runs (nights/weekends for the equity side) than before —
  intentional per step 8's tradeoff, but worth knowing if Actions usage
  becomes a concern.
- **yfinance from GitHub Actions' cloud IPs may get rate-limited or
  blocked** more than from a residential IP. The script already
  degrades gracefully on partial ticker failures, but if this becomes a
  persistent problem, the fix is a paid data provider, not more retries.
- **The hysteresis+cooldown version (step 4 above) has only been
  validated on synthetic data, never run against the real universe.**

## Explicitly rejected / not pursued

- **Redis / external DB for persistence** — considered, rejected in
  favor of committing the CSV to git: simpler, free, versioned, no extra
  accounts, and the data volume/access pattern doesn't need a real
  database.
- **Claude Cowork scheduled tasks as the execution engine** — considered
  since the user has Claude Pro. Rejected because Cowork's fastest
  cadence is hourly (not 15 min), and its execution model is an
  LLM-driven agent session rather than a deterministic pinned script —
  a meaningful mismatch for something meant to run identically every
  time. Decided to keep GitHub Actions as the sole execution engine.
