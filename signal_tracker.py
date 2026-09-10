# ============================================================
# Full self-contained notebook cell
# Mag7 + Semis + Energy + Crypto using Yahoo Finance
#
# UPDATED (Option B — structural risk controls):
#   1. Hard sector caps: max concurrent Semis / Crypto positions
#   2. Regime filter: only take new longs when QQQ > its 50D MA
#   3. Exit logic: stop moves to breakeven after +1R, then trails
#
# UPDATED AGAIN — runnable standalone (Colab OR GitHub Actions):
#   - no Jupyter "!pip install" magic (see requirements.txt)
#   - SIGNAL_DESK_MODE=tracker skips the heavy backtest/deep-dive
#     sections and only does the live signal log + scorecard, for
#     cheap frequent scheduled runs
#   - market-hours guard: no-ops outside 9:30am-4:00pm ET, Mon-Fri
#   - mark_to_market now uses intraday High/Low (not just Close), so
#     running this every 15 minutes actually catches a stop/target
#     the moment it's touched instead of only at end of day
#
# In Colab: run `!pip install -r requirements.txt` (or the packages
# below) in a cell BEFORE pasting this script.
# ============================================================

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import chi2
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings("ignore")

# "full"    -> everything: pipeline, backtest, deep-dive analysis, AND live tracking
# "tracker" -> only the live signal log + scorecard (cheap; what the scheduled
#              GitHub Actions job should use so it isn't rerunning a full
#              historical backtest every 15 minutes for no reason)
MODE = os.environ.get("SIGNAL_DESK_MODE", "full")


def is_market_open_now():
    """True during regular US market hours (9:30am-4:00pm ET), Mon-Fri.
    Does NOT know about market holidays (Thanksgiving, Christmas, etc.) —
    on those days this still returns True and the run will just find no
    new bar from Yahoo Finance, which is harmless but not free (still
    burns an API call + a few seconds of compute)."""
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    if now_ny.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    open_t = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_ny <= close_t

# ------------------------------------------------------------
# Universe
# ------------------------------------------------------------
TICKERS = [
    # Mag 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Semis
    "AMD", "AVGO", "TSM", "ASML", "QCOM", "MU", "AMAT", "LRCX", "KLAC",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "XLE",
    # Crypto
    "BTC-USD", "ETH-USD"
]

def get_group(t):
    if t in ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA"]:
        return "Mag7"
    elif t in ["AMD","AVGO","TSM","ASML","QCOM","MU","AMAT","LRCX","KLAC"]:
        return "Semis"
    elif t in ["XOM","CVX","COP","SLB","EOG","XLE"]:
        return "Energy"
    else:
        return "Crypto"

# ------------------------------------------------------------
# 1. Load data from Yahoo Finance
#    UPDATED — also returns High/Low series per symbol so
#    mark_to_market can check intraday stop/target touches, not just
#    the daily close
# ------------------------------------------------------------
def load_universe_yahoo(tickers, start="2023-07-01", end=None):
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    print("Downloading data from Yahoo Finance...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False
    )
    
    prices, volumes, highs, lows = {}, {}, {}, {}
    
    for t in tickers:
        try:
            df = data if len(tickers) == 1 else data[t]
            if df is None or df.empty or "Close" not in df.columns:
                print(f"  ✗ {t}")
                continue
            prices[t] = df["Close"].dropna()
            volumes[t] = df["Volume"].dropna()
            highs[t] = df["High"].dropna() if "High" in df.columns else df["Close"].dropna()
            lows[t] = df["Low"].dropna() if "Low" in df.columns else df["Close"].dropna()
            print(f"  ✓ {t:10s}  {len(prices[t])} bars")
        except Exception as e:
            print(f"  ✗ {t} – {e}")
    
    return prices, volumes, highs, lows


# ------------------------------------------------------------
# 1b. Regime filter (NEW — Option B)
#     UPDATED — hysteresis band instead of a bare MA cross, to stop
#     whipsaw-triggering the portfolio circuit breaker on ordinary noise
# ------------------------------------------------------------
def compute_regime_series(start="2023-07-01", ma_window=50, regime_symbol="QQQ", buffer_pct=0.02):
    """Returns a pd.Series of bool, indexed by date: True when regime_symbol is
    considered risk-on.

    Uses a hysteresis band around the moving average instead of a bare cross:
    once risk-off, price must close back ABOVE sma*(1+buffer_pct) to flip
    risk-on again; once risk-on, price must close BELOW sma*(1-buffer_pct) to
    flip risk-off. A bare cross (buffer_pct=0) whipsaws constantly whenever
    price oscillates within noise distance of its own average — the band
    requires a real move through the average before the regime actually
    flips, which is what a "circuit breaker" trigger should require.
    """
    qqq = yf.download(regime_symbol, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(qqq, pd.DataFrame):
        qqq = qqq.iloc[:, 0]
    sma = qqq.rolling(ma_window).mean()
    upper = sma * (1 + buffer_pct)
    lower = sma * (1 - buffer_pct)

    regime = pd.Series(index=qqq.index, dtype=bool)
    state = True
    for i in range(len(qqq)):
        if pd.isna(sma.iloc[i]):
            regime.iloc[i] = state
            continue
        if i == 0 or pd.isna(sma.iloc[i - 1]):
            # first valid reading: fall back to a plain comparison to seed state
            state = bool(qqq.iloc[i] > sma.iloc[i])
        elif state and qqq.iloc[i] < lower.iloc[i]:
            state = False
        elif (not state) and qqq.iloc[i] > upper.iloc[i]:
            state = True
        regime.iloc[i] = state

    return regime.rename("regime_on")


def apply_regime_gate(df_sig, regime_series):
    """Downgrades any 'long' signal to 'neutral' on dates where the regime
    filter is off. Keeps the pre-gate signal in `raw_signal` for transparency."""
    df_sig = df_sig.copy()
    df_sig["raw_signal"] = df_sig["signal"]

    def regime_ok(dt):
        try:
            val = regime_series.asof(dt)
            return bool(val) if pd.notna(val) else False
        except Exception:
            return False

    regime_map = {dt: regime_ok(dt) for dt in df_sig["date"].unique()}
    df_sig["regime_on"] = df_sig["date"].map(regime_map)
    df_sig["signal"] = np.where(
        (df_sig["raw_signal"] == "long") & (~df_sig["regime_on"]),
        "neutral",
        df_sig["raw_signal"],
    )
    return df_sig


# ------------------------------------------------------------
# 2. Feature & scoring helpers
# ------------------------------------------------------------
def rolling_robust_z(arr, window=60):
    T, d = arr.shape
    out = np.zeros_like(arr)
    for t in range(T):
        start = max(0, t - window + 1)
        sub = arr[start:t+1]
        med = np.nanmedian(sub, axis=0)
        mad = np.nanmedian(np.abs(sub - med), axis=0) + 1e-9
        out[t] = (arr[t] - med) / mad
    return out

def build_state_vector(price, volume, ret_lags=(1,5,20), vol_windows=(10,20)):
    logp = np.log(price)
    ret = logp.diff().fillna(0.0)
    df = pd.DataFrame(index=price.index)
    for k in ret_lags:
        df[f"r{k}"] = ret.rolling(k).sum().fillna(0.0)
    for w in vol_windows:
        df[f"rv{w}"] = ret.rolling(w).std().fillna(0.0)
    df["logvol"] = np.log(volume + 1).ffill()
    df["price"] = price
    df["atr"] = price.diff().abs().rolling(14).mean().bfill().fillna(price * 0.008)
    return df

def curvature_score(X, r=2):
    if X.shape[0] <= r + 1:
        return 0.0
    Xc = X - X.mean(0)
    C = np.cov(Xc, rowvar=False) + 1e-8 * np.eye(X.shape[1])
    eig = np.linalg.eigvalsh(C)[::-1]
    if len(eig) <= r:
        return 0.0
    return float(eig[r:].sum() / (eig[:r].sum() + 1e-12))

def build_momentum_subspace(X, k=3):
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.eye(X.shape[1])[:, :min(k, X.shape[1])]
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])].T

def projection_score(U, xt, xprev):
    P = U @ U.T
    pt = P @ xt
    pp = P @ xprev
    m = np.linalg.norm(pt) + 1e-12
    s = np.dot(pt, pp) / ((np.linalg.norm(pp) + 1e-12) * m)
    return float(m * max(0.0, s))

def mahalanobis_score(X, xt, shrink=0.15):
    if X.shape[0] < 12:
        return 0.0, 0.0
    mu = X.mean(0)
    emp = np.cov(X, rowvar=False)
    target = np.diag(np.diag(emp))
    cov = (1 - shrink) * emp + shrink * target + 1e-8 * np.eye(X.shape[1])
    try:
        inv = np.linalg.pinv(cov)
        D2 = float((xt - mu) @ inv @ (xt - mu))
        p = 1 - chi2.cdf(max(D2, 0), df=X.shape[1])
        return float(-np.log(p + 1e-12)), D2
    except:
        return 0.0, 0.0

def entry_stop_target(price, atr, stop_k=2.0, RR=2.5):
    stop = price - stop_k * atr
    if stop >= price * 0.998:
        stop = price * 0.99
    risk = price - stop
    target = price + RR * risk
    return float(stop), float(target), float(RR)

# ------------------------------------------------------------
# 3. Signal Pipeline
# ------------------------------------------------------------
def run_pipeline(prices, volumes, params):
    symbols = sorted(prices.keys())
    states = {
        s: build_state_vector(prices[s], volumes[s],
                              params["ret_lags"], params["vol_windows"])
        for s in symbols
    }

    common = states[symbols[0]].index
    for s in symbols[1:]:
        common = common.intersection(states[s].index)
    for s in symbols:
        states[s] = states[s].loc[common]

    T = len(common)
    feats = [c for c in states[symbols[0]].columns if c not in ("price", "atr")]
    n_sym, n_f = len(symbols), len(feats)

    X = np.zeros((T, n_sym, n_f))
    P = np.zeros((T, n_sym))
    A = np.zeros((T, n_sym))
    ADV = np.zeros((T, n_sym))

    for i, s in enumerate(symbols):
        X[:, i, :] = states[s][feats].values
        P[:, i] = states[s]["price"].values
        A[:, i] = states[s]["atr"].values
        ADV[:, i] = np.exp(states[s]["logvol"].values)

    Xn = np.zeros_like(X)
    for i in range(n_sym):
        Xn[:, i, :] = rolling_robust_z(X[:, i, :], params["window"])

    records = []
    W = params["window"]
    for t in range(W, T):
        for i, s in enumerate(symbols):
            Xw = Xn[t-W+1:t+1, i, :]
            xt = Xn[t, i, :]
            xp = Xn[t-1, i, :]
            kappa = curvature_score(Xw, params["r"])
            U = build_momentum_subspace(Xw, params["k"])
            mom = projection_score(U, xt, xp)
            anom, _ = mahalanobis_score(Xw, xt)
            px = P[t, i]
            atr = max(A[t, i], px * 0.006)
            stop, tgt, rr = entry_stop_target(px, atr, params["stop_k"], params["RR"])
            records.append({
                "date": common[t],
                "symbol": s,
                "kappa": kappa,
                "momentum": mom,
                "anomaly": anom,
                "price": px,
                "atr": atr,
                "stop": stop,
                "target": tgt,
                "rr": rr,
                "adv": ADV[t, i]
            })

    df = pd.DataFrame(records)

    def norm(g, col):
        v = g[col].values
        lo, hi = np.nanpercentile(v, [3, 97])
        if hi - lo < 1e-9:
            return np.zeros(len(v))
        return np.clip((v - lo) / (hi - lo), 0, 1)

    for col in ["kappa", "momentum", "anomaly"]:
        df[f"{col}_n"] = df.groupby("date", group_keys=False).apply(
            lambda g: pd.Series(norm(g, col), index=g.index)
        )

    w = params["weights"]
    df["conv"] = (w[0] * df["kappa_n"] +
                  w[1] * df["momentum_n"] +
                  w[2] * df["anomaly_n"])
    df["score"] = df.groupby("date")["conv"].transform(
        lambda x: 100 * (x - x.min()) / (x.max() - x.min() + 1e-12)
    )

    def decide(r):
        if r["adv"] < params["min_adv"] or r["rr"] < params["min_rr"]:
            return "skip"
        if r["momentum_n"] >= params["mom_thresh"] and r["score"] >= params["long_thresh"]:
            return "long"
        return "neutral"

    df["signal"] = df.apply(decide, axis=1)
    return df

# ------------------------------------------------------------
# 4. Backtester
#    UPDATED — sector caps + breakeven/trailing stop (Option B)
#    UPDATED AGAIN — portfolio regime circuit breaker + aggregate
#    risk-budget cap (addresses drawdown not being fixed by Option B)
# ------------------------------------------------------------
def run_backtest(df_sig, prices, params, capital=100_000., risk_frac=0.0075):
    """
    Structural risk controls (Option B):
      - hard cap on concurrent Semis positions (params["max_semis"])
      - hard cap on concurrent Crypto positions (params["max_crypto"])
      - stop moves to breakeven after +1R, then trails by stop_k*ATR off the
        highest price seen since entry (params["use_trailing"])

    Portfolio-level controls (new):
      - circuit breaker: when the regime flips from risk-on to risk-off
        while positions are already open, EVERY open position (regardless
        of sector) is either force-closed or cut in half, per
        params["breaker_action"] ("close" or "halve"). This is the piece
        Option B was missing: the regime filter only ever blocked *new*
        entries, it never touched exposure already on the book. Subject
        to params["breaker_cooldown_days"] so a choppy signal can't
        re-fire every few days.
      - aggregate risk cap: the dollar risk committed across ALL
        simultaneously open positions (sum of entry-to-stop risk in $) is
        capped at params["max_total_risk_frac"] of equity, independent of
        sector. A new position is sized down (not just skipped) to fit
        whatever risk budget remains that day.
    """
    max_semis = params.get("max_semis", 2)
    max_crypto = params.get("max_crypto", 1)
    use_trailing = params.get("use_trailing", True)
    breaker_enabled = params.get("breaker_enabled", True)
    breaker_action = params.get("breaker_action", "halve")  # "halve" or "close"
    max_total_risk_frac = params.get("max_total_risk_frac", 0.02)
    breaker_cooldown_days = params.get("breaker_cooldown_days", 10)

    dates = sorted(df_sig["date"].unique())
    equity = capital
    curve = []
    trades = []
    breaker_events = []
    open_pos = {}

    # date -> atr lookup per symbol, for trailing-stop updates
    atr_lookup = {sym: dict(zip(g["date"], g["atr"])) for sym, g in df_sig.groupby("symbol")}

    # date -> regime_on lookup (one value per date, same across symbols)
    if "regime_on" in df_sig.columns:
        regime_by_date = df_sig.groupby("date")["regime_on"].first().to_dict()
    else:
        regime_by_date = {}
    regime_prev = regime_by_date.get(dates[0], True)
    last_breaker_date = None

    def group_count(group_name):
        return sum(1 for s in open_pos if get_group(s) == group_name)

    def total_open_risk():
        return sum(p.get("risk_amount", 0.0) for p in open_pos.values())

    for i, dt in enumerate(dates[:-1]):
        ndt = dates[i + 1]
        regime_today = regime_by_date.get(ndt, regime_prev)

        # --- portfolio regime circuit breaker: fires on ON -> OFF flip,
        # subject to a minimum cooldown so a choppy signal can't re-fire
        # every few days even after hysteresis ---
        cooldown_ok = (last_breaker_date is None or
                       (ndt - last_breaker_date).days >= breaker_cooldown_days)
        if breaker_enabled and regime_prev and not regime_today and open_pos and cooldown_ok:
            hit_syms = []
            for sym in list(open_pos.keys()):
                if ndt not in prices[sym].index:
                    continue
                pos = open_pos[sym]
                px = float(prices[sym].loc[ndt])
                ret = (px - pos["entry"]) / pos["entry"]

                if breaker_action == "close":
                    pnl = ret * pos["notional"]
                    trades.append({
                        "symbol": sym, "entry_date": pos["edate"], "exit_date": ndt,
                        "entry": pos["entry"], "exit": px, "ret": ret, "pnl": pnl,
                        "reason": "regime_breaker"
                    })
                    equity += pnl
                    del open_pos[sym]
                else:  # "halve"
                    half_notional = pos["notional"] * 0.5
                    pnl = ret * half_notional
                    trades.append({
                        "symbol": sym, "entry_date": pos["edate"], "exit_date": ndt,
                        "entry": pos["entry"], "exit": px, "ret": ret, "pnl": pnl,
                        "reason": "regime_breaker_half"
                    })
                    equity += pnl
                    pos["notional"] -= half_notional
                    pos["risk_amount"] = pos.get("risk_amount", 0.0) * 0.5
                hit_syms.append(sym)
            if hit_syms:
                breaker_events.append({"date": ndt, "action": breaker_action, "symbols": hit_syms})
                last_breaker_date = ndt

        regime_prev = regime_today

        # Manage open positions
        to_close = []
        for sym, pos in open_pos.items():
            if ndt not in prices[sym].index:
                continue
            px = float(prices[sym].loc[ndt])

            # --- breakeven + trailing stop management ---
            pos["high"] = max(pos.get("high", pos["entry"]), px)
            if use_trailing and not pos.get("be_triggered") and px >= pos["entry"] + pos["risk"]:
                pos["stop"] = pos["entry"]          # move to breakeven at +1R
                pos["be_triggered"] = True
            if use_trailing and pos.get("be_triggered"):
                atr_today = atr_lookup.get(sym, {}).get(ndt)
                if atr_today is not None:
                    trail_stop = pos["high"] - params["stop_k"] * atr_today
                    pos["stop"] = max(pos["stop"], trail_stop)  # stop only ratchets up

            reason = None
            if px <= pos["stop"]:
                reason = "trail_stop" if pos.get("be_triggered") else "stop"
            elif px >= pos["target"]:
                reason = "target"
            elif (ndt - pos["edate"]).days >= params["max_hold"]:
                reason = "time"

            if reason:
                ret = (px - pos["entry"]) / pos["entry"]
                pnl = ret * pos["notional"]
                trades.append({
                    "symbol": sym,
                    "entry_date": pos["edate"],
                    "exit_date": ndt,
                    "entry": pos["entry"],
                    "exit": px,
                    "ret": ret,
                    "pnl": pnl,
                    "reason": reason
                })
                equity += pnl
                to_close.append(sym)

        for s in to_close:
            del open_pos[s]

        # New entries — now respects sector caps AND an aggregate
        # portfolio-level risk budget (sized down, not just skipped)
        cands = (df_sig[(df_sig["date"] == dt) & (df_sig["signal"] == "long")]
                 .sort_values("score", ascending=False))
        slots = params["max_pos"] - len(open_pos)

        for _, row in cands.iterrows():
            if slots <= 0:
                break
            sym = row["symbol"]
            if sym in open_pos or ndt not in prices[sym].index:
                continue
            grp = get_group(sym)
            if grp == "Semis" and group_count("Semis") >= max_semis:
                continue
            if grp == "Crypto" and group_count("Crypto") >= max_crypto:
                continue
            entry = float(prices[sym].loc[ndt])
            stop = row["stop"]
            risk_dist = entry - stop
            if risk_dist <= entry * 0.002:
                continue

            remaining_budget = max(0.0, equity * max_total_risk_frac - total_open_risk())
            if remaining_budget <= 0:
                continue  # portfolio risk cap fully used for today
            risk_amount = min(equity * risk_frac, remaining_budget)
            if risk_amount <= 0:
                continue

            shares = risk_amount / risk_dist
            notional = shares * entry
            open_pos[sym] = {
                "entry": entry,
                "stop": stop,
                "target": row["target"],
                "notional": notional,
                "edate": ndt,
                "risk": risk_dist,
                "risk_amount": risk_amount,
                "high": entry,
                "be_triggered": False,
            }
            slots -= 1

        curve.append({"date": ndt, "equity": equity, "n_open": len(open_pos)})

    # Force close remaining
    last = dates[-1]
    for sym, pos in list(open_pos.items()):
        if last in prices[sym].index:
            px = float(prices[sym].loc[last])
            ret = (px - pos["entry"]) / pos["entry"]
            pnl = ret * pos["notional"]
            trades.append({
                "symbol": sym,
                "entry_date": pos["edate"],
                "exit_date": last,
                "entry": pos["entry"],
                "exit": px,
                "ret": ret,
                "pnl": pnl,
                "reason": "eod"
            })
            equity += pnl

    return pd.DataFrame(trades), pd.DataFrame(curve), equity, breaker_events

# ------------------------------------------------------------
# 5. Metrics
# ------------------------------------------------------------
def compute_metrics(trades, curve, init_cap):
    if len(trades) == 0:
        return {"n_trades": 0}

    r = trades["ret"].values
    wins = r[r > 0]
    losses = r[r <= 0]
    n = len(r)
    wr = len(wins) / n
    aw = wins.mean() if len(wins) else 0.0
    al = abs(losses.mean()) if len(losses) else 1e-9
    exp = wr * aw - (1 - wr) * al

    gp = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gl = abs(trades.loc[trades["pnl"] <= 0, "pnl"].sum())
    pf = gp / gl if gl > 0 else np.inf

    eq = curve["equity"].values
    peak = np.maximum.accumulate(eq)
    mdd = ((peak - eq) / peak).max()
    tot_ret = eq[-1] / init_cap - 1
    days = (curve["date"].iloc[-1] - curve["date"].iloc[0]).days
    ann = (1 + tot_ret) ** (365.25 / max(days, 1)) - 1

    return {
        "n_trades": n,
        "win_rate": round(wr, 4),
        "avg_win": round(aw, 4),
        "avg_loss": round(al, 4),
        "expectancy": round(exp, 4),
        "profit_factor": round(pf, 3),
        "max_drawdown": round(mdd, 4),
        "total_return": round(tot_ret, 4),
        "ann_return_approx": round(ann, 4),
        "final_equity": round(eq[-1], 0),
        "realized_RR": round(aw / al, 2)
    }

# ------------------------------------------------------------
# 6. Execute everything
# ------------------------------------------------------------
params = {
    "ret_lags": (1, 5, 20),
    "vol_windows": (10, 20),
    "window": 60,
    "r": 2,
    "k": 3,
    "stop_k": 1.4,
    "RR": 2.8,
    "weights": (0.15, 0.65, 0.20),
    "min_adv": 1_500_000,
    "min_rr": 1.8,
    "mom_thresh": 0.65,
    "long_thresh": 75.0,
    "max_pos": 4,
    "max_hold": 7,
    "risk_frac": 0.0075,
    # --- structural risk controls (Option B) ---
    "max_semis": 2,        # hard cap: at most 2 concurrent Semis positions
    "max_crypto": 1,       # hard cap: at most 1 concurrent Crypto position
    "use_trailing": True,  # move stop to breakeven at +1R, then ATR-trail
    # --- portfolio-level controls (drawdown fix) ---
    "breaker_enabled": True,     # portfolio circuit breaker on regime ON->OFF flip
    "breaker_action": "halve",   # "halve" all open positions, or "close" them outright
    "max_total_risk_frac": 0.02, # cap: sum of $ risk across ALL open positions <= 2% of equity
    "regime_buffer_pct": 0.02,   # hysteresis band around the 50D MA (avoids whipsaw flips)
    "breaker_cooldown_days": 10, # minimum spacing between breaker activations
}

print("Loading real universe (Mag7 + Semis + Energy + Crypto) via Yahoo Finance...")

if MODE == "tracker" and not is_market_open_now():
    print("Market is closed right now (outside 9:30am-4:00pm ET, or a weekend) "
          "— skipping this run. (This check only applies in tracker mode; a "
          "manual full run will proceed regardless of market hours.)")
    import sys
    sys.exit(0)

prices, volumes, highs, lows = load_universe_yahoo(TICKERS, start="2023-07-01")
print(f"\nSuccessfully loaded {len(prices)} tickers\n")

if len(prices) >= 8:
    print("Building signals...")
    df_sig = run_pipeline(prices, volumes, params)

    print("Applying regime filter (QQQ vs 50D MA, with hysteresis band)...")
    regime_series = compute_regime_series(start="2023-07-01", buffer_pct=params["regime_buffer_pct"])
    df_sig = apply_regime_gate(df_sig, regime_series)

    last_date = df_sig["date"].max()
    regime_now = bool(df_sig[df_sig["date"] == last_date]["regime_on"].iloc[0])
    n_blocked = ((df_sig["raw_signal"] == "long") & (df_sig["signal"] == "neutral")).sum()
    print(f"Current regime: {'RISK-ON' if regime_now else 'RISK-OFF'} "
          f"(QQQ {'above' if regime_now else 'below'} its 50D MA)")
    print(f"Long signals downgraded to neutral by regime filter (all dates): {n_blocked}")
    print()
    print(df_sig["signal"].value_counts())
else:
    print("Not enough data loaded. Check ticker list or internet connection.")

if MODE == "full" and len(prices) >= 8:
    print("\nRunning backtest...")
    trades, curve, final, breaker_events = run_backtest(
        df_sig, prices, params,
        capital=100_000.0,
        risk_frac=params["risk_frac"]
    )

    print(f"\nPortfolio circuit breaker: {len(breaker_events)} activation(s) "
          f"(action = '{params['breaker_action']}')")
    for ev in breaker_events:
        print(f"  {ev['date'].date()}  hit {len(ev['symbols'])} open position(s): {ev['symbols']}")

    print("\n==================== RESULTS ====================")
    metrics = compute_metrics(trades, curve, 100_000.0)
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")

    print("\nRecent closed trades:")
    if len(trades) > 0:
        print(trades.tail(8).to_string(index=False))
    else:
        print("No trades generated")


# ============================================================
# SIGNAL ACCURACY DEEP DIVE — full mode only (needs the backtest's
# `trades`/`df_sig`; skipped entirely in tracker mode, which only
# needs the live signal log below)
# ============================================================
if MODE == "full" and len(prices) >= 8:
    print("="*60)
    print("SIGNAL ACCURACY ANALYSIS")
    print("="*60)

    # 1. Basic exit reason breakdown
    print("\n1. Exit Reason Breakdown")
    print(trades["reason"].value_counts(normalize=True).round(3) * 100)

    # 2. Performance by exit reason
    print("\n2. Avg Return by Exit Reason")
    print(trades.groupby("reason")["ret"].agg(["count", "mean", "median"]).round(4))

    # 3. Win rate & expectancy by symbol
    print("\n3. Performance by Symbol (Top & Bottom)")
    by_sym = trades.groupby("symbol").agg(
    trades=("ret", "count"),
    win_rate=("ret", lambda x: (x > 0).mean()),
    avg_ret=("ret", "mean"),
    total_pnl=("pnl", "sum")
    ).sort_values("total_pnl", ascending=False)

    print("\nBest symbols:")
    print(by_sym.head(8).round(3))
    print("\nWorst symbols:")
    print(by_sym.tail(8).round(3))

    # 4. Sector / Group analysis
    trades["group"] = trades["symbol"].apply(get_group)

    print("\n4. Performance by Group")
    print(trades.groupby("group").agg(
    trades=("ret", "count"),
    win_rate=("ret", lambda x: (x > 0).mean()),
    avg_ret=("ret", "mean"),
    total_pnl=("pnl", "sum"),
    avg_win=("ret", lambda x: x[x>0].mean()),
    avg_loss=("ret", lambda x: x[x<=0].mean())
    ).round(3))

    # 5. Conviction score vs outcome (if score is still available)
    # We need to merge score back onto trades
    sig_scores = df_sig[["date", "symbol", "score"]].copy()
    sig_scores = sig_scores.rename(columns={"date": "signal_date"})

    # Approximate: use entry_date - 1 business day as signal date
    trades["signal_date"] = trades["entry_date"] - pd.tseries.offsets.BDay(1)
    merged = trades.merge(sig_scores, left_on=["signal_date", "symbol"], right_on=["signal_date", "symbol"], how="left")

    if "score" in merged.columns:
        merged["score_quartile"] = pd.qcut(merged["score"].rank(method="first"), 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        print("\n5. Performance by Conviction Quartile")
        print(merged.groupby("score_quartile").agg(
        trades=("ret", "count"),
        win_rate=("ret", lambda x: (x > 0).mean()),
        avg_ret=("ret", "mean"),
        expectancy=("ret", "mean")
    ).round(4))

    # 6. Distribution of returns
    print("\n6. Return Distribution")
    print(trades["ret"].describe().round(4))
    print(f"\n% of trades > +10%: {(trades['ret'] > 0.10).mean()*100:.1f}%")
    print(f"% of trades < -5%:  {(trades['ret'] < -0.05).mean()*100:.1f}%")


# ============================================================
# 7. LIVE SIGNAL TRACKING — forward-test the actual signals
#    (NOT a backtest — this logs what the system says TODAY, then
#    checks on it with REAL future price data on later runs, so you
#    can judge signal quality without hindsight bias)
# ============================================================
LOG_COLUMNS = ["signal_date", "symbol", "group", "score", "entry", "stop", "target", "rr",
               "status", "resolved_date", "exit_price", "exit_reason", "ret", "days_held"]

def _get_log_path():
    """Where the signal log lives, in priority order:
      1. SIGNAL_LOG_PATH env var, if set (this is what the GitHub Actions
         workflow uses — a path inside the checked-out repo, so the workflow
         can git-commit the updated file back after each run).
      2. Google Drive, if running in Colab (survives between sessions
         without needing any repo/CI setup).
      3. A local, non-persistent fallback path otherwise.
    """
    env_path = os.environ.get("SIGNAL_LOG_PATH")
    if env_path:
        print(f"Signal log path (from SIGNAL_LOG_PATH): {env_path}")
        return env_path
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        path = '/content/drive/MyDrive/signal_desk_log.csv'
        print(f"Signal log will persist to Google Drive: {path}")
        return path
    except Exception:
        path = 'signal_desk_log.csv'
        print(f"Not running in Colab (or Drive mount failed) — log will NOT "
              f"persist between sessions unless you set SIGNAL_LOG_PATH: {path}")
        return path


def _load_log(log_path):
    try:
        log = pd.read_csv(log_path, parse_dates=["signal_date", "resolved_date"])
        # pandas infers dtype from the CSV's actual contents, so a column
        # that's entirely blank so far (e.g. exit_reason before anything has
        # resolved) gets read back as float64 NaN — force it back to a
        # flexible dtype or later string writes into it raise a TypeError
        if "exit_reason" in log.columns:
            log["exit_reason"] = log["exit_reason"].astype(object)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        log = pd.DataFrame(columns=LOG_COLUMNS)
    return log


def record_signals(df_sig, log_path):
    """Appends today's LIVE 'long' signals to the persistent log. Safe to
    call repeatedly on the same day — duplicates (same date + symbol) are
    skipped, not re-added."""
    log = _load_log(log_path)
    last_date = df_sig["date"].max()
    todays = df_sig[(df_sig["date"] == last_date) & (df_sig["signal"] == "long")].copy()

    if todays.empty:
        print(f"No long signals on {last_date.date()} — nothing to log.")
        return log

    already = set(zip(log["signal_date"], log["symbol"])) if len(log) else set()
    new_rows = []
    for _, row in todays.iterrows():
        key = (pd.Timestamp(last_date), row["symbol"])
        if key in already:
            continue
        new_rows.append({
            "signal_date": last_date, "symbol": row["symbol"], "group": get_group(row["symbol"]),
            "score": row["score"], "entry": row["price"], "stop": row["stop"], "target": row["target"],
            "rr": row["rr"], "status": "open", "resolved_date": pd.NaT, "exit_price": np.nan,
            "exit_reason": None, "ret": np.nan, "days_held": np.nan,
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # force these to stay as flexible (string/datetime) dtypes even though every
        # value starts out missing — otherwise pandas infers float64 from the all-None
        # column and later raises when a real string/date is written into it
        new_df["exit_reason"] = new_df["exit_reason"].astype(object)
        new_df["resolved_date"] = pd.to_datetime(new_df["resolved_date"])
        log = pd.concat([log, new_df], ignore_index=True)
        log.to_csv(log_path, index=False)
        print(f"Logged {len(new_rows)} new signal(s) for {last_date.date()}.")
    else:
        print(f"All of today's {len(todays)} signal(s) were already logged.")
    return log


def mark_to_market(log_path, prices, params, highs=None, lows=None):
    """Checks every still-open logged signal against the latest REAL price
    data available and resolves it (stop / target / time-out) using the
    exact same rules the backtester uses — but walked forward day by day
    on actual subsequent prices, not simulated ones.

    If `highs`/`lows` are supplied, resolution uses each day's intraday
    High/Low (not just the Close) — so a stop or target that was touched
    at any point during the day gets caught, which is the whole point of
    running this every 15 minutes instead of once at end of day. Falls
    back to Close-only if highs/lows aren't provided. Fill price on a
    touch is the stop/target level itself (the conservative assumption,
    not the day's actual worst/best print) — and the signal's OWN entry
    day is never used for resolution, since daily High/Low can't tell us
    whether the extreme happened before or after you actually entered.
    """
    log = _load_log(log_path)
    if log.empty:
        print("Log is empty — nothing to mark to market.")
        return log

    use_intraday = highs is not None and lows is not None
    max_hold = params["max_hold"]
    open_mask = log["status"] == "open"
    n_open_before = int(open_mask.sum())

    for idx in log[open_mask].index:
        sym = log.at[idx, "symbol"]
        if sym not in prices:
            continue
        sig_date = log.at[idx, "signal_date"]
        stop = log.at[idx, "stop"]
        target = log.at[idx, "target"]
        entry = log.at[idx, "entry"]
        px_series = prices[sym][prices[sym].index > sig_date]
        if px_series.empty:
            continue

        hi_series = highs[sym][highs[sym].index > sig_date] if use_intraday and sym in highs else None
        lo_series = lows[sym][lows[sym].index > sig_date] if use_intraday and sym in lows else None

        for dt, close_px in px_series.items():
            days_held = (dt - sig_date).days
            if hi_series is not None and lo_series is not None and dt in hi_series.index and dt in lo_series.index:
                day_high, day_low = float(hi_series.loc[dt]), float(lo_series.loc[dt])
            else:
                day_high = day_low = float(close_px)

            exit_reason, exit_px = None, None
            if day_low <= stop:
                exit_reason, exit_px = "stop", stop        # conservative fill: at the stop, not the day's low
            elif day_high >= target:
                exit_reason, exit_px = "target", target    # conservative fill: at the target, not the day's high
            elif days_held >= max_hold:
                exit_reason, exit_px = "time", float(close_px)

            if exit_reason:
                log.at[idx, "status"] = "closed"
                log.at[idx, "resolved_date"] = dt
                log.at[idx, "exit_price"] = float(exit_px)
                log.at[idx, "exit_reason"] = exit_reason
                log.at[idx, "ret"] = (float(exit_px) - entry) / entry
                log.at[idx, "days_held"] = days_held
                break

    n_resolved = n_open_before - int((log["status"] == "open").sum())
    log.to_csv(log_path, index=False)
    print(f"Marked-to-market: {n_resolved} signal(s) resolved this run, "
          f"{int((log['status']=='open').sum())} still open.")
    return log


def signal_scorecard(log_path):
    """The actual answer to 'are these signals any good' — computed from
    REAL signals the system issued and what actually happened to them
    afterward, not from a backtest re-running history."""
    log = _load_log(log_path)
    if log.empty:
        print("No signals logged yet.")
        return log

    closed = log[log["status"] == "closed"]
    n_open = int((log["status"] == "open").sum())

    print("=" * 60)
    print("LIVE SIGNAL SCORECARD (forward-tracked, not backtested)")
    print("=" * 60)
    print(f"Total signals logged : {len(log)}")
    print(f"Still open / pending : {n_open}")
    print(f"Resolved             : {len(closed)}")

    if len(closed) == 0:
        print("\nNo resolved signals yet — check back after positions have "
              "had time to hit stop / target / the time-out.")
        return log

    win_rate = (closed["ret"] > 0).mean()
    avg_ret = closed["ret"].mean()
    avg_days = closed["days_held"].mean()

    print(f"\nWin rate              : {win_rate:.1%}")
    print(f"Avg return per signal : {avg_ret:.2%}")
    print(f"Avg days held         : {avg_days:.1f}")

    print(f"\nExit reason breakdown:")
    print(closed["exit_reason"].value_counts())

    print(f"\nBy group:")
    print(closed.groupby("group").agg(
        n=("ret", "count"),
        win_rate=("ret", lambda x: (x > 0).mean()),
        avg_ret=("ret", "mean"),
    ).round(4))

    print(f"\nMost recent resolved signals:")
    print(closed.sort_values("resolved_date", ascending=False).head(10)
          [["symbol", "signal_date", "resolved_date", "score", "entry", "exit_price", "ret", "exit_reason"]]
          .to_string(index=False))

    if n_open > 0:
        print(f"\nStill open ({n_open}):")
        print(log[log["status"] == "open"]
              [["symbol", "signal_date", "score", "entry", "stop", "target"]]
              .to_string(index=False))

    return log


# Run the tracking workflow every time this cell runs:
#   1. log today's new signals (safe to re-run same day — no duplicates)
#   2. mark-to-market anything previously logged, using real prices since
#   3. print the scorecard
# Come back and re-run this cell periodically (daily/weekly) — the log
# accumulates real signal history over time, and the scorecard is your
# honest answer on whether the signals are actually any good.
if len(prices) >= 8:
    LOG_PATH = _get_log_path()
    record_signals(df_sig, LOG_PATH)
    mark_to_market(LOG_PATH, prices, params, highs=highs, lows=lows)
    signal_scorecard(LOG_PATH)
