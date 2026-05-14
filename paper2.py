"""paper2.py — paper-trading bot using the R80 champion strategy.

Same architectural reasoning as bot.py (paper 1):
  - Polls Binance Futures every candle close, evaluates signals
  - Maintains persistent state in JSON, sends Telegram updates
  - Tracks paper P&L from a fresh $100 balance starting at launch

But uses a different strategy than bot.py:
  - 4-signal mean-reversion ensemble (Williams %R DBB + strict RSI + strict
    Stoch + Bollinger pierce) on 4-hour candles across 125 USDT-perp pairs
  - Per trade: SL=3.2*ATR, TP=0.82*ATR, max_hold=20 bars, cooldown=10 bars after loss
  - Rolling weekly pair selection: re-rank every 7 days, trade top-15
  - 2 concurrent positions at 10x leverage, compounded
  - 70% drawdown circuit breaker (pause trading if balance < 30% of peak;
    resume only when balance recovers above the previous peak)
  - Starts fresh from $100 the moment this process boots
"""

import os
import json
import time
import math
import logging
import threading
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

MYT = timezone(timedelta(hours=8), name="MYT")

BINANCE_FAPI = "https://fapi.binance.com"
TELEGRAM_API = "https://api.telegram.org"

# Distinct env vars so paper2 can run alongside bot.py with a separate chat
TELEGRAM_BOT_TOKEN = (
    os.environ.get("PAPER2_TELEGRAM_BOT_TOKEN")
    or os.environ.get("TELEGRAM_BOT_TOKEN")
)
TELEGRAM_CHAT_ID = (
    os.environ.get("PAPER2_TELEGRAM_CHAT_ID")
    or os.environ.get("TELEGRAM_CHAT_ID")
)

STATE_PATH = os.environ.get("PAPER2_STATE_PATH", "/tmp/paper2_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("paper2")

# ---------------- Strategy config (R80 champion) ----------------

TIMEFRAME = "4h"
KLINE_LIMIT = 300  # enough for indicators + recent history
BAR_SECONDS = 4 * 3600

# Universe: 125 pairs from R80
PAIRS_UNIVERSE = [
    '1INCHUSDT', 'AAVEUSDT', 'ADAUSDT', 'AEVOUSDT', 'ALGOUSDT', 'ALTUSDT',
    'ANKRUSDT', 'APEUSDT', 'APTUSDT', 'ARBUSDT', 'ARKUSDT', 'AVAUSDT',
    'AVAXUSDT', 'BAKEUSDT', 'BANDUSDT', 'BCHUSDT', 'BELUSDT', 'BIGTIMEUSDT',
    'BMTUSDT', 'BNBUSDT', 'BTCUSDT', 'CATIUSDT', 'CFXUSDT', 'CGPTUSDT',
    'CHZUSDT', 'CKBUSDT', 'COMPUSDT', 'COSUSDT', 'COTIUSDT', 'CRVUSDT',
    'CTKUSDT', 'CTSIUSDT', 'DGBUSDT', 'DOGEUSDT', 'DOTUSDT', 'ENJUSDT',
    'ETCUSDT', 'ETHFIUSDT', 'ETHUSDT', 'FDUSDUSDT', 'FLMUSDT', 'FLOWUSDT',
    'GALAUSDT', 'GHSTUSDT', 'GMTUSDT', 'GMXUSDT', 'GRTUSDT', 'GUSDT',
    'HBARUSDT', 'HIVEUSDT', 'HMSTRUSDT', 'HOOKUSDT', 'HOTUSDT', 'IDUSDT',
    'ILVUSDT', 'IMXUSDT', 'INJUSDT', 'JOEUSDT', 'KNCUSDT', 'LDOUSDT',
    'LINKUSDT', 'LPTUSDT', 'LTCUSDT', 'MAGICUSDT', 'MASKUSDT', 'MAVUSDT',
    'MEMEUSDT', 'METISUSDT', 'MKRUSDT', 'MOVRUSDT', 'NEARUSDT', 'NMRUSDT',
    'NOTUSDT', 'OGNUSDT', 'OMNIUSDT', 'OMUSDT', 'ONEUSDT', 'ONGUSDT',
    'OXTUSDT', 'PEOPLEUSDT', 'PERPUSDT', 'PONDUSDT', 'PORTALUSDT',
    'POWRUSDT', 'PUNDIXUSDT', 'PYTHUSDT', 'QTUMUSDT', 'RAREUSDT',
    'RENDERUSDT', 'RLCUSDT', 'RPLUSDT', 'RUNEUSDT', 'RVNUSDT', 'SANDUSDT',
    'SEIUSDT', 'SFPUSDT', 'SHIBUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT',
    'SPELLUSDT', 'SSVUSDT', 'STEEMUSDT', 'STRKUSDT', 'SUIUSDT', 'SUNUSDT',
    'SUSHIUSDT', 'TLMUSDT', 'TRBUSDT', 'TRXUSDT', 'TURBOUSDT', 'TUSDUSDT',
    'UMAUSDT', 'UNIUSDT', 'USDPUSDT', 'USUALUSDT', 'VETUSDT', 'WANUSDT',
    'WBTCUSDT', 'XECUSDT', 'XRPUSDT', 'XTZUSDT', 'YFIUSDT', 'ZILUSDT',
    'ZRXUSDT',
]

# Strategy params (R80 champion)
ATR_PERIOD = 21
WR_PERIOD = 14
RSI_PERIOD = 14
STOCH_PERIOD = 14
BB_PERIOD = 20
BB_K = 2.5
SL_MULT = 3.2
TP_MULT = 0.82
MAX_HOLD_BARS = 20
COOLDOWN_BARS = 10  # bars to wait after a losing trade per pair

# DBB params
DBB_N_TOUCHES = 5
DBB_LOOKBACK = 17
DBB_CROSS_LEVEL = -84
DBB_STOCH_THRESH = 16

# RSI strict
RSI_LO = 22
RSI_HI = 78
RSI_N_TOUCH = 2

# Stoch strict
STOCH_LO = 15
STOCH_HI = 85
STOCH_N_TOUCH = 2

# Portfolio
TOP_N = 15
N_SLOTS = 2
LEVERAGE = 10
FEE_PER_SIDE = 0.0005  # Binance taker
ROLLING_LOOKBACK_DAYS = 7
ROLLING_REBALANCE_DAYS = 7
DD_CIRCUIT = 0.70  # halt when balance < (1-DD)*peak
STARTING_BALANCE = 100.0

# Scan cadence: poll every 5 minutes for new candle close events;
# the actual trade evaluation happens once per 4h candle close per pair.
SCAN_INTERVAL_SECONDS = 5 * 60

# ---------------- State ----------------

state_lock = threading.Lock()

state: dict = {
    "started_at": None,        # ISO datetime when bot first ran
    "balance": STARTING_BALANCE,
    "peak": STARTING_BALANCE,
    "halted": False,
    # slot_id -> position dict
    # position dict: {sym, dir, entry_price, entry_time(iso), entry_bar_close(iso),
    #                 sl, tp, max_hold_until(iso), margin, slot_id}
    "positions": [None, None],
    # sym -> iso timestamp until which the pair is in cooldown
    "cooldown_until": {},
    # iso timestamp of last successful pair rebalance (or None to force on start)
    "last_rebalance": None,
    # current top-N symbol list
    "top_pairs": [],
    # closed trades: [{sym,dir,entry,exit,pnl_pct,pnl_usd,balance_after,entry_time,exit_time,reason,margin,liquidated}]
    "closed_trades": [],
    # per-pair last evaluated candle close time iso, to avoid double-firing
    "last_eval_bar": {},
    # total signals fired (including those rejected because no slot)
    "stats": {
        "total_signals_fired": 0,
        "total_trades_opened": 0,
        "total_trades_closed": 0,
        "total_wins": 0,
        "total_losses": 0,
        "total_liquidations": 0,
        "total_dd_halts": 0,
        "total_dd_resumes": 0,
    },
}


def load_state() -> None:
    global state
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r") as f:
                loaded = json.load(f)
            # merge so new fields get defaults
            for k, v in loaded.items():
                state[k] = v
            log.info("Loaded state from %s (balance=$%.2f, peak=$%.2f, open=%d)",
                     STATE_PATH, state["balance"], state["peak"],
                     sum(1 for p in state["positions"] if p))
        else:
            log.info("No prior state at %s — starting fresh", STATE_PATH)
            state["started_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        log.exception("Failed to load state: %s", e)


def save_state() -> None:
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, default=str, indent=2)
    except Exception as e:
        log.exception("Failed to save state: %s", e)


# ---------------- Telegram ----------------

def send_telegram(text: str, parse_mode: str = "HTML") -> None:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log.info("[no-telegram] %s", text.replace("\n", " | "))
        return
    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            log.warning("telegram send failed: %s %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("telegram exception: %s", e)


# ---------------- Indicators ----------------

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = ATR_PERIOD) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = WR_PERIOD) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def stoch_k(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = STOCH_PERIOD) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return 100 * (close - ll) / (hh - ll).replace(0, np.nan)


def bbands(close: pd.Series, period: int = BB_PERIOD, k: float = BB_K):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ma + k * sd, ma, ma - k * sd


def precompute(df: pd.DataFrame) -> dict:
    upper, _, lower = bbands(df["close"], BB_PERIOD, BB_K)
    return {
        "open_time": df["open_time"],
        "close_time": df["close_time"],
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "atr": atr_wilder(df["high"], df["low"], df["close"], ATR_PERIOD),
        "wr": williams_r(df["high"], df["low"], df["close"], WR_PERIOD),
        "rsi": rsi(df["close"], RSI_PERIOD),
        "stoch": stoch_k(df["high"], df["low"], df["close"], STOCH_PERIOD),
        "bb_upper": upper, "bb_lower": lower,
    }


# ---------------- Signals (R80 ensemble) ----------------

def sig_dbb(p: dict, i: int):
    if i < DBB_LOOKBACK + 1:
        return None
    wp, wc = p["wr"].iloc[i - 1], p["wr"].iloc[i]
    if pd.isna(wp) or pd.isna(wc):
        return None
    recent = p["wr"].iloc[max(0, i - DBB_LOOKBACK):i].dropna()
    if len(recent) < DBB_LOOKBACK - 5:
        return None
    sc = p["stoch"].iloc[i]
    if pd.isna(sc):
        return None
    upper = -DBB_CROSS_LEVEL - 100  # = -16 when cross=-84
    long_touches = (recent <= DBB_CROSS_LEVEL).sum()
    short_touches = (recent >= upper).sum()
    if (wp <= DBB_CROSS_LEVEL and wc > DBB_CROSS_LEVEL
            and long_touches >= DBB_N_TOUCHES
            and sc < DBB_STOCH_THRESH + 20):
        return "LONG"
    if (wp >= upper and wc < upper
            and short_touches >= DBB_N_TOUCHES
            and sc > 100 - DBB_STOCH_THRESH - 20):
        return "SHORT"
    return None


def sig_rsi_strict(p: dict, i: int):
    if i < RSI_N_TOUCH + 1:
        return None
    rp, rc = p["rsi"].iloc[i - 1], p["rsi"].iloc[i]
    if pd.isna(rp) or pd.isna(rc):
        return None
    if rp <= RSI_LO and rc > RSI_LO:
        recent = p["rsi"].iloc[max(0, i - RSI_N_TOUCH - 1):i - 1]
        if (recent <= RSI_LO).sum() >= RSI_N_TOUCH:
            return "LONG"
    if rp >= RSI_HI and rc < RSI_HI:
        recent = p["rsi"].iloc[max(0, i - RSI_N_TOUCH - 1):i - 1]
        if (recent >= RSI_HI).sum() >= RSI_N_TOUCH:
            return "SHORT"
    return None


def sig_stoch_strict(p: dict, i: int):
    if i < STOCH_N_TOUCH + 1:
        return None
    sp, sc = p["stoch"].iloc[i - 1], p["stoch"].iloc[i]
    if pd.isna(sp) or pd.isna(sc):
        return None
    if sp <= STOCH_LO and sc > STOCH_LO:
        recent = p["stoch"].iloc[max(0, i - STOCH_N_TOUCH - 1):i - 1]
        if (recent <= STOCH_LO).sum() >= STOCH_N_TOUCH:
            return "LONG"
    if sp >= STOCH_HI and sc < STOCH_HI:
        recent = p["stoch"].iloc[max(0, i - STOCH_N_TOUCH - 1):i - 1]
        if (recent >= STOCH_HI).sum() >= STOCH_N_TOUCH:
            return "SHORT"
    return None


def sig_bb_touch(p: dict, i: int):
    if i < 2:
        return None
    cp, cc = p["close"].iloc[i - 1], p["close"].iloc[i]
    lp, lc = p["bb_lower"].iloc[i - 1], p["bb_lower"].iloc[i]
    up, uc = p["bb_upper"].iloc[i - 1], p["bb_upper"].iloc[i]
    if pd.isna(lp) or pd.isna(lc) or pd.isna(up) or pd.isna(uc):
        return None
    if cp <= lp and cc > lc:
        return "LONG"
    if cp >= up and cc < uc:
        return "SHORT"
    return None


def evaluate_ensemble(p: dict, i: int):
    """Return 'LONG'/'SHORT'/None. Any signal fires; conflicts resolved by majority."""
    votes = {"LONG": 0, "SHORT": 0}
    for sig_fn in (sig_dbb, sig_rsi_strict, sig_stoch_strict, sig_bb_touch):
        d = sig_fn(p, i)
        if d is not None:
            votes[d] += 1
    if votes["LONG"] > votes["SHORT"]:
        return "LONG", votes["LONG"]
    if votes["SHORT"] > votes["LONG"]:
        return "SHORT", votes["SHORT"]
    return None, 0


# ---------------- Data fetch ----------------

def fetch_klines(symbol: str, interval: str = TIMEFRAME,
                 limit: int = KLINE_LIMIT) -> pd.DataFrame | None:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                raw = r.json()
                if not raw:
                    return None
                df = pd.DataFrame(raw, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
                ])
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = df[c].astype(float)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                return df
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception as e:
            log.warning("fetch_klines %s attempt %d: %s", symbol, attempt, e)
            time.sleep(2 ** attempt)
    return None


# ---------------- Backtest (for ranking) ----------------

def backtest_pair(p: dict, df: pd.DataFrame, start_idx: int = 0) -> list[dict]:
    """Run signal logic on history from start_idx forward, returning a list of
    trade dicts {pnl_pct, entry_time, exit_time}. Used only for ranking — does
    NOT respect cooldown across pair boundaries (cooldown is for live trading)."""
    trades = []
    n = len(df)
    i = max(start_idx, 25)
    last_loss_end = None
    while i < n - 1:
        atr_v = p["atr"].iloc[i]
        if pd.isna(atr_v):
            i += 1
            continue
        # cooldown within this pair's history
        if last_loss_end is not None and i < last_loss_end + COOLDOWN_BARS:
            i += 1
            continue
        direction, _ = evaluate_ensemble(p, i)
        if direction is None:
            i += 1
            continue
        entry = float(p["close"].iloc[i])
        sl_d = SL_MULT * atr_v
        tp_d = TP_MULT * atr_v
        if direction == "LONG":
            sl, tp = entry - sl_d, entry + tp_d
        else:
            sl, tp = entry + sl_d, entry - tp_d
        exit_i = None
        result = None
        for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
            hh, ll = float(p["high"].iloc[j]), float(p["low"].iloc[j])
            if direction == "LONG":
                if ll <= sl:
                    exit_i, result = j, "SL"; break
                if hh >= tp:
                    exit_i, result = j, "TP"; break
            else:
                if hh >= sl:
                    exit_i, result = j, "SL"; break
                if ll <= tp:
                    exit_i, result = j, "TP"; break
        if exit_i is None:
            exit_i = min(i + MAX_HOLD_BARS, n - 1)
            result = "TIMEOUT"
        exit_px = float(p["close"].iloc[exit_i])
        pnl_pct = ((exit_px - entry) / entry) if direction == "LONG" else ((entry - exit_px) / entry)
        trades.append({
            "pnl_pct": pnl_pct,
            "entry_time": p["open_time"].iloc[i].isoformat(),
            "exit_time": p["open_time"].iloc[exit_i].isoformat(),
            "result": result,
        })
        if pnl_pct < 0:
            last_loss_end = exit_i
        else:
            last_loss_end = None
        i = exit_i + 1
    return trades


# ---------------- Pair selection ----------------

def select_top_pairs() -> list[str]:
    """Fetch klines for every pair in the universe, run the strategy over the
    last ROLLING_LOOKBACK_DAYS, and return the top TOP_N by cumulative raw PnL%.
    """
    log.info("Rebalancing top-%d pairs from %d-pair universe ...", TOP_N, len(PAIRS_UNIVERSE))
    cutoff = datetime.now(timezone.utc) - timedelta(days=ROLLING_LOOKBACK_DAYS)
    pair_scores: list[tuple[str, float]] = []
    for sym in PAIRS_UNIVERSE:
        df = fetch_klines(sym, TIMEFRAME, KLINE_LIMIT)
        if df is None or len(df) < 50:
            continue
        try:
            p = precompute(df)
            trades = backtest_pair(p, df, start_idx=0)
            recent = [t for t in trades
                      if pd.to_datetime(t["entry_time"]) >= cutoff]
            score = sum(t["pnl_pct"] for t in recent)
            pair_scores.append((sym, score))
        except Exception as e:
            log.warning("ranking failed for %s: %s", sym, e)
        time.sleep(0.05)  # polite rate limit
    pair_scores.sort(key=lambda kv: -kv[1])
    top = [s for s, _ in pair_scores[:TOP_N]]
    log.info("Top-%d pairs by 7d simulated PnL: %s", TOP_N, top)
    return top


def maybe_rebalance(force: bool = False) -> bool:
    """Re-pick top-N pairs if it's been ROLLING_REBALANCE_DAYS since last rebalance."""
    now = datetime.now(timezone.utc)
    last = state.get("last_rebalance")
    if not force and last:
        try:
            last_dt = datetime.fromisoformat(last)
            if (now - last_dt).total_seconds() < ROLLING_REBALANCE_DAYS * 86400:
                return False
        except Exception:
            pass
    new_top = select_top_pairs()
    if not new_top:
        log.warning("Rebalance produced empty top list; keeping previous")
        return False
    with state_lock:
        prev = set(state.get("top_pairs", []))
        cur = set(new_top)
        added = cur - prev
        removed = prev - cur
        state["top_pairs"] = new_top
        state["last_rebalance"] = now.isoformat()
    save_state()
    send_telegram(
        f"<b>📊 paper2 — weekly pair rebalance</b>\n"
        f"New top-{TOP_N}: {', '.join(new_top)}\n"
        f"➕ Added: {', '.join(sorted(added)) or 'none'}\n"
        f"➖ Dropped: {', '.join(sorted(removed)) or 'none'}"
    )
    return True


# ---------------- Position management ----------------

def free_slot_index() -> int | None:
    for i, p in enumerate(state["positions"]):
        if p is None:
            return i
    return None


def open_position(sym: str, direction: str, entry_price: float,
                  atr_v: float, entry_bar_time, slot_idx: int,
                  vote_count: int) -> None:
    margin = state["balance"] / N_SLOTS
    if margin < 1.0:
        return
    sl_d = SL_MULT * atr_v
    tp_d = TP_MULT * atr_v
    if direction == "LONG":
        sl, tp = entry_price - sl_d, entry_price + tp_d
    else:
        sl, tp = entry_price + sl_d, entry_price - tp_d
    entry_iso = (entry_bar_time if isinstance(entry_bar_time, str)
                 else pd.Timestamp(entry_bar_time).isoformat())
    max_hold_until = (pd.Timestamp(entry_bar_time) +
                      timedelta(seconds=MAX_HOLD_BARS * BAR_SECONDS)).isoformat()
    pos = {
        "sym": sym, "dir": direction,
        "entry_price": entry_price,
        "entry_time": entry_iso,
        "sl": sl, "tp": tp,
        "max_hold_until": max_hold_until,
        "margin": margin,
        "atr": atr_v,
        "votes": vote_count,
    }
    with state_lock:
        state["positions"][slot_idx] = pos
        state["stats"]["total_trades_opened"] += 1
    save_state()
    send_telegram(
        f"<b>🟢 paper2 OPEN {direction} {sym}</b>\n"
        f"Entry: ${entry_price:.6g}\n"
        f"SL: ${sl:.6g}  TP: ${tp:.6g}\n"
        f"Margin: ${margin:.2f}  Lev: {LEVERAGE}x  "
        f"Notional: ${margin*LEVERAGE:.2f}\n"
        f"Signal votes: {vote_count}/4\n"
        f"Slot {slot_idx+1}/{N_SLOTS}  Balance: ${state['balance']:.2f}"
    )


def close_position(slot_idx: int, exit_price: float, exit_time_iso: str,
                   reason: str) -> None:
    pos = state["positions"][slot_idx]
    if pos is None:
        return
    direction = pos["dir"]
    entry = pos["entry_price"]
    margin = pos["margin"]
    if direction == "LONG":
        pnl_pct = (exit_price - entry) / entry
    else:
        pnl_pct = (entry - exit_price) / entry
    # Apply leverage, subtract round-trip fees
    pnl_pct_net = pnl_pct - 2 * FEE_PER_SIDE
    pnl_frac_margin = pnl_pct_net * LEVERAGE
    # Liquidation: at 10x leverage with ~0.5% maintenance, a ~9.5% adverse
    # move wipes the margin. The strategy's SL at 3.2*ATR is often wider
    # than 9.5%; treat any pnl_frac_margin < -1.0 as full liquidation.
    liquidated = pnl_frac_margin <= -1.0
    if liquidated:
        pnl_frac_margin = -1.0
    pnl_usd = margin * pnl_frac_margin
    with state_lock:
        state["balance"] += pnl_usd
        if state["balance"] > state["peak"]:
            state["peak"] = state["balance"]
        state["positions"][slot_idx] = None
        # cooldown after loss
        if pnl_usd < 0:
            cd_until = (pd.Timestamp(exit_time_iso) +
                        timedelta(seconds=COOLDOWN_BARS * BAR_SECONDS)).isoformat()
            state["cooldown_until"][pos["sym"]] = cd_until
        trade_rec = {
            "sym": pos["sym"], "dir": direction,
            "entry": entry, "exit": exit_price,
            "pnl_pct": pnl_pct, "pnl_pct_net": pnl_pct_net,
            "pnl_usd": pnl_usd,
            "balance_after": state["balance"],
            "entry_time": pos["entry_time"], "exit_time": exit_time_iso,
            "reason": reason, "margin": margin,
            "liquidated": liquidated, "votes": pos.get("votes", 0),
        }
        state["closed_trades"].append(trade_rec)
        state["stats"]["total_trades_closed"] += 1
        if pnl_usd > 0:
            state["stats"]["total_wins"] += 1
        else:
            state["stats"]["total_losses"] += 1
        if liquidated:
            state["stats"]["total_liquidations"] += 1
        # DD circuit check
        if (not state["halted"]
                and state["balance"] < (1 - DD_CIRCUIT) * state["peak"]):
            state["halted"] = True
            state["stats"]["total_dd_halts"] += 1
            log.warning("DD circuit fired: balance $%.2f < %.0f%% of peak $%.2f",
                        state["balance"], (1 - DD_CIRCUIT) * 100, state["peak"])
    save_state()
    emoji = "✅" if pnl_usd > 0 else ("💀" if liquidated else "❌")
    halted_msg = ("\n<b>⛔ DD CIRCUIT TRIGGERED — TRADING PAUSED</b>"
                  if state["halted"] else "")
    send_telegram(
        f"<b>{emoji} paper2 CLOSE {direction} {pos['sym']} ({reason})</b>\n"
        f"Entry → Exit: ${entry:.6g} → ${exit_price:.6g} ({pnl_pct*100:+.2f}%)\n"
        f"Net (after {2*FEE_PER_SIDE*100:.2f}% fees): {pnl_pct_net*100:+.2f}%\n"
        f"Margin: ${margin:.2f} × {LEVERAGE}x  →  P&L: <b>${pnl_usd:+.2f}</b>"
        f"{' (LIQUIDATED)' if liquidated else ''}\n"
        f"Balance: <b>${state['balance']:.2f}</b>  "
        f"Peak: ${state['peak']:.2f}{halted_msg}"
    )


def check_position_exits(now_utc: datetime) -> None:
    """For each open position, fetch latest klines and check if SL/TP/timeout
    triggered since the last evaluated candle."""
    for slot_idx, pos in enumerate(state["positions"]):
        if pos is None:
            continue
        sym = pos["sym"]
        df = fetch_klines(sym, TIMEFRAME, limit=40)
        if df is None or len(df) < 2:
            continue
        entry_dt = pd.to_datetime(pos["entry_time"])
        max_hold_dt = pd.to_datetime(pos["max_hold_until"])
        sl, tp = pos["sl"], pos["tp"]
        direction = pos["dir"]
        # Iterate candles strictly AFTER the entry bar
        exit_price = None
        exit_time_iso = None
        reason = None
        for k in range(len(df)):
            bar_open = df["open_time"].iloc[k]
            if bar_open <= entry_dt:
                continue
            bar_close_time = df["close_time"].iloc[k]
            if bar_close_time > pd.Timestamp(now_utc):
                # candle not closed yet
                continue
            hh = float(df["high"].iloc[k])
            ll = float(df["low"].iloc[k])
            close_px = float(df["close"].iloc[k])
            if direction == "LONG":
                if ll <= sl:
                    exit_price, reason = sl, "SL"; exit_time_iso = bar_open.isoformat(); break
                if hh >= tp:
                    exit_price, reason = tp, "TP"; exit_time_iso = bar_open.isoformat(); break
            else:
                if hh >= sl:
                    exit_price, reason = sl, "SL"; exit_time_iso = bar_open.isoformat(); break
                if ll <= tp:
                    exit_price, reason = tp, "TP"; exit_time_iso = bar_open.isoformat(); break
            # max-hold timeout check
            if bar_open >= max_hold_dt:
                exit_price = close_px; reason = "TIMEOUT"
                exit_time_iso = bar_open.isoformat()
                break
        if exit_price is not None:
            close_position(slot_idx, exit_price, exit_time_iso, reason)


# ---------------- DD circuit ----------------

def check_dd_resume() -> None:
    """If halted and balance has recovered back above peak, resume trading."""
    if not state["halted"]:
        return
    # Resume only when balance recovers above peak (peak doesn't get updated
    # while halted because no new trades close into balance — except if open
    # positions resolve favorably; check directly)
    if state["balance"] >= state["peak"]:
        with state_lock:
            state["halted"] = False
            state["stats"]["total_dd_resumes"] += 1
        save_state()
        send_telegram(
            f"<b>🟢 paper2 DD CIRCUIT CLEARED — TRADING RESUMED</b>\n"
            f"Balance: ${state['balance']:.2f}  Peak: ${state['peak']:.2f}"
        )


# ---------------- Entry scanning ----------------

def scan_for_entries(now_utc: datetime) -> None:
    if state["halted"]:
        return
    free = free_slot_index()
    if free is None:
        return
    top = state.get("top_pairs", [])
    if not top:
        return
    cooldown = state.get("cooldown_until", {})
    held_syms = {p["sym"] for p in state["positions"] if p is not None}
    for sym in top:
        if sym in held_syms:
            continue
        cd = cooldown.get(sym)
        if cd:
            try:
                cd_dt = datetime.fromisoformat(cd)
                if datetime.now(timezone.utc) < cd_dt:
                    continue
            except Exception:
                pass
        df = fetch_klines(sym, TIMEFRAME, KLINE_LIMIT)
        if df is None or len(df) < 50:
            continue
        # Evaluate signal on the LAST CLOSED candle (index n-2 if last is forming)
        # close_time of last row indicates whether it has closed
        last_close = df["close_time"].iloc[-1]
        if last_close > pd.Timestamp(now_utc):
            # last row is still forming, use n-2
            idx = len(df) - 2
        else:
            idx = len(df) - 1
        if idx < 30:
            continue
        bar_iso = df["open_time"].iloc[idx].isoformat()
        # Avoid double-firing on the same bar
        if state["last_eval_bar"].get(sym) == bar_iso:
            continue
        state["last_eval_bar"][sym] = bar_iso
        try:
            p = precompute(df)
            direction, votes = evaluate_ensemble(p, idx)
        except Exception as e:
            log.warning("eval %s failed: %s", sym, e)
            continue
        if direction is None:
            continue
        atr_v = float(p["atr"].iloc[idx])
        if not (atr_v > 0):
            continue
        entry_price = float(p["close"].iloc[idx])
        entry_bar_time = df["open_time"].iloc[idx]
        state["stats"]["total_signals_fired"] += 1
        # Re-check free slot in case it changed
        slot = free_slot_index()
        if slot is None:
            return  # no room
        open_position(sym, direction, entry_price, atr_v, entry_bar_time, slot, votes)
        # break — only fire one new entry per scan to give the next one a chance
        # in the next 4h bar
        return


# ---------------- Telegram commands ----------------

def handle_command(cmd: str) -> None:
    cmd = cmd.strip().lower()
    if cmd in ("/paper2", "/status", "/p2"):
        send_status()
    elif cmd in ("/paper2history", "/history"):
        send_history()
    elif cmd in ("/paper2reset", "/reset"):
        do_reset()
    elif cmd in ("/paper2pairs", "/pairs"):
        top = state.get("top_pairs", [])
        send_telegram("<b>paper2 current top-15:</b>\n" +
                      ("\n".join(top) if top else "(no rebalance yet)"))
    elif cmd in ("/paper2rebalance", "/rebalance"):
        maybe_rebalance(force=True)
    elif cmd == "/help":
        send_telegram(
            "<b>paper2 commands</b>\n"
            "/paper2 — current balance, peak, open positions, stats\n"
            "/paper2history — recent closed trades\n"
            "/paper2pairs — current top-15 trading pairs\n"
            "/paper2rebalance — force a pair rebalance now\n"
            "/paper2reset — wipe state, start fresh from $100\n"
        )


def send_status() -> None:
    s = state
    n_open = sum(1 for p in s["positions"] if p)
    pnl = s["balance"] - STARTING_BALANCE
    ret_pct = (s["balance"] / STARTING_BALANCE - 1) * 100
    drawdown = (s["peak"] - s["balance"]) / s["peak"] * 100 if s["peak"] > 0 else 0
    halt_str = "⛔ <b>HALTED (DD circuit)</b>" if s["halted"] else "🟢 active"
    started = s.get("started_at", "?")
    stats = s.get("stats", {})
    pos_lines = []
    for i, p in enumerate(s["positions"]):
        if p:
            pos_lines.append(
                f"  Slot {i+1}: {p['dir']} {p['sym']} @ ${p['entry_price']:.6g} "
                f"(margin ${p['margin']:.2f})"
            )
    pos_str = "\n".join(pos_lines) if pos_lines else "  (no open positions)"
    n_closed = stats.get("total_trades_closed", 0)
    wins = stats.get("total_wins", 0)
    wr = (100.0 * wins / n_closed) if n_closed else 0
    send_telegram(
        f"<b>paper2 status</b>  [{halt_str}]\n"
        f"Started: {started}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Balance:  <b>${s['balance']:.2f}</b>  ({ret_pct:+.2f}% from $100)\n"
        f"Peak:     ${s['peak']:.2f}\n"
        f"Drawdown: {drawdown:.1f}% from peak\n"
        f"P&L:      ${pnl:+.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Open positions ({n_open}/{N_SLOTS}):\n{pos_str}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Trades closed: {n_closed}  |  Wins: {wins}  |  WR: {wr:.1f}%\n"
        f"Liquidations: {stats.get('total_liquidations', 0)}\n"
        f"DD halts: {stats.get('total_dd_halts', 0)}  resumes: {stats.get('total_dd_resumes', 0)}\n"
        f"Last rebalance: {s.get('last_rebalance', 'never')}"
    )


def send_history(n: int = 10) -> None:
    closed = state.get("closed_trades", [])
    if not closed:
        send_telegram("<b>paper2 history</b>\n(no closed trades yet)")
        return
    recent = closed[-n:][::-1]
    lines = []
    for t in recent:
        emoji = "✅" if t["pnl_usd"] > 0 else "❌"
        if t.get("liquidated"):
            emoji = "💀"
        lines.append(
            f"{emoji} {t['dir']} {t['sym']} {t['reason']} "
            f"${t['pnl_usd']:+.2f} → bal ${t['balance_after']:.2f}"
        )
    send_telegram("<b>paper2 last %d trades</b>\n%s" % (len(recent), "\n".join(lines)))


def do_reset() -> None:
    with state_lock:
        state["balance"] = STARTING_BALANCE
        state["peak"] = STARTING_BALANCE
        state["halted"] = False
        state["positions"] = [None] * N_SLOTS
        state["cooldown_until"] = {}
        state["last_rebalance"] = None
        state["top_pairs"] = []
        state["closed_trades"] = []
        state["last_eval_bar"] = {}
        state["stats"] = {
            "total_signals_fired": 0, "total_trades_opened": 0,
            "total_trades_closed": 0, "total_wins": 0, "total_losses": 0,
            "total_liquidations": 0, "total_dd_halts": 0, "total_dd_resumes": 0,
        }
        state["started_at"] = datetime.now(timezone.utc).isoformat()
    save_state()
    send_telegram(f"<b>paper2 reset</b> — fresh ${STARTING_BALANCE:.0f} from now.")


# ---------------- Telegram polling ----------------

_last_update_id = 0


def telegram_poll_loop() -> None:
    global _last_update_id
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log.info("Telegram disabled (no token/chat) — command polling off")
        return
    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    while True:
        try:
            r = requests.get(url, params={
                "offset": _last_update_id + 1,
                "timeout": 25,
            }, timeout=30)
            if r.status_code != 200:
                time.sleep(5)
                continue
            data = r.json()
            for upd in data.get("result", []):
                _last_update_id = upd["update_id"]
                msg = upd.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue
                if text.startswith("/"):
                    cmd = text.split(maxsplit=1)[0]
                    try:
                        handle_command(cmd)
                    except Exception as e:
                        log.exception("command handler failed: %s", e)
        except Exception as e:
            log.warning("telegram poll error: %s", e)
            time.sleep(5)


# ---------------- Main loop ----------------

def scan_once() -> None:
    now = datetime.now(timezone.utc)
    # 1. Resolve any open positions that hit exits
    check_position_exits(now)
    # 2. DD circuit resume check (in case open positions closed favorably)
    check_dd_resume()
    # 3. Time-based pair rebalance
    if not state.get("top_pairs"):
        maybe_rebalance(force=True)
    else:
        maybe_rebalance(force=False)
    # 4. Open new entries if slots available
    scan_for_entries(now)


def main() -> None:
    log.info("paper2 starting (state path: %s)", STATE_PATH)
    load_state()
    if not state.get("started_at"):
        state["started_at"] = datetime.now(timezone.utc).isoformat()
        save_state()
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    send_telegram(
        "<b>🚀 paper2 online</b>\n"
        "Strategy: 4-signal mean-reversion ensemble (W%R DBB + RSI strict + "
        "Stoch strict + BB pierce) on 4H bars across 125 USDT-perp pairs.\n"
        f"Per trade: SL={SL_MULT}×ATR, TP={TP_MULT}×ATR, max_hold={MAX_HOLD_BARS} "
        f"bars, cooldown={COOLDOWN_BARS} bars after loss.\n"
        f"Rolling top-{TOP_N} pair selection every {ROLLING_REBALANCE_DAYS} days "
        f"(lookback {ROLLING_LOOKBACK_DAYS}d).\n"
        f"{N_SLOTS} concurrent positions at {LEVERAGE}x leverage, compounded.\n"
        f"{int(DD_CIRCUIT*100)}% drawdown circuit breaker enabled.\n"
        f"Starting balance: ${state['balance']:.2f} "
        f"(started {state['started_at']}).\n"
        "Commands: /paper2 /paper2history /paper2pairs /paper2rebalance /paper2reset"
    )
    while True:
        start = time.time()
        try:
            scan_once()
        except Exception as e:
            log.exception("scan_once error: %s", e)
        elapsed = time.time() - start
        sleep_for = max(10, SCAN_INTERVAL_SECONDS - int(elapsed))
        log.info("paper2 scan done in %ds; sleeping %ds (balance $%.2f, halted=%s)",
                 int(elapsed), sleep_for, state["balance"], state["halted"])
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
