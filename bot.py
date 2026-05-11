import os
import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta

MYT = timezone(timedelta(hours=8), name="MYT")

import numpy as np
import pandas as pd
import requests

BINANCE_FAPI = "https://fapi.binance.com"
TELEGRAM_API = "https://api.telegram.org"

PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT",
    "APTUSDT", "OPUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT",
]

TIMEFRAME = "1h"
KLINE_LIMIT = 250
SCAN_INTERVAL_SECONDS = 60 * 60

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("binance-signal-bot")

last_signal_by_pair: dict[str, str] = {}
open_positions: dict[str, str] = {}
position_entry_price: dict[str, float] = {}
trade_results: list[bool] = []
closed_trades: list[dict] = []
position_entry_meta: dict[str, dict] = {}

# User-acceptance tracking (additive — independent of trade_results/closed_trades)
active_signals: dict[str, dict] = {}
accepted_completed: list[dict] = []
total_signals_sent_count: int = 0
total_skipped_count: int = 0
SIGNAL_TIMEOUT_SECONDS = 4 * 3600

# Operational state for /ping and daily summary scheduling
last_scan_time: int = 0  # unix seconds; updated when scan_once starts
last_daily_summary_date: str = ""  # YYYY-MM-DD in MYT

STATE_PATH = os.environ.get(
    "STATE_PATH",
    "/data/state.json" if os.path.isdir("/data") else "state.json",
)

PARAMS_PATH = os.environ.get(
    "PARAMS_PATH",
    "/data/params.json" if os.path.isdir("/data") else "params.json",
)

# Live ATR multipliers used in entry messages and SL distance calculations.
# These match the values backtest_strategy uses (the _Strategy class defaults)
# so live signal SL/TP equals the SL/TP that produced the 61.3% backtest WR.
DEFAULT_PARAMS: dict = {
    "sl_mult": 1.5,
    "tp_mult": 2.5,
}

params: dict = dict(DEFAULT_PARAMS)


def load_params() -> None:
    # SL/TP multipliers are now hardcoded to match backtest values, so we
    # ignore any saved params.json from earlier strategy iterations.
    log.info("Active params (hardcoded): %s", params)




def load_state() -> None:
    global total_signals_sent_count, total_skipped_count, last_daily_summary_date
    if not os.path.exists(STATE_PATH):
        log.info("No prior state at %s; starting fresh", STATE_PATH)
        return
    try:
        with open(STATE_PATH) as f:
            data = json.load(f)
        trade_results[:] = list(data.get("trade_results", []))
        open_positions.update(data.get("open_positions", {}))
        position_entry_price.update(data.get("position_entry_price", {}))
        last_signal_by_pair.update(data.get("last_signal_by_pair", {}))
        closed_trades[:] = list(data.get("closed_trades", []))
        position_entry_meta.update(data.get("position_entry_meta", {}))
        active_signals.update(data.get("active_signals", {}))
        accepted_completed[:] = list(data.get("accepted_completed", []))
        total_signals_sent_count = int(data.get("total_signals_sent_count", 0))
        total_skipped_count = int(data.get("total_skipped_count", 0))
        last_daily_summary_date = str(data.get("last_daily_summary_date", ""))
        log.info(
            "Loaded state: %d closed (rich), %d open, %d active signals, "
            "%d accepted completed, sent=%d skipped=%d",
            len(closed_trades), len(open_positions), len(active_signals),
            len(accepted_completed), total_signals_sent_count, total_skipped_count,
        )
    except Exception as e:
        log.warning("Failed to load state from %s: %s", STATE_PATH, e)


def save_state() -> None:
    try:
        parent = os.path.dirname(STATE_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "trade_results": trade_results,
                "open_positions": open_positions,
                "position_entry_price": position_entry_price,
                "last_signal_by_pair": last_signal_by_pair,
                "closed_trades": closed_trades,
                "position_entry_meta": position_entry_meta,
                "active_signals": active_signals,
                "accepted_completed": accepted_completed,
                "total_signals_sent_count": total_signals_sent_count,
                "total_skipped_count": total_skipped_count,
                "last_daily_summary_date": last_daily_summary_date,
            }, f, default=str)
        os.replace(tmp, STATE_PATH)
    except Exception as e:
        log.warning("Failed to save state to %s: %s", STATE_PATH, e)


def win_rate_text() -> str:
    if not trade_results:
        return "Current win rate: N/A (0 trades)"
    wins = sum(trade_results)
    total = len(trade_results)
    return f"Current win rate: {100 * wins / total:.1f}% ({wins}/{total})"


def send_telegram(text: str, reply_to_message_id: int | None = None) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing; skipping send")
        return
    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            log.error("Telegram send failed: %s %s", r.status_code, r.text)
    except requests.RequestException as e:
        log.error("Telegram request error: %s", e)


def send_telegram_with_buttons(text: str, signal_id: str) -> int | None:
    """Send a message with Accept/Skip inline buttons. Returns message_id or None."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing; skipping send")
        return None
    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ I'm In", "callback_data": f"accept_{signal_id}"},
            {"text": "❌ Skip", "callback_data": f"skip_{signal_id}"},
        ]]
    }
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": keyboard,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            log.error("Telegram send failed: %s %s", r.status_code, r.text)
            return None
        data = r.json()
        if data.get("ok"):
            return data["result"].get("message_id")
    except requests.RequestException as e:
        log.error("Telegram request error: %s", e)
    return None


def edit_telegram_message(message_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not message_id:
        return
    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "message_id": message_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            log.warning("Telegram edit failed: %s %s", r.status_code, r.text)
    except requests.RequestException as e:
        log.warning("Telegram edit error: %s", e)


def answer_callback_query(query_id: str, text: str = "") -> None:
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
    payload: dict = {"callback_query_id": query_id}
    if text:
        payload["text"] = text
    try:
        requests.post(url, json=payload, timeout=10)
    except requests.RequestException as e:
        log.warning("Telegram answerCallback error: %s", e)


def fetch_klines(symbol: str, interval: str = TIMEFRAME, limit: int = KLINE_LIMIT,
                 end_time: int | None = None) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ===== Strategy discovery framework =====

def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / rng


class _Strategy:
    """Wraps precompute + signal + optional exit functions."""
    def __init__(self, sid, name, precompute_fn, signal_fn, exit_fn=None,
                 sl_mult=1.5, tp_mult=2.5, group="discover"):
        self.id = sid
        self.name = name
        self._precompute = precompute_fn
        self._signal = signal_fn
        self._exit = exit_fn
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.group = group

    def precompute(self, df):
        p = self._precompute(df)
        if "atr" not in p:
            p["atr"] = atr(df["high"], df["low"], df["close"], 14)
        return p

    def entry_signal_at(self, p, i):
        try:
            return self._signal(p, i)
        except Exception:
            return None

    def exit_reasons_at(self, p, i, direction):
        if self._exit is None:
            return []
        try:
            r = self._exit(p, i, direction)
        except Exception:
            return []
        if r is False or r is None:
            return []
        if r is True:
            return [f"{self.name} exit"]
        if isinstance(r, str):
            return [r]
        if isinstance(r, list):
            return r
        return []

    def evaluate(self, df, symbol):
        p = self.precompute(df)
        n = len(df["close"])
        idx = n - 1
        direction = self.entry_signal_at(p, idx) if n >= 50 else None
        long_exits = self.exit_reasons_at(p, idx, "LONG") if n >= 50 else []
        short_exits = self.exit_reasons_at(p, idx, "SHORT") if n >= 50 else []
        price = float(df["close"].iloc[-1])
        atr_v = 0.0
        if "atr" in p and len(p["atr"]) > 0:
            v = p["atr"].iloc[-1]
            atr_v = float(v) if pd.notna(v) else 0.0
        result = {
            "direction": direction,
            "price": price,
            "atr": atr_v,
            "rsi": 0.0,
            "bb_upper": 0.0,
            "bb_middle": 0.0,
            "bb_lower": 0.0,
            "candle_time": df["close_time"].iloc[-1],
            "__strategy_long_exits": long_exits,
            "__strategy_short_exits": short_exits,
        }
        for key in ("rsi", "bb_upper", "bb_middle", "bb_lower"):
            if key in p and len(p[key]) > 0:
                v = p[key].iloc[-1]
                if pd.notna(v):
                    result[key] = float(v)
        return result

    def score_at(self, p, i):
        # Default: 100 if signal would fire, else 0
        try:
            sig = self._signal(p, i)
        except Exception:
            return 0, 0
        if sig == "LONG":
            return 100, 0
        if sig == "SHORT":
            return 0, 100
        return 0, 0


STRATEGIES: list[_Strategy] = []
STRATEGIES_BY_ID: dict[str, _Strategy] = {}


def _register(sid, name, precompute_fn, signal_fn, exit_fn=None,
              sl_mult=1.5, tp_mult=2.5, group="discover"):
    s = _Strategy(sid, name, precompute_fn, signal_fn, exit_fn,
                  sl_mult, tp_mult, group)
    STRATEGIES.append(s)
    STRATEGIES_BY_ID[sid] = s
    return s


# ===== /discover strategies (10) =====


def _s9_pre(df):
    return {
        "wr": compute_williams_r(df["high"], df["low"], df["close"], 14),
        "close": df["close"],
    }
def _s9_sig(p, i):
    if i < 1: return None
    wp = p["wr"].iloc[i-1]; wc = p["wr"].iloc[i]
    if pd.isna(wp) or pd.isna(wc): return None
    if wp <= -80 and wc > -80: return "LONG"
    if wp >= -20 and wc < -20: return "SHORT"
    return None
def _s9_exit(p, i, d):
    wc = p["wr"].iloc[i]
    if pd.isna(wc): return False
    if d == "LONG" and wc >= -50: return f"W%R hit -50 ({wc:.1f})"
    if d == "SHORT" and wc <= -50: return f"W%R hit -50 ({wc:.1f})"
    return False
_register("d9_williams_r", "Williams %R", _s9_pre, _s9_sig, _s9_exit)


def _w9_score(p, i):
    """Williams %R /check scoring: 4 weighted conditions, 100 pts total."""
    if i < 0 or "wr" not in p:
        return 0, 0
    wr = p["wr"]
    n = len(wr)
    if i >= n:
        return 0, 0
    wr_now = wr.iloc[i]
    wr_prev = wr.iloc[i - 1] if i >= 1 else float("nan")
    wr_prev2 = wr.iloc[i - 2] if i >= 2 else float("nan")

    long_score = 0
    short_score = 0

    # Condition 1 (40 pts): in extreme zone
    if pd.notna(wr_now):
        if wr_now < -80:
            long_score += 40
        if wr_now > -20:
            short_score += 40

    # Condition 2 (25 pts): cross of threshold within last 2 candles
    if pd.notna(wr_now) and pd.notna(wr_prev):
        long_cross_now = wr_prev <= -80 and wr_now > -80
        short_cross_now = wr_prev >= -20 and wr_now < -20
        long_cross_prev = (
            pd.notna(wr_prev2) and wr_prev2 <= -80 and wr_prev > -80
        )
        short_cross_prev = (
            pd.notna(wr_prev2) and wr_prev2 >= -20 and wr_prev < -20
        )
        if long_cross_now or long_cross_prev:
            long_score += 25
        if short_cross_now or short_cross_prev:
            short_score += 25

    # Condition 3 (20 pts): turning in reversal direction
    if pd.notna(wr_now) and pd.notna(wr_prev):
        if wr_now > wr_prev:
            long_score += 20
        if wr_now < wr_prev:
            short_score += 20

    # Condition 4 (15 pts): ATR >= 0.3% of price (volatility filter)
    if "atr" in p and "close" in p and i < len(p["atr"]) and i < len(p["close"]):
        a = p["atr"].iloc[i]
        c = p["close"].iloc[i]
        if pd.notna(a) and pd.notna(c) and c > 0 and a >= 0.003 * c:
            long_score += 15
            short_score += 15

    return long_score, short_score


# Override score_at on the registered Williams %R instance.
# Assigning a plain function bypasses descriptor binding so no `self` is needed.
STRATEGIES_BY_ID["d9_williams_r"].score_at = lambda p, i: _w9_score(p, i)
STRATEGIES_BY_ID["d9_williams_r"].backtest_wr = 61.3
STRATEGIES_BY_ID["d9_williams_r"].score_description = [
    "40 pts: W%R is in the extreme zone (below -80 for LONG, above -20 for SHORT)",
    "25 pts: W%R crossed the threshold within the last 2 candles (fresh signal)",
    "20 pts: W%R is turning in the reversal direction (rising for LONG, falling for SHORT)",
    "15 pts: ATR is at least 0.3% of current price (sufficient volatility to trade)",
]


# Strategy 2 needs `close` access - rewrite signal closure

# ===== Strategy persistence and discovery loop =====

STRATEGY_PATH = os.environ.get(
    "STRATEGY_PATH",
    "/data/strategy.json" if os.path.isdir("/data") else "strategy.json",
)
active_strategy_id: str | None = None


FORCED_STRATEGY_ID = "d9_williams_r"


def load_strategy() -> None:
    """Force Williams %R as the permanent live strategy regardless of saved file."""
    global active_strategy_id
    if FORCED_STRATEGY_ID not in STRATEGIES_BY_ID:
        log.error("Forced strategy %s not in registry!", FORCED_STRATEGY_ID)
        return

    saved_id = None
    if os.path.exists(STRATEGY_PATH):
        try:
            with open(STRATEGY_PATH) as f:
                saved_id = json.load(f).get("id")
        except Exception as e:
            log.warning("Failed to read %s: %s", STRATEGY_PATH, e)

    active_strategy_id = FORCED_STRATEGY_ID
    if saved_id != FORCED_STRATEGY_ID:
        log.info(
            "Forcing active strategy to Williams %%R (was %s); rewriting %s",
            saved_id, STRATEGY_PATH,
        )
        save_strategy(FORCED_STRATEGY_ID, {
            "name": STRATEGIES_BY_ID[FORCED_STRATEGY_ID].name,
            "forced": True,
            "opt_wr": 61.3, "opt_total": 6274,
            "val_wr": 57.4, "val_total": 3009,
        })
    else:
        log.info("Active strategy: Williams %%R (matches saved file)")


def save_strategy(sid: str, meta: dict) -> None:
    try:
        parent = os.path.dirname(STRATEGY_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = STRATEGY_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"id": sid, "meta": meta}, f, default=str)
        os.replace(tmp, STRATEGY_PATH)
    except Exception as e:
        log.warning("Failed to save strategy: %s", e)



def evaluate(df: pd.DataFrame, symbol: str) -> dict:
    sid = active_strategy_id
    if sid and sid in STRATEGIES_BY_ID:
        return STRATEGIES_BY_ID[sid].evaluate(df, symbol)
    # Should never happen — Williams %R is hardcoded as the active strategy.
    return {
        "direction": None,
        "price": float(df["close"].iloc[-1]),
        "atr": 0.0,
        "candle_time": df["close_time"].iloc[-1],
        "__strategy_long_exits": [],
        "__strategy_short_exits": [],
    }


def _detect_sl_tp_hit(df, position: str, entry, sl_distance: float,
                      tp_distance: float, entry_time_str: str):
    """Walk candles after entry_time. Return (type, price) on first SL or TP
    hit, else None. SL takes priority if both hit in the same candle."""
    if not entry_time_str or entry is None or sl_distance <= 0:
        return None
    try:
        entry_ts = pd.to_datetime(entry_time_str, utc=True)
    except Exception:
        return None
    entry = float(entry)
    if position == "LONG":
        sl_price = entry - sl_distance
        tp_price = entry + tp_distance
    else:
        sl_price = entry + sl_distance
        tp_price = entry - tp_distance
    relevant = df[df["close_time"] > entry_ts]
    if len(relevant) == 0:
        return None
    highs = relevant["high"].values
    lows = relevant["low"].values
    for i in range(len(relevant)):
        h = float(highs[i])
        l = float(lows[i])
        if position == "LONG":
            sl_hit = l <= sl_price
            tp_hit = h >= tp_price
        else:
            sl_hit = h >= sl_price
            tp_hit = l <= tp_price
        if sl_hit:
            return ("SL", sl_price)
        if tp_hit:
            return ("TP", tp_price)
    return None


def _close_position_at(symbol: str, position: str, exit_price: float,
                      exit_time_str: str, reason_label: str,
                      hit_r: float | None = None) -> bool:
    """Pop position state and append rich close records. Returns True if accepted."""
    entry = position_entry_price.pop(symbol, None)
    meta = position_entry_meta.pop(symbol, {})
    if entry is None:
        open_positions.pop(symbol, None)
        last_signal_by_pair.pop(symbol, None)
        save_state()
        return False

    sl_dist = float(meta.get("sl_distance", 0.0))
    if position == "LONG":
        win_bool = exit_price > float(entry)
    else:
        win_bool = exit_price < float(entry)

    if hit_r is not None:
        r_mult = float(hit_r)
    elif sl_dist > 0:
        if position == "LONG":
            r_mult = (exit_price - float(entry)) / sl_dist
        else:
            r_mult = (float(entry) - exit_price) / sl_dist
    else:
        r_mult = 0.0

    trade_results.append(win_bool)
    record = {
        "symbol": symbol, "direction": position,
        "entry_price": float(entry), "exit_price": float(exit_price),
        "entry_time": meta.get("entry_time", ""),
        "exit_time": exit_time_str,
        "sl_distance": sl_dist, "r": r_mult,
        "win": win_bool,
        "exit_reason": reason_label,
    }
    closed_trades.append(record)

    sid = meta.get("signal_id")
    accepted = False
    if sid and sid in active_signals:
        sig = active_signals.pop(sid)
        if sig.get("status") == "accepted":
            accepted = True
            acc_record = dict(record)
            acc_record["entry_time"] = sig.get("entry_time", record["entry_time"])
            accepted_completed.append(acc_record)

    open_positions.pop(symbol, None)
    last_signal_by_pair.pop(symbol, None)
    save_state()
    return accepted


def _fmt_price(p: float) -> str:
    """Format a price as USD with sensible precision based on magnitude."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "$?"
    if p <= 0:
        return f"${p}"
    if p >= 1000:
        return f"${p:,.2f}"
    if p >= 1:
        return f"${p:.4f}"
    if p >= 0.001:
        return f"${p:.5f}"
    return f"${p:.7f}"


def format_message(symbol: str, r: dict) -> str:
    ts = datetime.now(MYT).strftime("%Y-%m-%d %H:%M MYT")
    price = float(r["price"])
    atr_v = float(r["atr"])
    tp_mult = params["tp_mult"]
    sl_mult = params["sl_mult"]
    direction = r["direction"]
    if direction == "LONG":
        emoji = "🟢"
        action_label = "Buy at"
        tp = price + tp_mult * atr_v
        sl = price - sl_mult * atr_v
        why = "Price was very oversold — expecting a bounce up."
    else:
        emoji = "🔴"
        action_label = "Sell short at"
        tp = price - tp_mult * atr_v
        sl = price + sl_mult * atr_v
        why = "Price was very overbought — expecting a drop down."
    return (
        f"{emoji} <b>{direction}: {symbol}</b>\n"
        f"{action_label}: {_fmt_price(price)}\n"
        f"Take profit: {_fmt_price(tp)}\n"
        f"Stop loss: {_fmt_price(sl)}\n"
        f"Why: {why}\n"
        f"Sent: {ts}"
    )


def check_long_exit(r: dict) -> list[str]:
    return r.get("__strategy_long_exits", [])


def check_short_exit(r: dict) -> list[str]:
    return r.get("__strategy_short_exits", [])


def format_close_message(symbol: str, direction: str, r: dict,
                         reasons: list[str], entry_price: float | None = None) -> str:
    ts = datetime.now(MYT).strftime("%Y-%m-%d %H:%M MYT")
    exit_price = float(r["price"])
    pct = None
    if entry_price and entry_price > 0:
        if direction == "LONG":
            pct = (exit_price - entry_price) / entry_price * 100
        else:
            pct = (entry_price - exit_price) / entry_price * 100
    if direction == "LONG":
        action_label = "Sell at"
        plain_reason = "Bounce target reached — time to close."
    else:
        action_label = "Buy back at"
        plain_reason = "Drop target reached — time to close."
    if pct is None:
        emoji = "⚪"
        result_line = ""
    elif pct >= 0:
        emoji = "✅"
        result_line = f"Profit: +{pct:.2f}% from entry\n"
    else:
        emoji = "❌"
        result_line = f"Loss: {pct:.2f}% from entry\n"
    lines = [
        f"{emoji} <b>CLOSE {direction}: {symbol}</b>",
        f"{action_label}: {_fmt_price(exit_price)}",
    ]
    if result_line:
        lines.append(result_line.rstrip("\n"))
    lines.append(f"Reason: {plain_reason}")
    lines.append(f"Time: {ts}")
    return "\n".join(lines)


def score_pair(df: pd.DataFrame, symbol: str) -> tuple[str, int]:
    sid = active_strategy_id
    if sid and sid in STRATEGIES_BY_ID:
        strat = STRATEGIES_BY_ID[sid]
        p = strat.precompute(df)
        n = len(df["close"])
        ls, ss = strat.score_at(p, n - 1)
        if ls >= ss:
            return "LONG", int(ls)
        return "SHORT", int(ss)
    return "LONG", 0


def _handle_signal_callback(cb: dict, target_chat: str) -> None:
    cb_id = cb.get("id", "")
    data = cb.get("data", "") or ""
    msg_obj = cb.get("message") or {}
    from_chat = str(msg_obj.get("chat", {}).get("id", ""))
    if from_chat != target_chat:
        answer_callback_query(cb_id, "Unauthorized")
        return
    if "_" not in data:
        answer_callback_query(cb_id, "Bad data")
        return
    action, _, sid = data.partition("_")
    if sid not in active_signals:
        answer_callback_query(cb_id, "Signal expired or unknown")
        return
    sig = active_signals[sid]
    cur_status = sig.get("status", "")
    if cur_status != "pending":
        answer_callback_query(cb_id, f"Already {cur_status}")
        return
    msg_id = sig.get("message_id")
    original = sig.get("original_text", "")
    global total_skipped_count
    if action == "accept":
        sig["status"] = "accepted"
        save_state()
        if msg_id:
            edit_telegram_message(msg_id, original + "\n\n<b>✅ Accepted</b>")
        answer_callback_query(cb_id, "Trade accepted")
        log.info("signal accepted: %s", sid)
    elif action == "skip":
        sig["status"] = "skipped"
        total_skipped_count += 1
        save_state()
        if msg_id:
            edit_telegram_message(msg_id, original + "\n\n<b>❌ Skipped</b>")
        answer_callback_query(cb_id, "Trade skipped")
        log.info("signal skipped: %s", sid)
    else:
        answer_callback_query(cb_id, "Unknown action")


def _expire_old_signals() -> None:
    global total_skipped_count
    now = int(time.time())
    expired_ids = []
    for sid, sig in list(active_signals.items()):
        if sig.get("status") != "pending":
            continue
        if now - int(sig.get("sent_at", now)) > SIGNAL_TIMEOUT_SECONDS:
            expired_ids.append(sid)
    if not expired_ids:
        return
    for sid in expired_ids:
        sig = active_signals[sid]
        sig["status"] = "skipped"
        sig["expired"] = True
        total_skipped_count += 1
        msg_id = sig.get("message_id")
        if msg_id:
            edit_telegram_message(
                msg_id,
                sig.get("original_text", "") + "\n\n<b>⌛ Expired (4h timeout)</b>",
            )
    save_state()
    log.info("Expired %d pending signals", len(expired_ids))


def handle_reset_command(reply_to_message_id: int | None = None) -> None:
    global total_signals_sent_count, total_skipped_count
    log.info("Processing /reset command")
    closed_trades.clear()
    accepted_completed.clear()
    trade_results[:] = []
    total_signals_sent_count = 0
    total_skipped_count = 0
    save_state()
    send_telegram(
        "✅ Win rate and trade history reset. Tracking fresh from now.",
        reply_to_message_id=reply_to_message_id,
    )


PAPER_MARGIN = 100.0       # USDT margin per trade
PAPER_LEVERAGE = 10        # 10x leverage
PAPER_NOTIONAL = PAPER_MARGIN * PAPER_LEVERAGE  # = $1000 effective position size
PAPER_FEE_PCT_ONE_SIDE = 0.0004  # Binance Futures taker 0.04% per side


def handle_paper_command(reply_to_message_id: int | None = None) -> None:
    """Show paper-trading stats assuming every closed trade was auto-executed
    at fixed $100 notional with Binance taker fees."""
    log.info("Processing /paper command")
    if not closed_trades:
        send_telegram(
            "No closed trades yet. Paper P&L starts populating after the "
            "first signal closes (SL hit, TP hit, or strategy exit).",
            reply_to_message_id=reply_to_message_id,
        )
        return

    notional = PAPER_NOTIONAL
    fee_rt = 2 * PAPER_FEE_PCT_ONE_SIDE  # round-trip
    n = len(closed_trades)
    wins = 0
    losses = 0
    gross_total = 0.0
    fees_total = 0.0
    best = None
    worst = None
    per_pair: dict[str, dict] = {}

    for t in closed_trades:
        try:
            entry = float(t["entry_price"])
            exit_p = float(t["exit_price"])
        except (TypeError, ValueError, KeyError):
            continue
        if entry <= 0:
            continue
        direction = t.get("direction", "")
        if direction == "LONG":
            pct = (exit_p - entry) / entry
        elif direction == "SHORT":
            pct = (entry - exit_p) / entry
        else:
            continue
        gross_pnl = notional * pct
        fee_dollars = notional * fee_rt
        net_pnl = gross_pnl - fee_dollars

        gross_total += gross_pnl
        fees_total += fee_dollars
        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        if best is None or net_pnl > best["pnl"]:
            best = {"pnl": net_pnl, "symbol": t.get("symbol", "?"),
                    "dir": direction, "time": t.get("exit_time", "")}
        if worst is None or net_pnl < worst["pnl"]:
            worst = {"pnl": net_pnl, "symbol": t.get("symbol", "?"),
                     "dir": direction, "time": t.get("exit_time", "")}

        sym = t.get("symbol", "?")
        if sym not in per_pair:
            per_pair[sym] = {"trades": 0, "pnl": 0.0}
        per_pair[sym]["trades"] += 1
        per_pair[sym]["pnl"] += net_pnl

    net_total = gross_total - fees_total
    wr = (100.0 * wins / n) if n else 0.0
    avg_trade = net_total / n if n else 0.0
    pct_on_margin = (net_total / (PAPER_MARGIN * n) * 100) if n else 0.0

    active = STRATEGIES_BY_ID.get(active_strategy_id) if active_strategy_id else None
    backtest_wr = float(getattr(active, "backtest_wr", 61.3)) if active else 61.3
    diff = wr - backtest_wr

    # Accepted comparison
    acc_n = len(accepted_completed)
    acc_wins = sum(1 for t in accepted_completed if t.get("win"))
    acc_wr = (100.0 * acc_wins / acc_n) if acc_n else None

    body = (
        "━━━━━━━━━━━━━━━━━━━\n"
        f"Margin per trade:   ${PAPER_MARGIN:.0f} USDT\n"
        f"Leverage:           {PAPER_LEVERAGE}x\n"
        f"Effective notional: ${notional:.0f}\n"
        f"Fees (round-trip):  {fee_rt*100:.2f}% (Binance taker)\n"
        "\n"
        f"Total trades:       {n}\n"
        f"Wins:               {wins}\n"
        f"Losses:             {losses}\n"
        f"Win rate:           {wr:.1f}%\n"
        "\n"
        f"Gross P&L:          {gross_total:+.2f} USDT\n"
        f"Fees paid:          -{fees_total:.2f} USDT\n"
        f"Net P&L:            {net_total:+.2f} USDT\n"
        f"Return on margin:   {pct_on_margin:+.2f}% per ${PAPER_MARGIN:.0f}\n"
        f"Avg trade:          {avg_trade:+.2f} USDT\n"
    )
    if best is not None:
        body += (
            "\n"
            f"Best:  {best['symbol']} {best['dir']} {best['pnl']:+.2f} USD "
            f"({str(best['time'])[:10]})\n"
            f"Worst: {worst['symbol']} {worst['dir']} {worst['pnl']:+.2f} USD "
            f"({str(worst['time'])[:10]})\n"
        )
    body += (
        "━━━━━━━━━━━━━━━━━━━\n"
        f"Backtest WR: {backtest_wr:.1f}%\n"
        f"Paper WR:    {wr:.1f}% ({diff:+.1f} pp vs backtest)\n"
    )
    if acc_wr is not None:
        body += f"Your accepted WR: {acc_wr:.1f}% (n={acc_n})\n"

    msg = (
        f"📄 <b>Paper Trading — Williams %R</b>\n"
        f"<pre>{body}</pre>"
    )
    send_telegram(msg, reply_to_message_id=reply_to_message_id)


def handle_commands_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /commands command")
    msg = (
        "<b>📖 Available commands</b>\n\n"
        "<b>Trade tracking</b>\n"
        "<code>/positions</code> — open positions you accepted, with live P&L\n"
        "<code>/win</code> — performance stats on your accepted trades\n"
        "<code>/paper</code> — what auto-trading every signal would have earned\n"
        "<code>/reset</code> — wipe win/loss history\n"
        "\n"
        "<b>Manual position control</b>\n"
        "<code>/cancel SYMBOL</code> — order never filled, drop from tracking (no stats impact)\n"
        "<code>/close SYMBOL</code> — you exited manually, record close at market\n"
        "<code>/reopen SYMBOL LONG|SHORT [price]</code> — re-attach a position\n"
        "(also <code>/cancel all</code>, <code>/close all</code>)\n"
        "\n"
        "<b>Info / health</b>\n"
        "<code>/check</code> — confluence scores across all 20 pairs\n"
        "<code>/ping</code> — bot online? last/next scan? open count?\n"
        "<code>/commands</code> — this list\n"
        "\n"
        "<b>On each signal</b>\n"
        "✅ I'm In — bot will message you when conditions say close\n"
        "❌ Skip — bot ignores it for your stats\n"
        "(no tap within 4h = auto-skip)"
    )
    send_telegram(msg, reply_to_message_id=reply_to_message_id)


def handle_ping_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /ping command")
    now = int(time.time())
    if last_scan_time:
        age = now - last_scan_time
        if age < 60:
            ago = f"{age}s ago"
        elif age < 3600:
            ago = f"{age // 60}m ago"
        else:
            ago = f"{age // 3600}h {(age % 3600) // 60}m ago"
        next_in = SCAN_INTERVAL_SECONDS - age
        if next_in <= 0:
            next_str = "any moment"
        elif next_in < 60:
            next_str = f"in {next_in}s"
        else:
            next_str = f"in ~{next_in // 60}m"
    else:
        ago = "scan hasn't run yet"
        next_str = "starting up"

    active = STRATEGIES_BY_ID.get(active_strategy_id) if active_strategy_id else None
    strat_name = active.name if active else "(unknown)"
    n_open = len(open_positions)
    n_pending = sum(
        1 for s in active_signals.values() if s.get("status") == "pending"
    )
    msg = (
        f"✅ <b>Bot online</b>\n"
        f"Strategy: {strat_name}\n"
        f"Last scan: {ago}\n"
        f"Next scan: {next_str}\n"
        f"Active positions: {n_open}\n"
        f"Pending signals: {n_pending}"
    )
    send_telegram(msg, reply_to_message_id=reply_to_message_id)


def _send_daily_summary() -> None:
    now = datetime.now(MYT)
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight = today_midnight - timedelta(days=1)

    def in_yesterday(time_str: str) -> bool:
        if not time_str:
            return False
        try:
            t = pd.to_datetime(time_str, utc=True).tz_convert(MYT)
            return yesterday_midnight <= t < today_midnight
        except Exception:
            return False

    y_acc_closed = [
        t for t in accepted_completed if in_yesterday(t.get("exit_time", ""))
    ]
    y_wins = sum(1 for t in y_acc_closed if t.get("win"))
    y_losses = len(y_acc_closed) - y_wins

    y_total_entries = sum(
        1 for t in closed_trades if in_yesterday(t.get("entry_time", ""))
    ) + sum(
        1 for s in active_signals.values()
        if in_yesterday(s.get("entry_time", ""))
    )

    y_accepted_entries = sum(
        1 for t in accepted_completed if in_yesterday(t.get("entry_time", ""))
    ) + sum(
        1 for s in active_signals.values()
        if s.get("status") == "accepted" and in_yesterday(s.get("entry_time", ""))
    )

    net_pct = 0.0
    for t in y_acc_closed:
        try:
            entry = float(t.get("entry_price", 0))
            exit_p = float(t.get("exit_price", 0))
            if entry <= 0:
                continue
            if t.get("direction") == "LONG":
                net_pct += (exit_p - entry) / entry * 100
            else:
                net_pct += (entry - exit_p) / entry * 100
        except Exception:
            pass

    n_open = len(open_positions)
    n_pending = sum(
        1 for s in active_signals.values() if s.get("status") == "pending"
    )

    msg = (
        f"📅 <b>Daily summary — {now.strftime('%Y-%m-%d')}</b>\n\n"
        f"<b>Yesterday</b>\n"
        f"Signals fired: {y_total_entries}\n"
        f"You accepted: {y_accepted_entries}\n"
        f"Closed: {len(y_acc_closed)} ({y_wins} won, {y_losses} lost)\n"
        f"Net result on closed: {net_pct:+.2f}%\n\n"
        f"<b>Right now</b>\n"
        f"Open positions: {n_open}\n"
        f"Pending response: {n_pending}\n\n"
        f"Have a good day."
    )
    send_telegram(msg)


def _maybe_send_daily_summary() -> None:
    global last_daily_summary_date
    now = datetime.now(MYT)
    today_str = now.strftime("%Y-%m-%d")
    if now.hour < 8:
        return
    if last_daily_summary_date == today_str:
        return
    try:
        _send_daily_summary()
    except Exception as e:
        log.warning("daily summary send failed: %s", e)
        return
    last_daily_summary_date = today_str
    save_state()


def handle_win_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /win command")

    def _is_win(t: dict) -> bool:
        if "win" in t:
            return bool(t["win"])
        direction = t.get("direction", "")
        try:
            entry = float(t.get("entry_price", 0))
            exit_p = float(t.get("exit_price", 0))
        except (TypeError, ValueError):
            return False
        if direction == "LONG":
            return exit_p > entry
        if direction == "SHORT":
            return exit_p < entry
        return float(t.get("r", 0)) > 0

    def _fmt_date(s: str) -> str:
        if not s:
            return "?"
        try:
            return str(s)[:10]
        except Exception:
            return str(s)

    # Section 1 — Accepted trades performance
    n_acc_closed = len(accepted_completed)
    acc_winners = [t for t in accepted_completed if _is_win(t)]
    acc_losers = [t for t in accepted_completed if not _is_win(t)]
    n_acc_win = len(acc_winners)
    n_acc_loss = len(acc_losers)
    acc_win_rate = (100.0 * n_acc_win / n_acc_closed) if n_acc_closed else 0.0
    acc_rated = [t for t in accepted_completed if float(t.get("r", 0)) != 0]
    acc_rated_winners = [t for t in acc_rated if _is_win(t)]
    acc_rated_losers = [t for t in acc_rated if not _is_win(t)]
    acc_avg_winner = (
        sum(float(t["r"]) for t in acc_rated_winners) / len(acc_rated_winners)
        if acc_rated_winners else 0.0
    )
    acc_avg_loser = (
        sum(float(t["r"]) for t in acc_rated_losers) / len(acc_rated_losers)
        if acc_rated_losers else 0.0
    )
    acc_expectancy = (
        sum(float(t["r"]) for t in acc_rated) / len(acc_rated)
        if acc_rated else 0.0
    )
    if acc_rated:
        acc_best = max(acc_rated, key=lambda t: float(t.get("r", 0)))
        acc_worst = min(acc_rated, key=lambda t: float(t.get("r", 0)))
    else:
        acc_best = acc_worst = None

    # Section 2 — All signals counters
    n_sent = total_signals_sent_count
    # Accepted = anything that got status accepted: those still open (status accepted in
    # active_signals) plus those now in accepted_completed
    n_accepted_open = sum(
        1 for s in active_signals.values() if s.get("status") == "accepted"
    )
    n_accepted_total = n_acc_closed + n_accepted_open
    n_skipped = total_skipped_count
    n_pending = sum(
        1 for s in active_signals.values() if s.get("status") == "pending"
    )
    accept_rate = (100.0 * n_accepted_total / n_sent) if n_sent else 0.0

    # Date range from earliest signal seen
    times = [s.get("entry_time", "") for s in accepted_completed if s.get("entry_time")]
    times.extend(
        s.get("entry_time", "") for s in active_signals.values() if s.get("entry_time")
    )
    first_date = min(times) if times else ""
    today = datetime.now(MYT).strftime("%Y-%m-%d")

    active = STRATEGIES_BY_ID.get(active_strategy_id) if active_strategy_id else None
    strat_name = active.name if active else "(unknown)"
    backtest_wr = float(getattr(active, "backtest_wr", 61.3)) if active else 61.3

    # Build the accepted-trades section
    if n_acc_closed == 0 and n_accepted_open == 0 and n_sent == 0:
        send_telegram(
            "No signals tracked yet. Once the bot fires a signal and you tap "
            "✅ I'm In or ❌ Skip, your live stats will start populating.",
            reply_to_message_id=reply_to_message_id,
        )
        return

    if acc_best is not None and acc_worst is not None:
        best_line = (
            f"Best:        {acc_best['symbol']} {float(acc_best['r']):+.2f} R "
            f"({_fmt_date(acc_best.get('exit_time', ''))})"
        )
        worst_line = (
            f"Worst:       {acc_worst['symbol']} {float(acc_worst['r']):+.2f} R "
            f"({_fmt_date(acc_worst.get('exit_time', ''))})"
        )
    else:
        best_line = "Best:        (no rated accepted trades yet)"
        worst_line = "Worst:       (no rated accepted trades yet)"

    section1 = (
        "Section 1: Accepted Trades\n"
        "─────────────────────\n"
        f"Total accepted:        {n_accepted_total}\n"
        f" ├ Closed:             {n_acc_closed}\n"
        f" └ Still open:         {n_accepted_open}\n"
        f"Wins:                  {n_acc_win}\n"
        f"Losses:                {n_acc_loss}\n"
        f"Win Rate:              {acc_win_rate:.1f}%\n"
        f"Avg winner:           {acc_avg_winner:+.2f} R\n"
        f"Avg loser:            {acc_avg_loser:+.2f} R\n"
        f"Expectancy:           {acc_expectancy:+.2f} R\n"
        f"{best_line}\n"
        f"{worst_line}\n"
    )

    section2 = (
        "Section 2: All Signals Sent\n"
        "─────────────────────\n"
        f"Total signals sent:    {n_sent}\n"
        f"Accepted:              {n_accepted_total}\n"
        f"Skipped:               {n_skipped}\n"
        f"Pending response:      {n_pending}\n"
        f"Acceptance rate:       {accept_rate:.1f}%\n"
    )

    diff = acc_win_rate - backtest_wr
    cmp_line = (
        f"Backtest WR: {backtest_wr:.1f}% | "
        f"Your accepted trades WR: {acc_win_rate:.1f}% "
        f"({diff:+.1f} pp)"
    )

    body = (
        "━━━━━━━━━━━━━━━━━━━\n"
        f"From: {_fmt_date(first_date)} to {today}\n\n"
        f"{section1}\n"
        f"{section2}\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        f"{cmp_line}"
    )
    msg = (
        f"\U0001F4CA <b>Live Performance — {strat_name}</b>\n"
        f"<pre>{body}</pre>"
    )
    send_telegram(msg, reply_to_message_id=reply_to_message_id)


def handle_reopen_command(args: list[str], reply_to_message_id: int | None = None) -> None:
    """Manually re-attach a position to the bot's tracking.
    Usage: /reopen SYMBOL LONG|SHORT [entry_price]"""
    log.info("Processing /reopen command: %s", args)
    if len(args) < 2:
        send_telegram(
            "Usage: <code>/reopen SYMBOL LONG|SHORT [entry_price]</code>\n\n"
            "Use this if you cancelled but your order actually filled, or "
            "if you took a trade outside the bot and want it to manage the "
            "close for you.\n\n"
            "Examples:\n"
            "<code>/reopen BNBUSDT LONG</code> (uses current market price)\n"
            "<code>/reopen BNBUSDT LONG 567.50</code> (specify entry)",
            reply_to_message_id=reply_to_message_id,
        )
        return

    symbol = args[0].upper()
    direction = args[1].upper()
    if direction not in ("LONG", "SHORT"):
        send_telegram(
            "Direction must be <code>LONG</code> or <code>SHORT</code>.",
            reply_to_message_id=reply_to_message_id,
        )
        return
    if symbol not in PAIRS:
        send_telegram(
            f"<b>{symbol}</b> is not in the bot's pair list.",
            reply_to_message_id=reply_to_message_id,
        )
        return
    if symbol in open_positions:
        send_telegram(
            f"Already tracking a {open_positions[symbol]} position on "
            f"<b>{symbol}</b>. Use <code>/cancel {symbol}</code> first if "
            "you want to replace it.",
            reply_to_message_id=reply_to_message_id,
        )
        return

    entry_price = None
    if len(args) >= 3:
        try:
            entry_price = float(args[2])
        except ValueError:
            send_telegram(
                f"Invalid entry price: <code>{args[2]}</code>",
                reply_to_message_id=reply_to_message_id,
            )
            return

    try:
        df = fetch_klines(symbol)
        if len(df) < 14:
            send_telegram(
                f"Not enough data to set up {symbol}.",
                reply_to_message_id=reply_to_message_id,
            )
            return
        atr_series = atr(df["high"], df["low"], df["close"], 14)
        atr_at_entry = float(atr_series.iloc[-1])
        if entry_price is None:
            entry_price = float(df["close"].iloc[-1])
        entry_time = str(df["close_time"].iloc[-1])
    except Exception as e:
        send_telegram(
            f"Failed to fetch data for {symbol}: {e}",
            reply_to_message_id=reply_to_message_id,
        )
        return

    sl_mult_v = float(params.get("sl_mult", 1.0))
    tp_mult_v = float(params.get("tp_mult", 2.0))
    sl_distance = atr_at_entry * sl_mult_v
    if direction == "LONG":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_mult_v * atr_at_entry
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_mult_v * atr_at_entry

    signal_id = f"{symbol}_{direction}_{int(time.time() * 1000)}_manual"
    sent_at_ts = int(time.time())

    open_positions[symbol] = direction
    position_entry_price[symbol] = float(entry_price)
    position_entry_meta[symbol] = {
        "entry_time": entry_time,
        "atr": atr_at_entry,
        "sl_mult": sl_mult_v,
        "sl_distance": sl_distance,
        "signal_id": signal_id,
    }
    last_signal_by_pair[symbol] = direction
    active_signals[signal_id] = {
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "entry_price": float(entry_price),
        "entry_time": entry_time,
        "atr_at_entry": atr_at_entry,
        "sl_distance": sl_distance,
        "sent_at": sent_at_ts,
        "status": "accepted",
        "original_text": f"Manually reopened {direction} {symbol}",
        "message_id": None,
        "manual_reopen": True,
    }
    save_state()

    emoji = "🟢" if direction == "LONG" else "🔴"
    send_telegram(
        f"{emoji} <b>Reopened: {direction} {symbol}</b>\n"
        f"Entry: {_fmt_price(entry_price)}\n"
        f"Take profit: {_fmt_price(tp_price)}\n"
        f"Stop loss: {_fmt_price(sl_price)}\n\n"
        f"The bot will tell you when to close. Use /positions to verify.",
        reply_to_message_id=reply_to_message_id,
    )
    log.info("/reopen: %s %s @ %.6g", symbol, direction, entry_price)


def handle_cancel_command(args: list[str], reply_to_message_id: int | None = None) -> None:
    """Remove a position from tracking without recording a trade.
    Use when a signal fired but your order didn't fill (so you're not in it)."""
    log.info("Processing /cancel command: %s", args)
    if not args:
        send_telegram(
            "Usage: <code>/cancel SYMBOL</code> or <code>/cancel all</code>\n\n"
            "Use this if the signal fired and you tapped ✅ I'm In, but your "
            "order never filled and you're not actually in the trade. "
            "The bot will forget the position and resume sending new signals.",
            reply_to_message_id=reply_to_message_id,
        )
        return
    target = args[0].upper()
    if target == "ALL":
        if not open_positions:
            send_telegram("No open positions to cancel.",
                          reply_to_message_id=reply_to_message_id)
            return
        symbols = list(open_positions.keys())
    else:
        if target not in open_positions:
            send_telegram(
                f"No open position for <b>{target}</b>.",
                reply_to_message_id=reply_to_message_id,
            )
            return
        symbols = [target]

    cancelled: list[tuple[str, str]] = []
    for sym in symbols:
        position = open_positions.get(sym)
        if position is None:
            continue
        meta = position_entry_meta.pop(sym, {})
        position_entry_price.pop(sym, None)
        last_signal_by_pair.pop(sym, None)
        open_positions.pop(sym, None)
        sid = meta.get("signal_id")
        if sid and sid in active_signals:
            active_signals.pop(sid)
        cancelled.append((sym, position))
        log.info("/cancel: %s %s removed from tracking (no trade recorded)",
                 sym, position)
    save_state()

    if not cancelled:
        send_telegram("Nothing cancelled.", reply_to_message_id=reply_to_message_id)
        return
    lines = ["🗑️ <b>Cancelled (not recorded as a trade):</b>"]
    for sym, pos in cancelled:
        lines.append(f"• {pos} {sym}")
    lines.append("")
    lines.append("The bot will resume sending new signals.")
    send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)


def handle_close_command(args: list[str], reply_to_message_id: int | None = None) -> None:
    """Manually close a position. Usage: /close BTCUSDT [or /close all]"""
    log.info("Processing /close command: %s", args)
    if not args:
        send_telegram(
            "Usage: <code>/close SYMBOL</code> or <code>/close all</code>",
            reply_to_message_id=reply_to_message_id,
        )
        return
    target = args[0].upper()
    if target == "ALL":
        if not open_positions:
            send_telegram("No open positions to close.",
                          reply_to_message_id=reply_to_message_id)
            return
        symbols = list(open_positions.keys())
    else:
        if target not in open_positions:
            send_telegram(
                f"No open position for <b>{target}</b>.",
                reply_to_message_id=reply_to_message_id,
            )
            return
        symbols = [target]

    closed = []
    for sym in symbols:
        position = open_positions.get(sym)
        if position is None:
            continue
        # Use the latest available price as the exit
        try:
            df = fetch_klines(sym, limit=2)
            exit_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
            exit_time = str(df["close_time"].iloc[-1]) if len(df) > 0 else ""
        except Exception as e:
            log.warning("/close %s price fetch failed: %s", sym, e)
            exit_price = float(position_entry_price.get(sym, 0.0))
            exit_time = datetime.now(MYT).isoformat()
        _close_position_at(sym, position, exit_price, exit_time,
                           "Manually closed via /close")
        closed.append((sym, position, exit_price))
        log.info("/close: %s %s closed manually at %.6g", sym, position, exit_price)

    if not closed:
        send_telegram("Nothing closed.", reply_to_message_id=reply_to_message_id)
        return
    lines = ["✅ <b>Manually closed:</b>"]
    for sym, pos, px in closed:
        lines.append(f"• {pos} {sym} at {_fmt_price(px)}")
    send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)


def handle_positions_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /positions command")
    # Only show positions the user explicitly accepted. Skipped, pending,
    # expired, or pre-deploy positions are tracked internally but not
    # displayed here — they aren't trades the user is actually in.
    accepted_open: list[tuple[str, str]] = []
    for symbol, direction in open_positions.items():
        meta = position_entry_meta.get(symbol, {})
        sid = meta.get("signal_id")
        if not sid or sid not in active_signals:
            continue
        if active_signals[sid].get("status") != "accepted":
            continue
        accepted_open.append((symbol, direction))

    if not accepted_open:
        send_telegram(
            "No accepted open positions right now.",
            reply_to_message_id=reply_to_message_id,
        )
        return

    lines = ["📊 <b>Open Positions</b>", ""]
    for symbol, direction in accepted_open:
        entry = position_entry_price.get(symbol)
        current = None
        try:
            df = fetch_klines(symbol, limit=2)
            if len(df) > 0:
                current = float(df["close"].iloc[-1])
        except Exception as e:
            log.warning("/positions %s price fetch failed: %s", symbol, e)

        emoji = "🟢" if direction == "LONG" else "🔴"
        lines.append(f"{emoji} <b>{direction} {symbol}</b>")
        if entry:
            lines.append(f"Entry: {_fmt_price(entry)}")
        if current is not None:
            lines.append(f"Now: {_fmt_price(current)}")
            if entry and entry > 0:
                if direction == "LONG":
                    pct = (current - entry) / entry * 100
                else:
                    pct = (entry - current) / entry * 100
                if pct >= 0:
                    lines.append(f"P&L: <b>+{pct:.2f}%</b>")
                else:
                    lines.append(f"P&L: <b>{pct:.2f}%</b>")
        lines.append("")

    send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)


def handle_check_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /check command")
    rankings: list[tuple[str, str, int]] = []
    for symbol in PAIRS:
        try:
            df = fetch_klines(symbol)
            if len(df) < 120:
                continue
            direction, score = score_pair(df, symbol)
            rankings.append((symbol, direction, score))
        except Exception as e:
            log.warning("/check %s failed: %s", symbol, e)

    rankings.sort(key=lambda x: x[2], reverse=True)
    ts = datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S MYT")
    lines = [f"<b>Confluence check — {ts}</b>"]

    active = None
    sid = active_strategy_id
    if sid and sid in STRATEGIES_BY_ID:
        active = STRATEGIES_BY_ID[sid]
    if active is not None:
        wr = getattr(active, "backtest_wr", None)
        wr_str = f" | Backtest WR: {wr:.1f}%" if wr is not None else ""
        lines.append(f"Strategy: {active.name}{wr_str}")
    lines.append("")

    for symbol, direction, score in rankings:
        filled = round(score / 10)
        bar = "█" * filled + "░" * (10 - filled)
        lines.append(f"{symbol} {direction} {score}% {bar}")

    descriptions = getattr(active, "score_description", None) if active else None
    if descriptions:
        lines.append("")
        lines.append("<b>Score breakdown:</b>")
        for d in descriptions:
            lines.append(f"• {d}")

    lines.append("")
    lines.append("<i>100% = a live signal would fire right now.</i>")
    send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)
    log.info("/check reply sent (%d pairs)", len(rankings))





def telegram_poll_loop() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram polling disabled (missing credentials)")
        return
    target_chat = str(TELEGRAM_CHAT_ID)
    url = f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    offset = 0
    try:
        r = requests.get(url, params={"offset": -1, "timeout": 0}, timeout=10)
        if r.ok:
            results = r.json().get("result", [])
            if results:
                offset = results[-1]["update_id"] + 1
    except Exception as e:
        log.warning("Failed to drain pending updates: %s", e)
    log.info("Telegram /check listener started")
    while True:
        try:
            r = requests.get(
                url,
                params={"offset": offset, "timeout": 2},
                timeout=10,
            )
            if r.status_code != 200:
                log.warning("getUpdates failed: %s %s", r.status_code, r.text)
                time.sleep(2)
                continue
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                cb = upd.get("callback_query")
                if cb:
                    try:
                        _handle_signal_callback(cb, target_chat)
                    except Exception as e:
                        log.exception("callback handler failed: %s", e)
                    continue
                msg = upd.get("message")
                if not msg:
                    continue
                if str(msg.get("chat", {}).get("id", "")) != target_chat:
                    continue
                text = (msg.get("text") or "").strip()
                parts = text.split()
                if not parts:
                    continue
                cmd = parts[0].split("@")[0]
                if cmd == "/check":
                    handle_check_command(msg.get("message_id"))
                elif cmd == "/win":
                    handle_win_command(msg.get("message_id"))
                elif cmd == "/reset":
                    handle_reset_command(msg.get("message_id"))
                elif cmd == "/ping":
                    handle_ping_command(msg.get("message_id"))
                elif cmd == "/paper":
                    handle_paper_command(msg.get("message_id"))
                elif cmd == "/commands":
                    handle_commands_command(msg.get("message_id"))
                elif cmd == "/positions":
                    handle_positions_command(msg.get("message_id"))
                elif cmd == "/close":
                    handle_close_command(parts[1:], msg.get("message_id"))
                elif cmd == "/cancel":
                    handle_cancel_command(parts[1:], msg.get("message_id"))
                elif cmd == "/reopen":
                    handle_reopen_command(parts[1:], msg.get("message_id"))
            try:
                _expire_old_signals()
            except Exception as e:
                log.warning("expire sweep failed: %s", e)
        except Exception as e:
            log.warning("Telegram poll error: %s", e)
            time.sleep(5)
            continue
        time.sleep(2)


def scan_once() -> None:
    global last_scan_time
    last_scan_time = int(time.time())
    log.info("Starting scan of %d pairs", len(PAIRS))
    for symbol in PAIRS:
        try:
            df = fetch_klines(symbol)
            if len(df) < 120:
                log.warning("%s: not enough candles (%d)", symbol, len(df))
                continue
            result = evaluate(df, symbol)

            position = open_positions.get(symbol)

            # SL/TP detection on candles after entry. If hit, close at the
            # SL or TP price and skip the strategy-exit check below.
            if position is not None:
                meta_peek = position_entry_meta.get(symbol, {})
                entry_peek = position_entry_price.get(symbol)
                sl_dist_peek = float(meta_peek.get("sl_distance", 0.0))
                sl_mult_v = float(meta_peek.get("sl_mult", 1.0)) or 1.0
                tp_mult_p = float(params.get("tp_mult", 2.0))
                tp_dist_peek = sl_dist_peek * (tp_mult_p / sl_mult_v)
                entry_time_str = meta_peek.get("entry_time", "")
                sl_tp = _detect_sl_tp_hit(
                    df, position, entry_peek, sl_dist_peek,
                    tp_dist_peek, entry_time_str,
                )
                if sl_tp:
                    hit_type, hit_price = sl_tp
                    sid_peek = meta_peek.get("signal_id")
                    accepted_peek = bool(
                        sid_peek and sid_peek in active_signals
                        and active_signals[sid_peek].get("status") == "accepted"
                    )
                    if accepted_peek and entry_peek and entry_peek > 0:
                        if position == "LONG":
                            pct = (hit_price - entry_peek) / entry_peek * 100
                        else:
                            pct = (entry_peek - hit_price) / entry_peek * 100
                        pct_str = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
                        emoji = "❌" if hit_type == "SL" else "✅"
                        reason_msg = ("Stop loss triggered"
                                      if hit_type == "SL" else "Take profit hit")
                        ts = datetime.now(MYT).strftime("%Y-%m-%d %H:%M MYT")
                        send_telegram(
                            f"{emoji} <b>CLOSE {position}: {symbol}</b>\n"
                            f"Closed at: {_fmt_price(hit_price)}\n"
                            f"P&L: {pct_str} from entry\n"
                            f"Reason: {reason_msg}\n"
                            f"Time: {ts}"
                        )
                    elif not accepted_peek:
                        log.info("%s: %s %s hit (silent — not accepted)",
                                 symbol, position, hit_type)
                    last_ct = str(df["close_time"].iloc[-1])
                    r_val = -1.0 if hit_type == "SL" else (
                        tp_mult_p / sl_mult_v if sl_mult_v > 0 else 0.0
                    )
                    reason_label = (f"SL hit at {_fmt_price(hit_price)}"
                                    if hit_type == "SL"
                                    else f"TP hit at {_fmt_price(hit_price)}")
                    _close_position_at(symbol, position, hit_price,
                                       last_ct, reason_label, hit_r=r_val)
                    log.info("%s: %s %s hit, closed at %.6g",
                             symbol, position, hit_type, hit_price)
                    position = None  # skip strategy-exit branch

            if position == "LONG":
                reasons = check_long_exit(result)
                if reasons:
                    meta_peek = position_entry_meta.get(symbol, {})
                    sid_peek = meta_peek.get("signal_id")
                    accepted = bool(
                        sid_peek and sid_peek in active_signals
                        and active_signals[sid_peek].get("status") == "accepted"
                    )
                    if accepted:
                        entry_for_msg = position_entry_price.get(symbol)
                        send_telegram(format_close_message(
                            symbol, "LONG", result, reasons, entry_for_msg
                        ))
                    else:
                        log.info("%s: LONG close (silent — not accepted)", symbol)
                    entry = position_entry_price.pop(symbol, None)
                    meta = position_entry_meta.pop(symbol, {})
                    exit_price = float(result["price"])
                    exit_time = str(result.get("candle_time", ""))
                    if entry is not None:
                        win_bool = exit_price > entry
                        trade_results.append(win_bool)
                        sl_dist = float(meta.get("sl_distance", 0.0))
                        r_mult = ((exit_price - entry) / sl_dist) if sl_dist > 0 else 0.0
                        closed_trades.append({
                            "symbol": symbol, "direction": "LONG",
                            "entry_price": float(entry), "exit_price": exit_price,
                            "entry_time": meta.get("entry_time", ""),
                            "exit_time": exit_time,
                            "sl_distance": sl_dist, "r": r_mult,
                            "win": win_bool,
                            "exit_reason": "; ".join(reasons),
                        })
                        # Accepted-trade tracking (additive, separate from above)
                        sid = meta.get("signal_id")
                        if sid and sid in active_signals:
                            sig = active_signals.pop(sid)
                            if sig.get("status") == "accepted":
                                accepted_completed.append({
                                    "symbol": symbol, "direction": "LONG",
                                    "entry_price": float(entry),
                                    "exit_price": exit_price,
                                    "entry_time": sig.get("entry_time", ""),
                                    "exit_time": exit_time,
                                    "sl_distance": sl_dist, "r": r_mult,
                                    "win": win_bool,
                                    "exit_reason": "; ".join(reasons),
                                })
                    open_positions.pop(symbol, None)
                    last_signal_by_pair.pop(symbol, None)
                    save_state()
                    log.info("%s: CLOSE LONG sent (%s)", symbol, "; ".join(reasons))
            elif position == "SHORT":
                reasons = check_short_exit(result)
                if reasons:
                    meta_peek = position_entry_meta.get(symbol, {})
                    sid_peek = meta_peek.get("signal_id")
                    accepted = bool(
                        sid_peek and sid_peek in active_signals
                        and active_signals[sid_peek].get("status") == "accepted"
                    )
                    if accepted:
                        entry_for_msg = position_entry_price.get(symbol)
                        send_telegram(format_close_message(
                            symbol, "SHORT", result, reasons, entry_for_msg
                        ))
                    else:
                        log.info("%s: SHORT close (silent — not accepted)", symbol)
                    entry = position_entry_price.pop(symbol, None)
                    meta = position_entry_meta.pop(symbol, {})
                    exit_price = float(result["price"])
                    exit_time = str(result.get("candle_time", ""))
                    if entry is not None:
                        win_bool = exit_price < entry
                        trade_results.append(win_bool)
                        sl_dist = float(meta.get("sl_distance", 0.0))
                        r_mult = ((entry - exit_price) / sl_dist) if sl_dist > 0 else 0.0
                        closed_trades.append({
                            "symbol": symbol, "direction": "SHORT",
                            "entry_price": float(entry), "exit_price": exit_price,
                            "entry_time": meta.get("entry_time", ""),
                            "exit_time": exit_time,
                            "sl_distance": sl_dist, "r": r_mult,
                            "win": win_bool,
                            "exit_reason": "; ".join(reasons),
                        })
                        sid = meta.get("signal_id")
                        if sid and sid in active_signals:
                            sig = active_signals.pop(sid)
                            if sig.get("status") == "accepted":
                                accepted_completed.append({
                                    "symbol": symbol, "direction": "SHORT",
                                    "entry_price": float(entry),
                                    "exit_price": exit_price,
                                    "entry_time": sig.get("entry_time", ""),
                                    "exit_time": exit_time,
                                    "sl_distance": sl_dist, "r": r_mult,
                                    "win": win_bool,
                                    "exit_reason": "; ".join(reasons),
                                })
                    open_positions.pop(symbol, None)
                    last_signal_by_pair.pop(symbol, None)
                    save_state()
                    log.info("%s: CLOSE SHORT sent (%s)", symbol, "; ".join(reasons))

            direction = result["direction"]
            if direction is None:
                continue
            if last_signal_by_pair.get(symbol) == direction:
                log.info("%s: duplicate %s signal, skipping", symbol, direction)
                continue
            # If an accepted trade is currently open, suppress new entries
            # entirely. The bot keeps scanning but doesn't message the user
            # until the accepted one is closed.
            has_open_accepted = any(
                s.get("status") == "accepted"
                for s in active_signals.values()
            )
            if has_open_accepted:
                log.info(
                    "%s: %s signal suppressed — accepted trade still open",
                    symbol, direction,
                )
                continue
            last_signal_by_pair[symbol] = direction
            open_positions[symbol] = direction
            position_entry_price[symbol] = result["price"]
            atr_at_entry = float(result.get("atr", 0.0) or 0.0)
            sl_mult_v = float(params.get("sl_mult", 1.0) or 1.0)
            sent_at_ts = int(time.time())
            signal_id = f"{symbol}_{direction}_{int(time.time() * 1000)}"
            position_entry_meta[symbol] = {
                "entry_time": str(result.get("candle_time", "")),
                "atr": atr_at_entry,
                "sl_mult": sl_mult_v,
                "sl_distance": atr_at_entry * sl_mult_v,
                "signal_id": signal_id,
            }
            msg = format_message(symbol, result)
            active_signals[signal_id] = {
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": direction,
                "entry_price": float(result["price"]),
                "entry_time": str(result.get("candle_time", "")),
                "atr_at_entry": atr_at_entry,
                "sl_distance": atr_at_entry * sl_mult_v,
                "sent_at": sent_at_ts,
                "status": "pending",
                "original_text": msg,
                "message_id": None,
            }
            global total_signals_sent_count
            total_signals_sent_count += 1
            save_state()
            message_id = send_telegram_with_buttons(msg, signal_id)
            if message_id is not None:
                active_signals[signal_id]["message_id"] = message_id
                save_state()
            log.info("%s: %s signal sent (sid=%s, msg=%s)",
                     symbol, direction, signal_id, message_id)
        except requests.RequestException as e:
            log.error("%s: network error: %s", symbol, e)
        except Exception as e:
            log.exception("%s: evaluation error: %s", symbol, e)
        time.sleep(0.25)


def main() -> None:
    log.info("binance-signal-bot starting (state path: %s)", STATE_PATH)
    load_state()
    load_params()
    load_strategy()
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    send_telegram(
        "<b>binance-signal-bot online</b>\n"
        "Active strategy: <b>Williams %R</b> (hardcoded permanent default).\n"
        "LONG when W%R(14) crosses above -80 from below; "
        "SHORT when W%R(14) crosses below -20 from above; "
        "exit when W%R reaches -50 or ATR-based SL is hit.\n"
        "Backtest: 61.3% WR on optimization (6274 trades), "
        "57.4% WR on validation (3009 trades).\n"
        f"Timeframe: {TIMEFRAME} across {len(PAIRS)} Binance Futures pairs.\n"
        f"{win_rate_text()}\n"
        f"Started: {datetime.now(MYT).strftime('%Y-%m-%d %H:%M:%S MYT')}"
    )
    while True:
        start = time.time()
        try:
            scan_once()
        except Exception as e:
            log.exception("scan loop error: %s", e)
        try:
            _maybe_send_daily_summary()
        except Exception as e:
            log.warning("daily summary scheduler failed: %s", e)
        elapsed = time.time() - start
        sleep_for = max(0, SCAN_INTERVAL_SECONDS - int(elapsed))
        log.info("Scan complete in %ds; sleeping %ds", int(elapsed), sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
