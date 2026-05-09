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

STATE_PATH = os.environ.get(
    "STATE_PATH",
    "/data/state.json" if os.path.isdir("/data") else "state.json",
)

PARAMS_PATH = os.environ.get(
    "PARAMS_PATH",
    "/data/params.json" if os.path.isdir("/data") else "params.json",
)

DEFAULT_PARAMS: dict = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bb_period": 20,
    "bb_std": 2.0,
    "sl_mult": 1.0,
    "tp_mult": 2.0,
}

params: dict = dict(DEFAULT_PARAMS)


def load_params() -> None:
    if not os.path.exists(PARAMS_PATH):
        log.info("No saved params at %s; using defaults", PARAMS_PATH)
        return
    try:
        with open(PARAMS_PATH) as f:
            saved = json.load(f)
        for k in DEFAULT_PARAMS:
            if k in saved:
                params[k] = saved[k]
        log.info("Loaded params: %s", params)
    except Exception as e:
        log.warning("Failed to load params from %s: %s", PARAMS_PATH, e)


def save_params() -> None:
    try:
        parent = os.path.dirname(PARAMS_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = PARAMS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(params, f)
        os.replace(tmp, PARAMS_PATH)
    except Exception as e:
        log.warning("Failed to save params to %s: %s", PARAMS_PATH, e)


def params_text() -> str:
    return (
        f"RSI {params['rsi_oversold']}/{params['rsi_overbought']}, "
        f"BB({params['bb_period']}, {params['bb_std']}), "
        f"SL {params['sl_mult']}xATR, "
        f"TP {params['tp_mult']}xATR"
    )


def load_state() -> None:
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
        log.info(
            "Loaded state: %d trades, %d open positions",
            len(trade_results), len(open_positions),
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
            }, f)
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


def fetch_klines_paginated(symbol: str, interval: str, target: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    end_ms = int(time.time() * 1000)
    fetched = 0
    while fetched < target:
        chunk = fetch_klines(symbol, interval=interval, limit=1000, end_time=end_ms)
        if chunk.empty:
            break
        chunks.append(chunk)
        fetched += len(chunk)
        oldest_open_ms = int(chunk["open_time"].iloc[0].timestamp() * 1000)
        end_ms = oldest_open_ms - 1
        if len(chunk) < 1000:
            break
        time.sleep(0.1)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window=period).mean()
    sd = close.rolling(window=period).std()
    upper = middle + std * sd
    lower = middle - std * sd
    return upper, middle, lower


# ===== Strategy discovery framework =====

def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / rng


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                       k_period: int = 14, k_smooth: int = 3, d_smooth: int = 3):
    ll = low.rolling(window=k_period).min()
    hh = high.rolling(window=k_period).max()
    raw_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k = raw_k.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d


def compute_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 10, mult: float = 3.0):
    hl2 = (high + low) / 2
    atr_v = atr(high, low, close, period)
    bu = (hl2 + mult * atr_v).values
    bl = (hl2 - mult * atr_v).values
    cl = close.values
    n = len(close)
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    for i in range(n):
        if i == 0 or np.isnan(bu[i]):
            fu[i] = bu[i]
            fl[i] = bl[i]
            direction[i] = 1
            continue
        if np.isnan(fu[i - 1]):
            fu[i] = bu[i]
        elif bu[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]:
            fu[i] = bu[i]
        else:
            fu[i] = fu[i - 1]
        if np.isnan(fl[i - 1]):
            fl[i] = bl[i]
        elif bl[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]:
            fl[i] = bl[i]
        else:
            fl[i] = fl[i - 1]
        prev = direction[i - 1]
        if prev == -1 and cl[i] > fu[i]:
            direction[i] = 1
        elif prev == 1 and cl[i] < fl[i]:
            direction[i] = -1
        else:
            direction[i] = prev
    line = np.where(direction == 1, fl, fu)
    return pd.Series(line, index=close.index), pd.Series(direction, index=close.index)


def compute_heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series):
    n = len(close)
    ha_close = (open_ + high + low + close) / 4
    ha_close_v = ha_close.values
    o = open_.values
    h = high.values
    l = low.values
    ha_open = np.full(n, np.nan)
    ha_high = np.full(n, np.nan)
    ha_low = np.full(n, np.nan)
    ha_open[0] = (o[0] + close.iloc[0]) / 2
    ha_high[0] = h[0]
    ha_low[0] = l[0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close_v[i - 1]) / 2
        ha_high[i] = max(h[i], ha_open[i], ha_close_v[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close_v[i])
    return (
        pd.Series(ha_open, index=close.index),
        pd.Series(ha_high, index=close.index),
        pd.Series(ha_low, index=close.index),
        ha_close,
    )


def compute_rolling_vwap(df: pd.DataFrame, reset_period: int = 24) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    group = pd.Series(np.arange(len(df)) // reset_period, index=df.index)
    cumul_tp_vol = tp_vol.groupby(group).cumsum()
    cumul_vol = df["volume"].groupby(group).cumsum()
    return cumul_tp_vol / cumul_vol.replace(0, np.nan)


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

def _s1_pre(df):
    return {"rsi": rsi(df["close"], 14)}
def _s1_sig(p, i):
    if i < 1: return None
    pr = p["rsi"].iloc[i-1]; cu = p["rsi"].iloc[i]
    if pd.isna(pr) or pd.isna(cu): return None
    if pr < 30 and cu >= 30: return "LONG"
    if pr > 70 and cu <= 70: return "SHORT"
    return None
def _s1_exit(p, i, d):
    cu = p["rsi"].iloc[i]
    if pd.isna(cu): return False
    if d == "LONG" and cu >= 50: return f"RSI hit 50 ({cu:.1f})"
    if d == "SHORT" and cu <= 50: return f"RSI hit 50 ({cu:.1f})"
    return False
_register("d1_rsi_meanrev", "Pure RSI Mean Reversion", _s1_pre, _s1_sig, _s1_exit)


def _s2_pre(df):
    u, m, l = compute_bollinger_bands(df["close"], 20, 2.0)
    width = (u - l) / m.replace(0, np.nan)
    return {
        "bb_upper": u, "bb_middle": m, "bb_lower": l,
        "width": width, "width_avg": width.rolling(20).mean(),
        "vol": df["volume"], "vol_sma": df["volume"].rolling(20).mean(),
    }
def _s2_sig(p, i):
    if i < 5: return None
    w = p["width"]; wa = p["width_avg"]
    squeeze = all(
        pd.notna(w.iloc[i-k]) and pd.notna(wa.iloc[i-k]) and w.iloc[i-k] < wa.iloc[i-k]
        for k in range(1, 6)
    )
    if not squeeze: return None
    c = p["bb_upper"].iloc[i]; b = p["bb_lower"].iloc[i]
    cl = float(p["bb_middle"].iloc[i])
    vol = p["vol"].iloc[i]; vs = p["vol_sma"].iloc[i]
    if pd.isna(vol) or pd.isna(vs) or vs <= 0: return None
    vol_ok = vol >= 1.5 * vs
    if not vol_ok: return None
    # Need actual close
    return None  # signal_fn doesn't have access to close directly; handled via wrapper
_register("d2_bb_squeeze", "BB Squeeze Breakout", _s2_pre, _s2_sig, None,
          sl_mult=1.5, tp_mult=2.5)


def _s3_pre(df):
    return {
        "ema9": ema(df["close"], 9),
        "ema21": ema(df["close"], 21),
        "vol": df["volume"],
        "vol_sma": df["volume"].rolling(20).mean(),
    }
def _s3_sig(p, i):
    if i < 1: return None
    e9p = p["ema9"].iloc[i-1]; e9c = p["ema9"].iloc[i]
    e21p = p["ema21"].iloc[i-1]; e21c = p["ema21"].iloc[i]
    if any(pd.isna(x) for x in (e9p, e9c, e21p, e21c)): return None
    vol = p["vol"].iloc[i]; vs = p["vol_sma"].iloc[i]
    vol_ok = pd.notna(vs) and pd.notna(vol) and vol > vs
    if not vol_ok: return None
    if e9p <= e21p and e9c > e21c: return "LONG"
    if e9p >= e21p and e9c < e21c: return "SHORT"
    return None
def _s3_exit(p, i, d):
    if i < 1: return False
    e9p = p["ema9"].iloc[i-1]; e9c = p["ema9"].iloc[i]
    e21p = p["ema21"].iloc[i-1]; e21c = p["ema21"].iloc[i]
    if any(pd.isna(x) for x in (e9p, e9c, e21p, e21c)): return False
    if d == "LONG" and e9p >= e21p and e9c < e21c: return "EMA9 crossed below EMA21"
    if d == "SHORT" and e9p <= e21p and e9c > e21c: return "EMA9 crossed above EMA21"
    return False
_register("d3_ema_cross_vol", "EMA9/21 Crossover with Volume", _s3_pre, _s3_sig, _s3_exit)


def _s4_pre(df):
    return {
        "vwap": compute_rolling_vwap(df, 24),
        "rsi": rsi(df["close"], 14),
        "close": df["close"],
    }
def _s4_sig(p, i):
    c = p["close"].iloc[i]
    v = p["vwap"].iloc[i]
    r = p["rsi"].iloc[i]
    if pd.isna(c) or pd.isna(v) or pd.isna(r) or v <= 0: return None
    diff_pct = (c - v) / v
    if diff_pct < -0.015 and r < 40: return "LONG"
    if diff_pct > 0.015 and r > 60: return "SHORT"
    return None
def _s4_exit(p, i, d):
    if i < 1: return False
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    vp = p["vwap"].iloc[i-1]; vc = p["vwap"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, vp, vc)): return False
    if d == "LONG" and cp < vp and cc >= vc: return "Price returned to VWAP"
    if d == "SHORT" and cp > vp and cc <= vc: return "Price returned to VWAP"
    return False
_register("d4_vwap_reversion", "VWAP Reversion", _s4_pre, _s4_sig, _s4_exit)


def _s5_pre(df):
    k, d = compute_stochastic(df["high"], df["low"], df["close"], 14, 3, 3)
    return {"k": k, "d": d}
def _s5_sig(p, i):
    if i < 1: return None
    kp = p["k"].iloc[i-1]; kc = p["k"].iloc[i]
    dp = p["d"].iloc[i-1]; dc = p["d"].iloc[i]
    if any(pd.isna(x) for x in (kp, kc, dp, dc)): return None
    if kp <= dp and kc > dc and kc < 20: return "LONG"
    if kp >= dp and kc < dc and kc > 80: return "SHORT"
    return None
def _s5_exit(p, i, d):
    kc = p["k"].iloc[i]
    if pd.isna(kc): return False
    if d == "LONG" and (kc >= 50 or kc >= 80): return f"Stoch K {kc:.1f}"
    if d == "SHORT" and (kc <= 50 or kc <= 20): return f"Stoch K {kc:.1f}"
    return False
_register("d5_stochastic", "Stochastic Crossover", _s5_pre, _s5_sig, _s5_exit)


def _s6_pre(df):
    u, m, l = compute_bollinger_bands(df["close"], 20, 2.0)
    return {
        "rsi": rsi(df["close"], 14),
        "bb_middle": m, "bb_upper": u, "bb_lower": l,
        "low": df["low"], "high": df["high"], "close": df["close"],
    }
def _s6_sig(p, i):
    if i < 5: return None
    r = p["rsi"]; lo = p["low"]; hi = p["high"]
    rc = r.iloc[i]; rb = r.iloc[i-5]
    lc = lo.iloc[i]; lb = lo.iloc[i-5]
    hc = hi.iloc[i]; hb = hi.iloc[i-5]
    if any(pd.isna(x) for x in (rc, rb, lc, lb, hc, hb)): return None
    if rc < 45 and lc < lb and rc > rb: return "LONG"
    if rc > 55 and hc > hb and rc < rb: return "SHORT"
    return None
def _s6_exit(p, i, d):
    c = p["close"].iloc[i]; m = p["bb_middle"].iloc[i]
    if pd.isna(c) or pd.isna(m): return False
    if d == "LONG" and c >= m: return "Reached BB middle"
    if d == "SHORT" and c <= m: return "Reached BB middle"
    return False
_register("d6_rsi_divergence", "RSI Divergence", _s6_pre, _s6_sig, _s6_exit)


def _s7_pre(df):
    line, direction = compute_supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    return {"st_line": line, "st_dir": direction, "close": df["close"]}
def _s7_sig(p, i):
    if i < 1: return None
    dp = p["st_dir"].iloc[i-1]; dc = p["st_dir"].iloc[i]
    if pd.isna(dp) or pd.isna(dc): return None
    if dp == -1 and dc == 1: return "LONG"
    if dp == 1 and dc == -1: return "SHORT"
    return None
def _s7_exit(p, i, d):
    if i < 1: return False
    dp = p["st_dir"].iloc[i-1]; dc = p["st_dir"].iloc[i]
    if pd.isna(dp) or pd.isna(dc): return False
    if d == "LONG" and dp == 1 and dc == -1: return "Supertrend flipped down"
    if d == "SHORT" and dp == -1 and dc == 1: return "Supertrend flipped up"
    return False
_register("d7_supertrend", "Supertrend", _s7_pre, _s7_sig, _s7_exit)


def _s8_pre(df):
    ho, hh, hl, hc = compute_heikin_ashi(df["open"], df["high"], df["low"], df["close"])
    return {"ha_open": ho, "ha_close": hc}
def _s8_sig(p, i):
    if i < 2: return None
    ho = p["ha_open"]; hc = p["ha_close"]
    g = lambda k: pd.notna(ho.iloc[k]) and pd.notna(hc.iloc[k]) and hc.iloc[k] > ho.iloc[k]
    r = lambda k: pd.notna(ho.iloc[k]) and pd.notna(hc.iloc[k]) and hc.iloc[k] < ho.iloc[k]
    if g(i) and g(i-1) and g(i-2): return "LONG"
    if r(i) and r(i-1) and r(i-2): return "SHORT"
    return None
def _s8_exit(p, i, d):
    ho = p["ha_open"].iloc[i]; hc = p["ha_close"].iloc[i]
    if pd.isna(ho) or pd.isna(hc): return False
    if d == "LONG" and hc < ho: return "First red HA candle"
    if d == "SHORT" and hc > ho: return "First green HA candle"
    return False
_register("d8_heikin_ashi", "Heikin Ashi Trend", _s8_pre, _s8_sig, _s8_exit)


def _s9_pre(df):
    return {"wr": compute_williams_r(df["high"], df["low"], df["close"], 14)}
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


def _s10_pre(df):
    return {
        "rsi3": rsi(df["close"], 3),
        "rsi": rsi(df["close"], 14),
    }
def _s10_sig(p, i):
    if i < 1: return None
    r3p = p["rsi3"].iloc[i-1]; r3c = p["rsi3"].iloc[i]
    r14 = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (r3p, r3c, r14)): return None
    if r3p < 20 and r3c >= 20 and r14 < 50: return "LONG"
    if r3p > 80 and r3c <= 80 and r14 > 50: return "SHORT"
    return None
def _s10_exit(p, i, d):
    r3 = p["rsi3"].iloc[i]
    if pd.isna(r3): return False
    if d == "LONG" and r3 >= 80: return f"RSI(3) hit 80 ({r3:.1f})"
    if d == "SHORT" and r3 <= 20: return f"RSI(3) hit 20 ({r3:.1f})"
    return False
_register("d10_double_rsi", "Double RSI", _s10_pre, _s10_sig, _s10_exit)


# Strategy 2 needs `close` access - rewrite signal closure
def _s2_sig_v2(p, i):
    if i < 5: return None
    w = p["width"]; wa = p["width_avg"]
    if any(pd.isna(w.iloc[i-k]) or pd.isna(wa.iloc[i-k]) or w.iloc[i-k] >= wa.iloc[i-k]
           for k in range(1, 6)):
        return None
    cl = float(p["bb_middle"].iloc[i])  # not actually close but used as proxy for direction
    upper = p["bb_upper"].iloc[i]; lower = p["bb_lower"].iloc[i]
    if pd.isna(upper) or pd.isna(lower): return None
    vol = p["vol"].iloc[i]; vs = p["vol_sma"].iloc[i]
    if pd.isna(vol) or pd.isna(vs) or vs <= 0 or vol < 1.5 * vs: return None
    # Compare close to upper/lower band — close stored as middle's underlying series? we need close.
    # Since precompute doesn't include close, we add it.
    return None  # will be replaced below


def _s2_pre_v2(df):
    u, m, l = compute_bollinger_bands(df["close"], 20, 2.0)
    width = (u - l) / m.replace(0, np.nan)
    return {
        "bb_upper": u, "bb_middle": m, "bb_lower": l,
        "width": width, "width_avg": width.rolling(20).mean(),
        "vol": df["volume"], "vol_sma": df["volume"].rolling(20).mean(),
        "close": df["close"],
    }
def _s2_sig_final(p, i):
    if i < 5: return None
    w = p["width"]; wa = p["width_avg"]
    for k in range(1, 6):
        wv = w.iloc[i-k]; wav = wa.iloc[i-k]
        if pd.isna(wv) or pd.isna(wav) or wv >= wav:
            return None
    upper = p["bb_upper"].iloc[i]; lower = p["bb_lower"].iloc[i]
    cl = p["close"].iloc[i]
    vol = p["vol"].iloc[i]; vs = p["vol_sma"].iloc[i]
    if any(pd.isna(x) for x in (upper, lower, cl, vol, vs)) or vs <= 0:
        return None
    if vol < 1.5 * vs: return None
    if cl > upper: return "LONG"
    if cl < lower: return "SHORT"
    return None

# Re-register strategy 2 with the corrected versions
STRATEGIES_BY_ID["d2_bb_squeeze"]._precompute = _s2_pre_v2
STRATEGIES_BY_ID["d2_bb_squeeze"]._signal = _s2_sig_final


# ===== /discover2 batch 1: strategies 1-25 =====

def _wma(s: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()
    return s.rolling(period).apply(
        lambda x: float(np.dot(x, weights) / wsum) if not np.any(np.isnan(x)) else float("nan"),
        raw=True,
    )


def _roc(s: pd.Series, period: int) -> pd.Series:
    return 100 * (s - s.shift(period)) / s.shift(period).replace(0, np.nan)


def _crosses(prev_a, cur_a, prev_b, cur_b, direction="up"):
    if any(pd.isna(x) for x in (prev_a, cur_a, prev_b, cur_b)):
        return False
    if direction == "up":
        return prev_a <= prev_b and cur_a > cur_b
    return prev_a >= prev_b and cur_a < cur_b


# 1. Chandelier Exit
def _d2s1_pre(df):
    a = atr(df["high"], df["low"], df["close"], 22)
    hh = df["high"].rolling(22).max()
    ll = df["low"].rolling(22).min()
    return {
        "atr": a, "long_stop": hh - 3 * a, "short_stop": ll + 3 * a,
        "close": df["close"],
    }
def _d2s1_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    lsp = p["long_stop"].iloc[i-1]; lsc = p["long_stop"].iloc[i]
    ssp = p["short_stop"].iloc[i-1]; ssc = p["short_stop"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, lsp, lsc, ssp, ssc)): return None
    if cp <= lsp and cc > lsc: return "LONG"
    if cp >= ssp and cc < ssc: return "SHORT"
    return None
_register("e1_chandelier", "Chandelier Exit", _d2s1_pre, _d2s1_sig, None, group="discover2")


# 2. Keltner Channel reversion
def _d2s2_pre(df):
    em = ema(df["close"], 20)
    a = atr(df["high"], df["low"], df["close"], 14)
    return {"ema20": em, "atr": a, "kc_upper": em + 2 * a, "kc_lower": em - 2 * a, "close": df["close"]}
def _d2s2_sig(p, i):
    c = p["close"].iloc[i]; u = p["kc_upper"].iloc[i]; l = p["kc_lower"].iloc[i]
    if any(pd.isna(x) for x in (c, u, l)): return None
    if c < l: return "LONG"
    if c > u: return "SHORT"
    return None
def _d2s2_exit(p, i, d):
    c = p["close"].iloc[i]; m = p["ema20"].iloc[i]
    if pd.isna(c) or pd.isna(m): return False
    if d == "LONG" and c >= m: return "Reached EMA20"
    if d == "SHORT" and c <= m: return "Reached EMA20"
    return False
_register("e2_keltner", "Keltner Channel Reversion", _d2s2_pre, _d2s2_sig, _d2s2_exit, group="discover2")


# 3. Elder Ray
def _d2s3_pre(df):
    em = ema(df["close"], 13)
    return {"ema13": em, "bull": df["high"] - em, "bear": df["low"] - em}
def _d2s3_sig(p, i):
    if i < 2: return None
    em = p["ema13"]; bear = p["bear"]; bull = p["bull"]
    if any(pd.isna(x) for x in (em.iloc[i], em.iloc[i-1], bear.iloc[i], bear.iloc[i-1])):
        return None
    em_rising = em.iloc[i] > em.iloc[i-1]
    em_falling = em.iloc[i] < em.iloc[i-1]
    bear_rising_neg = bear.iloc[i] < 0 and bear.iloc[i] > bear.iloc[i-1]
    bull_falling_pos = bull.iloc[i] > 0 and bull.iloc[i] < bull.iloc[i-1]
    if em_rising and bear_rising_neg: return "LONG"
    if em_falling and bull_falling_pos: return "SHORT"
    return None
_register("e3_elder_ray", "Elder Ray", _d2s3_pre, _d2s3_sig, None, group="discover2")


# 4. Coppock Curve
def _d2s4_pre(df):
    rc = _roc(df["close"], 14) + _roc(df["close"], 11)
    return {"coppock": _wma(rc, 10)}
def _d2s4_sig(p, i):
    if i < 1: return None
    cp = p["coppock"].iloc[i-1]; cc = p["coppock"].iloc[i]
    if pd.isna(cp) or pd.isna(cc): return None
    if cp <= 0 and cc > 0: return "LONG"
    if cp >= 0 and cc < 0: return "SHORT"
    return None
_register("e4_coppock", "Coppock Curve", _d2s4_pre, _d2s4_sig, None, group="discover2")


# 5. Detrended Price Oscillator
def _d2s5_pre(df, period=20):
    shift = period // 2 + 1
    sma = df["close"].rolling(period).mean()
    return {"dpo": df["close"] - sma.shift(shift)}
def _d2s5_sig(p, i):
    if i < 1: return None
    pp = p["dpo"].iloc[i-1]; pc = p["dpo"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e5_dpo", "Detrended Price Oscillator", _d2s5_pre, _d2s5_sig, None, group="discover2")


# 6. Mass Index
def _d2s6_pre(df):
    rng = df["high"] - df["low"]
    e1 = ema(rng, 9); e2 = ema(e1, 9)
    ratio = e1 / e2.replace(0, np.nan)
    mi = ratio.rolling(25).sum()
    em9 = ema(df["close"], 9)
    return {"mi": mi, "ema9": em9}
def _d2s6_sig(p, i):
    if i < 5: return None
    mi = p["mi"]; em = p["ema9"]
    cur = mi.iloc[i]
    if pd.isna(cur): return None
    # need any of last 5 to have been > 27
    spike = any(pd.notna(mi.iloc[i-k]) and mi.iloc[i-k] > 27 for k in (1, 2, 3, 4, 5))
    if not spike: return None
    if cur >= 26.5: return None
    if pd.isna(em.iloc[i]) or pd.isna(em.iloc[i-1]): return None
    em_up = em.iloc[i] > em.iloc[i-1]
    if em_up: return "LONG"
    return "SHORT"
_register("e6_mass_index", "Mass Index", _d2s6_pre, _d2s6_sig, None, group="discover2")


# 7. Vortex Indicator
def _d2s7_pre(df, period=14):
    h = df["high"]; l = df["low"]; c = df["close"]
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    vmp = (h - l.shift(1)).abs()
    vmn = (l - h.shift(1)).abs()
    vi_p = vmp.rolling(period).sum() / tr.rolling(period).sum().replace(0, np.nan)
    vi_n = vmn.rolling(period).sum() / tr.rolling(period).sum().replace(0, np.nan)
    return {"vi_p": vi_p, "vi_n": vi_n}
def _d2s7_sig(p, i):
    if i < 1: return None
    if _crosses(p["vi_p"].iloc[i-1], p["vi_p"].iloc[i], p["vi_n"].iloc[i-1], p["vi_n"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["vi_p"].iloc[i-1], p["vi_p"].iloc[i], p["vi_n"].iloc[i-1], p["vi_n"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e7_vortex", "Vortex Indicator", _d2s7_pre, _d2s7_sig, None, group="discover2")


# 8. TRIX
def _d2s8_pre(df, period=15):
    e1 = ema(df["close"], period); e2 = ema(e1, period); e3 = ema(e2, period)
    return {"trix": 100 * (e3 - e3.shift(1)) / e3.shift(1).replace(0, np.nan)}
def _d2s8_sig(p, i):
    if i < 1: return None
    pp = p["trix"].iloc[i-1]; pc = p["trix"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e8_trix", "TRIX", _d2s8_pre, _d2s8_sig, None, group="discover2")


# 9. Know Sure Thing (KST)
def _d2s9_pre(df):
    r1 = _roc(df["close"], 10).rolling(10).mean()
    r2 = _roc(df["close"], 15).rolling(10).mean()
    r3 = _roc(df["close"], 20).rolling(10).mean()
    r4 = _roc(df["close"], 30).rolling(15).mean()
    kst = r1 + 2 * r2 + 3 * r3 + 4 * r4
    return {"kst": kst, "kst_sig": kst.rolling(9).mean()}
def _d2s9_sig(p, i):
    if i < 1: return None
    if _crosses(p["kst"].iloc[i-1], p["kst"].iloc[i], p["kst_sig"].iloc[i-1], p["kst_sig"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["kst"].iloc[i-1], p["kst"].iloc[i], p["kst_sig"].iloc[i-1], p["kst_sig"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e9_kst", "Know Sure Thing", _d2s9_pre, _d2s9_sig, None, group="discover2")


# 10. Aroon
def _d2s10_pre(df, period=25):
    high = df["high"]; low = df["low"]
    aroon_up = high.rolling(period + 1).apply(
        lambda x: 100 * (period - (period - int(np.argmax(x)))) / period, raw=True
    )
    aroon_dn = low.rolling(period + 1).apply(
        lambda x: 100 * (period - (period - int(np.argmin(x)))) / period, raw=True
    )
    return {"aup": aroon_up, "adn": aroon_dn}
def _d2s10_sig(p, i):
    if i < 1: return None
    if _crosses(p["aup"].iloc[i-1], p["aup"].iloc[i], p["adn"].iloc[i-1], p["adn"].iloc[i], "up"):
        if p["aup"].iloc[i] > 70: return "LONG"
    if _crosses(p["aup"].iloc[i-1], p["aup"].iloc[i], p["adn"].iloc[i-1], p["adn"].iloc[i], "down"):
        if p["adn"].iloc[i] > 70: return "SHORT"
    return None
_register("e10_aroon", "Aroon Crossover", _d2s10_pre, _d2s10_sig, None, group="discover2")


# 11. Chande Momentum Oscillator
def _d2s11_pre(df, period=14):
    diff = df["close"].diff()
    up = diff.clip(lower=0).rolling(period).sum()
    dn = (-diff.clip(upper=0)).rolling(period).sum()
    cmo = 100 * (up - dn) / (up + dn).replace(0, np.nan)
    return {"cmo": cmo}
def _d2s11_sig(p, i):
    if i < 1: return None
    pp = p["cmo"].iloc[i-1]; pc = p["cmo"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= -50 and pc > -50: return "LONG"
    if pp >= 50 and pc < 50: return "SHORT"
    return None
_register("e11_cmo", "Chande Momentum", _d2s11_pre, _d2s11_sig, None, group="discover2")


# 12. Price Oscillator (PPO-like)
def _d2s12_pre(df):
    e10 = ema(df["close"], 10); e30 = ema(df["close"], 30)
    return {"po": 100 * (e10 - e30) / e30.replace(0, np.nan)}
def _d2s12_sig(p, i):
    if i < 1: return None
    pp = p["po"].iloc[i-1]; pc = p["po"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e12_price_osc", "Price Oscillator", _d2s12_pre, _d2s12_sig, None, group="discover2")


# 13. Inertia (Linear regression of RVI)
def _d2s13_pre(df, period=14):
    co = (df["close"] - df["open"]).rolling(4).mean()
    hl = (df["high"] - df["low"]).rolling(4).mean()
    rvi = co / hl.replace(0, np.nan)
    inertia = rvi.rolling(period).apply(
        lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1) + np.mean(x)) if not np.any(np.isnan(x)) else float("nan"),
        raw=True,
    )
    # Scale 0-100 by passing through sigmoid-like normalize to 50 +/- 50*tanh
    inertia_scaled = 50 + 50 * np.tanh(inertia)
    return {"inertia": pd.Series(inertia_scaled, index=df.index)}
def _d2s13_sig(p, i):
    if i < 1: return None
    pp = p["inertia"].iloc[i-1]; pc = p["inertia"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 50 and pc > 50: return "LONG"
    if pp >= 50 and pc < 50: return "SHORT"
    return None
_register("e13_inertia", "Inertia", _d2s13_pre, _d2s13_sig, None, group="discover2")


# 14. Relative Vigor Index
def _d2s14_pre(df):
    co = df["close"] - df["open"]
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    rvi_n = (co + 2 * co.shift(1) + 2 * co.shift(2) + co.shift(3)) / 6
    rvi_d = (hl + 2 * hl.shift(1) + 2 * hl.shift(2) + hl.shift(3)) / 6
    rvi = rvi_n / rvi_d
    sig = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6
    return {"rvi": rvi, "sig": sig}
def _d2s14_sig(p, i):
    if i < 1: return None
    rp = p["rvi"].iloc[i-1]; rc = p["rvi"].iloc[i]
    sp = p["sig"].iloc[i-1]; sc = p["sig"].iloc[i]
    if any(pd.isna(x) for x in (rp, rc, sp, sc)): return None
    if rp <= sp and rc > sc and rc < 0: return "LONG"
    if rp >= sp and rc < sc and rc > 0: return "SHORT"
    return None
_register("e14_rvi", "Relative Vigor Index", _d2s14_pre, _d2s14_sig, None, group="discover2")


# 15. Psychological Line
def _d2s15_pre(df, period=12):
    rising = (df["close"] > df["close"].shift(1)).astype(float)
    return {"psy": 100 * rising.rolling(period).mean()}
def _d2s15_sig(p, i):
    if i < 1: return None
    pp = p["psy"].iloc[i-1]; pc = p["psy"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pc < 30 and pc > pp: return "LONG"
    if pc > 70 and pc < pp: return "SHORT"
    return None
_register("e15_psy_line", "Psychological Line", _d2s15_pre, _d2s15_sig, None, group="discover2")


# 16. Vertical Horizontal Filter
def _d2s16_pre(df, period=28):
    high_n = df["close"].rolling(period).max()
    low_n = df["close"].rolling(period).min()
    rng = high_n - low_n
    sum_diff = df["close"].diff().abs().rolling(period).sum()
    vhf = rng / sum_diff.replace(0, np.nan)
    return {"vhf": vhf, "rsi": rsi(df["close"], 14)}
def _d2s16_sig(p, i):
    v = p["vhf"].iloc[i]; r = p["rsi"].iloc[i]
    if pd.isna(v) or pd.isna(r): return None
    if v < 0.35 and r < 40: return "LONG"
    if v < 0.35 and r > 60: return "SHORT"
    return None
_register("e16_vhf", "VHF + RSI", _d2s16_pre, _d2s16_sig, None, group="discover2")


# 17. Price Channel Breakout
def _d2s17_pre(df, period=20):
    return {
        "hh": df["high"].rolling(period).max().shift(1),
        "ll": df["low"].rolling(period).min().shift(1),
        "close": df["close"],
    }
def _d2s17_sig(p, i):
    c = p["close"].iloc[i]; hh = p["hh"].iloc[i]; ll = p["ll"].iloc[i]
    if any(pd.isna(x) for x in (c, hh, ll)): return None
    if c > hh: return "LONG"
    if c < ll: return "SHORT"
    return None
_register("e17_price_channel", "Price Channel Breakout", _d2s17_pre, _d2s17_sig, None, group="discover2")


# 18. Donchian Channel midline reversion
def _d2s18_pre(df, period=20):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    mid = (hh + ll) / 2
    return {"hh": hh, "ll": ll, "mid": mid, "close": df["close"], "low": df["low"], "high": df["high"]}
def _d2s18_sig(p, i):
    c = p["close"].iloc[i]; lo = p["low"].iloc[i]; hi = p["high"].iloc[i]
    ll = p["ll"].iloc[i]; hh = p["hh"].iloc[i]; m = p["mid"].iloc[i]
    if any(pd.isna(x) for x in (c, lo, hi, ll, hh, m)): return None
    if lo <= ll and c > m: return "LONG"
    if hi >= hh and c < m: return "SHORT"
    return None
def _d2s18_exit(p, i, d):
    c = p["close"].iloc[i]; m = p["mid"].iloc[i]
    if pd.isna(c) or pd.isna(m): return False
    if d == "LONG" and c >= p["hh"].iloc[i]: return "Reached upper Donchian"
    if d == "SHORT" and c <= p["ll"].iloc[i]: return "Reached lower Donchian"
    return False
_register("e18_donchian_mid", "Donchian Mid Reversion", _d2s18_pre, _d2s18_sig, _d2s18_exit, group="discover2")


# 19. Linear Regression Slope
def _d2s19_pre(df, period=20):
    def _slope(x):
        if np.any(np.isnan(x)): return float("nan")
        return float(np.polyfit(np.arange(len(x)), x, 1)[0])
    return {"slope": df["close"].rolling(period).apply(_slope, raw=True)}
def _d2s19_sig(p, i):
    if i < 1: return None
    pp = p["slope"].iloc[i-1]; pc = p["slope"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e19_lr_slope", "Linear Regression Slope", _d2s19_pre, _d2s19_sig, None, group="discover2")


# 20. R-squared filter with RSI
def _d2s20_pre(df, period=14):
    def _r2(x):
        if np.any(np.isnan(x)): return float("nan")
        n = len(x); xs = np.arange(n)
        ss_tot = float(np.var(x) * n)
        if ss_tot == 0: return 1.0
        slope, intercept = np.polyfit(xs, x, 1)
        pred = slope * xs + intercept
        ss_res = float(np.sum((x - pred) ** 2))
        return float(1 - ss_res / ss_tot)
    return {"r2": df["close"].rolling(period).apply(_r2, raw=True), "rsi": rsi(df["close"], 14)}
def _d2s20_sig(p, i):
    r2 = p["r2"].iloc[i]; r = p["rsi"].iloc[i]
    if pd.isna(r2) or pd.isna(r): return None
    if r2 < 0.35 and r < 35: return "LONG"
    if r2 < 0.35 and r > 65: return "SHORT"
    return None
_register("e20_r2_rsi", "R-squared + RSI", _d2s20_pre, _d2s20_sig, None, group="discover2")


# 21. Adaptive RSI (Kaufman efficiency ratio adapts period)
def _d2s21_pre(df, base=14):
    diff = df["close"].diff().abs()
    direction = (df["close"] - df["close"].shift(10)).abs()
    volatility = diff.rolling(10).sum().replace(0, np.nan)
    er = (direction / volatility).clip(0, 1).fillna(0.5)
    # adaptive period 5..25
    adaptive_period = (5 + (1 - er) * 20).round().astype(int)
    # Approximate adaptive RSI by interpolating between rsi(5) and rsi(25)
    rsi5 = rsi(df["close"], 5)
    rsi25 = rsi(df["close"], 25)
    blend = (adaptive_period - 5) / 20
    arsi = rsi5 * (1 - blend) + rsi25 * blend
    return {"arsi": arsi}
def _d2s21_sig(p, i):
    if i < 1: return None
    pp = p["arsi"].iloc[i-1]; pc = p["arsi"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 30 and pc > 30: return "LONG"
    if pp >= 70 and pc < 70: return "SHORT"
    return None
_register("e21_adaptive_rsi", "Adaptive RSI", _d2s21_pre, _d2s21_sig, None, group="discover2")


# 22. Chaikin Money Flow
def _d2s22_pre(df, period=20):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mfv = mfm * df["volume"]
    cmf = mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)
    return {"cmf": cmf}
def _d2s22_sig(p, i):
    if i < 1: return None
    pp = p["cmf"].iloc[i-1]; pc = p["cmf"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0.05 and pc > 0.05: return "LONG"
    if pp >= -0.05 and pc < -0.05: return "SHORT"
    return None
_register("e22_cmf", "Chaikin Money Flow", _d2s22_pre, _d2s22_sig, None, group="discover2")


# 23. Force Index
def _d2s23_pre(df):
    fi = (df["close"] - df["close"].shift(1)) * df["volume"]
    return {"fi": ema(fi, 13)}
def _d2s23_sig(p, i):
    if i < 1: return None
    pp = p["fi"].iloc[i-1]; pc = p["fi"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e23_force_index", "Force Index", _d2s23_pre, _d2s23_sig, None, group="discover2")


# 24. Ease of Movement
def _d2s24_pre(df, period=14):
    mid = (df["high"] + df["low"]) / 2
    box = df["volume"] / (df["high"] - df["low"]).replace(0, np.nan)
    raw = (mid - mid.shift(1)) / box.replace(0, np.nan)
    return {"emv": raw.rolling(period).mean()}
def _d2s24_sig(p, i):
    if i < 1: return None
    pp = p["emv"].iloc[i-1]; pc = p["emv"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e24_emv", "Ease of Movement", _d2s24_pre, _d2s24_sig, None, group="discover2")


# 25. Negative Volume Index
def _d2s25_pre(df, ema_period=255):
    n = len(df)
    nvi = np.zeros(n)
    nvi[0] = 1000
    closes = df["close"].values
    vols = df["volume"].values
    for i in range(1, n):
        if vols[i] < vols[i-1]:
            pct = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0
            nvi[i] = nvi[i-1] * (1 + pct)
        else:
            nvi[i] = nvi[i-1]
    nvi_s = pd.Series(nvi, index=df.index)
    return {"nvi": nvi_s, "nvi_ema": ema(nvi_s, ema_period)}
def _d2s25_sig(p, i):
    if i < 1: return None
    pp = p["nvi"].iloc[i-1]; pc = p["nvi"].iloc[i]
    ep = p["nvi_ema"].iloc[i-1]; ec = p["nvi_ema"].iloc[i]
    if any(pd.isna(x) for x in (pp, pc, ep, ec)): return None
    if pp <= ep and pc > ec: return "LONG"
    if pp >= ep and pc < ec: return "SHORT"
    return None
_register("e25_nvi", "Negative Volume Index", _d2s25_pre, _d2s25_sig, None, group="discover2")


# 26. Positive Volume Index (mirror of NVI)
def _d2s26_pre(df, ema_period=255):
    n = len(df)
    pvi = np.zeros(n)
    pvi[0] = 1000
    closes = df["close"].values
    vols = df["volume"].values
    for i in range(1, n):
        if vols[i] > vols[i-1]:
            pct = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0
            pvi[i] = pvi[i-1] * (1 + pct)
        else:
            pvi[i] = pvi[i-1]
    s = pd.Series(pvi, index=df.index)
    return {"pvi": s, "pvi_ema": ema(s, ema_period)}
def _d2s26_sig(p, i):
    if i < 1: return None
    pp = p["pvi"].iloc[i-1]; pc = p["pvi"].iloc[i]
    ep = p["pvi_ema"].iloc[i-1]; ec = p["pvi_ema"].iloc[i]
    if any(pd.isna(x) for x in (pp, pc, ep, ec)): return None
    if pp <= ep and pc > ec: return "LONG"
    if pp >= ep and pc < ec: return "SHORT"
    return None
_register("e26_pvi", "Positive Volume Index", _d2s26_pre, _d2s26_sig, None, group="discover2")


# 27. Ulcer Index mean reversion
def _d2s27_pre(df, period=14):
    rolling_max = df["close"].rolling(period).max()
    pct_dd = 100 * (df["close"] - rolling_max) / rolling_max.replace(0, np.nan)
    sq = pct_dd ** 2
    ui = np.sqrt(sq.rolling(period).mean())
    return {"ui": ui, "rsi": rsi(df["close"], 14)}
def _d2s27_sig(p, i):
    if i < 5: return None
    cur = p["ui"].iloc[i]
    if pd.isna(cur) or cur >= 5: return None
    spike = any(pd.notna(p["ui"].iloc[i-k]) and p["ui"].iloc[i-k] > 10 for k in range(1, 6))
    if not spike: return None
    return "LONG"
def _d2s27_exit(p, i, d):
    r = p["rsi"].iloc[i]
    if pd.isna(r): return False
    if d == "LONG" and r > 60: return f"RSI > 60 ({r:.1f})"
    return False
_register("e27_ulcer", "Ulcer Index Reversion", _d2s27_pre, _d2s27_sig, _d2s27_exit, group="discover2")


# 28. Stochastic RSI crossover
def _d2s28_pre(df):
    rs = rsi(df["close"], 14)
    rmin = rs.rolling(14).min()
    rmax = rs.rolling(14).max()
    rng = (rmax - rmin).replace(0, np.nan)
    k_raw = 100 * (rs - rmin) / rng
    k = k_raw.rolling(3).mean()
    d = k.rolling(3).mean()
    return {"srsi_k": k, "srsi_d": d}
def _d2s28_sig(p, i):
    if i < 1: return None
    if _crosses(p["srsi_k"].iloc[i-1], p["srsi_k"].iloc[i],
                p["srsi_d"].iloc[i-1], p["srsi_d"].iloc[i], "up"):
        if pd.notna(p["srsi_k"].iloc[i]) and p["srsi_k"].iloc[i] < 20: return "LONG"
    if _crosses(p["srsi_k"].iloc[i-1], p["srsi_k"].iloc[i],
                p["srsi_d"].iloc[i-1], p["srsi_d"].iloc[i], "down"):
        if pd.notna(p["srsi_k"].iloc[i]) and p["srsi_k"].iloc[i] > 80: return "SHORT"
    return None
_register("e28_stoch_rsi", "Stochastic RSI Crossover", _d2s28_pre, _d2s28_sig, None, group="discover2")


# 29. Fisher Transform
def _d2s29_pre(df, period=10):
    hl2 = (df["high"] + df["low"]) / 2
    hh = hl2.rolling(period).max()
    ll = hl2.rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    raw = 2 * ((hl2 - ll) / rng - 0.5)
    raw = raw.clip(-0.999, 0.999).fillna(0)
    val = pd.Series(np.zeros(len(df)), index=df.index)
    fish = pd.Series(np.zeros(len(df)), index=df.index)
    rv = raw.values
    vv = val.values; fv = fish.values
    for i in range(1, len(df)):
        vv[i] = 0.66 * rv[i] + 0.67 * vv[i-1]
        vv[i] = max(min(vv[i], 0.999), -0.999)
        fv[i] = 0.5 * np.log((1 + vv[i]) / (1 - vv[i])) + 0.5 * fv[i-1]
    return {"fisher": pd.Series(fv, index=df.index)}
def _d2s29_sig(p, i):
    if i < 1: return None
    pp = p["fisher"].iloc[i-1]; pc = p["fisher"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e29_fisher", "Fisher Transform", _d2s29_pre, _d2s29_sig, None, group="discover2")


# 30. Schaff Trend Cycle (simplified: STC of MACD via double stoch)
def _d2s30_pre(df, fast=23, slow=50, cycle=10):
    macd_line = ema(df["close"], fast) - ema(df["close"], slow)
    mmin = macd_line.rolling(cycle).min()
    mmax = macd_line.rolling(cycle).max()
    rng1 = (mmax - mmin).replace(0, np.nan)
    pf = 100 * (macd_line - mmin) / rng1
    pf_smooth = pf.ewm(alpha=0.5, adjust=False).mean()
    pmin = pf_smooth.rolling(cycle).min()
    pmax = pf_smooth.rolling(cycle).max()
    rng2 = (pmax - pmin).replace(0, np.nan)
    stc_raw = 100 * (pf_smooth - pmin) / rng2
    stc = stc_raw.ewm(alpha=0.5, adjust=False).mean()
    return {"stc": stc}
def _d2s30_sig(p, i):
    if i < 1: return None
    pp = p["stc"].iloc[i-1]; pc = p["stc"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 25 and pc > 25: return "LONG"
    if pp >= 75 and pc < 75: return "SHORT"
    return None
_register("e30_stc", "Schaff Trend Cycle", _d2s30_pre, _d2s30_sig, None, group="discover2")


# 31. Ehlers Cybernetic cycle (approx via detrended sinewave)
def _d2s31_pre(df, period=10):
    detrended = df["close"] - ema(df["close"], period * 2)
    cycle = pd.Series(np.zeros(len(df)), index=df.index)
    cv = cycle.values
    dv = detrended.values
    for i in range(2, len(df)):
        if np.isnan(dv[i]) or np.isnan(dv[i-1]) or np.isnan(dv[i-2]): continue
        cv[i] = 0.5 * (dv[i] - dv[i-2]) + 0.5 * cv[i-1]
    return {"cycle": pd.Series(cv, index=df.index)}
def _d2s31_sig(p, i):
    if i < 1: return None
    pp = p["cycle"].iloc[i-1]; pc = p["cycle"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0 and pc > 0: return "LONG"
    if pp >= 0 and pc < 0: return "SHORT"
    return None
_register("e31_cybernetic", "Ehlers Cybernetic Cycle", _d2s31_pre, _d2s31_sig, None, group="discover2")


# 32. Zero Lag EMA crossover
def _zlema(s, period):
    lag = (period - 1) // 2
    return ema(2 * s - s.shift(lag), period)
def _d2s32_pre(df):
    return {"z9": _zlema(df["close"], 9), "z21": _zlema(df["close"], 21)}
def _d2s32_sig(p, i):
    if i < 1: return None
    if _crosses(p["z9"].iloc[i-1], p["z9"].iloc[i], p["z21"].iloc[i-1], p["z21"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["z9"].iloc[i-1], p["z9"].iloc[i], p["z21"].iloc[i-1], p["z21"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e32_zlema", "ZLEMA Crossover", _d2s32_pre, _d2s32_sig, None, group="discover2")


# 33. Hull MA crossover
def _hma(s, period):
    half = max(2, period // 2)
    sqp = max(2, int(np.sqrt(period)))
    return _wma(2 * _wma(s, half) - _wma(s, period), sqp)
def _d2s33_pre(df):
    return {"h9": _hma(df["close"], 9), "h21": _hma(df["close"], 21)}
def _d2s33_sig(p, i):
    if i < 1: return None
    if _crosses(p["h9"].iloc[i-1], p["h9"].iloc[i], p["h21"].iloc[i-1], p["h21"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["h9"].iloc[i-1], p["h9"].iloc[i], p["h21"].iloc[i-1], p["h21"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e33_hma", "Hull MA Crossover", _d2s33_pre, _d2s33_sig, None, group="discover2")


# 34. DEMA crossover
def _dema(s, period):
    e = ema(s, period); ee = ema(e, period)
    return 2 * e - ee
def _d2s34_pre(df):
    return {"d9": _dema(df["close"], 9), "d21": _dema(df["close"], 21)}
def _d2s34_sig(p, i):
    if i < 1: return None
    if _crosses(p["d9"].iloc[i-1], p["d9"].iloc[i], p["d21"].iloc[i-1], p["d21"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["d9"].iloc[i-1], p["d9"].iloc[i], p["d21"].iloc[i-1], p["d21"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e34_dema", "DEMA Crossover", _d2s34_pre, _d2s34_sig, None, group="discover2")


# 35. TEMA crossover
def _tema(s, period):
    e = ema(s, period); ee = ema(e, period); eee = ema(ee, period)
    return 3 * e - 3 * ee + eee
def _d2s35_pre(df):
    return {"t9": _tema(df["close"], 9), "t21": _tema(df["close"], 21)}
def _d2s35_sig(p, i):
    if i < 1: return None
    if _crosses(p["t9"].iloc[i-1], p["t9"].iloc[i], p["t21"].iloc[i-1], p["t21"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["t9"].iloc[i-1], p["t9"].iloc[i], p["t21"].iloc[i-1], p["t21"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e35_tema", "TEMA Crossover", _d2s35_pre, _d2s35_sig, None, group="discover2")


# 36. Laguerre RSI (gamma 0.5)
def _d2s36_pre(df, gamma=0.5):
    n = len(df)
    L0 = np.zeros(n); L1 = np.zeros(n); L2 = np.zeros(n); L3 = np.zeros(n)
    cv = df["close"].values
    lrsi = np.zeros(n)
    for i in range(1, n):
        L0[i] = (1 - gamma) * cv[i] + gamma * L0[i-1]
        L1[i] = -gamma * L0[i] + L0[i-1] + gamma * L1[i-1]
        L2[i] = -gamma * L1[i] + L1[i-1] + gamma * L2[i-1]
        L3[i] = -gamma * L2[i] + L2[i-1] + gamma * L3[i-1]
        cu = max(L0[i] - L1[i], 0) + max(L1[i] - L2[i], 0) + max(L2[i] - L3[i], 0)
        cd = max(L1[i] - L0[i], 0) + max(L2[i] - L1[i], 0) + max(L3[i] - L2[i], 0)
        lrsi[i] = cu / (cu + cd) if (cu + cd) > 0 else 0
    return {"lrsi": pd.Series(lrsi, index=df.index)}
def _d2s36_sig(p, i):
    if i < 1: return None
    pp = p["lrsi"].iloc[i-1]; pc = p["lrsi"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 0.2 and pc > 0.2: return "LONG"
    if pp >= 0.8 and pc < 0.8: return "SHORT"
    return None
_register("e36_laguerre_rsi", "Laguerre RSI", _d2s36_pre, _d2s36_sig, None, group="discover2")


# 37. Rainbow oscillator
def _d2s37_pre(df):
    smas = [df["close"].rolling(p).mean() for p in range(2, 11)]
    rainbow = sum(smas) / len(smas)
    return {"rainbow": rainbow, "close": df["close"]}
def _d2s37_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    rp = p["rainbow"].iloc[i-1]; rc = p["rainbow"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, rp, rc)): return None
    if cp <= rp and cc > rc: return "LONG"
    if cp >= rp and cc < rc: return "SHORT"
    return None
_register("e37_rainbow", "Rainbow Oscillator", _d2s37_pre, _d2s37_sig, None, group="discover2")


# 38. Gann HiLo activator
def _d2s38_pre(df):
    return {
        "sma_h": df["high"].rolling(3).mean(),
        "sma_l": df["low"].rolling(3).mean(),
        "close": df["close"],
    }
def _d2s38_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    sh = p["sma_h"].iloc[i]; sl = p["sma_l"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, sh, sl)): return None
    if cp <= sh and cc > sh: return "LONG"
    if cp >= sl and cc < sl: return "SHORT"
    return None
_register("e38_gann_hilo", "Gann HiLo Activator", _d2s38_pre, _d2s38_sig, None, group="discover2")


# 39. VIDYA (CMO-adapted EMA)
def _d2s39_pre(df, period=14):
    diff = df["close"].diff()
    up = diff.clip(lower=0).rolling(period).sum()
    dn = (-diff.clip(upper=0)).rolling(period).sum()
    cmo = ((up - dn) / (up + dn).replace(0, np.nan)).abs().fillna(0)
    n = len(df); vidya = np.zeros(n); cv = df["close"].values; ad = cmo.values
    alpha = 2 / (period + 1)
    vidya[0] = cv[0]
    for i in range(1, n):
        a = alpha * ad[i] if not np.isnan(ad[i]) else alpha * 0.5
        vidya[i] = a * cv[i] + (1 - a) * vidya[i-1]
    return {"vidya": pd.Series(vidya, index=df.index), "rsi": rsi(df["close"], 14)}
def _d2s39_sig(p, i):
    if i < 1: return None
    pp = p["vidya"].iloc[i-1]; pc = p["vidya"].iloc[i]
    r = p["rsi"].iloc[i]
    if pd.isna(pp) or pd.isna(pc) or pd.isna(r): return None
    if pc > pp and r < 55: return "LONG"
    if pc < pp and r > 45: return "SHORT"
    return None
_register("e39_vidya", "VIDYA", _d2s39_pre, _d2s39_sig, None, group="discover2")


# 40. FRAMA (fractal adaptive)
def _d2s40_pre(df, period=16):
    n = len(df); cv = df["close"].values
    half = period // 2
    frama = np.zeros(n); frama[:period] = cv[:period]
    hv = df["high"].values; lv = df["low"].values
    for i in range(period, n):
        n1 = (np.max(hv[i-period:i-half]) - np.min(lv[i-period:i-half])) / half
        n2 = (np.max(hv[i-half:i]) - np.min(lv[i-half:i])) / half
        n3 = (np.max(hv[i-period:i]) - np.min(lv[i-period:i])) / period
        if n1 > 0 and n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        else:
            d = 1.5
        alpha = max(0.01, min(1, np.exp(-4.6 * (d - 1))))
        frama[i] = alpha * cv[i] + (1 - alpha) * frama[i-1]
    return {"frama": pd.Series(frama, index=df.index), "close": df["close"]}
def _d2s40_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    fp = p["frama"].iloc[i-1]; fc = p["frama"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, fp, fc)): return None
    if cp <= fp and cc > fc: return "LONG"
    if cp >= fp and cc < fc: return "SHORT"
    return None
_register("e40_frama", "FRAMA", _d2s40_pre, _d2s40_sig, None, group="discover2")


# 41. McGinley Dynamic
def _d2s41_pre(df, period=14):
    n = len(df); md = np.zeros(n); cv = df["close"].values
    md[0] = cv[0]
    for i in range(1, n):
        if md[i-1] == 0:
            md[i] = cv[i]
        else:
            md[i] = md[i-1] + (cv[i] - md[i-1]) / (period * (cv[i] / md[i-1]) ** 4)
    return {"md": pd.Series(md, index=df.index), "close": df["close"]}
def _d2s41_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    mp = p["md"].iloc[i-1]; mc = p["md"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, mp, mc)): return None
    if cp <= mp and cc > mc: return "LONG"
    if cp >= mp and cc < mc: return "SHORT"
    return None
_register("e41_mcginley", "McGinley Dynamic", _d2s41_pre, _d2s41_sig, None, group="discover2")


# 42. Ehlers Instantaneous Trendline (approx via WMA + smoothing)
def _d2s42_pre(df, alpha=0.07):
    n = len(df); cv = df["close"].values
    it = np.zeros(n)
    it[0] = cv[0]; it[1] = cv[1] if n > 1 else cv[0]
    for i in range(2, n):
        it[i] = (alpha - alpha**2/4) * cv[i] + 0.5 * alpha**2 * cv[i-1] - (alpha - 0.75 * alpha**2) * cv[i-2] + 2 * (1 - alpha) * it[i-1] - (1 - alpha)**2 * it[i-2]
    return {"itrend": pd.Series(it, index=df.index), "close": df["close"]}
def _d2s42_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    tp = p["itrend"].iloc[i-1]; tc = p["itrend"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, tp, tc)): return None
    if cp <= tp and cc > tc: return "LONG"
    if cp >= tp and cc < tc: return "SHORT"
    return None
_register("e42_itrend", "Ehlers Instantaneous Trendline", _d2s42_pre, _d2s42_sig, None, group="discover2")


# 43. Sinewave (approx via cycle + 45-deg leading)
def _d2s43_pre(df, period=10):
    detr = df["close"] - df["close"].rolling(period).mean()
    n = len(df)
    sin_v = np.zeros(n); lead = np.zeros(n)
    dv = detr.values
    for i in range(period, n):
        if np.isnan(dv[i]): continue
        # Phase from arctan ratio of lagged values
        try:
            phase = np.arctan2(dv[i] - dv[i-period//4], dv[i] + 1e-9)
        except Exception:
            phase = 0
        sin_v[i] = np.sin(phase)
        lead[i] = np.sin(phase + np.pi / 4)
    return {"sine": pd.Series(sin_v, index=df.index),
            "lead": pd.Series(lead, index=df.index)}
def _d2s43_sig(p, i):
    if i < 1: return None
    if _crosses(p["sine"].iloc[i-1], p["sine"].iloc[i], p["lead"].iloc[i-1], p["lead"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["sine"].iloc[i-1], p["sine"].iloc[i], p["lead"].iloc[i-1], p["lead"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e43_sinewave", "Ehlers Sinewave", _d2s43_pre, _d2s43_sig, None, group="discover2")


# 44. Even Better Sinewave
def _d2s44_pre(df, period=20):
    detr = df["close"] - df["close"].rolling(period).mean()
    rms = np.sqrt((detr ** 2).rolling(period).mean()).replace(0, np.nan)
    ebs = (detr / rms).fillna(0).clip(-2, 2) / 2
    return {"ebs": ebs}
def _d2s44_sig(p, i):
    if i < 1: return None
    pp = p["ebs"].iloc[i-1]; pc = p["ebs"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= -0.9 and pc > -0.9: return "LONG"
    if pp >= 0.9 and pc < 0.9: return "SHORT"
    return None
_register("e44_ebs", "Even Better Sinewave", _d2s44_pre, _d2s44_sig, None, group="discover2")


# 45. Recursive Median Filter
def _d2s45_pre(df, period=5, alpha=0.3):
    raw_med = df["close"].rolling(period).median()
    n = len(df); rm = np.zeros(n); rv = raw_med.values; cv = df["close"].values
    rm[0] = cv[0]
    for i in range(1, n):
        if np.isnan(rv[i]):
            rm[i] = rm[i-1]
        else:
            rm[i] = alpha * rv[i] + (1 - alpha) * rm[i-1]
    return {"rmedian": pd.Series(rm, index=df.index), "close": df["close"]}
def _d2s45_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    rp = p["rmedian"].iloc[i-1]; rc = p["rmedian"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, rp, rc)): return None
    if cp <= rp and cc > rc: return "LONG"
    if cp >= rp and cc < rc: return "SHORT"
    return None
_register("e45_rec_median", "Recursive Median Filter", _d2s45_pre, _d2s45_sig, None, group="discover2")


# 46. Volatility breakout
def _d2s46_pre(df):
    a = atr(df["high"], df["low"], df["close"], 14)
    return {
        "atr": a,
        "trig_up": df["close"].shift(1) + 0.7 * a.shift(1),
        "trig_dn": df["close"].shift(1) - 0.7 * a.shift(1),
        "high": df["high"], "low": df["low"],
    }
def _d2s46_sig(p, i):
    h = p["high"].iloc[i]; l = p["low"].iloc[i]
    tu = p["trig_up"].iloc[i]; td = p["trig_dn"].iloc[i]
    if any(pd.isna(x) for x in (h, l, tu, td)): return None
    if h >= tu: return "LONG"
    if l <= td: return "SHORT"
    return None
_register("e46_vol_breakout", "Volatility Breakout", _d2s46_pre, _d2s46_sig, None, group="discover2")


# 47. Opening Range Breakout (24-bar windows on 1H)
def _d2s47_pre(df, period=24):
    group = pd.Series(np.arange(len(df)) // period, index=df.index)
    first_high = df.groupby(group)["high"].transform("first")
    first_low = df.groupby(group)["low"].transform("first")
    bar_in_window = pd.Series(np.arange(len(df)) % period, index=df.index)
    return {"fh": first_high, "fl": first_low, "bar": bar_in_window,
            "close": df["close"]}
def _d2s47_sig(p, i):
    if p["bar"].iloc[i] == 0: return None  # don't trigger on first bar
    c = p["close"].iloc[i]; fh = p["fh"].iloc[i]; fl = p["fl"].iloc[i]
    if any(pd.isna(x) for x in (c, fh, fl)): return None
    if c > fh: return "LONG"
    if c < fl: return "SHORT"
    return None
def _d2s47_exit(p, i, d):
    if p["bar"].iloc[i] == 23: return "End of UTC day"
    return False
_register("e47_orb", "Opening Range Breakout (24h)", _d2s47_pre, _d2s47_sig, _d2s47_exit, group="discover2")


# 48. Three bar reversal
def _d2s48_pre(df):
    return {"open": df["open"], "close": df["close"]}
def _d2s48_sig(p, i):
    if i < 3: return None
    o = p["open"]; c = p["close"]
    o0, c0 = o.iloc[i-3], c.iloc[i-3]
    o1, c1 = o.iloc[i-2], c.iloc[i-2]
    o2, c2 = o.iloc[i-1], c.iloc[i-1]
    o3, c3 = o.iloc[i], c.iloc[i]
    if any(pd.isna(x) for x in (o0, c0, o1, c1, o2, c2, o3, c3)): return None
    bear3 = c0 < o0 and c1 < o1 and c2 < o2
    bull3 = c0 > o0 and c1 > o1 and c2 > o2
    mid0 = (o0 + c0) / 2
    if bear3 and c3 > o3 and c3 > mid0: return "LONG"
    if bull3 and c3 < o3 and c3 < mid0: return "SHORT"
    return None
_register("e48_three_bar", "Three Bar Reversal", _d2s48_pre, _d2s48_sig, None, group="discover2")


# 49. Inside bar breakout
def _d2s49_pre(df):
    return {"high": df["high"], "low": df["low"], "close": df["close"]}
def _d2s49_sig(p, i):
    if i < 2: return None
    h = p["high"]; l = p["low"]; c = p["close"]
    h0, l0 = h.iloc[i-2], l.iloc[i-2]
    h1, l1 = h.iloc[i-1], l.iloc[i-1]
    cc = c.iloc[i]; ch = h.iloc[i]; cl = l.iloc[i]
    if any(pd.isna(x) for x in (h0, l0, h1, l1, cc, ch, cl)): return None
    inside = h1 <= h0 and l1 >= l0
    if not inside: return None
    if ch > h1 and cc > h1: return "LONG"
    if cl < l1 and cc < l1: return "SHORT"
    return None
_register("e49_inside_bar", "Inside Bar Breakout", _d2s49_pre, _d2s49_sig, None, group="discover2")


# 50. Pivot point reversion (daily, 24-bar window)
def _d2s50_pre(df, period=24):
    group = pd.Series(np.arange(len(df)) // period, index=df.index)
    prev_high = df.groupby(group)["high"].transform("max").shift(period)
    prev_low = df.groupby(group)["low"].transform("min").shift(period)
    prev_close = df["close"].shift(period)
    pivot = (prev_high + prev_low + prev_close) / 3
    s1 = 2 * pivot - prev_high
    r1 = 2 * pivot - prev_low
    return {"pivot": pivot, "s1": s1, "r1": r1, "rsi": rsi(df["close"], 14),
            "low": df["low"], "high": df["high"]}
def _d2s50_sig(p, i):
    lo = p["low"].iloc[i]; hi = p["high"].iloc[i]
    s1 = p["s1"].iloc[i]; r1 = p["r1"].iloc[i]
    r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (lo, hi, s1, r1, r)): return None
    if lo <= s1 and r < 40: return "LONG"
    if hi >= r1 and r > 60: return "SHORT"
    return None
_register("e50_pivot_rev", "Pivot Point Reversion", _d2s50_pre, _d2s50_sig, None, group="discover2")


# 51. Candlestick hammer
def _d2s51_pre(df):
    return {"o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"],
            "rsi": rsi(df["close"], 14)}
def _d2s51_sig(p, i):
    o = p["o"].iloc[i]; h = p["h"].iloc[i]; l = p["l"].iloc[i]; c = p["c"].iloc[i]
    r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (o, h, l, c, r)): return None
    body = abs(c - o); rng = h - l
    if rng <= 0: return None
    upper_wick = h - max(o, c); lower_wick = min(o, c) - l
    body_top = max(o, c)
    in_upper_third = (body_top - l) >= 2 * rng / 3
    if lower_wick >= 2 * body and in_upper_third and r < 50: return "LONG"
    if upper_wick >= 2 * body and (h - max(o, c)) >= 0 and (max(o,c) - l) <= rng / 3 and r > 50:
        return "SHORT"
    return None
_register("e51_hammer", "Hammer Candle", _d2s51_pre, _d2s51_sig, None, group="discover2")


# 52. Engulfing pattern
def _d2s52_pre(df):
    return {"o": df["open"], "c": df["close"], "v": df["volume"],
            "vsma": df["volume"].rolling(20).mean()}
def _d2s52_sig(p, i):
    if i < 1: return None
    o0 = p["o"].iloc[i-1]; c0 = p["c"].iloc[i-1]
    o1 = p["o"].iloc[i]; c1 = p["c"].iloc[i]
    v = p["v"].iloc[i]; vs = p["vsma"].iloc[i]
    if any(pd.isna(x) for x in (o0, c0, o1, c1, v, vs)): return None
    if v <= vs: return None
    if c0 < o0 and c1 > o1 and o1 <= c0 and c1 >= o0: return "LONG"
    if c0 > o0 and c1 < o1 and o1 >= c0 and c1 <= o0: return "SHORT"
    return None
_register("e52_engulfing", "Engulfing Pattern", _d2s52_pre, _d2s52_sig, None, group="discover2")


# 53. Morning star simplified
def _d2s53_pre(df):
    return {"o": df["open"], "c": df["close"]}
def _d2s53_sig(p, i):
    if i < 2: return None
    o = p["o"]; c = p["c"]
    o0, c0 = o.iloc[i-2], c.iloc[i-2]
    o1, c1 = o.iloc[i-1], c.iloc[i-1]
    o2, c2 = o.iloc[i], c.iloc[i]
    if any(pd.isna(x) for x in (o0, c0, o1, c1, o2, c2)): return None
    body0 = abs(c0 - o0); body1 = abs(c1 - o1); body2 = abs(c2 - o2)
    mid0 = (o0 + c0) / 2
    if body0 == 0: return None
    if c0 < o0 and body1 < body0 / 2 and c2 > o2 and c2 > mid0: return "LONG"
    if c0 > o0 and body1 < body0 / 2 and c2 < o2 and c2 < mid0: return "SHORT"
    return None
_register("e53_morning_star", "Morning/Evening Star", _d2s53_pre, _d2s53_sig, None, group="discover2")


# 54. Three white soldiers / black crows
def _d2s54_pre(df):
    return {"o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"]}
def _d2s54_sig(p, i):
    if i < 2: return None
    o = p["o"]; h = p["h"]; l = p["l"]; c = p["c"]
    bull = lambda k: c.iloc[k] > o.iloc[k] and (c.iloc[k] - l.iloc[k]) > 0.7 * (h.iloc[k] - l.iloc[k])
    bear = lambda k: c.iloc[k] < o.iloc[k] and (h.iloc[k] - c.iloc[k]) > 0.7 * (h.iloc[k] - l.iloc[k])
    inside = lambda k_curr, k_prev: o.iloc[k_curr] >= min(o.iloc[k_prev], c.iloc[k_prev]) and o.iloc[k_curr] <= max(o.iloc[k_prev], c.iloc[k_prev])
    try:
        if bull(i) and bull(i-1) and bull(i-2) and inside(i, i-1) and inside(i-1, i-2): return "LONG"
        if bear(i) and bear(i-1) and bear(i-2) and inside(i, i-1) and inside(i-1, i-2): return "SHORT"
    except Exception:
        return None
    return None
_register("e54_three_soldiers", "Three Soldiers/Crows", _d2s54_pre, _d2s54_sig, None, group="discover2")


# 55. Doji reversal
def _d2s55_pre(df):
    return {"o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"],
            "rsi": rsi(df["close"], 14)}
def _d2s55_sig(p, i):
    if i < 2: return None
    o = p["o"]; h = p["h"]; l = p["l"]; c = p["c"]
    o2, c2 = o.iloc[i], c.iloc[i]; h2, l2 = h.iloc[i], l.iloc[i]
    o0, c0 = o.iloc[i-2], c.iloc[i-2]
    o1, c1 = o.iloc[i-1], c.iloc[i-1]
    r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (o0, c0, o1, c1, o2, c2, h2, l2, r)): return None
    rng = h2 - l2
    if rng <= 0: return None
    body_pct = abs(c2 - o2) / rng
    if body_pct >= 0.1: return None
    if c0 < o0 and c1 < o1 and r < 45: return "LONG"
    if c0 > o0 and c1 > o1 and r > 55: return "SHORT"
    return None
_register("e55_doji_reversal", "Doji Reversal", _d2s55_pre, _d2s55_sig, None, group="discover2")


# 56. Tweezer bottom/top
def _d2s56_pre(df):
    return {"o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"]}
def _d2s56_sig(p, i):
    if i < 1: return None
    l0, l1 = p["l"].iloc[i-1], p["l"].iloc[i]
    h0, h1 = p["h"].iloc[i-1], p["h"].iloc[i]
    o0, c0 = p["o"].iloc[i-1], p["c"].iloc[i-1]
    o1, c1 = p["o"].iloc[i], p["c"].iloc[i]
    if any(pd.isna(x) for x in (l0, l1, h0, h1, o0, c0, o1, c1)): return None
    if l0 > 0 and abs(l0 - l1) / l0 < 0.001 and c0 < o0 and c1 > o1: return "LONG"
    if h0 > 0 and abs(h0 - h1) / h0 < 0.001 and c0 > o0 and c1 < o1: return "SHORT"
    return None
_register("e56_tweezer", "Tweezer Bottom/Top", _d2s56_pre, _d2s56_sig, None, group="discover2")


# 57. Harami pattern
def _d2s57_pre(df):
    return {"o": df["open"], "c": df["close"], "rsi": rsi(df["close"], 14)}
def _d2s57_sig(p, i):
    if i < 1: return None
    o0, c0 = p["o"].iloc[i-1], p["c"].iloc[i-1]
    o1, c1 = p["o"].iloc[i], p["c"].iloc[i]
    r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (o0, c0, o1, c1, r)): return None
    big_bear = c0 < o0 and (o0 - c0) > 0
    inside_bull = c1 > o1 and o1 >= c0 and c1 <= o0
    big_bull = c0 > o0 and (c0 - o0) > 0
    inside_bear = c1 < o1 and o1 <= c0 and c1 >= o0
    if big_bear and inside_bull and r < 50: return "LONG"
    if big_bull and inside_bear and r > 50: return "SHORT"
    return None
_register("e57_harami", "Harami Pattern", _d2s57_pre, _d2s57_sig, None, group="discover2")


# 58. Kicker pattern
def _d2s58_pre(df):
    return {"o": df["open"], "c": df["close"], "rsi": rsi(df["close"], 14)}
def _d2s58_sig(p, i):
    if i < 1: return None
    o0, c0 = p["o"].iloc[i-1], p["c"].iloc[i-1]
    o1, c1 = p["o"].iloc[i], p["c"].iloc[i]
    r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (o0, c0, o1, c1, r)): return None
    if c0 < o0 and c1 > o1 and o1 > o0 and r < 55: return "LONG"
    if c0 > o0 and c1 < o1 and o1 < o0 and r > 45: return "SHORT"
    return None
_register("e58_kicker", "Kicker Pattern", _d2s58_pre, _d2s58_sig, None, group="discover2")


# 59. OBV divergence
def _d2s59_pre(df):
    return {"obv": obv(df["close"], df["volume"]), "low": df["low"], "high": df["high"]}
def _d2s59_sig(p, i):
    if i < 10: return None
    obvc = p["obv"].iloc[i]; obvb = p["obv"].iloc[i-10]
    lc = p["low"].iloc[i]; lb = p["low"].iloc[i-10]
    hc = p["high"].iloc[i]; hb = p["high"].iloc[i-10]
    if any(pd.isna(x) for x in (obvc, obvb, lc, lb, hc, hb)): return None
    if lc < lb and obvc > obvb: return "LONG"
    if hc > hb and obvc < obvb: return "SHORT"
    return None
_register("e59_obv_div", "OBV Divergence", _d2s59_pre, _d2s59_sig, None, group="discover2")


# 60. Accumulation/Distribution divergence
def _d2s60_pre(df):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mfv = (mfm * df["volume"]).fillna(0)
    return {"ad": mfv.cumsum(), "close": df["close"]}
def _d2s60_sig(p, i):
    if i < 5: return None
    ac = p["ad"].iloc[i]; ab = p["ad"].iloc[i-5]
    cc = p["close"].iloc[i]; cb = p["close"].iloc[i-5]
    if any(pd.isna(x) for x in (ac, ab, cc, cb)): return None
    if ac > ab and cc < cb: return "LONG"
    if ac < ab and cc > cb: return "SHORT"
    return None
_register("e60_ad_div", "AD Divergence", _d2s60_pre, _d2s60_sig, None, group="discover2")


# 61. Money Flow Index reversion
def _d2s61_pre(df, period=14):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = typical * df["volume"]
    delta = typical.diff()
    pos_mf = pd.Series(np.where(delta > 0, raw_mf, 0), index=df.index)
    neg_mf = pd.Series(np.where(delta < 0, raw_mf, 0), index=df.index)
    mfr = pos_mf.rolling(period).sum() / neg_mf.rolling(period).sum().replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return {"mfi": mfi}
def _d2s61_sig(p, i):
    if i < 1: return None
    pp = p["mfi"].iloc[i-1]; pc = p["mfi"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= 20 and pc > 20: return "LONG"
    if pp >= 80 and pc < 80: return "SHORT"
    return None
_register("e61_mfi", "MFI Reversion", _d2s61_pre, _d2s61_sig, None, group="discover2")


# 62. CCI cross +/- 100
def _cci(df, period=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: float(np.mean(np.abs(x - x.mean()))) if not np.any(np.isnan(x)) else float("nan"), raw=False)
    return (tp - sma) / (0.015 * md.replace(0, np.nan))
def _d2s62_pre(df):
    return {"cci": _cci(df, 20)}
def _d2s62_sig(p, i):
    if i < 1: return None
    pp = p["cci"].iloc[i-1]; pc = p["cci"].iloc[i]
    if pd.isna(pp) or pd.isna(pc): return None
    if pp <= -100 and pc > -100: return "LONG"
    if pp >= 100 and pc < 100: return "SHORT"
    return None
_register("e62_cci", "CCI Cross +/-100", _d2s62_pre, _d2s62_sig, None, group="discover2")


# 63. CCI extreme reversion
def _d2s63_pre(df):
    return {"cci": _cci(df, 20)}
def _d2s63_sig(p, i):
    if i < 2: return None
    cur = p["cci"].iloc[i]; prev = p["cci"].iloc[i-1]; prev2 = p["cci"].iloc[i-2]
    if any(pd.isna(x) for x in (cur, prev, prev2)): return None
    if prev < -200 and cur > prev: return "LONG"
    if prev > 200 and cur < prev: return "SHORT"
    return None
_register("e63_cci_extreme", "CCI Extreme Reversion", _d2s63_pre, _d2s63_sig, None, group="discover2")


# 64. Woodies CCI (zero cross after 6+ bars below)
def _d2s64_pre(df):
    return {"cci": _cci(df, 20)}
def _d2s64_sig(p, i):
    if i < 7: return None
    cur = p["cci"].iloc[i]; prev = p["cci"].iloc[i-1]
    if pd.isna(cur) or pd.isna(prev): return None
    if prev <= 0 and cur > 0:
        prior = [p["cci"].iloc[i-k] for k in range(2, 8)]
        if all(pd.notna(v) and v < 0 for v in prior): return "LONG"
    if prev >= 0 and cur < 0:
        prior = [p["cci"].iloc[i-k] for k in range(2, 8)]
        if all(pd.notna(v) and v > 0 for v in prior): return "SHORT"
    return None
_register("e64_woodies_cci", "Woodies CCI", _d2s64_pre, _d2s64_sig, None, group="discover2")


# 65. DPO reversion (% of price)
def _d2s65_pre(df, period=20):
    shift = period // 2 + 1
    sma = df["close"].rolling(period).mean()
    dpo = df["close"] - sma.shift(shift)
    dpo_pct = 100 * dpo / df["close"]
    return {"dpo_pct": dpo_pct}
def _d2s65_sig(p, i):
    if i < 1: return None
    cur = p["dpo_pct"].iloc[i]; prev = p["dpo_pct"].iloc[i-1]
    if pd.isna(cur) or pd.isna(prev): return None
    if prev < -2 and cur > prev: return "LONG"
    if prev > 2 and cur < prev: return "SHORT"
    return None
_register("e65_dpo_rev", "DPO % Reversion", _d2s65_pre, _d2s65_sig, None, group="discover2")


# 66. Price - MA distance
def _d2s66_pre(df):
    sma = df["close"].rolling(50).mean()
    diff_pct = 100 * (df["close"] - sma) / sma.replace(0, np.nan)
    return {"diff_pct": diff_pct, "rsi": rsi(df["close"], 14)}
def _d2s66_sig(p, i):
    d = p["diff_pct"].iloc[i]; r = p["rsi"].iloc[i]
    if pd.isna(d) or pd.isna(r): return None
    if d < -3 and r < 40: return "LONG"
    if d > 3 and r > 60: return "SHORT"
    return None
_register("e66_price_ma_dist", "Price-MA Distance", _d2s66_pre, _d2s66_sig, None, group="discover2")


# 67. ROC10 zero cross + ROC3 confirmation
def _d2s67_pre(df):
    return {"roc10": _roc(df["close"], 10), "roc3": _roc(df["close"], 3)}
def _d2s67_sig(p, i):
    if i < 1: return None
    pp = p["roc10"].iloc[i-1]; pc = p["roc10"].iloc[i]
    r3 = p["roc3"].iloc[i]
    if any(pd.isna(x) for x in (pp, pc, r3)): return None
    if pp <= 0 and pc > 0 and r3 > 0: return "LONG"
    if pp >= 0 and pc < 0 and r3 < 0: return "SHORT"
    return None
_register("e67_roc_momentum", "ROC Momentum", _d2s67_pre, _d2s67_sig, None, group="discover2")


# 68. Triple ROC
def _d2s68_pre(df):
    return {"r5": _roc(df["close"], 5), "r10": _roc(df["close"], 10), "r20": _roc(df["close"], 20)}
def _d2s68_sig(p, i):
    if i < 1: return None
    r5c = p["r5"].iloc[i]; r10c = p["r10"].iloc[i]; r20c = p["r20"].iloc[i]
    r5p = p["r5"].iloc[i-1]; r10p = p["r10"].iloc[i-1]; r20p = p["r20"].iloc[i-1]
    if any(pd.isna(x) for x in (r5c, r10c, r20c, r5p, r10p, r20p)): return None
    if r5c > 0 and r10c > 0 and r20c > 0 and (r5p < 0 or r10p < 0 or r20p < 0): return "LONG"
    if r5c < 0 and r10c < 0 and r20c < 0 and (r5p > 0 or r10p > 0 or r20p > 0): return "SHORT"
    return None
_register("e68_triple_roc", "Triple ROC", _d2s68_pre, _d2s68_sig, None, group="discover2")


# 69. Momentum with signal
def _d2s69_pre(df, period=10):
    mom = df["close"] - df["close"].shift(period)
    return {"mom": mom, "sig": ema(mom, 3)}
def _d2s69_sig(p, i):
    if i < 1: return None
    if _crosses(p["mom"].iloc[i-1], p["mom"].iloc[i], p["sig"].iloc[i-1], p["sig"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["mom"].iloc[i-1], p["mom"].iloc[i], p["sig"].iloc[i-1], p["sig"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e69_momentum_sig", "Momentum + Signal", _d2s69_pre, _d2s69_sig, None, group="discover2")


# 70. Disparity Index
def _d2s70_pre(df, period=14):
    sma = df["close"].rolling(period).mean()
    return {"disp": 100 * (df["close"] - sma) / sma.replace(0, np.nan)}
def _d2s70_sig(p, i):
    if i < 1: return None
    cur = p["disp"].iloc[i]; prev = p["disp"].iloc[i-1]
    if pd.isna(cur) or pd.isna(prev): return None
    if cur < -3 and cur > prev: return "LONG"
    if cur > 3 and cur < prev: return "SHORT"
    return None
_register("e70_disparity", "Disparity Index", _d2s70_pre, _d2s70_sig, None, group="discover2")


# 71. Price acceleration
def _d2s71_pre(df):
    accel = df["close"].diff().diff()
    return {"accel": accel, "close": df["close"]}
def _d2s71_sig(p, i):
    if i < 2: return None
    pa = p["accel"].iloc[i-1]; ca = p["accel"].iloc[i]
    cp = p["close"].iloc[i-1]; cpp = p["close"].iloc[i-2]
    if any(pd.isna(x) for x in (pa, ca, cp, cpp)): return None
    falling_before = cp < cpp
    rising_before = cp > cpp
    if pa <= 0 and ca > 0 and falling_before: return "LONG"
    if pa >= 0 and ca < 0 and rising_before: return "SHORT"
    return None
_register("e71_accel", "Price Acceleration", _d2s71_pre, _d2s71_sig, None, group="discover2")


# 72. Bandwidth percentile
def _d2s72_pre(df, lookback=252):
    u, m, l = compute_bollinger_bands(df["close"], 20, 2.0)
    bw = (u - l) / m.replace(0, np.nan)
    pct = bw.rolling(lookback).rank(pct=True)
    return {"bw_pct": pct, "rsi": rsi(df["close"], 14)}
def _d2s72_sig(p, i):
    bw = p["bw_pct"].iloc[i]; r = p["rsi"].iloc[i]
    if pd.isna(bw) or pd.isna(r): return None
    if bw < 0.20 and r < 45: return "LONG"
    if bw < 0.20 and r > 55: return "SHORT"
    return None
_register("e72_bw_pct", "Bandwidth Percentile + RSI", _d2s72_pre, _d2s72_sig, None, group="discover2")


# 73. Volatility ratio
def _d2s73_pre(df):
    a = atr(df["high"], df["low"], df["close"], 14)
    return {"vr": a / a.rolling(20).mean().replace(0, np.nan)}
def _d2s73_sig(p, i):
    if i < 10: return None
    cur = p["vr"].iloc[i]
    if pd.isna(cur) or cur >= 0.7: return None
    spike = any(pd.notna(p["vr"].iloc[i-k]) and p["vr"].iloc[i-k] > 1.3 for k in range(1, 11))
    if not spike: return None
    return "LONG"
_register("e73_vol_ratio", "Volatility Ratio Reversion", _d2s73_pre, _d2s73_sig, None, group="discover2")


# 74. Historical volatility reversion
def _d2s74_pre(df, period=10):
    log_ret = np.log(df["close"] / df["close"].shift(1))
    hv = log_ret.rolling(period).std() * np.sqrt(8760) * 100  # annualize for hourly
    return {"hv": hv}
def _d2s74_sig(p, i):
    if i < 5: return None
    cur = p["hv"].iloc[i]
    if pd.isna(cur) or cur >= 30: return None
    spike = any(pd.notna(p["hv"].iloc[i-k]) and p["hv"].iloc[i-k] > 60 for k in range(1, 6))
    if not spike: return None
    return "LONG"
def _d2s74_exit(p, i, d):
    if i < 1: return False
    cur = p["hv"].iloc[i]; prev = p["hv"].iloc[i-1]
    if pd.isna(cur) or pd.isna(prev): return False
    if d == "LONG" and cur > prev: return f"HV rising ({cur:.1f})"
    return False
_register("e74_hv_rev", "Historical Volatility Reversion", _d2s74_pre, _d2s74_sig, _d2s74_exit, group="discover2")


# 75. Z-score mean reversion
def _d2s75_pre(df, period=20):
    sma = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std().replace(0, np.nan)
    return {"z": (df["close"] - sma) / sd}
def _d2s75_sig(p, i):
    if i < 1: return None
    cur = p["z"].iloc[i]; prev = p["z"].iloc[i-1]
    if pd.isna(cur) or pd.isna(prev): return None
    if prev < -2 and cur > prev: return "LONG"
    if prev > 2 and cur < prev: return "SHORT"
    return None
_register("e75_zscore", "Z-Score Mean Reversion", _d2s75_pre, _d2s75_sig, None, group="discover2")


# 76. Percentile rank reversion
def _d2s76_pre(df, period=100):
    return {"pct": df["close"].rolling(period).rank(pct=True) * 100}
def _d2s76_sig(p, i):
    cur = p["pct"].iloc[i]
    if pd.isna(cur): return None
    if cur < 10: return "LONG"
    if cur > 90: return "SHORT"
    return None
def _d2s76_exit(p, i, d):
    cur = p["pct"].iloc[i]
    if pd.isna(cur): return False
    if d == "LONG" and cur >= 50: return f"Percentile {cur:.0f}"
    if d == "SHORT" and cur <= 50: return f"Percentile {cur:.0f}"
    return False
_register("e76_percentile", "Percentile Rank Reversion", _d2s76_pre, _d2s76_sig, _d2s76_exit, group="discover2")


# 77. Hurst exponent filter (variance method, simplified)
def _hurst(x):
    n = len(x)
    if n < 20 or np.any(np.isnan(x)): return float("nan")
    try:
        lags = [2, 4, 8, 16]
        tau = []
        for lag in lags:
            diffs = np.subtract(x[lag:], x[:-lag])
            sigma = np.std(diffs)
            if sigma <= 0: return float("nan")
            tau.append(np.log(sigma))
        slope = np.polyfit(np.log(lags), tau, 1)[0]
        return float(slope)
    except Exception:
        return float("nan")
def _d2s77_pre(df):
    h = df["close"].rolling(100).apply(_hurst, raw=True)
    return {"h": h, "rsi": rsi(df["close"], 14)}
def _d2s77_sig(p, i):
    h = p["h"].iloc[i]; r = p["rsi"].iloc[i]
    if pd.isna(h) or pd.isna(r): return None
    if h < 0.45 and r < 35: return "LONG"
    if h < 0.45 and r > 65: return "SHORT"
    return None
_register("e77_hurst", "Hurst + RSI", _d2s77_pre, _d2s77_sig, None, group="discover2")


# 78. Autocorrelation reversal
def _d2s78_pre(df, period=20):
    ret = df["close"].pct_change()
    ac = ret.rolling(period).apply(
        lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1]) if not np.any(np.isnan(x)) else float("nan"),
        raw=True,
    )
    return {"ac": ac, "ret": ret}
def _d2s78_sig(p, i):
    ac = p["ac"].iloc[i]; r = p["ret"].iloc[i]
    if pd.isna(ac) or pd.isna(r): return None
    if ac < -0.3 and r < 0: return "LONG"
    if ac < -0.3 and r > 0: return "SHORT"
    return None
_register("e78_autocorr", "Autocorrelation Reversal", _d2s78_pre, _d2s78_sig, None, group="discover2")


# 79. Regime filter (ADX > 25 EMA cross, else RSI reversion)
def _d2s79_adx(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    up = high - high.shift(1); dn = low.shift(1) - low
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index)
    a = 1 / period
    atr_w = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr_w.replace(0, np.nan)
    ndi = 100 * ndm.ewm(alpha=a, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()
def _d2s79_pre(df):
    return {
        "adx": _d2s79_adx(df["high"], df["low"], df["close"], 50),
        "ema9": ema(df["close"], 9), "ema21": ema(df["close"], 21),
        "rsi": rsi(df["close"], 14),
    }
def _d2s79_sig(p, i):
    if i < 1: return None
    a = p["adx"].iloc[i]
    if pd.isna(a): return None
    if a > 25:
        e9p = p["ema9"].iloc[i-1]; e9c = p["ema9"].iloc[i]
        e21p = p["ema21"].iloc[i-1]; e21c = p["ema21"].iloc[i]
        if any(pd.isna(x) for x in (e9p, e9c, e21p, e21c)): return None
        if e9p <= e21p and e9c > e21c: return "LONG"
        if e9p >= e21p and e9c < e21c: return "SHORT"
    else:
        rp = p["rsi"].iloc[i-1]; rc = p["rsi"].iloc[i]
        if pd.isna(rp) or pd.isna(rc): return None
        if rp < 30 and rc >= 30: return "LONG"
        if rp > 70 and rc <= 70: return "SHORT"
    return None
_register("e79_regime", "Regime Filter", _d2s79_pre, _d2s79_sig, None, group="discover2")


# 80. Kalman filter crossover
def _d2s80_pre(df, q=0.01, r=1.0):
    n = len(df); cv = df["close"].values
    x = np.zeros(n); P = np.zeros(n)
    x[0] = cv[0]; P[0] = 1.0
    for i in range(1, n):
        Pp = P[i-1] + q
        K = Pp / (Pp + r)
        x[i] = x[i-1] + K * (cv[i] - x[i-1])
        P[i] = (1 - K) * Pp
    return {"k": pd.Series(x, index=df.index), "close": df["close"]}
def _d2s80_sig(p, i):
    if i < 1: return None
    cp = p["close"].iloc[i-1]; cc = p["close"].iloc[i]
    kp = p["k"].iloc[i-1]; kc = p["k"].iloc[i]
    if any(pd.isna(x) for x in (cp, cc, kp, kc)): return None
    if cp <= kp and cc > kc: return "LONG"
    if cp >= kp and cc < kc: return "SHORT"
    return None
_register("e80_kalman", "Kalman Filter Crossover", _d2s80_pre, _d2s80_sig, None, group="discover2")


# 81. LSMA crossover (linreg endpoint)
def _lsma(s, period):
    def _ep(x):
        if np.any(np.isnan(x)): return float("nan")
        slope, intercept = np.polyfit(np.arange(len(x)), x, 1)
        return float(slope * (len(x) - 1) + intercept)
    return s.rolling(period).apply(_ep, raw=True)
def _d2s81_pre(df):
    return {"l9": _lsma(df["close"], 9), "l21": _lsma(df["close"], 21)}
def _d2s81_sig(p, i):
    if i < 1: return None
    if _crosses(p["l9"].iloc[i-1], p["l9"].iloc[i], p["l21"].iloc[i-1], p["l21"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["l9"].iloc[i-1], p["l9"].iloc[i], p["l21"].iloc[i-1], p["l21"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e81_lsma", "LSMA Crossover", _d2s81_pre, _d2s81_sig, None, group="discover2")


# 82. Weighted close momentum
def _d2s82_pre(df):
    wc = (df["high"] + df["low"] + 2 * df["close"]) / 4
    return {"s5": wc.rolling(5).mean(), "s20": wc.rolling(20).mean()}
def _d2s82_sig(p, i):
    if i < 1: return None
    if _crosses(p["s5"].iloc[i-1], p["s5"].iloc[i], p["s20"].iloc[i-1], p["s20"].iloc[i], "up"):
        return "LONG"
    if _crosses(p["s5"].iloc[i-1], p["s5"].iloc[i], p["s20"].iloc[i-1], p["s20"].iloc[i], "down"):
        return "SHORT"
    return None
_register("e82_weighted_close", "Weighted Close Momentum", _d2s82_pre, _d2s82_sig, None, group="discover2")


# 83. Volume-weighted RSI
def _d2s83_pre(df, period=14):
    delta = df["close"].diff() * df["volume"]
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return {"vwrsi": (100 - 100 / (1 + rs)).fillna(50)}
def _d2s83_sig(p, i):
    cur = p["vwrsi"].iloc[i]
    if pd.isna(cur): return None
    if cur < 30: return "LONG"
    if cur > 70: return "SHORT"
    return None
_register("e83_vw_rsi", "Volume-Weighted RSI", _d2s83_pre, _d2s83_sig, None, group="discover2")


# 84. Tick volume pressure
def _d2s84_pre(df, period=10):
    up = (df["close"] > df["close"].shift(1)).astype(float)
    return {"ratio": up.rolling(period).mean()}
def _d2s84_sig(p, i):
    cur = p["ratio"].iloc[i]
    if pd.isna(cur): return None
    if cur < 0.3: return "LONG"
    if cur > 0.7: return "SHORT"
    return None
_register("e84_tick_pressure", "Tick Volume Pressure", _d2s84_pre, _d2s84_sig, None, group="discover2")


# 85. Price density
def _d2s85_pre(df, period=20):
    a = atr(df["high"], df["low"], df["close"], 14)
    closes = df["close"]
    def _density(window, atr_v, cur):
        if pd.isna(atr_v) or atr_v <= 0: return float("nan")
        return float(np.sum(np.abs(window - cur) <= atr_v))
    density = pd.Series(np.full(len(df), np.nan), index=df.index)
    cv = closes.values; av = a.values
    for i in range(period, len(df)):
        if pd.isna(av[i]): continue
        density.iloc[i] = float(np.sum(np.abs(cv[i-period:i] - cv[i]) <= av[i]))
    hh = df["high"].rolling(period).max().shift(1)
    ll = df["low"].rolling(period).min().shift(1)
    return {"density": density, "hh": hh, "ll": ll, "close": df["close"]}
def _d2s85_sig(p, i):
    den = p["density"].iloc[i]; c = p["close"].iloc[i]
    hh = p["hh"].iloc[i]; ll = p["ll"].iloc[i]
    if any(pd.isna(x) for x in (den, c, hh, ll)): return None
    if den < 12: return None
    if c > hh: return "LONG"
    if c < ll: return "SHORT"
    return None
_register("e85_density", "Price Density Breakout", _d2s85_pre, _d2s85_sig, None, group="discover2")


# 86. Gap and go
def _d2s86_pre(df):
    return {"o": df["open"], "c": df["close"], "v": df["volume"],
            "vsma": df["volume"].rolling(20).mean()}
def _d2s86_sig(p, i):
    if i < 1: return None
    o = p["o"].iloc[i]; c = p["c"].iloc[i]
    cp = p["c"].iloc[i-1]; v = p["v"].iloc[i]; vs = p["vsma"].iloc[i]
    if any(pd.isna(x) for x in (o, c, cp, v, vs)): return None
    if cp <= 0 or vs <= 0: return None
    gap = (o - cp) / cp
    if gap > 0.005 and c > o and v > vs: return "LONG"
    if gap < -0.005 and c < o and v > vs: return "SHORT"
    return None
_register("e86_gap_go", "Gap and Go", _d2s86_pre, _d2s86_sig, None, group="discover2")


# 87. Mean reversion after spike
def _d2s87_pre(df):
    a = atr(df["high"], df["low"], df["close"], 14)
    rng = df["high"] - df["low"]
    return {"o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"],
            "atr_avg": a, "rng": rng}
def _d2s87_sig(p, i):
    if i < 1: return None
    a = p["atr_avg"].iloc[i]; rp = p["rng"].iloc[i-1]
    if pd.isna(a) or pd.isna(rp) or a <= 0: return None
    if rp < 3 * a: return None
    cp = p["c"].iloc[i-1]; op = p["o"].iloc[i-1]
    cc = p["c"].iloc[i]; oc = p["o"].iloc[i]
    if any(pd.isna(x) for x in (cp, op, cc, oc)): return None
    spike_up = cp > op  # bullish spike
    spike_dn = cp < op  # bearish spike
    if spike_up:
        midspike = (op + cp) / 2
        retrace = (cp - cc) / (cp - op + 1e-9) if cp > op else 0
        if retrace > 0.5: return "SHORT"
    if spike_dn:
        midspike = (op + cp) / 2
        retrace = (cc - cp) / (op - cp + 1e-9) if op > cp else 0
        if retrace > 0.5: return "LONG"
    return None
_register("e87_spike_rev", "Spike Mean Reversion", _d2s87_pre, _d2s87_sig, None, group="discover2")


# 88. Consecutive closes
def _d2s88_pre(df):
    return {"o": df["open"], "c": df["close"], "rsi": rsi(df["close"], 14)}
def _d2s88_sig(p, i):
    if i < 4: return None
    c = p["c"]; o = p["o"]
    bears = all(c.iloc[i-k] < c.iloc[i-k-1] for k in range(1, 5))
    bulls = all(c.iloc[i-k] > c.iloc[i-k-1] for k in range(1, 5))
    r = p["rsi"].iloc[i]
    if pd.isna(r): return None
    cur_bull = c.iloc[i] > o.iloc[i]
    cur_bear = c.iloc[i] < o.iloc[i]
    if bears and cur_bull and r < 45: return "LONG"
    if bulls and cur_bear and r > 55: return "SHORT"
    return None
_register("e88_consec_closes", "Consecutive Closes Reversal", _d2s88_pre, _d2s88_sig, None, group="discover2")


# 89. HL channel position
def _d2s89_pre(df, period=20):
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    rng = (hh - ll).replace(0, np.nan)
    return {"pos": 100 * (df["close"] - ll) / rng}
def _d2s89_sig(p, i):
    cur = p["pos"].iloc[i]
    if pd.isna(cur): return None
    if cur < 10: return "LONG"
    if cur > 90: return "SHORT"
    return None
def _d2s89_exit(p, i, d):
    cur = p["pos"].iloc[i]
    if pd.isna(cur): return False
    if d == "LONG" and cur >= 50: return f"HL pos {cur:.0f}"
    if d == "SHORT" and cur <= 50: return f"HL pos {cur:.0f}"
    return False
_register("e89_hl_pos", "HL Channel Position", _d2s89_pre, _d2s89_sig, _d2s89_exit, group="discover2")


# 90. Relative strength pairs (per-pair simplification: ROC vs its rolling mean ROC)
def _d2s90_pre(df):
    roc = _roc(df["close"], 10)
    return {"roc": roc, "roc_avg": roc.rolling(50).mean(), "rsi": rsi(df["close"], 14)}
def _d2s90_sig(p, i):
    rc = p["roc"].iloc[i]; ra = p["roc_avg"].iloc[i]; r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (rc, ra, r)): return None
    if rc < ra - 5 and r < 40: return "LONG"
    if rc > ra + 5 and r > 60: return "SHORT"
    return None
_register("e90_rel_strength", "Relative Strength Reversion", _d2s90_pre, _d2s90_sig, None, group="discover2")


# 91. Volume profile reversion (POC = price level with most volume in 50 bars)
def _d2s91_pre(df, period=50):
    closes = df["close"].values; vols = df["volume"].values
    vah = np.full(len(df), np.nan)
    val = np.full(len(df), np.nan)
    for i in range(period, len(df)):
        c_window = closes[i-period:i]; v_window = vols[i-period:i]
        if np.any(np.isnan(c_window)) or np.any(np.isnan(v_window)): continue
        try:
            bins = np.linspace(c_window.min(), c_window.max(), 10)
            idx = np.digitize(c_window, bins) - 1
            idx = np.clip(idx, 0, 9)
            bin_vol = np.zeros(10)
            for k in range(len(c_window)):
                bin_vol[idx[k]] += v_window[k]
            poc_bin = int(np.argmax(bin_vol))
            poc_price = (bins[poc_bin] + bins[min(poc_bin + 1, 9)]) / 2
            vah[i] = poc_price * 1.005
            val[i] = poc_price * 0.995
        except Exception:
            continue
    return {"vah": pd.Series(vah, index=df.index), "val": pd.Series(val, index=df.index),
            "close": df["close"]}
def _d2s91_sig(p, i):
    c = p["close"].iloc[i]; vah = p["vah"].iloc[i]; val = p["val"].iloc[i]
    if any(pd.isna(x) for x in (c, vah, val)): return None
    if c < val * 0.98: return "LONG"
    if c > vah * 1.02: return "SHORT"
    return None
_register("e91_vol_profile", "Volume Profile Reversion", _d2s91_pre, _d2s91_sig, None, group="discover2")


# 92. Candle body momentum
def _d2s92_pre(df):
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return {"body_ratio": (body / rng).rolling(5).mean(),
            "o": df["open"], "c": df["close"]}
def _d2s92_sig(p, i):
    if i < 2: return None
    br = p["body_ratio"].iloc[i]
    o = p["o"]; c = p["c"]
    if pd.isna(br) or br < 0.6: return None
    if all(c.iloc[i-k] > o.iloc[i-k] for k in range(0, 3)): return "LONG"
    if all(c.iloc[i-k] < o.iloc[i-k] for k in range(0, 3)): return "SHORT"
    return None
_register("e92_body_momentum", "Body Momentum", _d2s92_pre, _d2s92_sig, None, group="discover2")


# 93. Wick rejection
def _d2s93_pre(df):
    return {"o": df["open"], "h": df["high"], "l": df["low"], "c": df["close"],
            "vsma": df["volume"].rolling(20).mean(), "v": df["volume"]}
def _d2s93_sig(p, i):
    o = p["o"].iloc[i]; h = p["h"].iloc[i]; l = p["l"].iloc[i]; c = p["c"].iloc[i]
    v = p["v"].iloc[i]; vs = p["vsma"].iloc[i]
    if any(pd.isna(x) for x in (o, h, l, c, v, vs)): return None
    if vs <= 0 or v <= vs: return None
    upper = h - max(o, c); lower = min(o, c) - l
    if lower > 3 * upper and c > o: return "LONG"
    if upper > 3 * lower and c < o: return "SHORT"
    return None
_register("e93_wick_rejection", "Wick Rejection", _d2s93_pre, _d2s93_sig, None, group="discover2")


# 94. Support/Resistance bounce
def _d2s94_pre(df, period=50):
    closes = df["close"]
    sup = pd.Series(np.full(len(df), np.nan), index=df.index)
    res = pd.Series(np.full(len(df), np.nan), index=df.index)
    lv = df["low"].values; hv = df["high"].values
    for i in range(period, len(df)):
        l_w = lv[i-period:i]; h_w = hv[i-period:i]
        if np.any(np.isnan(l_w)): continue
        sorted_lows = np.sort(l_w)[:3]
        sorted_highs = np.sort(h_w)[-3:]
        sup.iloc[i] = float(np.mean(sorted_lows))
        res.iloc[i] = float(np.mean(sorted_highs))
    return {"sup": sup, "res": res, "close": closes, "rsi": rsi(closes, 14)}
def _d2s94_sig(p, i):
    c = p["close"].iloc[i]; s = p["sup"].iloc[i]; rr = p["res"].iloc[i]
    r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (c, s, rr, r)): return None
    if abs(c - s) / s < 0.003 and r < 50: return "LONG"
    if abs(c - rr) / rr < 0.003 and r > 50: return "SHORT"
    return None
_register("e94_sr_bounce", "Support/Resistance Bounce", _d2s94_pre, _d2s94_sig, None, group="discover2")


# 95. Volatility contraction expansion
def _d2s95_pre(df):
    a5 = atr(df["high"], df["low"], df["close"], 5)
    a20 = atr(df["high"], df["low"], df["close"], 20)
    return {"ratio": a5 / a20.replace(0, np.nan), "o": df["open"], "c": df["close"]}
def _d2s95_sig(p, i):
    if i < 1: return None
    ratio_prev = p["ratio"].iloc[i-1]
    if pd.isna(ratio_prev) or ratio_prev >= 0.7: return None
    o = p["o"].iloc[i]; c = p["c"].iloc[i]
    if pd.isna(o) or pd.isna(c): return None
    if c > o: return "LONG"
    if c < o: return "SHORT"
    return None
_register("e95_vol_ce", "Volatility Contraction/Expansion", _d2s95_pre, _d2s95_sig, None, group="discover2")


# 96. Return distribution skew
def _d2s96_pre(df, period=20):
    ret = df["close"].pct_change()
    skew = ret.rolling(period).skew()
    return {"skew": skew}
def _d2s96_sig(p, i):
    cur = p["skew"].iloc[i]
    if pd.isna(cur): return None
    if cur < -1: return "LONG"
    if cur > 1: return "SHORT"
    return None
_register("e96_skew", "Return Skew Reversion", _d2s96_pre, _d2s96_sig, None, group="discover2")


# 97. Entropy filter (proxy: rolling normalized std of returns)
def _d2s97_pre(df, period=20):
    ret = df["close"].pct_change()
    entropy_proxy = ret.rolling(period).std() * np.sqrt(period)
    return {"entropy": entropy_proxy, "rsi": rsi(df["close"], 14)}
def _d2s97_sig(p, i):
    if i < 1: return None
    pp = p["entropy"].iloc[i-1]; pc = p["entropy"].iloc[i]
    r = p["rsi"].iloc[i]
    if any(pd.isna(x) for x in (pp, pc, r)): return None
    rising = pc > pp
    if rising and r < 45: return "LONG"
    if rising and r > 55: return "SHORT"
    return None
_register("e97_entropy", "Entropy Filter", _d2s97_pre, _d2s97_sig, None, group="discover2")


# 98. Cross-asset momentum (use same pair's ROC3 as proxy; cannot read other pairs in per-pair backtest)
def _d2s98_pre(df):
    return {"roc3": _roc(df["close"], 3), "rsi": rsi(df["close"], 14)}
def _d2s98_sig(p, i):
    r3 = p["roc3"].iloc[i]; r = p["rsi"].iloc[i]
    if pd.isna(r3) or pd.isna(r): return None
    if r3 > 0 and r < 35: return "LONG"
    if r3 < 0 and r > 65: return "SHORT"
    return None
_register("e98_cross_momentum", "Cross-Asset Momentum (proxy)", _d2s98_pre, _d2s98_sig, None, group="discover2")


# 99. Combination mean reversion
def _d2s99_pre(df):
    u, m, l = compute_bollinger_bands(df["close"], 20, 2.0)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    rmf = typical * df["volume"]
    delta = typical.diff()
    pos_mf = pd.Series(np.where(delta > 0, rmf, 0), index=df.index)
    neg_mf = pd.Series(np.where(delta < 0, rmf, 0), index=df.index)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return {
        "rsi": rsi(df["close"], 14), "bb_lower": l, "bb_upper": u,
        "mfi": mfi, "cci": _cci(df, 20),
        "wr": compute_williams_r(df["high"], df["low"], df["close"], 14),
        "close": df["close"],
    }
def _d2s99_sig(p, i):
    r = p["rsi"].iloc[i]; bl = p["bb_lower"].iloc[i]; bu = p["bb_upper"].iloc[i]
    mfi = p["mfi"].iloc[i]; cci = p["cci"].iloc[i]; wr = p["wr"].iloc[i]
    c = p["close"].iloc[i]
    if any(pd.isna(x) for x in (r, bl, bu, mfi, cci, wr, c)): return None
    long_conds = sum([r < 35, c < bl, mfi < 25, cci < -100, wr < -80])
    short_conds = sum([r > 65, c > bu, mfi > 75, cci > 100, wr > -20])
    if long_conds >= 3: return "LONG"
    if short_conds >= 3: return "SHORT"
    return None
_register("e99_combo_meanrev", "Combination Mean Reversion", _d2s99_pre, _d2s99_sig, None, group="discover2")


# 100. Adaptive regime (Hurst switch between Combo and Supertrend)
def _d2s100_pre(df):
    return {
        "hurst": df["close"].rolling(100).apply(_hurst, raw=True),
        "combo": _d2s99_pre(df),
        "st": dict(zip(("st_line", "st_dir"),
                       compute_supertrend(df["high"], df["low"], df["close"], 10, 3.0))),
        "close": df["close"],
    }
def _d2s100_sig(p, i):
    h = p["hurst"].iloc[i]
    if pd.isna(h): return None
    if h < 0.5:
        return _d2s99_sig(p["combo"], i)
    else:
        if i < 1: return None
        dp = p["st"]["st_dir"].iloc[i-1]; dc = p["st"]["st_dir"].iloc[i]
        if pd.isna(dp) or pd.isna(dc): return None
        if dp == -1 and dc == 1: return "LONG"
        if dp == 1 and dc == -1: return "SHORT"
        return None
_register("e100_adaptive_regime", "Adaptive Regime (Hurst switch)", _d2s100_pre, _d2s100_sig, None, group="discover2")


def handle_discover2_command(reply_to_message_id: int | None = None) -> None:
    global discover_running
    with discover_lock:
        if discover_running:
            send_telegram("Discovery already in progress.",
                          reply_to_message_id=reply_to_message_id)
            return
        discover_running = True
    n_strats = len([s for s in STRATEGIES if s.group == "discover2"])
    send_telegram(
        f"<b>Strategy discovery 2 started</b>\n"
        f"Testing {n_strats} unconventional strategies on 6mo of 1H data across 20 pairs. "
        "Pass = opt WR >= 60% AND val WR >= 55% AND >= 100 trades on opt window. "
        "This will take quite a while.",
        reply_to_message_id=reply_to_message_id,
    )
    threading.Thread(
        target=_run_discover, args=(reply_to_message_id, "discover2"), daemon=True
    ).start()


# ===== Strategy persistence and discovery loop =====

STRATEGY_PATH = os.environ.get(
    "STRATEGY_PATH",
    "/data/strategy.json" if os.path.isdir("/data") else "strategy.json",
)
active_strategy_id: str | None = None
discover_running = False
discover_lock = threading.Lock()


def load_strategy() -> None:
    global active_strategy_id
    if not os.path.exists(STRATEGY_PATH):
        log.info("No saved strategy at %s; using default", STRATEGY_PATH)
        return
    try:
        with open(STRATEGY_PATH) as f:
            data = json.load(f)
        sid = data.get("id")
        if sid in STRATEGIES_BY_ID:
            active_strategy_id = sid
            log.info("Active strategy: %s (%s)", sid, STRATEGIES_BY_ID[sid].name)
        else:
            log.warning("Saved strategy %s not in registry; ignoring", sid)
    except Exception as e:
        log.warning("Failed to load strategy: %s", e)


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


def backtest_strategy(strat: _Strategy, df: pd.DataFrame) -> list[dict]:
    p = strat.precompute(df)
    n = len(df["close"])
    trades: list[dict] = []
    open_pos: dict | None = None
    sl_mult = strat.sl_mult
    tp_mult = strat.tp_mult
    close = df["close"]
    high_arr = df["high"]
    low_arr = df["low"]
    start = 50
    for i in range(start, n):
        c = float(close.iloc[i])
        h = float(high_arr.iloc[i])
        l = float(low_arr.iloc[i])
        atr_v = p["atr"].iloc[i] if "atr" in p else float("nan")
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        atr_v = float(atr_v)

        if open_pos is not None:
            d = open_pos["direction"]
            entry = open_pos["entry"]
            tp = open_pos["tp"]
            sl = open_pos["sl"]
            risk = open_pos["risk"]
            if d == "LONG":
                sl_hit = l <= sl
                tp_hit = h >= tp
            else:
                sl_hit = h >= sl
                tp_hit = l <= tp
            if sl_hit:
                trades.append({"direction": d, "entry": entry, "exit": sl,
                               "rr": -1.0, "result": "loss"})
                open_pos = None
                continue
            exits = strat.exit_reasons_at(p, i, d)
            if exits:
                if d == "LONG":
                    rr = (c - entry) / risk
                else:
                    rr = (entry - c) / risk
                trades.append({"direction": d, "entry": entry, "exit": c,
                               "rr": rr, "result": "win" if rr > 0 else "loss"})
                open_pos = None
                continue
            if tp_hit:
                trades.append({"direction": d, "entry": entry, "exit": tp,
                               "rr": tp_mult / sl_mult, "result": "win"})
                open_pos = None
                continue
            continue

        sig = strat.entry_signal_at(p, i)
        if sig is None:
            continue
        risk = sl_mult * atr_v
        if sig == "LONG":
            tp_v = c + tp_mult * atr_v
            sl_v = c - sl_mult * atr_v
        else:
            tp_v = c - tp_mult * atr_v
            sl_v = c + sl_mult * atr_v
        open_pos = {"direction": sig, "entry": c, "tp": tp_v,
                    "sl": sl_v, "risk": risk}
    return trades


def handle_discover_command(reply_to_message_id: int | None = None) -> None:
    global discover_running
    with discover_lock:
        if discover_running:
            send_telegram("Discovery already in progress.",
                          reply_to_message_id=reply_to_message_id)
            return
        discover_running = True
    send_telegram(
        "<b>Strategy discovery started</b>\n"
        "Testing 10 well-known strategies on 6 months of 1H data across 20 pairs. "
        "Pass = opt WR >= 60% AND val WR >= 55% AND >= 100 trades on opt window.",
        reply_to_message_id=reply_to_message_id,
    )
    threading.Thread(
        target=_run_discover, args=(reply_to_message_id, "discover"), daemon=True
    ).start()


def _run_discover(reply_to_message_id: int | None, group: str) -> None:
    global active_strategy_id, discover_running
    try:
        strategies = [s for s in STRATEGIES if s.group == group]
        if not strategies:
            send_telegram(f"No strategies registered for {group}.",
                          reply_to_message_id=reply_to_message_id)
            return

        log.info("%s: fetching 6mo data for %d pairs", group, len(PAIRS))
        all_data = {}
        for symbol in PAIRS:
            try:
                df = fetch_klines_paginated(symbol, "1h", target=4320)
                if len(df) < 400:
                    continue
                all_data[symbol] = df
            except Exception as e:
                log.exception("%s %s fetch failed: %s", group, symbol, e)

        if not all_data:
            send_telegram(f"{group} failed: no pair data available.",
                          reply_to_message_id=reply_to_message_id)
            return

        opt_data = {}
        val_data = {}
        opt_dates = []
        val_dates = []
        for sym, df in all_data.items():
            split = (len(df) * 4) // 6
            opt_data[sym] = df.iloc[:split].reset_index(drop=True)
            val_data[sym] = df.iloc[split:].reset_index(drop=True)
            opt_dates.append((opt_data[sym]["close_time"].iloc[0],
                              opt_data[sym]["close_time"].iloc[-1]))
            val_dates.append((val_data[sym]["close_time"].iloc[0],
                              val_data[sym]["close_time"].iloc[-1]))
        opt_start = min(d[0] for d in opt_dates)
        opt_end = max(d[1] for d in opt_dates)
        val_start = min(d[0] for d in val_dates)
        val_end = max(d[1] for d in val_dates)

        all_results = []
        progress_every = 2 if group == "discover" else 10
        for idx, strat in enumerate(strategies, 1):
            opt_total = 0; opt_wins = 0
            val_total = 0; val_wins = 0
            for sym in opt_data:
                try:
                    t = backtest_strategy(strat, opt_data[sym])
                    opt_total += len(t)
                    opt_wins += sum(1 for x in t if x["result"] == "win")
                except Exception as e:
                    log.exception("%s opt %s/%s failed: %s", group, strat.id, sym, e)
                try:
                    t = backtest_strategy(strat, val_data[sym])
                    val_total += len(t)
                    val_wins += sum(1 for x in t if x["result"] == "win")
                except Exception as e:
                    log.exception("%s val %s/%s failed: %s", group, strat.id, sym, e)
            opt_wr = (100 * opt_wins / opt_total) if opt_total else 0.0
            val_wr = (100 * val_wins / val_total) if val_total else 0.0
            overfit = abs(opt_wr - val_wr) > 15
            passed = opt_wr >= 60 and val_wr >= 55 and opt_total >= 100
            result = {
                "idx": idx, "id": strat.id, "name": strat.name,
                "opt_total": opt_total, "opt_wr": opt_wr,
                "val_total": val_total, "val_wr": val_wr,
                "overfit": overfit, "passed": passed,
            }
            all_results.append(result)
            log.info(
                "%s %d/%d %s: opt=%.1f%%(%d) val=%.1f%%(%d) %s",
                group, idx, len(strategies), strat.id,
                opt_wr, opt_total, val_wr, val_total,
                "PASS" if passed else "fail",
            )
            if passed or idx % progress_every == 0 or idx == len(strategies):
                tag = "<b>PASSED</b>" if passed else (
                    "overfit" if overfit else "fail"
                )
                send_telegram(
                    f"{group} {idx}/{len(strategies)} <b>{strat.name}</b>: "
                    f"opt {opt_wr:.1f}% (n={opt_total}), "
                    f"val {val_wr:.1f}% (n={val_total}) - {tag}",
                    reply_to_message_id=reply_to_message_id,
                )
            if passed:
                active_strategy_id = strat.id
                save_strategy(strat.id, result)
                send_telegram(
                    f"<b>Winner: {strat.name}</b>\n"
                    f"ID: {strat.id}\n"
                    f"Opt: {opt_start.strftime('%Y-%m-%d')} -> "
                    f"{opt_end.strftime('%Y-%m-%d')} | "
                    f"Val: {val_start.strftime('%Y-%m-%d')} -> "
                    f"{val_end.strftime('%Y-%m-%d')}\n"
                    f"Opt WR: {opt_wr:.2f}% on {opt_total} trades\n"
                    f"Val WR: {val_wr:.2f}% on {val_total} trades\n"
                    f"<i>Activated for live scanning.</i>",
                    reply_to_message_id=reply_to_message_id,
                )
                return

        # No winner — pick best by combined avg
        scored = [r for r in all_results if r["opt_total"] > 0]
        if not scored:
            send_telegram(f"{group} finished: no strategy produced trades.",
                          reply_to_message_id=reply_to_message_id)
            return
        scored.sort(key=lambda r: (r["opt_wr"] + r["val_wr"]) / 2, reverse=True)
        best = scored[0]
        active_strategy_id = best["id"]
        save_strategy(best["id"], best)
        all_results.sort(key=lambda r: (r["opt_wr"] + r["val_wr"]) / 2, reverse=True)
        lines = [
            f"<b>{group} complete: no strategy met thresholds</b>",
            f"Best by combined avg: <b>{best['name']}</b> "
            f"(opt {best['opt_wr']:.1f}%, val {best['val_wr']:.1f}%) - activated",
            "",
            "<pre>",
            f"{'#':<3}{'name':<30}{'opt%':>6}{'val%':>6}{'optN':>6}",
        ]
        for r in all_results[:30]:
            n_short = r["name"][:29]
            lines.append(
                f"{r['idx']:<3}{n_short:<30}{r['opt_wr']:>5.1f}%"
                f"{r['val_wr']:>5.1f}%{r['opt_total']:>6}"
            )
        lines.append("</pre>")
        send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)
    except Exception as e:
        log.exception("%s failed: %s", group, e)
        send_telegram(f"<b>{group} failed</b>\n{e}",
                      reply_to_message_id=reply_to_message_id)
    finally:
        with discover_lock:
            discover_running = False


_4h_trend_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


def get_4h_trend(symbol: str) -> str:
    with _cache_lock:
        if symbol in _4h_trend_cache:
            return _4h_trend_cache[symbol]
    df = fetch_klines(symbol, interval="4h", limit=50)
    ema100_4h = ema(df["close"], 100).iloc[-1]
    price = df["close"].iloc[-1]
    trend = "up" if price > ema100_4h else "down"
    with _cache_lock:
        _4h_trend_cache[symbol] = trend
    return trend


def evaluate(df: pd.DataFrame, symbol: str) -> dict:
    sid = active_strategy_id
    if sid and sid in STRATEGIES_BY_ID:
        try:
            return STRATEGIES_BY_ID[sid].evaluate(df, symbol)
        except Exception as e:
            log.exception("strategy %s evaluate failed, falling back: %s", sid, e)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi14 = rsi(close, 14)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(
        close, params["bb_period"], params["bb_std"]
    )
    atr14 = atr(high, low, close, 14)

    price = float(close.iloc[-1])
    rsi_now = rsi14.iloc[-1]
    bb_upper_now = bb_upper.iloc[-1]
    bb_middle_now = bb_middle.iloc[-1]
    bb_lower_now = bb_lower.iloc[-1]
    atr_now = atr14.iloc[-1]

    rsi_oversold = params["rsi_oversold"]
    rsi_overbought = params["rsi_overbought"]

    long_entry = (
        pd.notna(rsi_now) and pd.notna(bb_lower_now)
        and rsi_now < rsi_oversold and price < float(bb_lower_now)
    )
    short_entry = (
        pd.notna(rsi_now) and pd.notna(bb_upper_now)
        and rsi_now > rsi_overbought and price > float(bb_upper_now)
    )

    direction = None
    if long_entry:
        direction = "LONG"
    elif short_entry:
        direction = "SHORT"

    return {
        "direction": direction,
        "price": price,
        "rsi": float(rsi_now) if pd.notna(rsi_now) else 0.0,
        "bb_upper": float(bb_upper_now) if pd.notna(bb_upper_now) else 0.0,
        "bb_middle": float(bb_middle_now) if pd.notna(bb_middle_now) else 0.0,
        "bb_lower": float(bb_lower_now) if pd.notna(bb_lower_now) else 0.0,
        "atr": float(atr_now) if pd.notna(atr_now) else 0.0,
        "candle_time": df["close_time"].iloc[-1],
    }


def format_message(symbol: str, r: dict) -> str:
    ts = datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S MYT")
    price = r["price"]
    atr_v = r["atr"]
    tp_mult = params["tp_mult"]
    sl_mult = params["sl_mult"]
    if r["direction"] == "LONG":
        tp = price + tp_mult * atr_v
        sl = price - sl_mult * atr_v
    else:
        tp = price - tp_mult * atr_v
        sl = price + sl_mult * atr_v
    return (
        f"<b>{r['direction']} signal: {symbol}</b>\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Price: {price:.6g}\n"
        f"RSI(14): {r['rsi']:.2f}\n"
        f"BB lower/mid/upper: {r['bb_lower']:.6g} / {r['bb_middle']:.6g} / {r['bb_upper']:.6g}\n"
        f"ATR(14): {atr_v:.6g}\n"
        f"Take Profit: {tp:.6g}\n"
        f"Stop Loss: {sl:.6g}\n"
        f"Time: {ts}"
    )


def check_long_exit(r: dict) -> list[str]:
    if "__strategy_long_exits" in r:
        return r["__strategy_long_exits"]
    reasons = []
    rsi_overbought = params["rsi_overbought"]
    if r["rsi"] > rsi_overbought:
        reasons.append(f"RSI > {rsi_overbought} ({r['rsi']:.2f})")
    if r["bb_middle"] > 0 and r["price"] > r["bb_middle"]:
        reasons.append("Price closed above BB middle")
    return reasons


def check_short_exit(r: dict) -> list[str]:
    if "__strategy_short_exits" in r:
        return r["__strategy_short_exits"]
    reasons = []
    rsi_oversold = params["rsi_oversold"]
    if r["rsi"] < rsi_oversold:
        reasons.append(f"RSI < {rsi_oversold} ({r['rsi']:.2f})")
    if r["bb_middle"] > 0 and r["price"] < r["bb_middle"]:
        reasons.append("Price closed below BB middle")
    return reasons


def format_close_message(symbol: str, direction: str, r: dict, reasons: list[str]) -> str:
    ts = datetime.now(MYT).strftime("%Y-%m-%d %H:%M:%S MYT")
    return (
        f"<b>CLOSE {direction}: {symbol}</b>\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Price: {r['price']:.6g}\n"
        f"Trigger: {'; '.join(reasons)}\n"
        f"Time: {ts}"
    )


def score_pair(df: pd.DataFrame, symbol: str) -> tuple[str, int]:
    sid = active_strategy_id
    if sid and sid in STRATEGIES_BY_ID:
        try:
            strat = STRATEGIES_BY_ID[sid]
            p = strat.precompute(df)
            n = len(df["close"])
            ls, ss = strat.score_at(p, n - 1)
            if ls >= ss:
                return "LONG", int(ls)
            return "SHORT", int(ss)
        except Exception as e:
            log.warning("score_pair %s failed: %s", sid, e)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi14 = rsi(close, 14)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(
        close, params["bb_period"], params["bb_std"]
    )
    atr14 = atr(high, low, close, 14)

    price = float(close.iloc[-1])
    rsi_now = rsi14.iloc[-1]
    bb_upper_now = bb_upper.iloc[-1]
    bb_lower_now = bb_lower.iloc[-1]
    bb_middle_now = bb_middle.iloc[-1]
    atr_now = atr14.iloc[-1]
    low_now = float(low.iloc[-1])
    high_now = float(high.iloc[-1])

    rsi_oversold = params["rsi_oversold"]
    rsi_overbought = params["rsi_overbought"]

    rsi_long = bool(pd.notna(rsi_now) and rsi_now < rsi_oversold)
    rsi_short = bool(pd.notna(rsi_now) and rsi_now > rsi_overbought)

    bb_outside_long = bool(
        pd.notna(bb_lower_now) and (price <= float(bb_lower_now) or low_now <= float(bb_lower_now))
    )
    bb_outside_short = bool(
        pd.notna(bb_upper_now) and (price >= float(bb_upper_now) or high_now >= float(bb_upper_now))
    )

    returning_long = bool(
        pd.notna(bb_lower_now) and pd.notna(bb_middle_now)
        and price > float(bb_lower_now) and price < float(bb_middle_now)
    )
    returning_short = bool(
        pd.notna(bb_upper_now) and pd.notna(bb_middle_now)
        and price < float(bb_upper_now) and price > float(bb_middle_now)
    )

    atr_ok = bool(pd.notna(atr_now) and float(atr_now) >= 0.003 * price)

    long_score = 0
    if rsi_long: long_score += 40
    if bb_outside_long: long_score += 40
    if returning_long: long_score += 10
    if atr_ok: long_score += 10

    short_score = 0
    if rsi_short: short_score += 40
    if bb_outside_short: short_score += 40
    if returning_short: short_score += 10
    if atr_ok: short_score += 10

    if long_score >= short_score:
        return "LONG", long_score
    return "SHORT", short_score


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
    lines = [f"<b>Confluence check — {ts}</b>", ""]
    for symbol, direction, score in rankings:
        filled = round(score / 10)
        bar = "█" * filled + "░" * (10 - filled)
        lines.append(f"{symbol} {direction} {score}% {bar}")
    lines.append("")
    lines.append("<i>100% = a live signal would fire right now.</i>")
    send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)
    log.info("/check reply sent (%d pairs)", len(rankings))


def _precompute_bt(df_1h: pd.DataFrame, df_4h: pd.DataFrame | None = None,
                   bb_stds: list[float] | None = None,
                   bb_periods: list[int] | None = None) -> dict:
    close = df_1h["close"]
    high_arr = df_1h["high"]
    low_arr = df_1h["low"]

    if bb_stds is None:
        bb_stds = [params["bb_std"]]
    if bb_periods is None:
        bb_periods = [params["bb_period"]]
    bb_cache = {
        (p, s): compute_bollinger_bands(close, p, s)
        for p in bb_periods for s in bb_stds
    }

    return {
        "close": close,
        "high": high_arr,
        "low": low_arr,
        "rsi14": rsi(close, 14),
        "bb_cache": bb_cache,
        "atr14": atr(high_arr, low_arr, close, 14),
    }


def _simulate_with_precomp(symbol: str, precomp: dict, sim_params: dict,
                           log_entries: bool = False) -> tuple[list[dict], dict]:
    close = precomp["close"]
    high_arr = precomp["high"]
    low_arr = precomp["low"]
    rsi14 = precomp["rsi14"]
    atr14 = precomp["atr14"]

    bb_period = sim_params["bb_period"]
    bb_std = sim_params["bb_std"]
    bb_key = (bb_period, bb_std)
    if bb_key not in precomp["bb_cache"]:
        precomp["bb_cache"][bb_key] = compute_bollinger_bands(close, bb_period, bb_std)
    bb_upper, bb_middle, bb_lower = precomp["bb_cache"][bb_key]

    rsi_oversold = sim_params["rsi_oversold"]
    rsi_overbought = sim_params["rsi_overbought"]
    sl_mult = sim_params["sl_mult"]
    tp_mult = sim_params["tp_mult"]
    tp_hit_rr = tp_mult / sl_mult

    n = len(close)
    trades: list[dict] = []
    open_pos: dict | None = None
    debug = {
        "candles_evaluated": 0,
        "conditions_met_sum": 0,
        "rejected_by_4h_only": 0,
        "fired_conditions_sum": 0,
        "fired_c1": 0,
        "fired_c2": 0,
        "fired_c3": 0,
        "fired_c4": 0,
        "eval_c1": 0,
        "eval_c2": 0,
        "eval_c3": 0,
        "eval_c4": 0,
        "atr_too_low": 0,
    }

    start_idx = max(bb_period, 14) + 1
    for i in range(start_idx, n):
        c = float(close.iloc[i])
        h = float(high_arr.iloc[i])
        l = float(low_arr.iloc[i])

        if open_pos is not None:
            d = open_pos["direction"]
            entry = open_pos["entry"]
            tp = open_pos["tp"]
            sl = open_pos["sl"]
            risk = open_pos["risk"]

            if d == "LONG":
                tp_hit = h >= tp
                sl_hit = l <= sl
            else:
                tp_hit = l <= tp
                sl_hit = h >= sl

            if sl_hit:
                trades.append({"direction": d, "entry": entry, "exit": sl, "rr": -1.0, "result": "loss"})
                open_pos = None
                continue
            if tp_hit:
                trades.append({"direction": d, "entry": entry, "exit": tp, "rr": tp_hit_rr, "result": "win"})
                open_pos = None
                continue

            rsi_cur = float(rsi14.iloc[i])
            bb_mid_cur = float(bb_middle.iloc[i]) if pd.notna(bb_middle.iloc[i]) else float("nan")

            if d == "LONG":
                exit_signal = rsi_cur > rsi_overbought or (
                    pd.notna(bb_mid_cur) and c > bb_mid_cur
                )
            else:
                exit_signal = rsi_cur < rsi_oversold or (
                    pd.notna(bb_mid_cur) and c < bb_mid_cur
                )
            if exit_signal:
                if d == "LONG":
                    rr = (c - entry) / risk if risk > 0 else 0.0
                else:
                    rr = (entry - c) / risk if risk > 0 else 0.0
                trades.append({"direction": d, "entry": entry, "exit": c, "rr": rr,
                               "result": "win" if rr > 0 else "loss"})
                open_pos = None
            continue

        atr_cur = float(atr14.iloc[i])
        if pd.isna(atr_cur):
            continue
        bb_upper_cur = float(bb_upper.iloc[i]) if pd.notna(bb_upper.iloc[i]) else float("nan")
        bb_lower_cur = float(bb_lower.iloc[i]) if pd.notna(bb_lower.iloc[i]) else float("nan")
        rsi_cur = float(rsi14.iloc[i])
        if pd.isna(bb_upper_cur) or pd.isna(bb_lower_cur) or pd.isna(rsi_cur):
            continue

        debug["candles_evaluated"] += 1
        rsi_long = rsi_cur < rsi_oversold
        rsi_short = rsi_cur > rsi_overbought
        bb_long = c < bb_lower_cur
        bb_short = c > bb_upper_cur
        atr_ok = atr_cur >= 0.003 * c

        if rsi_long or rsi_short: debug["eval_c1"] += 1
        if bb_long or bb_short: debug["eval_c2"] += 1
        if atr_ok: debug["eval_c4"] += 1
        else: debug["atr_too_low"] += 1

        long_fires = rsi_long and bb_long
        short_fires = rsi_short and bb_short

        if long_fires:
            risk = sl_mult * atr_cur
            open_pos = {
                "direction": "LONG", "entry": c,
                "tp": c + tp_mult * atr_cur, "sl": c - sl_mult * atr_cur, "risk": risk,
            }
            debug["fired_c1"] += 1
            debug["fired_c2"] += 1
            debug["fired_conditions_sum"] += 2
            if log_entries:
                log.info(
                    "backtest %s LONG entry @ %.6g | rsi=%.2f bb_low=%.6g atr=%.6g",
                    symbol, c, rsi_cur, bb_lower_cur, atr_cur,
                )
        elif short_fires:
            risk = sl_mult * atr_cur
            open_pos = {
                "direction": "SHORT", "entry": c,
                "tp": c - tp_mult * atr_cur, "sl": c + sl_mult * atr_cur, "risk": risk,
            }
            debug["fired_c1"] += 1
            debug["fired_c2"] += 1
            debug["fired_conditions_sum"] += 2
            if log_entries:
                log.info(
                    "backtest %s SHORT entry @ %.6g | rsi=%.2f bb_up=%.6g atr=%.6g",
                    symbol, c, rsi_cur, bb_upper_cur, atr_cur,
                )

    return trades, debug


def backtest_pair(symbol: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame | None = None,
                  sim_params: dict | None = None) -> tuple[list[dict], dict]:
    if sim_params is None:
        sim_params = dict(params)
    precomp = _precompute_bt(
        df_1h, df_4h,
        bb_stds=[sim_params["bb_std"]],
        bb_periods=[sim_params["bb_period"]],
    )
    return _simulate_with_precomp(symbol, precomp, sim_params, log_entries=True)


def handle_backtest_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /backtest command")
    send_telegram(
        "<b>Backtest started (backtesting.py engine)</b>\n"
        "Running BTCUSDT 6-month POC first. If Sharpe > 1.0 and Expectancy > 0, "
        "will expand to all 20 pairs. Otherwise stops and reports POC stats.",
        reply_to_message_id=reply_to_message_id,
    )
    threading.Thread(
        target=_run_backtest, args=(reply_to_message_id,), daemon=True
    ).start()


def _klines_to_bt_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["close_time", "open", "high", "low", "close", "volume"]].copy()
    out = out.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    out = out.set_index("close_time")
    out.index = pd.DatetimeIndex(out.index).tz_convert(None)
    return out[["Open", "High", "Low", "Close", "Volume"]]


def _build_mean_reversion_strategy():
    from backtesting import Strategy

    class BbRsiStrategy(Strategy):
        bb_period = 20
        bb_std = 2.0
        rsi_oversold = 30
        rsi_overbought = 70
        sl_mult = 1.0
        tp_mult = 2.0

        def init(self):
            close = pd.Series(self.data.Close)
            high = pd.Series(self.data.High)
            low = pd.Series(self.data.Low)

            self.rsi14 = self.I(lambda: rsi(close, 14).values, name="RSI14")
            bb_u, bb_m, bb_l = compute_bollinger_bands(close, self.bb_period, self.bb_std)
            self.bb_upper = self.I(lambda: bb_u.values, name="BB_upper")
            self.bb_middle = self.I(lambda: bb_m.values, name="BB_middle")
            self.bb_lower = self.I(lambda: bb_l.values, name="BB_lower")
            self.atr14 = self.I(lambda: atr(high, low, close, 14).values, name="ATR14")

        def next(self):
            n = len(self.data.Close)
            if n < max(self.bb_period, 14) + 1:
                return

            price = float(self.data.Close[-1])
            rsi_now = float(self.rsi14[-1])
            bb_lower_now = float(self.bb_lower[-1])
            bb_upper_now = float(self.bb_upper[-1])
            bb_middle_now = float(self.bb_middle[-1])
            atr_val = float(self.atr14[-1])

            if any(np.isnan(x) for x in (rsi_now, bb_lower_now, bb_upper_now,
                                          bb_middle_now, atr_val)):
                return

            if self.position:
                if self.position.is_long:
                    if rsi_now > self.rsi_overbought or price > bb_middle_now:
                        self.position.close()
                else:
                    if rsi_now < self.rsi_oversold or price < bb_middle_now:
                        self.position.close()
                return

            if rsi_now < self.rsi_oversold and price < bb_lower_now:
                tp = price + self.tp_mult * atr_val
                sl = price - self.sl_mult * atr_val
                self.buy(sl=sl, tp=tp)
            elif rsi_now > self.rsi_overbought and price > bb_upper_now:
                tp = price - self.tp_mult * atr_val
                sl = price + self.sl_mult * atr_val
                self.sell(sl=sl, tp=tp)

    return BbRsiStrategy


def _run_pair_library_backtest(symbol: str) -> dict | None:
    from backtesting import Backtest

    df = fetch_klines_paginated(symbol, "1h", target=4320)
    if len(df) < 200:
        log.warning("backtest %s: insufficient data (%d)", symbol, len(df))
        return None

    bt_df = _klines_to_bt_df(df)
    Strategy = _build_mean_reversion_strategy()
    bt = Backtest(
        bt_df, Strategy,
        cash=10000, commission=0.001, exclusive_orders=True,
    )
    stats = bt.run(
        bb_period=params["bb_period"],
        bb_std=params["bb_std"],
        rsi_oversold=params["rsi_oversold"],
        rsi_overbought=params["rsi_overbought"],
        sl_mult=params["sl_mult"],
        tp_mult=params["tp_mult"],
    )

    def _f(key, default=0.0):
        v = stats.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        if v != v:  # NaN
            return default
        return v

    return {
        "symbol": symbol,
        "trades": int(_f("# Trades", 0)),
        "wr": _f("Win Rate [%]"),
        "sharpe": _f("Sharpe Ratio"),
        "max_dd": _f("Max. Drawdown [%]"),
        "expectancy": _f("Expectancy [%]"),
        "ret": _f("Return [%]"),
        "sqn": _f("SQN"),
        "start": str(bt_df.index[0].date()),
        "end": str(bt_df.index[-1].date()),
    }


def _run_backtest(reply_to_message_id: int | None) -> None:
    try:
        log.info("library backtest: starting BTCUSDT POC")
        btc = _run_pair_library_backtest("BTCUSDT")
        if btc is None:
            send_telegram(
                "Library backtest aborted: insufficient BTCUSDT data.",
                reply_to_message_id=reply_to_message_id,
            )
            return

        poc_msg = (
            "<b>Library backtest POC: BTCUSDT</b>\n"
            f"Range: {btc['start']} -> {btc['end']}\n"
            f"Trades: {btc['trades']}\n"
            f"Win Rate: {btc['wr']:.2f}%\n"
            f"Sharpe Ratio: {btc['sharpe']:.2f}\n"
            f"Max Drawdown: {btc['max_dd']:.2f}%\n"
            f"Expectancy: {btc['expectancy']:+.3f}%\n"
            f"Return: {btc['ret']:+.2f}%\n"
            f"SQN: {btc['sqn']:.2f}"
        )
        send_telegram(poc_msg, reply_to_message_id=reply_to_message_id)
        log.info(
            "POC: trades=%d wr=%.2f sharpe=%.2f exp=%.3f maxdd=%.2f",
            btc["trades"], btc["wr"], btc["sharpe"], btc["expectancy"], btc["max_dd"],
        )

        passes = btc["sharpe"] > 1.0 and btc["expectancy"] > 0
        if not passes:
            why = []
            if btc["sharpe"] <= 1.0:
                why.append(f"Sharpe {btc['sharpe']:.2f} <= 1.0")
            if btc["expectancy"] <= 0:
                why.append(f"Expectancy {btc['expectancy']:+.3f}% <= 0")
            send_telegram(
                "POC did not meet thresholds (" + "; ".join(why) + "). "
                "Reporting stats as is, awaiting further instructions.",
                reply_to_message_id=reply_to_message_id,
            )
            return

        send_telegram(
            "POC passed (Sharpe > 1.0 AND Expectancy > 0). "
            "Expanding to all 20 pairs - this takes a few minutes.",
            reply_to_message_id=reply_to_message_id,
        )

        all_results: list[dict] = [btc]
        for symbol in PAIRS:
            if symbol == "BTCUSDT":
                continue
            try:
                r = _run_pair_library_backtest(symbol)
                if r is None:
                    continue
                all_results.append(r)
                log.info(
                    "backtest %s: trades=%d wr=%.2f sharpe=%.2f exp=%.3f",
                    symbol, r["trades"], r["wr"], r["sharpe"], r["expectancy"],
                )
            except Exception as e:
                log.exception("backtest %s failed: %s", symbol, e)

        all_results.sort(key=lambda r: r["sharpe"], reverse=True)
        total_trades = sum(r["trades"] for r in all_results)
        wr_weighted = (
            sum(r["wr"] * r["trades"] for r in all_results) / total_trades
            if total_trades else 0.0
        )
        sharpe_avg = sum(r["sharpe"] for r in all_results) / len(all_results)
        exp_avg = sum(r["expectancy"] for r in all_results) / len(all_results)
        ret_total = sum(r["ret"] for r in all_results)

        lines = ["<b>20-pair library backtest complete</b>"]
        lines.append("<pre>")
        lines.append(
            f"{'Pair':<10}{'T':>4}{'WR%':>7}{'Shrp':>7}{'DD%':>7}{'Exp%':>8}{'Ret%':>8}"
        )
        for r in all_results:
            lines.append(
                f"{r['symbol']:<10}{r['trades']:>4}"
                f"{r['wr']:>6.1f}%{r['sharpe']:>7.2f}"
                f"{r['max_dd']:>6.1f}%{r['expectancy']:>+7.2f}%{r['ret']:>+7.2f}%"
            )
        lines.append("</pre>")
        lines.append("")
        lines.append(f"Total trades: {total_trades}")
        lines.append(f"Weighted win rate: {wr_weighted:.2f}%")
        lines.append(f"Average Sharpe: {sharpe_avg:.2f}")
        lines.append(f"Average expectancy: {exp_avg:+.3f}%")
        lines.append(f"Sum of returns: {ret_total:+.2f}%")
        lines.append("")
        lines.append("<i>Past performance does not guarantee future results.</i>")
        send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)
    except Exception as e:
        log.exception("library backtest failed: %s", e)
        send_telegram(
            f"<b>Library backtest failed</b>\n{e}",
            reply_to_message_id=reply_to_message_id,
        )


def handle_optimize_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /optimize command")
    send_telegram(
        "<b>Optimization started</b>\n"
        "Fetching 6 months of 1H data for 20 pairs and grid-searching "
        "729 parameter combinations on a 4-month optimization window, then "
        "validating the top 5 on a held-out 2-month window. "
        "Will take several minutes.",
        reply_to_message_id=reply_to_message_id,
    )
    threading.Thread(
        target=_run_optimize, args=(reply_to_message_id,), daemon=True
    ).start()


def _run_optimize(reply_to_message_id: int | None) -> None:
    import itertools
    try:
        opt_precomp: dict[str, dict] = {}
        val_precomp: dict[str, dict] = {}
        opt_dates: list = []
        val_dates: list = []

        bb_period_grid = [15, 20, 25]
        bb_std_grid = [1.5, 2.0, 2.5]
        for symbol in PAIRS:
            try:
                df_1h = fetch_klines_paginated(symbol, "1h", target=4320)
                if len(df_1h) < 400:
                    log.warning("optimize %s: insufficient 1h data (%d)", symbol, len(df_1h))
                    continue
                split_1h = (len(df_1h) * 4) // 6
                df_1h_opt = df_1h.iloc[:split_1h].reset_index(drop=True)
                df_1h_val = df_1h.iloc[split_1h:].reset_index(drop=True)
                if len(df_1h_val) < 200:
                    log.warning("optimize %s: validation window too short", symbol)
                    continue
                opt_precomp[symbol] = _precompute_bt(
                    df_1h_opt, bb_stds=bb_std_grid, bb_periods=bb_period_grid
                )
                val_precomp[symbol] = _precompute_bt(
                    df_1h_val, bb_stds=bb_std_grid, bb_periods=bb_period_grid
                )
                opt_dates.append((df_1h_opt["close_time"].iloc[0],
                                  df_1h_opt["close_time"].iloc[-1]))
                val_dates.append((df_1h_val["close_time"].iloc[0],
                                  df_1h_val["close_time"].iloc[-1]))
            except Exception as e:
                log.exception("optimize %s fetch failed: %s", symbol, e)

        if not opt_precomp:
            send_telegram("Optimize failed: no pair data available.",
                          reply_to_message_id=reply_to_message_id)
            return

        opt_start = min(d[0] for d in opt_dates)
        opt_end = max(d[1] for d in opt_dates)
        val_start = min(d[0] for d in val_dates)
        val_end = max(d[1] for d in val_dates)

        rsi_oversold_grid = [25, 30, 35]
        rsi_overbought_grid = [65, 70, 75]
        sl_grid = [0.75, 1.0, 1.5]
        tp_grid = [1.5, 2.0, 2.5]

        combos = []
        for ro, rb, bp, bs, sm, tm in itertools.product(
            rsi_oversold_grid, rsi_overbought_grid,
            bb_period_grid, bb_std_grid, sl_grid, tp_grid,
        ):
            combos.append({
                "rsi_oversold": ro,
                "rsi_overbought": rb,
                "bb_period": bp,
                "bb_std": bs,
                "sl_mult": sm,
                "tp_mult": tm,
            })

        log.info("optimize: %d combos x %d pairs on opt window",
                 len(combos), len(opt_precomp))

        opt_results: list[dict] = []
        for idx, combo in enumerate(combos):
            total_trades = 0
            total_wins = 0
            for symbol, precomp in opt_precomp.items():
                try:
                    trades, _ = _simulate_with_precomp(
                        symbol, precomp, combo, log_entries=False
                    )
                    total_trades += len(trades)
                    total_wins += sum(1 for t in trades if t["result"] == "win")
                except Exception as e:
                    log.exception("optimize %s combo %d failed: %s", symbol, idx, e)
            if total_trades < 50:
                continue
            wr = 100.0 * total_wins / total_trades
            opt_results.append({
                "params": combo,
                "trades": total_trades,
                "wins": total_wins,
                "wr_opt": wr,
            })
            if (idx + 1) % 50 == 0:
                log.info("optimize: %d/%d combos evaluated", idx + 1, len(combos))

        if not opt_results:
            send_telegram(
                "Optimize finished but no parameter combination produced 50+ trades. "
                "Try widening the data window or relaxing filters.",
                reply_to_message_id=reply_to_message_id,
            )
            return

        opt_results.sort(key=lambda x: x["wr_opt"], reverse=True)
        top5 = opt_results[:5]
        log.info("optimize: top opt-window WR = %.2f%%", top5[0]["wr_opt"])

        for entry in top5:
            v_trades = 0
            v_wins = 0
            for symbol, precomp in val_precomp.items():
                try:
                    trades, _ = _simulate_with_precomp(
                        symbol, precomp, entry["params"], log_entries=False
                    )
                    v_trades += len(trades)
                    v_wins += sum(1 for t in trades if t["result"] == "win")
                except Exception as e:
                    log.exception("validate %s failed: %s", symbol, e)
            entry["val_trades"] = v_trades
            entry["val_wins"] = v_wins
            entry["wr_val"] = (100.0 * v_wins / v_trades) if v_trades else 0.0
            entry["wr_avg"] = (entry["wr_opt"] + entry["wr_val"]) / 2

        top5.sort(key=lambda x: x["wr_avg"], reverse=True)
        winner = top5[0]

        params.update(winner["params"])
        save_params()
        log.info("optimize: new parameters active: %s", params)

        def fmt_combo(p: dict) -> str:
            return (
                f"rsi{p['rsi_oversold']:>2}/{p['rsi_overbought']:>2} "
                f"bb({p['bb_period']:>2},{p['bb_std']:.1f}) "
                f"sl{p['sl_mult']:.2f} "
                f"tp{p['tp_mult']:.2f}"
            )

        lines = ["<b>Optimization complete</b>"]
        lines.append(
            f"Opt window: {opt_start.strftime('%Y-%m-%d')} -> "
            f"{opt_end.strftime('%Y-%m-%d')}"
        )
        lines.append(
            f"Val window: {val_start.strftime('%Y-%m-%d')} -> "
            f"{val_end.strftime('%Y-%m-%d')}"
        )
        lines.append(f"Pairs evaluated: {len(opt_precomp)} / {len(PAIRS)}")
        lines.append(f"Combos with >=50 trades: {len(opt_results)} / {len(combos)}")
        lines.append("")
        lines.append("<b>Top 5 from opt window (sorted by combined avg)</b>")
        lines.append("<pre>")
        lines.append(f"{'#':<2}{'combo':<36}{'opt%':>7}{'val%':>7}{'avg%':>7}{'trades':>8}")
        for i, e in enumerate(top5, 1):
            lines.append(
                f"{i:<2}{fmt_combo(e['params']):<36}"
                f"{e['wr_opt']:>6.1f}%{e['wr_val']:>6.1f}%{e['wr_avg']:>6.1f}%"
                f"{e['trades']:>8}"
            )
        lines.append("</pre>")
        lines.append("")
        lines.append(f"<b>Winner:</b> {fmt_combo(winner['params'])}")
        lines.append(
            f"Opt WR: {winner['wr_opt']:.1f}% | "
            f"Val WR: {winner['wr_val']:.1f}% | "
            f"Avg: {winner['wr_avg']:.1f}%"
        )
        lines.append("")
        lines.append("<b>New live parameters active:</b>")
        lines.append(f"<pre>{params_text()}</pre>")
        lines.append("")
        lines.append("<i>Past performance does not guarantee future results.</i>")

        send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)
    except Exception as e:
        log.exception("optimize failed: %s", e)
        send_telegram(
            f"<b>Optimize failed</b>\n{e}",
            reply_to_message_id=reply_to_message_id,
        )


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
                elif cmd == "/backtest":
                    handle_backtest_command(msg.get("message_id"))
                elif cmd == "/optimize":
                    handle_optimize_command(msg.get("message_id"))
                elif cmd == "/discover":
                    handle_discover_command(msg.get("message_id"))
                elif cmd == "/discover2":
                    handle_discover2_command(msg.get("message_id"))
        except Exception as e:
            log.warning("Telegram poll error: %s", e)
            time.sleep(5)
            continue
        time.sleep(2)


def scan_once() -> None:
    log.info("Starting scan of %d pairs", len(PAIRS))
    with _cache_lock:
        _4h_trend_cache.clear()
    for symbol in PAIRS:
        try:
            df = fetch_klines(symbol)
            if len(df) < 120:
                log.warning("%s: not enough candles (%d)", symbol, len(df))
                continue
            result = evaluate(df, symbol)

            position = open_positions.get(symbol)
            if position == "LONG":
                reasons = check_long_exit(result)
                if reasons:
                    send_telegram(format_close_message(symbol, "LONG", result, reasons))
                    entry = position_entry_price.pop(symbol, None)
                    if entry is not None:
                        trade_results.append(result["price"] > entry)
                    open_positions.pop(symbol, None)
                    save_state()
                    log.info("%s: CLOSE LONG sent (%s)", symbol, "; ".join(reasons))
            elif position == "SHORT":
                reasons = check_short_exit(result)
                if reasons:
                    send_telegram(format_close_message(symbol, "SHORT", result, reasons))
                    entry = position_entry_price.pop(symbol, None)
                    if entry is not None:
                        trade_results.append(result["price"] < entry)
                    open_positions.pop(symbol, None)
                    save_state()
                    log.info("%s: CLOSE SHORT sent (%s)", symbol, "; ".join(reasons))

            direction = result["direction"]
            if direction is None:
                continue
            if last_signal_by_pair.get(symbol) == direction:
                log.info("%s: duplicate %s signal, skipping", symbol, direction)
                continue
            last_signal_by_pair[symbol] = direction
            open_positions[symbol] = direction
            position_entry_price[symbol] = result["price"]
            save_state()
            msg = format_message(symbol, result)
            send_telegram(msg)
            log.info("%s: %s signal sent", symbol, direction)
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
        f"Watching {len(PAIRS)} pairs on {TIMEFRAME}.\n"
        "Strategy: BbRsi (proven Freqtrade community mean reversion). "
        f"LONG = RSI < {params['rsi_oversold']} AND close < lower BB. "
        f"SHORT = RSI > {params['rsi_overbought']} AND close > upper BB.\n"
        "TP at 2x ATR, SL at 1x ATR (2:1 reward:risk).\n"
        f"Live params: {params_text()}\n"
        f"{win_rate_text()}\n"
        f"Started: {datetime.now(MYT).strftime('%Y-%m-%d %H:%M:%S MYT')}"
    )
    while True:
        start = time.time()
        try:
            scan_once()
        except Exception as e:
            log.exception("scan loop error: %s", e)
        elapsed = time.time() - start
        sleep_for = max(0, SCAN_INTERVAL_SECONDS - int(elapsed))
        log.info("Scan complete in %ds; sleeping %ds", int(elapsed), sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
