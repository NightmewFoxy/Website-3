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
