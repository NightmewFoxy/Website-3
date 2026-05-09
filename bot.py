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
    "rsi_oversold": 25,
    "rsi_overbought": 75,
    "bb_std": 2.0,
    "volume_mult": 1.1,
    "sl_mult": 1.0,
    "tp_mult": 1.5,
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
        f"BB std {params['bb_std']}, "
        f"vol x{params['volume_mult']}, "
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


def compute_ema_slope(series: pd.Series, period: int = 50, lookback: int = 3) -> pd.Series:
    e = ema(series, period)
    return e - e.shift(lookback)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    alpha = 1 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean()


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
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    rsi14 = rsi(close, 14)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, 20, params["bb_std"])
    ema50 = ema(close, 50)
    ema50_slope = ema50 - ema50.shift(3)
    obv_line = obv(close, volume)
    obv_ema = ema(obv_line, 20)
    atr14 = atr(high, low, close, 14)

    price = float(close.iloc[-1])
    rsi_now = rsi14.iloc[-1]
    rsi_prev1 = rsi14.iloc[-2] if len(rsi14) >= 2 else float("nan")
    rsi_prev2 = rsi14.iloc[-3] if len(rsi14) >= 3 else float("nan")
    bb_upper_now = bb_upper.iloc[-1]
    bb_lower_now = bb_lower.iloc[-1]
    bb_middle_now = bb_middle.iloc[-1]
    slope_now = ema50_slope.iloc[-1]
    obv_now = obv_line.iloc[-1]
    obv_ema_now = obv_ema.iloc[-1]
    atr_now = atr14.iloc[-1]

    recent_rsi_low = any(
        pd.notna(rsi14.iloc[-i]) and rsi14.iloc[-i] < 30 for i in (1, 2, 3) if len(rsi14) >= i
    )
    recent_rsi_high = any(
        pd.notna(rsi14.iloc[-i]) and rsi14.iloc[-i] > 70 for i in (1, 2, 3) if len(rsi14) >= i
    )
    recent_bb_long_touch = any(
        pd.notna(low.iloc[-i]) and pd.notna(bb_lower.iloc[-i]) and low.iloc[-i] <= bb_lower.iloc[-i]
        for i in (1, 2, 3) if len(low) >= i
    )
    recent_bb_short_touch = any(
        pd.notna(high.iloc[-i]) and pd.notna(bb_upper.iloc[-i]) and high.iloc[-i] >= bb_upper.iloc[-i]
        for i in (1, 2, 3) if len(high) >= i
    )

    near_lower = (
        pd.notna(bb_lower_now)
        and abs(price - float(bb_lower_now)) <= 0.005 * price
    )
    near_upper = (
        pd.notna(bb_upper_now)
        and abs(price - float(bb_upper_now)) <= 0.005 * price
    )

    c1_long = bool((pd.notna(rsi_now) and rsi_now < 35) or recent_rsi_low)
    c2_long = bool(near_lower or recent_bb_long_touch)
    c3_long = bool(obv_now > obv_ema_now)
    c4_long = bool(pd.notna(slope_now) and slope_now > -0.005 * price)

    c1_short = bool((pd.notna(rsi_now) and rsi_now > 65) or recent_rsi_high)
    c2_short = bool(near_upper or recent_bb_short_touch)
    c3_short = bool(obv_now < obv_ema_now)
    c4_short = bool(pd.notna(slope_now) and slope_now < 0.005 * price)

    long_count = sum([c1_long, c2_long, c3_long, c4_long])
    short_count = sum([c1_short, c2_short, c3_short, c4_short])

    direction = None
    if long_count >= 3 and long_count >= short_count:
        direction = "LONG"
    elif short_count >= 3:
        direction = "SHORT"

    obv_bull = obv_now > obv_ema_now
    obv_bear = obv_now < obv_ema_now

    return {
        "direction": direction,
        "price": price,
        "rsi": float(rsi_now) if pd.notna(rsi_now) else 0.0,
        "bb_upper": float(bb_upper_now) if pd.notna(bb_upper_now) else 0.0,
        "bb_middle": float(bb_middle_now) if pd.notna(bb_middle_now) else 0.0,
        "bb_lower": float(bb_lower_now) if pd.notna(bb_lower_now) else 0.0,
        "ema50_slope": float(slope_now) if pd.notna(slope_now) else 0.0,
        "obv": float(obv_now),
        "obv_ema": float(obv_ema_now),
        "obv_trend": "bullish" if obv_bull else ("bearish" if obv_bear else "flat"),
        "obv_bull": bool(obv_bull),
        "obv_bear": bool(obv_bear),
        "atr": float(atr_now),
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
        f"OBV trend: {r['obv_trend']}\n"
        f"EMA50 slope (3-bar): {r['ema50_slope']:+.6g}\n"
        f"ATR(14): {atr_v:.6g}\n"
        f"Take Profit: {tp:.6g}\n"
        f"Stop Loss: {sl:.6g}\n"
        f"Time: {ts}"
    )


def check_long_exit(r: dict) -> list[str]:
    reasons = []
    if r["rsi"] > 60:
        reasons.append(f"RSI > 60 ({r['rsi']:.2f})")
    if r["bb_middle"] > 0 and r["price"] >= r["bb_middle"]:
        reasons.append("Price reached BB middle")
    if r["obv_bear"]:
        reasons.append("OBV dropped below EMA")
    if len(reasons) < 2:
        return []
    return reasons


def check_short_exit(r: dict) -> list[str]:
    reasons = []
    if r["rsi"] < 40:
        reasons.append(f"RSI < 40 ({r['rsi']:.2f})")
    if r["bb_middle"] > 0 and r["price"] <= r["bb_middle"]:
        reasons.append("Price reached BB middle")
    if r["obv_bull"]:
        reasons.append("OBV rose above EMA")
    if len(reasons) < 2:
        return []
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
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi14 = rsi(close, 14)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, 20, params["bb_std"])
    ema50 = ema(close, 50)
    ema50_slope = ema50 - ema50.shift(3)
    obv_line = obv(close, volume)
    obv_ema_v = ema(obv_line, 20)
    volume_sma20 = volume.rolling(window=20).mean()

    price = close.iloc[-1]
    low_now, low_prev = low.iloc[-1], low.iloc[-2]
    high_now, high_prev = high.iloc[-1], high.iloc[-2]
    rsi_now, rsi_prev = rsi14.iloc[-1], rsi14.iloc[-2]
    bb_upper_now, bb_upper_prev = bb_upper.iloc[-1], bb_upper.iloc[-2]
    bb_lower_now, bb_lower_prev = bb_lower.iloc[-1], bb_lower.iloc[-2]
    slope_now = ema50_slope.iloc[-1]
    obv_now = obv_line.iloc[-1]
    obv_ema_now = obv_ema_v.iloc[-1]
    vol_now = volume.iloc[-1]
    vol_sma_now = volume_sma20.iloc[-1]

    rsi_oversold = params["rsi_oversold"]
    rsi_overbought = params["rsi_overbought"]

    rsi_long_revert = (rsi_now < rsi_oversold or rsi_prev < rsi_oversold) and rsi_now > rsi_prev
    rsi_short_revert = (rsi_now > rsi_overbought or rsi_prev > rsi_overbought) and rsi_now < rsi_prev
    bb_long_touch = pd.notna(bb_lower_now) and pd.notna(bb_lower_prev) and (
        low_now <= bb_lower_now or low_prev <= bb_lower_prev
    )
    bb_short_touch = pd.notna(bb_upper_now) and pd.notna(bb_upper_prev) and (
        high_now >= bb_upper_now or high_prev >= bb_upper_prev
    )
    inside_long = pd.notna(bb_lower_now) and price > bb_lower_now
    inside_short = pd.notna(bb_upper_now) and price < bb_upper_now
    obv_bull = obv_now > obv_ema_now
    obv_bear = obv_now < obv_ema_now
    volume_ok = bool(pd.notna(vol_sma_now) and vol_now >= params["volume_mult"] * vol_sma_now)
    slope_long_ok = bool(pd.notna(slope_now) and slope_now >= 0)
    slope_short_ok = bool(pd.notna(slope_now) and slope_now <= 0)

    long_score = 0
    if rsi_long_revert: long_score += 25
    if bb_long_touch: long_score += 25
    if obv_bull: long_score += 15
    if volume_ok: long_score += 10
    if slope_long_ok: long_score += 15
    if inside_long: long_score += 10

    short_score = 0
    if rsi_short_revert: short_score += 25
    if bb_short_touch: short_score += 25
    if obv_bear: short_score += 15
    if volume_ok: short_score += 10
    if slope_short_ok: short_score += 15
    if inside_short: short_score += 10

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
                   bb_stds: list[float] | None = None) -> dict:
    close = df_1h["close"]
    high_arr = df_1h["high"]
    low_arr = df_1h["low"]
    volume = df_1h["volume"]

    if bb_stds is None:
        bb_stds = [params["bb_std"]]
    bb_cache = {s: compute_bollinger_bands(close, 20, s) for s in bb_stds}

    obv_line = obv(close, volume)
    return {
        "close": close,
        "high": high_arr,
        "low": low_arr,
        "volume": volume,
        "rsi14": rsi(close, 14),
        "bb_cache": bb_cache,
        "ema50": ema(close, 50),
        "obv": obv_line,
        "obv_ema": ema(obv_line, 20),
        "atr14": atr(high_arr, low_arr, close, 14),
        "vol_sma20": volume.rolling(window=20).mean(),
    }


def _simulate_with_precomp(symbol: str, precomp: dict, sim_params: dict,
                           log_entries: bool = False) -> tuple[list[dict], dict]:
    close = precomp["close"]
    high_arr = precomp["high"]
    low_arr = precomp["low"]
    volume = precomp["volume"]
    rsi14 = precomp["rsi14"]
    bb_std_key = sim_params["bb_std"]
    if bb_std_key not in precomp["bb_cache"]:
        precomp["bb_cache"][bb_std_key] = compute_bollinger_bands(close, 20, bb_std_key)
    bb_upper, bb_middle, bb_lower = precomp["bb_cache"][bb_std_key]
    ema50 = precomp["ema50"]
    obv_line = precomp["obv"]
    obv_ema_v = precomp["obv_ema"]
    atr14 = precomp["atr14"]
    vol_sma20 = precomp["vol_sma20"]

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
    }

    start_idx = 60
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
            obv_cur = float(obv_line.iloc[i])
            obv_ema_cur = float(obv_ema_v.iloc[i])

            if d == "LONG":
                conds = [
                    rsi_cur > 60,
                    pd.notna(bb_mid_cur) and c >= bb_mid_cur,
                    obv_cur < obv_ema_cur,
                ]
                if sum(conds) >= 2:
                    rr = (c - entry) / risk if risk > 0 else 0.0
                    trades.append({"direction": d, "entry": entry, "exit": c, "rr": rr,
                                   "result": "win" if rr > 0 else "loss"})
                    open_pos = None
            else:
                conds = [
                    rsi_cur < 40,
                    pd.notna(bb_mid_cur) and c <= bb_mid_cur,
                    obv_cur > obv_ema_cur,
                ]
                if sum(conds) >= 2:
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
        if pd.isna(bb_upper_cur) or pd.isna(bb_lower_cur):
            continue

        ema50_cur = float(ema50.iloc[i])
        ema50_back = float(ema50.iloc[i - 3]) if i >= 3 else float("nan")
        if pd.isna(ema50_cur) or pd.isna(ema50_back):
            continue
        slope = ema50_cur - ema50_back

        rsi_cur = float(rsi14.iloc[i])
        rsi_recent_low = any(
            i - k >= 0 and pd.notna(rsi14.iloc[i - k]) and rsi14.iloc[i - k] < 30
            for k in (0, 1, 2)
        )
        rsi_recent_high = any(
            i - k >= 0 and pd.notna(rsi14.iloc[i - k]) and rsi14.iloc[i - k] > 70
            for k in (0, 1, 2)
        )

        bb_long_touch_recent = any(
            i - k >= 0
            and pd.notna(low_arr.iloc[i - k]) and pd.notna(bb_lower.iloc[i - k])
            and low_arr.iloc[i - k] <= bb_lower.iloc[i - k]
            for k in (0, 1, 2)
        )
        bb_short_touch_recent = any(
            i - k >= 0
            and pd.notna(high_arr.iloc[i - k]) and pd.notna(bb_upper.iloc[i - k])
            and high_arr.iloc[i - k] >= bb_upper.iloc[i - k]
            for k in (0, 1, 2)
        )
        near_lower = abs(c - bb_lower_cur) <= 0.005 * c
        near_upper = abs(c - bb_upper_cur) <= 0.005 * c

        obv_cur = float(obv_line.iloc[i])
        obv_ema_cur = float(obv_ema_v.iloc[i])

        c1_long = (rsi_cur < 35) or rsi_recent_low
        c2_long = near_lower or bb_long_touch_recent
        c3_long = obv_cur > obv_ema_cur
        c4_long = slope > -0.005 * c

        c1_short = (rsi_cur > 65) or rsi_recent_high
        c2_short = near_upper or bb_short_touch_recent
        c3_short = obv_cur < obv_ema_cur
        c4_short = slope < 0.005 * c

        long_arr = [c1_long, c2_long, c3_long, c4_long]
        short_arr = [c1_short, c2_short, c3_short, c4_short]
        long_count = sum(long_arr)
        short_count = sum(short_arr)
        debug["candles_evaluated"] += 1
        debug["conditions_met_sum"] += max(long_count, short_count)

        long_fires = long_count >= 3 and long_count >= short_count
        short_fires = (not long_fires) and short_count >= 3

        if long_fires:
            risk = sl_mult * atr_cur
            open_pos = {
                "direction": "LONG", "entry": c,
                "tp": c + tp_mult * atr_cur, "sl": c - sl_mult * atr_cur, "risk": risk,
            }
            debug["fired_conditions_sum"] += long_count
            if c1_long: debug["fired_c1"] += 1
            if c2_long: debug["fired_c2"] += 1
            if c3_long: debug["fired_c3"] += 1
            if c4_long: debug["fired_c4"] += 1
            if log_entries:
                log.info(
                    "backtest %s LONG entry @ %.6g | rsi=%.2f bb_low=%.6g slope=%+.6g | "
                    "c1=%s c2=%s c3=%s c4=%s",
                    symbol, c, rsi_cur, bb_lower_cur, slope,
                    c1_long, c2_long, c3_long, c4_long,
                )
        elif short_fires:
            risk = sl_mult * atr_cur
            open_pos = {
                "direction": "SHORT", "entry": c,
                "tp": c - tp_mult * atr_cur, "sl": c + sl_mult * atr_cur, "risk": risk,
            }
            debug["fired_conditions_sum"] += short_count
            if c1_short: debug["fired_c1"] += 1
            if c2_short: debug["fired_c2"] += 1
            if c3_short: debug["fired_c3"] += 1
            if c4_short: debug["fired_c4"] += 1
            if log_entries:
                log.info(
                    "backtest %s SHORT entry @ %.6g | rsi=%.2f bb_up=%.6g slope=%+.6g | "
                    "c1=%s c2=%s c3=%s c4=%s",
                    symbol, c, rsi_cur, bb_upper_cur, slope,
                    c1_short, c2_short, c3_short, c4_short,
                )

    return trades, debug


def backtest_pair(symbol: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame | None = None,
                  sim_params: dict | None = None) -> tuple[list[dict], dict]:
    if sim_params is None:
        sim_params = dict(params)
    precomp = _precompute_bt(df_1h, df_4h, bb_stds=[sim_params["bb_std"]])
    return _simulate_with_precomp(symbol, precomp, sim_params, log_entries=True)


def handle_backtest_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /backtest command")
    send_telegram(
        "<b>Backtest started</b>\n"
        "Fetching ~3 months of 1H data for 20 pairs and simulating the mean-reversion strategy. "
        "Results will arrive in a few minutes.",
        reply_to_message_id=reply_to_message_id,
    )
    threading.Thread(
        target=_run_backtest, args=(reply_to_message_id,), daemon=True
    ).start()


def _run_backtest(reply_to_message_id: int | None) -> None:
    try:
        results: list[dict] = []
        date_min = None
        date_max = None
        agg_debug = {
            "candles_evaluated": 0,
            "conditions_met_sum": 0,
            "rejected_by_4h_only": 0,
            "fired_conditions_sum": 0,
            "fired_c1": 0,
            "fired_c2": 0,
            "fired_c3": 0,
            "fired_c4": 0,
        }
        for symbol in PAIRS:
            try:
                df_1h = fetch_klines_paginated(symbol, "1h", target=2160)
                if len(df_1h) < 200:
                    log.warning("backtest %s: insufficient 1h data (%d)", symbol, len(df_1h))
                    continue
                first_ct = df_1h["close_time"].iloc[0]
                last_ct = df_1h["close_time"].iloc[-1]
                date_min = first_ct if date_min is None or first_ct < date_min else date_min
                date_max = last_ct if date_max is None or last_ct > date_max else date_max

                trades, debug = backtest_pair(symbol, df_1h)
                for k in agg_debug:
                    agg_debug[k] += debug[k]
                wins = sum(1 for t in trades if t["result"] == "win")
                losses = sum(1 for t in trades if t["result"] == "loss")
                total = len(trades)
                wr = (100.0 * wins / total) if total else 0.0
                avg_rr = (sum(t["rr"] for t in trades) / total) if total else 0.0
                results.append({
                    "symbol": symbol, "total": total, "wins": wins,
                    "losses": losses, "wr": wr, "avg_rr": avg_rr,
                })
                log.info(
                    "backtest %s: %d trades, %.1f%% WR, avg RR %+.2f | candles=%d, avg_conds=%.2f/5",
                    symbol, total, wr, avg_rr,
                    debug["candles_evaluated"],
                    debug["conditions_met_sum"] / debug["candles_evaluated"]
                    if debug["candles_evaluated"] else 0.0,
                )
            except Exception as e:
                log.exception("backtest %s failed: %s", symbol, e)

        results.sort(key=lambda x: (x["wr"], x["total"]), reverse=True)

        lines = ["<b>Backtest results</b>"]
        if date_min is not None and date_max is not None:
            lines.append(
                f"Range: {date_min.strftime('%Y-%m-%d')} -> {date_max.strftime('%Y-%m-%d')}"
            )
        lines.append("")
        lines.append("<pre>")
        lines.append(f"{'Pair':<10}{'T':>4}{'W':>4}{'L':>4}{'WR%':>7}{'avgRR':>8}")
        for r in results:
            lines.append(
                f"{r['symbol']:<10}{r['total']:>4}{r['wins']:>4}{r['losses']:>4}"
                f"{r['wr']:>6.1f}%{r['avg_rr']:>+8.2f}"
            )
        lines.append("</pre>")

        total_t = sum(r["total"] for r in results)
        total_w = sum(r["wins"] for r in results)
        overall_wr = (100.0 * total_w / total_t) if total_t else 0.0
        lines.append(
            f"<b>Overall:</b> {total_w}/{total_t} wins = {overall_wr:.1f}% across all pairs"
        )

        lines.append("")
        lines.append("<b>DEBUG</b>")
        avg_fired = (
            agg_debug["fired_conditions_sum"] / total_t if total_t else 0.0
        )
        avg_evaluated = (
            agg_debug["conditions_met_sum"] / agg_debug["candles_evaluated"]
            if agg_debug["candles_evaluated"] else 0.0
        )
        lines.append(f"Avg conditions met per fired signal: {avg_fired:.2f} / 4")
        lines.append(f"Avg conditions met per evaluated candle: {avg_evaluated:.2f} / 4")
        lines.append(f"Candles evaluated for entry: {agg_debug['candles_evaluated']:,}")
        lines.append(
            f"Per-condition fires across {total_t} entries: "
            f"c1 RSI extreme {agg_debug['fired_c1']}, "
            f"c2 BB touch/near {agg_debug['fired_c2']}, "
            f"c3 OBV aligned {agg_debug['fired_c3']}, "
            f"c4 slope OK {agg_debug['fired_c4']}"
        )

        lines.append("")
        lines.append("<i>Past performance does not guarantee future results.</i>")

        send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)
        log.info(
            "Backtest complete: %d total trades, %.1f%% overall WR, "
            "avg_fired=%.2f, avg_eval=%.2f",
            total_t, overall_wr, avg_fired, avg_evaluated,
        )
    except Exception as e:
        log.exception("backtest failed: %s", e)
        send_telegram(
            f"<b>Backtest failed</b>\n{e}",
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
                opt_precomp[symbol] = _precompute_bt(df_1h_opt, bb_stds=bb_std_grid)
                val_precomp[symbol] = _precompute_bt(df_1h_val, bb_stds=bb_std_grid)
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

        rsi_oversold_grid = [20, 25, 30]
        rsi_overbought_grid = [70, 75, 80]
        vol_grid = [1.0, 1.1, 1.2]
        sl_grid = [0.75, 1.0, 1.25]
        tp_grid = [1.0, 1.5, 2.0]

        combos = []
        for ro, rb, bb_s, v, sm, tm in itertools.product(
            rsi_oversold_grid, rsi_overbought_grid, bb_std_grid,
            vol_grid, sl_grid, tp_grid,
        ):
            combos.append({
                "rsi_oversold": ro,
                "rsi_overbought": rb,
                "bb_std": bb_s,
                "volume_mult": v,
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
                f"bb{p['bb_std']:.1f} "
                f"vol{p['volume_mult']:.2f} "
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
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    send_telegram(
        "<b>binance-signal-bot online</b>\n"
        f"Watching {len(PAIRS)} pairs on {TIMEFRAME}.\n"
        "Strategy: Mean reversion using RSI extremes and Bollinger Band touches "
        "(OBV + volume + EMA50 slope confirmation).\n"
        "Targeting high win rate at 1.5:1 reward:risk.\n"
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
