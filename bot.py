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
    "adx_threshold": 25,
    "rsi_lo": 35,
    "rsi_hi": 65,
    "volume_mult": 1.2,
    "sl_mult": 2.0,
    "tp_mult": 3.0,
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
        f"ADX>{params['adx_threshold']}, "
        f"RSI {params['rsi_lo']}-{params['rsi_hi']}, "
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

    ema100 = ema(close, 100)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    obv_line = obv(close, volume)
    obv_ema = ema(obv_line, 20)
    atr14 = atr(high, low, close, 14)
    adx14 = compute_adx(high, low, close, 14)
    volume_sma20 = volume.rolling(window=20).mean()

    price = close.iloc[-1]
    ema100_now = ema100.iloc[-1]
    rsi_now, rsi_prev = rsi14.iloc[-1], rsi14.iloc[-2]
    macd_now, macd_prev = macd_line.iloc[-1], macd_line.iloc[-2]
    sig_now, sig_prev = signal_line.iloc[-1], signal_line.iloc[-2]
    hist_now, hist_prev = hist.iloc[-1], hist.iloc[-2]
    obv_now = obv_line.iloc[-1]
    obv_ema_now = obv_ema.iloc[-1]
    atr_now = atr14.iloc[-1]
    adx_now = adx14.iloc[-1]
    vol_now = volume.iloc[-1]
    vol_sma_now = volume_sma20.iloc[-1]

    macd_cross_up = macd_prev <= sig_prev and macd_now > sig_now
    macd_cross_down = macd_prev >= sig_prev and macd_now < sig_now
    hist_growing_up = hist_now > hist_prev
    hist_growing_down = hist_now < hist_prev

    rsi_in_range = params["rsi_lo"] <= rsi_now <= params["rsi_hi"]
    rsi_long_bounce = rsi_prev < 30 and rsi_now > rsi_prev
    rsi_short_rollover = rsi_prev > 70 and rsi_now < rsi_prev
    rsi_long_ok = rsi_in_range or rsi_long_bounce
    rsi_short_ok = rsi_in_range or rsi_short_rollover

    obv_bull = obv_now > obv_ema_now
    obv_bear = obv_now < obv_ema_now

    volume_ok = bool(pd.notna(vol_sma_now) and vol_now >= params["volume_mult"] * vol_sma_now)
    adx_ok = bool(pd.notna(adx_now) and adx_now >= params["adx_threshold"])

    trend_4h = get_4h_trend(symbol)

    direction = None
    if (
        price > ema100_now
        and trend_4h == "up"
        and macd_cross_up
        and hist_growing_up
        and rsi_long_ok
        and obv_bull
        and volume_ok
        and adx_ok
    ):
        direction = "LONG"
    elif (
        price < ema100_now
        and trend_4h == "down"
        and macd_cross_down
        and hist_growing_down
        and rsi_short_ok
        and obv_bear
        and volume_ok
        and adx_ok
    ):
        direction = "SHORT"

    return {
        "direction": direction,
        "price": float(price),
        "ema100": float(ema100_now),
        "rsi": float(rsi_now),
        "macd": float(macd_now),
        "macd_signal": float(sig_now),
        "obv": float(obv_now),
        "obv_ema": float(obv_ema_now),
        "obv_trend": "bullish" if obv_bull else ("bearish" if obv_bear else "flat"),
        "atr": float(atr_now),
        "adx": float(adx_now) if pd.notna(adx_now) else 0.0,
        "macd_cross_up": bool(macd_cross_up),
        "macd_cross_down": bool(macd_cross_down),
        "obv_bull": bool(obv_bull),
        "obv_bear": bool(obv_bear),
        "trend_4h": trend_4h,
        "volume_ok": volume_ok,
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
        f"EMA100: {r['ema100']:.6g}\n"
        f"RSI(14): {r['rsi']:.2f}\n"
        f"MACD: {r['macd']:.6g} | signal: {r['macd_signal']:.6g}\n"
        f"OBV trend: {r['obv_trend']}\n"
        f"ATR(14): {atr_v:.6g}\n"
        f"Take Profit: {tp:.6g}\n"
        f"Stop Loss: {sl:.6g}\n"
        f"Time: {ts}"
    )


def check_long_exit(r: dict) -> list[str]:
    reasons = []
    if r["macd_cross_down"]:
        reasons.append("MACD crossed below signal")
    if r["rsi"] > 70:
        reasons.append(f"RSI overbought ({r['rsi']:.2f})")
    if r["obv_bear"]:
        reasons.append("OBV dropped below EMA")
    if len(reasons) < 2:
        return []
    return reasons


def check_short_exit(r: dict) -> list[str]:
    reasons = []
    if r["macd_cross_up"]:
        reasons.append("MACD crossed above signal")
    if r["rsi"] < 30:
        reasons.append(f"RSI oversold ({r['rsi']:.2f})")
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
    volume = df["volume"]

    ema100 = ema(close, 100)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    obv_line = obv(close, volume)
    obv_ema_v = ema(obv_line, 20)
    volume_sma20 = volume.rolling(window=20).mean()

    price = close.iloc[-1]
    ema100_now = ema100.iloc[-1]
    rsi_now, rsi_prev = rsi14.iloc[-1], rsi14.iloc[-2]
    macd_now = macd_line.iloc[-1]
    sig_now = signal_line.iloc[-1]
    hist_now, hist_prev = hist.iloc[-1], hist.iloc[-2]
    obv_now = obv_line.iloc[-1]
    obv_ema_now = obv_ema_v.iloc[-1]
    vol_now = volume.iloc[-1]
    vol_sma_now = volume_sma20.iloc[-1]

    trend_4h = get_4h_trend(symbol)

    rsi_in_range = 35 <= rsi_now <= 65
    rsi_long_bounce = rsi_prev < 30 and rsi_now > rsi_prev
    rsi_short_rollover = rsi_prev > 70 and rsi_now < rsi_prev
    volume_ok = bool(pd.notna(vol_sma_now) and vol_now >= 1.2 * vol_sma_now)

    long_score = 0
    if price > ema100_now: long_score += 20
    if trend_4h == "up": long_score += 20
    if macd_now > sig_now: long_score += 15
    if hist_now > hist_prev: long_score += 10
    if rsi_in_range or rsi_long_bounce: long_score += 15
    if obv_now > obv_ema_now: long_score += 10
    if volume_ok: long_score += 10

    short_score = 0
    if price < ema100_now: short_score += 20
    if trend_4h == "down": short_score += 20
    if macd_now < sig_now: short_score += 15
    if hist_now < hist_prev: short_score += 10
    if rsi_in_range or rsi_short_rollover: short_score += 15
    if obv_now < obv_ema_now: short_score += 10
    if volume_ok: short_score += 10

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


def _precompute_bt(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> dict:
    close = df_1h["close"]
    high_arr = df_1h["high"]
    low_arr = df_1h["low"]
    volume = df_1h["volume"]

    macd_line, sig_line, hist = macd(close)
    obv_line = obv(close, volume)

    ema100_4h_full = ema(df_4h["close"], 100)
    ema100_4h_indexed = pd.Series(ema100_4h_full.values, index=df_4h["close_time"])
    ema100_4h_aligned = ema100_4h_indexed.reindex(df_1h["close_time"], method="ffill").values

    return {
        "close": close,
        "high": high_arr,
        "low": low_arr,
        "volume": volume,
        "ema100_1h": ema(close, 100),
        "rsi14": rsi(close, 14),
        "macd_line": macd_line,
        "sig_line": sig_line,
        "hist": hist,
        "obv": obv_line,
        "obv_ema": ema(obv_line, 20),
        "atr14": atr(high_arr, low_arr, close, 14),
        "adx14": compute_adx(high_arr, low_arr, close, 14),
        "vol_sma20": volume.rolling(window=20).mean(),
        "ema100_4h_aligned": ema100_4h_aligned,
    }


def _simulate_with_precomp(symbol: str, precomp: dict, sim_params: dict,
                           log_entries: bool = False) -> tuple[list[dict], dict]:
    close = precomp["close"]
    high_arr = precomp["high"]
    low_arr = precomp["low"]
    volume = precomp["volume"]
    ema100_1h = precomp["ema100_1h"]
    rsi14 = precomp["rsi14"]
    macd_line = precomp["macd_line"]
    sig_line = precomp["sig_line"]
    hist = precomp["hist"]
    obv_line = precomp["obv"]
    obv_ema_v = precomp["obv_ema"]
    atr14 = precomp["atr14"]
    adx14 = precomp["adx14"]
    vol_sma20 = precomp["vol_sma20"]
    ema100_4h_aligned = precomp["ema100_4h_aligned"]

    rsi_lo = sim_params["rsi_lo"]
    rsi_hi = sim_params["rsi_hi"]
    vol_mult = sim_params["volume_mult"]
    adx_thr = sim_params["adx_threshold"]
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
    }

    for i in range(100, n):
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

            macd_cur = float(macd_line.iloc[i]); macd_pr = float(macd_line.iloc[i - 1])
            sig_cur = float(sig_line.iloc[i]); sig_pr = float(sig_line.iloc[i - 1])
            cross_down = macd_pr >= sig_pr and macd_cur < sig_cur
            cross_up = macd_pr <= sig_pr and macd_cur > sig_cur
            rsi_cur = float(rsi14.iloc[i])
            obv_cur = float(obv_line.iloc[i])
            obv_ema_cur = float(obv_ema_v.iloc[i])

            if d == "LONG":
                cnt = sum([cross_down, rsi_cur > 70, obv_cur < obv_ema_cur])
                if cnt >= 2:
                    rr = (c - entry) / risk if risk > 0 else 0.0
                    trades.append({"direction": d, "entry": entry, "exit": c, "rr": rr,
                                   "result": "win" if rr > 0 else "loss"})
                    open_pos = None
            else:
                cnt = sum([cross_up, rsi_cur < 30, obv_cur > obv_ema_cur])
                if cnt >= 2:
                    rr = (entry - c) / risk if risk > 0 else 0.0
                    trades.append({"direction": d, "entry": entry, "exit": c, "rr": rr,
                                   "result": "win" if rr > 0 else "loss"})
                    open_pos = None
            continue

        ema100_now = float(ema100_1h.iloc[i])
        ema100_4h_now = ema100_4h_aligned[i]
        atr_cur = float(atr14.iloc[i])
        adx_cur = float(adx14.iloc[i]) if pd.notna(adx14.iloc[i]) else float("nan")
        vol_cur = float(volume.iloc[i])
        vol_sma_cur = float(vol_sma20.iloc[i]) if pd.notna(vol_sma20.iloc[i]) else float("nan")

        if pd.isna(ema100_4h_now) or pd.isna(atr_cur) or pd.isna(vol_sma_cur) or pd.isna(adx_cur):
            continue

        macd_cur = float(macd_line.iloc[i]); macd_pr = float(macd_line.iloc[i - 1])
        sig_cur = float(sig_line.iloc[i]); sig_pr = float(sig_line.iloc[i - 1])
        hist_cur = float(hist.iloc[i]); hist_pr = float(hist.iloc[i - 1])
        rsi_cur = float(rsi14.iloc[i]); rsi_pr = float(rsi14.iloc[i - 1])
        obv_cur = float(obv_line.iloc[i]); obv_ema_cur = float(obv_ema_v.iloc[i])

        cross_up = macd_pr <= sig_pr and macd_cur > sig_cur
        cross_down = macd_pr >= sig_pr and macd_cur < sig_cur
        hist_growing_up = hist_cur > hist_pr
        hist_growing_down = hist_cur < hist_pr
        rsi_in_range = rsi_lo <= rsi_cur <= rsi_hi
        rsi_long_bounce = rsi_pr < 30 and rsi_cur > rsi_pr
        rsi_short_rollover = rsi_pr > 70 and rsi_cur < rsi_pr
        rsi_long_ok = rsi_in_range or rsi_long_bounce
        rsi_short_ok = rsi_in_range or rsi_short_rollover
        obv_bull = obv_cur > obv_ema_cur
        obv_bear = obv_cur < obv_ema_cur
        volume_ok = vol_cur >= vol_mult * vol_sma_cur
        ema4h = float(ema100_4h_now)
        trend_up = c > ema4h
        trend_down = c < ema4h
        adx_ok = adx_cur >= adx_thr

        c_long = [c > ema100_now, trend_up, cross_up, hist_growing_up,
                  rsi_long_ok, obv_bull, volume_ok, adx_ok]
        c_short = [c < ema100_now, trend_down, cross_down, hist_growing_down,
                   rsi_short_ok, obv_bear, volume_ok, adx_ok]
        long_count = sum(c_long)
        short_count = sum(c_short)
        debug["candles_evaluated"] += 1
        debug["conditions_met_sum"] += max(long_count, short_count)

        if c_long[0] and not c_long[1] and all(c_long[2:]):
            debug["rejected_by_4h_only"] += 1
        if c_short[0] and not c_short[1] and all(c_short[2:]):
            debug["rejected_by_4h_only"] += 1

        if all(c_long):
            risk = sl_mult * atr_cur
            open_pos = {
                "direction": "LONG", "entry": c,
                "tp": c + tp_mult * atr_cur, "sl": c - sl_mult * atr_cur, "risk": risk,
            }
            debug["fired_conditions_sum"] += long_count
            if log_entries:
                log.info(
                    "backtest %s LONG entry @ %.6g | rsi=%.2f vol=%.2fx adx=%.2f",
                    symbol, c, rsi_cur,
                    vol_cur / vol_sma_cur if vol_sma_cur else 0.0, adx_cur,
                )
        elif all(c_short):
            risk = sl_mult * atr_cur
            open_pos = {
                "direction": "SHORT", "entry": c,
                "tp": c - tp_mult * atr_cur, "sl": c + sl_mult * atr_cur, "risk": risk,
            }
            debug["fired_conditions_sum"] += short_count
            if log_entries:
                log.info(
                    "backtest %s SHORT entry @ %.6g | rsi=%.2f vol=%.2fx adx=%.2f",
                    symbol, c, rsi_cur,
                    vol_cur / vol_sma_cur if vol_sma_cur else 0.0, adx_cur,
                )

    return trades, debug


def backtest_pair(symbol: str, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                  sim_params: dict | None = None) -> tuple[list[dict], dict]:
    if sim_params is None:
        sim_params = dict(params)
    precomp = _precompute_bt(df_1h, df_4h)
    return _simulate_with_precomp(symbol, precomp, sim_params, log_entries=True)


def handle_backtest_command(reply_to_message_id: int | None = None) -> None:
    log.info("Processing /backtest command")
    send_telegram(
        "<b>Backtest started</b>\n"
        "Fetching ~3 months of 1H + 4H data for 20 pairs and simulating the strategy. "
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
        }
        for symbol in PAIRS:
            try:
                df_1h = fetch_klines_paginated(symbol, "1h", target=2160)
                df_4h = fetch_klines_paginated(symbol, "4h", target=540)
                if len(df_1h) < 200 or len(df_4h) < 50:
                    log.warning("backtest %s: insufficient data (1h=%d, 4h=%d)",
                                symbol, len(df_1h), len(df_4h))
                    continue
                first_ct = df_1h["close_time"].iloc[0]
                last_ct = df_1h["close_time"].iloc[-1]
                date_min = first_ct if date_min is None or first_ct < date_min else date_min
                date_max = last_ct if date_max is None or last_ct > date_max else date_max

                trades, debug = backtest_pair(symbol, df_1h, df_4h)
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
                    "backtest %s: %d trades, %.1f%% WR, avg RR %+.2f | "
                    "candles=%d, avg_conds=%.2f/8, 4h_rejections=%d",
                    symbol, total, wr, avg_rr,
                    debug["candles_evaluated"],
                    debug["conditions_met_sum"] / debug["candles_evaluated"]
                    if debug["candles_evaluated"] else 0.0,
                    debug["rejected_by_4h_only"],
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
        lines.append(f"Avg conditions met per fired signal: {avg_fired:.2f} / 8")
        lines.append(f"Avg conditions met per evaluated candle: {avg_evaluated:.2f} / 8")
        lines.append(f"Candles evaluated for entry: {agg_debug['candles_evaluated']:,}")
        lines.append(f"Trades rejected by 4H filter alone: {agg_debug['rejected_by_4h_only']:,}")

        lines.append("")
        lines.append("<i>Past performance does not guarantee future results.</i>")

        send_telegram("\n".join(lines), reply_to_message_id=reply_to_message_id)
        log.info(
            "Backtest complete: %d total trades, %.1f%% overall WR, "
            "avg_fired=%.2f, avg_eval=%.2f, 4h_rejections=%d",
            total_t, overall_wr, avg_fired, avg_evaluated,
            agg_debug["rejected_by_4h_only"],
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
        "Fetching 6 months of 1H + 4H data for 20 pairs and grid-searching "
        "243 parameter combinations on a 4-month optimization window, then "
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

        for symbol in PAIRS:
            try:
                df_1h = fetch_klines_paginated(symbol, "1h", target=4320)
                df_4h = fetch_klines_paginated(symbol, "4h", target=1080)
                if len(df_1h) < 400 or len(df_4h) < 100:
                    log.warning("optimize %s: insufficient data (1h=%d, 4h=%d)",
                                symbol, len(df_1h), len(df_4h))
                    continue
                split_1h = (len(df_1h) * 4) // 6
                split_4h = (len(df_4h) * 4) // 6
                df_1h_opt = df_1h.iloc[:split_1h].reset_index(drop=True)
                df_1h_val = df_1h.iloc[split_1h:].reset_index(drop=True)
                df_4h_opt = df_4h.iloc[:split_4h].reset_index(drop=True)
                df_4h_val = df_4h.iloc[split_4h:].reset_index(drop=True)
                if len(df_1h_val) < 200 or len(df_4h_val) < 50:
                    log.warning("optimize %s: validation window too short", symbol)
                    continue
                opt_precomp[symbol] = _precompute_bt(df_1h_opt, df_4h_opt)
                val_precomp[symbol] = _precompute_bt(df_1h_val, df_4h_val)
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

        adx_grid = [20, 25, 30]
        rsi_grid = [(30, 70), (35, 65), (40, 60)]
        vol_grid = [1.1, 1.2, 1.5]
        sl_grid = [1.5, 2.0, 2.5]
        tp_grid = [2.5, 3.0, 3.5]

        combos = []
        for adx_t, rsi_r, v, sm, tm in itertools.product(
            adx_grid, rsi_grid, vol_grid, sl_grid, tp_grid
        ):
            combos.append({
                "adx_threshold": adx_t,
                "rsi_lo": rsi_r[0],
                "rsi_hi": rsi_r[1],
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
                f"adx>{p['adx_threshold']:>2} "
                f"rsi{p['rsi_lo']:>2}-{p['rsi_hi']:>2} "
                f"vol{p['volume_mult']:.1f} "
                f"sl{p['sl_mult']:.1f} "
                f"tp{p['tp_mult']:.1f}"
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
        "Strategy: 4H trend confirmation + 1H entry timing "
        "(EMA100 + RSI + MACD + OBV + volume spike + ADX).\n"
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
