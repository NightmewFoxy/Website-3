import os
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BINANCE_FAPI = "https://fapi.binance.com"
TELEGRAM_API = "https://api.telegram.org"

PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT",
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


def send_telegram(text: str) -> None:
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
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            log.error("Telegram send failed: %s %s", r.status_code, r.text)
    except requests.RequestException as e:
        log.error("Telegram request error: %s", e)


def fetch_klines(symbol: str, interval: str = TIMEFRAME, limit: int = KLINE_LIMIT) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
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


def evaluate(df: pd.DataFrame) -> dict:
    close = df["close"]
    volume = df["volume"]

    ema100 = ema(close, 100)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, _ = macd(close)
    obv_line = obv(close, volume)
    obv_ema = ema(obv_line, 20)

    price = close.iloc[-1]
    ema100_now = ema100.iloc[-1]
    rsi_now, rsi_prev = rsi14.iloc[-1], rsi14.iloc[-2]
    macd_now, macd_prev = macd_line.iloc[-1], macd_line.iloc[-2]
    sig_now, sig_prev = signal_line.iloc[-1], signal_line.iloc[-2]
    obv_now = obv_line.iloc[-1]
    obv_ema_now = obv_ema.iloc[-1]

    macd_cross_up = macd_prev <= sig_prev and macd_now > sig_now
    macd_cross_down = macd_prev >= sig_prev and macd_now < sig_now

    rsi_long_ok = rsi_now < 60 or (rsi_prev < 30 and rsi_now > rsi_prev)
    rsi_short_ok = rsi_now > 40 or (rsi_prev > 70 and rsi_now < rsi_prev)

    obv_bull = obv_now > obv_ema_now
    obv_bear = obv_now < obv_ema_now

    direction = None
    if price > ema100_now and macd_cross_up and rsi_long_ok and obv_bull:
        direction = "LONG"
    elif price < ema100_now and macd_cross_down and rsi_short_ok and obv_bear:
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
        "candle_time": df["close_time"].iloc[-1],
    }


def format_message(symbol: str, r: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"<b>{r['direction']} signal: {symbol}</b>\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Price: {r['price']:.6g}\n"
        f"EMA100: {r['ema100']:.6g}\n"
        f"RSI(14): {r['rsi']:.2f}\n"
        f"MACD: {r['macd']:.6g} | signal: {r['macd_signal']:.6g}\n"
        f"OBV trend: {r['obv_trend']}\n"
        f"Time: {ts}"
    )


def scan_once() -> None:
    log.info("Starting scan of %d pairs", len(PAIRS))
    for symbol in PAIRS:
        try:
            df = fetch_klines(symbol)
            if len(df) < 120:
                log.warning("%s: not enough candles (%d)", symbol, len(df))
                continue
            result = evaluate(df)
            direction = result["direction"]
            if direction is None:
                continue
            if last_signal_by_pair.get(symbol) == direction:
                log.info("%s: duplicate %s signal, skipping", symbol, direction)
                continue
            last_signal_by_pair[symbol] = direction
            msg = format_message(symbol, result)
            send_telegram(msg)
            log.info("%s: %s signal sent", symbol, direction)
        except requests.RequestException as e:
            log.error("%s: network error: %s", symbol, e)
        except Exception as e:
            log.exception("%s: evaluation error: %s", symbol, e)
        time.sleep(0.25)


def main() -> None:
    log.info("binance-signal-bot starting")
    send_telegram(
        "<b>binance-signal-bot online</b>\n"
        f"Watching {len(PAIRS)} pairs on {TIMEFRAME}.\n"
        f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
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
