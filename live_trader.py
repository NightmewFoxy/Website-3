"""live_trader.py — Binance Futures REST adapter for the R80 portfolio bot.

This module is the ONLY place that places real orders. bot.py imports it and
calls `is_live()` before deciding whether to hit the exchange. If env vars
aren't set, every public function is a no-op and the bot stays in paper mode.

Required env vars to go live:
  BINANCE_API_KEY        — Futures-enabled API key
  BINANCE_API_SECRET     — Secret for the same key
  R80_LIVE_TRADING=1     — Master enable. Default is paper.
  BINANCE_TESTNET=1      — Optional: route to futures testnet instead of mainnet

Order flow per trade:
  1. set_leverage(symbol, 10)
  2. place_market_entry(symbol, side, quantity)
  3. place_stop_market(symbol, opposite_side, stop_price)  ← SL, closePosition=true
  4. place_take_profit_market(symbol, opposite_side, stop_price) ← TP, closePosition=true

SL/TP orders persist at the exchange. Even if this process dies, your
position has a stop in the market. On the next scan, the bot checks
position size via /fapi/v2/positionRisk — if it's zero, the SL or TP fired,
and the bot reconciles state by fetching the realized P&L from order history.
"""

from __future__ import annotations

import os
import time
import hmac
import math
import json
import hashlib
import logging
from urllib.parse import urlencode

import requests

log = logging.getLogger("live_trader")

API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
TESTNET = os.environ.get("BINANCE_TESTNET", "0") == "1"
LIVE_TRADING_ENABLED = os.environ.get("R80_LIVE_TRADING", "0") == "1"

MAINNET_BASE = "https://fapi.binance.com"
TESTNET_BASE = "https://testnet.binancefuture.com"
BASE_URL = TESTNET_BASE if TESTNET else MAINNET_BASE

# In-memory cache of symbol filters (LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL)
_filters_cache: dict[str, dict] = {}
_filters_cached_at: float = 0.0
FILTERS_TTL_SECONDS = 6 * 3600


def is_live() -> bool:
    """True only if all of LIVE_TRADING, API_KEY, API_SECRET are configured."""
    return bool(LIVE_TRADING_ENABLED and API_KEY and API_SECRET)


def _sign(params: dict) -> str:
    qs = urlencode(params, doseq=True)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"


def _public_get(path: str, params: dict | None = None) -> dict | list | None:
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, params=params or {}, timeout=15)
        if r.status_code != 200:
            log.warning("Binance public GET %s failed: %s %s",
                        path, r.status_code, r.text[:300])
            return None
        return r.json()
    except Exception as e:
        log.warning("Binance public GET %s exception: %s", path, e)
        return None


def _signed_request(method: str, path: str,
                    params: dict | None = None) -> dict | None:
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 5000
    qs = _sign(p)
    url = f"{BASE_URL}{path}?{qs}"
    headers = {"X-MBX-APIKEY": API_KEY}
    try:
        r = requests.request(method, url, headers=headers, timeout=15)
        body = r.text
        try:
            data = r.json()
        except Exception:
            data = {"raw": body}
        if r.status_code != 200:
            log.warning("Binance %s %s failed: %s %s",
                        method, path, r.status_code, body[:300])
            return data if isinstance(data, dict) else None
        return data
    except Exception as e:
        log.warning("Binance %s %s exception: %s", method, path, e)
        return None


# ---------------- Account & balance ----------------

def get_futures_balance() -> float | None:
    """Return total walletBalance (USDT) on the futures account, or None."""
    data = _signed_request("GET", "/fapi/v2/balance", {})
    if not isinstance(data, list):
        return None
    for asset in data:
        if asset.get("asset") == "USDT":
            try:
                return float(asset.get("balance", 0))
            except (TypeError, ValueError):
                return None
    return None


def get_position(symbol: str) -> dict | None:
    """Return the position info dict for `symbol`, or None."""
    data = _signed_request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
    if not isinstance(data, list):
        return None
    for p in data:
        if p.get("symbol") == symbol:
            return p
    return None


def position_quantity(symbol: str) -> float:
    """Net position size for `symbol` (positive=long, negative=short, 0=flat)."""
    pos = get_position(symbol)
    if pos is None:
        return 0.0
    try:
        return float(pos.get("positionAmt", 0))
    except (TypeError, ValueError):
        return 0.0


def validate_api_key() -> bool:
    """Cheap sanity check: can we read our balance?"""
    bal = get_futures_balance()
    return bal is not None


# ---------------- Exchange info / precision ----------------

def _refresh_filters() -> None:
    global _filters_cached_at
    data = _public_get("/fapi/v1/exchangeInfo")
    if not isinstance(data, dict):
        return
    out = {}
    for s in data.get("symbols", []):
        sym = s.get("symbol")
        if not sym:
            continue
        d = {"price_tick": None, "lot_step": None, "min_qty": None,
             "min_notional": None}
        for f in s.get("filters", []):
            t = f.get("filterType")
            if t == "PRICE_FILTER":
                d["price_tick"] = float(f.get("tickSize", 0))
            elif t == "LOT_SIZE":
                d["lot_step"] = float(f.get("stepSize", 0))
                d["min_qty"] = float(f.get("minQty", 0))
            elif t == "MIN_NOTIONAL":
                try:
                    d["min_notional"] = float(f.get("notional", 0))
                except (TypeError, ValueError):
                    pass
        out[sym] = d
    if out:
        _filters_cache.update(out)
        _filters_cached_at = time.time()


def get_filters(symbol: str) -> dict | None:
    if (time.time() - _filters_cached_at) > FILTERS_TTL_SECONDS or symbol not in _filters_cache:
        _refresh_filters()
    return _filters_cache.get(symbol)


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    # Floor to step to avoid LOT_SIZE rejections
    return math.floor(value / step) * step


def _format_quantity(quantity: float, step: float) -> str:
    if step <= 0:
        return str(quantity)
    # Determine decimal places from step size
    s = f"{step:.10f}".rstrip("0").rstrip(".")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return f"{quantity:.{decimals}f}"


def _format_price(price: float, tick: float) -> str:
    if tick <= 0:
        return str(price)
    rounded = round(price / tick) * tick
    s = f"{tick:.10f}".rstrip("0").rstrip(".")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return f"{rounded:.{decimals}f}"


def compute_quantity(symbol: str, margin_usdt: float, leverage: int,
                     mark_price: float) -> tuple[float, str] | None:
    """Return (qty_float, qty_str) honoring LOT_SIZE and MIN_NOTIONAL; None if
    margin is too small to satisfy MIN_NOTIONAL."""
    filt = get_filters(symbol)
    if not filt:
        return None
    notional = margin_usdt * leverage
    raw_qty = notional / max(mark_price, 1e-12)
    step = filt.get("lot_step") or 0.0
    qty = _round_step(raw_qty, step)
    min_qty = filt.get("min_qty") or 0.0
    if qty < min_qty:
        log.warning("compute_quantity %s: qty %s < min_qty %s", symbol, qty, min_qty)
        return None
    min_notional = filt.get("min_notional") or 0.0
    if qty * mark_price < min_notional:
        log.warning("compute_quantity %s: notional %s < min_notional %s",
                    symbol, qty * mark_price, min_notional)
        return None
    return qty, _format_quantity(qty, step)


# ---------------- Leverage & margin mode ----------------

def set_leverage(symbol: str, leverage: int) -> bool:
    """Set leverage for `symbol`. Idempotent — re-setting same value is OK."""
    res = _signed_request("POST", "/fapi/v1/leverage", {
        "symbol": symbol, "leverage": leverage,
    })
    if not isinstance(res, dict):
        return False
    # Success returns {"leverage": 10, "maxNotionalValue": ..., "symbol": ...}
    return "leverage" in res


def set_margin_type_isolated(symbol: str) -> None:
    """Switch symbol to ISOLATED margin so liquidations don't cascade.
    -4046 means already isolated — that's fine."""
    res = _signed_request("POST", "/fapi/v1/marginType", {
        "symbol": symbol, "marginType": "ISOLATED",
    })
    if isinstance(res, dict) and res.get("code") and res.get("code") != -4046:
        log.warning("set_margin_type_isolated %s: %s", symbol, res)


# ---------------- Orders ----------------

def place_market_entry(symbol: str, side: str,
                       quantity: float) -> dict | None:
    """Place a MARKET entry. `side` is 'BUY' or 'SELL'. Returns order resp."""
    filt = get_filters(symbol) or {}
    step = filt.get("lot_step") or 0.0
    qty_str = _format_quantity(quantity, step)
    return _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol, "side": side, "type": "MARKET",
        "quantity": qty_str, "newOrderRespType": "RESULT",
    })


def place_stop_loss(symbol: str, opposite_side: str,
                    stop_price: float) -> dict | None:
    """STOP_MARKET reduceOnly closePosition=true. Triggers a full market close
    when stop_price is hit."""
    filt = get_filters(symbol) or {}
    tick = filt.get("price_tick") or 0.0
    return _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol, "side": opposite_side, "type": "STOP_MARKET",
        "stopPrice": _format_price(stop_price, tick),
        "closePosition": "true",
        "workingType": "MARK_PRICE",
    })


def place_take_profit(symbol: str, opposite_side: str,
                      stop_price: float) -> dict | None:
    """TAKE_PROFIT_MARKET reduceOnly closePosition=true."""
    filt = get_filters(symbol) or {}
    tick = filt.get("price_tick") or 0.0
    return _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol, "side": opposite_side, "type": "TAKE_PROFIT_MARKET",
        "stopPrice": _format_price(stop_price, tick),
        "closePosition": "true",
        "workingType": "MARK_PRICE",
    })


def place_market_close(symbol: str, opposite_side: str,
                       quantity: float) -> dict | None:
    """Force-close an existing position by sending an opposing MARKET reduceOnly."""
    filt = get_filters(symbol) or {}
    step = filt.get("lot_step") or 0.0
    qty_str = _format_quantity(abs(quantity), step)
    return _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol, "side": opposite_side, "type": "MARKET",
        "quantity": qty_str, "reduceOnly": "true",
        "newOrderRespType": "RESULT",
    })


def cancel_all_open_orders(symbol: str) -> dict | None:
    """Cancel any outstanding orders for the symbol (used to clean up the
    untriggered SL/TP after the other side fires, or before timeout-close)."""
    return _signed_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})


def get_user_trades(symbol: str, since_ms: int | None = None,
                    limit: int = 50) -> list | None:
    """Recent fills for the symbol. Used to reconcile exit price when an SL/TP
    fires on the exchange."""
    params = {"symbol": symbol, "limit": limit}
    if since_ms is not None:
        params["startTime"] = since_ms
    data = _signed_request("GET", "/fapi/v1/userTrades", params)
    return data if isinstance(data, list) else None


# ---------------- High-level helpers ----------------

def execute_entry(symbol: str, direction: str, leverage: int,
                  margin_usdt: float, sl_price: float, tp_price: float,
                  mark_price: float) -> dict | None:
    """Open a position with SL+TP at the exchange. Returns a dict with:
      - {"ok": True, "fill_price": float, "quantity": float, "order_id": int,
         "sl_order_id": int|None, "tp_order_id": int|None}
    or {"ok": False, "error": str} on any failure."""
    if not is_live():
        return {"ok": False, "error": "live trading disabled"}
    # Ensure isolated margin so blowups don't drain the whole account
    set_margin_type_isolated(symbol)
    # Set leverage
    if not set_leverage(symbol, leverage):
        return {"ok": False, "error": "set_leverage failed"}
    qty_pair = compute_quantity(symbol, margin_usdt, leverage, mark_price)
    if qty_pair is None:
        return {"ok": False, "error": "quantity too small for filters"}
    qty, _ = qty_pair
    side = "BUY" if direction == "LONG" else "SELL"
    opp = "SELL" if direction == "LONG" else "BUY"
    entry_resp = place_market_entry(symbol, side, qty)
    if not isinstance(entry_resp, dict) or entry_resp.get("code"):
        return {"ok": False, "error": f"entry rejected: {entry_resp}"}
    try:
        fill_price = float(entry_resp.get("avgPrice")
                           or entry_resp.get("price")
                           or mark_price)
        entry_order_id = int(entry_resp.get("orderId", 0))
    except (TypeError, ValueError):
        fill_price = mark_price
        entry_order_id = 0
    sl_resp = place_stop_loss(symbol, opp, sl_price)
    tp_resp = place_take_profit(symbol, opp, tp_price)
    sl_id = (sl_resp.get("orderId") if isinstance(sl_resp, dict)
             and not sl_resp.get("code") else None)
    tp_id = (tp_resp.get("orderId") if isinstance(tp_resp, dict)
             and not tp_resp.get("code") else None)
    return {
        "ok": True,
        "fill_price": fill_price,
        "quantity": qty,
        "order_id": entry_order_id,
        "sl_order_id": sl_id,
        "tp_order_id": tp_id,
    }


def execute_close(symbol: str, direction: str,
                  quantity: float) -> dict | None:
    """Force-close an open position (used on TIMEOUT). Cancels SL/TP first."""
    if not is_live():
        return {"ok": False, "error": "live trading disabled"}
    cancel_all_open_orders(symbol)
    opp = "SELL" if direction == "LONG" else "BUY"
    resp = place_market_close(symbol, opp, quantity)
    if not isinstance(resp, dict) or resp.get("code"):
        return {"ok": False, "error": f"close rejected: {resp}"}
    try:
        fill_price = float(resp.get("avgPrice") or resp.get("price") or 0)
    except (TypeError, ValueError):
        fill_price = 0
    return {"ok": True, "fill_price": fill_price,
            "order_id": int(resp.get("orderId", 0))}


def reconcile_exit(symbol: str, entry_time_ms: int) -> dict | None:
    """If position is flat on the exchange, find the most recent closing fill
    after entry_time_ms and return its price and realized PnL."""
    if not is_live():
        return None
    pos = get_position(symbol)
    if pos is None:
        return None
    try:
        amt = float(pos.get("positionAmt", 0))
    except (TypeError, ValueError):
        amt = 0.0
    if abs(amt) > 0:
        return {"closed": False, "size": amt}
    trades = get_user_trades(symbol, since_ms=entry_time_ms, limit=50)
    if not trades:
        return {"closed": True, "fill_price": None, "realized_pnl": None}
    closing = [t for t in trades if t.get("realizedPnl") not in (None, "0")]
    if not closing:
        # Last trade is probably the exit even if realizedPnl wasn't set;
        # fall back to the most recent trade
        last = trades[-1]
        try:
            return {"closed": True,
                    "fill_price": float(last.get("price", 0)),
                    "realized_pnl": float(last.get("realizedPnl", 0)),
                    "fee": float(last.get("commission", 0))}
        except (TypeError, ValueError):
            return {"closed": True, "fill_price": None, "realized_pnl": None}
    realized = sum(float(t.get("realizedPnl", 0)) for t in closing)
    fee = sum(float(t.get("commission", 0)) for t in closing)
    last_fill = float(closing[-1].get("price", 0))
    # Cancel any leftover (untriggered SL or TP)
    cancel_all_open_orders(symbol)
    return {"closed": True, "fill_price": last_fill,
            "realized_pnl": realized, "fee": fee}
