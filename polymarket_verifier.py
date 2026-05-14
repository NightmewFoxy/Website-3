"""polymarket_verifier.py — backtest whether Claude's AI probability estimates
beat Polymarket crowd prices on resolved historical markets.

Pipeline:
1. Fetch resolved binary YES/NO markets from Polymarket Gamma API (last 6 months)
2. Filter: binary, $1000+ volume, has clean resolution
3. Train/validate split: closed >4mo ago = opt, closed <2mo ago = val
4. For each market:
   - Get crowd price at market open (CLOB prices-history endpoint)
   - Ask Claude (with web search) to estimate probability using only pre-open info
   - If confidence is medium/high AND |claude - crowd| > 0.10, place Kelly-sized bet
   - Record outcome vs resolution
5. Report win rate, EV, calibration, Sharpe to console + Telegram

Trigger via Telegram: /polyverify (full ~30-60min) or /polyverify_fast (top 50 by volume, ~10min)
"""

import os
import json
import time
import math
import threading
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

log = logging.getLogger("polyverify")

# ---------------- Config ----------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get(
    "POLYVERIFY_MODEL",
    "claude-opus-4-5",  # Opus 4.5 for superforecasting-grade calibration on probability estimates
)
PROGRESS_PATH = (
    "/data/polyverify_progress.json"
    if os.path.isdir("/data")
    else "polyverify_progress.json"
)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

STARTING_BANKROLL = 100.0
MIN_VOLUME = 1000.0
MIN_EDGE = 0.10
KELLY_CAP = 0.25
EDGE_OPT_MONTHS = 4   # markets closed >4 months ago = optimization window
EDGE_VAL_MONTHS = 2   # markets closed <2 months ago = validation window
RATE_LIMIT_SLEEP = 20  # seconds between Anthropic calls (3/min)

SYSTEM_PROMPT = (
    "You are a superforecaster estimating probabilities for prediction market "
    "questions. You must only use information available before the date provided. "
    "Return your answer as JSON with fields: probability (0.0 to 1.0), confidence "
    "(low/medium/high), reasoning (one sentence). Return only valid JSON, no other text."
)


# ---------------- Polymarket data ----------------

def fetch_resolved_markets(target_count: int = 500) -> list[dict]:
    """Fetch resolved markets from Gamma API, paginating until we have target_count
    or run out. Returns markets with the resolution date populated."""
    out: list[dict] = []
    offset = 0
    page_size = 100
    while len(out) < target_count:
        try:
            r = requests.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "closed": "true",
                    "limit": page_size,
                    "offset": offset,
                    "order": "endDate",
                    "ascending": "false",
                },
                timeout=30,
            )
            if r.status_code != 200:
                log.warning("gamma markets %s: %s", r.status_code, r.text[:200])
                break
            data = r.json()
            if not isinstance(data, list) or not data:
                break
            out.extend(data)
            offset += page_size
            if len(data) < page_size:
                break
        except Exception as e:
            log.warning("gamma fetch err at offset %s: %s", offset, e)
            break
        time.sleep(0.3)
    return out[:target_count]


def parse_outcomes(market: dict) -> list[str]:
    raw = market.get("outcomes", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    return [str(o).strip() for o in raw]


def parse_outcome_prices(market: dict) -> list[float] | None:
    raw = market.get("outcomePrices", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    try:
        return [float(raw[0]), float(raw[1])]
    except (TypeError, ValueError):
        return None


def parse_clob_token_ids(market: dict) -> list[str]:
    raw = market.get("clobTokenIds", [])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    return [str(t) for t in raw]


def is_binary_yes_no(market: dict) -> bool:
    outcomes = parse_outcomes(market)
    if len(outcomes) != 2:
        return False
    return set(o.lower() for o in outcomes) == {"yes", "no"}


def get_resolution(market: dict) -> str | None:
    """Returns 'YES', 'NO', or None if not cleanly resolved."""
    prices = parse_outcome_prices(market)
    if prices is None:
        return None
    p_yes, p_no = prices
    if p_yes > 0.99 and p_no < 0.01:
        return "YES"
    if p_no > 0.99 and p_yes < 0.01:
        return "NO"
    return None


def parse_iso_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    # Normalize "+00" (no minutes) to "+00:00"
    if len(s) >= 3 and s[-3] in ("+", "-") and ":" not in s[-3:]:
        s = s + ":00"
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def get_close_dt(market: dict) -> datetime | None:
    """When the market actually resolved/closed for trading. `endDate` is the
    nominal deadline (often in the future); `closedTime` is the real one."""
    return (
        parse_iso_dt(market.get("closedTime"))
        or parse_iso_dt(market.get("updatedAt"))
        or parse_iso_dt(market.get("endDate"))
    )


def get_open_price(market: dict) -> float | None:
    """Fetch crowd price at market open (within the first hour after startDate).
    Uses CLOB prices-history with the YES token from clobTokenIds."""
    tokens = parse_clob_token_ids(market)
    if len(tokens) < 1:
        return None
    yes_token = tokens[0]  # convention: index 0 = YES, index 1 = NO
    start_dt = parse_iso_dt(market.get("startDate")) or parse_iso_dt(market.get("createdAt"))
    if start_dt is None:
        return None
    start_ts = int(start_dt.timestamp())
    end_ts = start_ts + 3600
    try:
        r = requests.get(
            f"{CLOB_BASE}/prices-history",
            params={"market": yes_token, "startTs": start_ts, "endTs": end_ts},
            timeout=30,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        history = data.get("history", [])
        if not history:
            # Fallback: try a longer window
            r2 = requests.get(
                f"{CLOB_BASE}/prices-history",
                params={"market": yes_token, "startTs": start_ts, "endTs": start_ts + 86400},
                timeout=30,
            )
            if r2.status_code == 200:
                history = r2.json().get("history", [])
        if not history:
            return None
        return float(history[0]["p"])
    except Exception:
        return None


def get_market_volume(market: dict) -> float:
    for key in ("volume", "volumeNum", "volume24hr"):
        v = market.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def has_real_trading(market: dict) -> bool:
    """Liquidity proxy when Gamma API doesn't return volume. For a RESOLVED
    market: `lastTradePrice` near 0 or 1 is expected (it converged to the
    resolution). The real signal is that `lastTradePrice` is non-null AND
    `spread` is tight enough that the orderbook was being actively quoted."""
    ltp = market.get("lastTradePrice")
    if ltp is None:
        return False
    try:
        ltp = float(ltp)
    except (TypeError, ValueError):
        return False
    if ltp == 0:
        # Distinguish "resolved NO" (genuine final price) from "never traded"
        # by requiring a tight spread.
        try:
            spread = float(market.get("spread", 1.0))
        except (TypeError, ValueError):
            return False
        if spread > 0.10:
            return False
    return True


# ---------------- Claude estimation ----------------

def estimate_probability(
    client: "anthropic.Anthropic",
    question: str,
    open_date: datetime,
) -> dict | None:
    """Call Claude with web search to estimate probability. Returns
    {"probability": float, "confidence": str, "reasoning": str} or None on error."""
    user_prompt = (
        f"Question: {question}\n\n"
        f"Only use information available before {open_date.strftime('%Y-%m-%d')}. "
        f"What is the probability this resolves YES?"
    )
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APIStatusError as e:
        log.warning("Anthropic API %s: %s", e.status_code, str(e)[:200])
        return None
    except Exception as e:
        log.warning("Anthropic call failed: %s", e)
        return None

    # Extract text from response (may be after tool_use blocks)
    text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    text = text.strip()
    if not text:
        return None

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Find the JSON object in the response
    try:
        # Try to find { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    prob = parsed.get("probability")
    conf = str(parsed.get("confidence", "")).lower().strip()
    try:
        prob = float(prob)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= prob <= 1.0):
        return None
    if conf not in ("low", "medium", "high"):
        conf = "low"
    return {
        "probability": prob,
        "confidence": conf,
        "reasoning": str(parsed.get("reasoning", ""))[:200],
    }


# ---------------- Kelly + simulation ----------------

def kelly_bet(claude_prob: float, crowd_price: float, bankroll: float) -> dict | None:
    """Returns {"side": "YES"/"NO", "fraction": float, "stake": float} or None.

    Kelly fraction = edge / odds. Bet YES if claude_prob > crowd + MIN_EDGE; bet NO if
    claude_prob < crowd - MIN_EDGE. Capped at KELLY_CAP of bankroll."""
    edge = claude_prob - crowd_price
    if edge > MIN_EDGE:
        # Bet YES at crowd_price
        odds_against = (1.0 - crowd_price) / max(crowd_price, 1e-9)
        # Kelly = (p * b - q) / b where b = odds_against, p = claude_prob, q = 1-p
        b = odds_against
        if b <= 0:
            return None
        f = (claude_prob * b - (1.0 - claude_prob)) / b
        f = max(0.0, min(f, KELLY_CAP))
        if f <= 0:
            return None
        return {"side": "YES", "fraction": f, "stake": bankroll * f,
                "crowd_price": crowd_price, "edge": edge}
    if edge < -MIN_EDGE:
        # Bet NO at (1 - crowd_price)
        no_price = 1.0 - crowd_price
        no_prob = 1.0 - claude_prob
        odds_against = crowd_price / max(no_price, 1e-9)
        b = odds_against
        if b <= 0:
            return None
        f = (no_prob * b - (1.0 - no_prob)) / b
        f = max(0.0, min(f, KELLY_CAP))
        if f <= 0:
            return None
        return {"side": "NO", "fraction": f, "stake": bankroll * f,
                "crowd_price": no_price, "edge": -edge}
    return None


def settle_bet(bet: dict, resolution: str, bankroll: float) -> float:
    """Returns delta to bankroll (positive on win, negative on loss)."""
    won = bet["side"] == resolution
    crowd_price = bet["crowd_price"]
    fraction = bet["fraction"]
    if won:
        # Profit = stake * (1/price - 1)
        return bankroll * fraction * (1.0 / max(crowd_price, 1e-9) - 1.0)
    else:
        return -bankroll * fraction


# ---------------- Persistence ----------------

def load_progress() -> dict:
    if not os.path.exists(PROGRESS_PATH):
        return {"completed": {}}
    try:
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"completed": {}}


def save_progress(progress: dict) -> None:
    try:
        parent = os.path.dirname(PROGRESS_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = PROGRESS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f, default=str)
        os.replace(tmp, PROGRESS_PATH)
    except Exception as e:
        log.warning("save_progress failed: %s", e)


# ---------------- Main verifier ----------------

def run_verifier(
    fast: bool = False,
    send_telegram_fn=None,
    print_progress=True,
) -> str:
    """Run the verifier end-to-end. Returns the formatted report string.

    fast=True: only top 50 highest-volume resolved markets from the last 2 months.
    send_telegram_fn: optional callable taking a string to push to Telegram.
    """
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        msg = "POLYVERIFY: ANTHROPIC_API_KEY not set or anthropic SDK missing"
        log.error(msg)
        if send_telegram_fn:
            send_telegram_fn(msg)
        return msg

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    progress = load_progress()
    completed_ids = set(progress.get("completed", {}).keys())

    if print_progress:
        log.info("Fetching resolved markets ...")
    target_count = 500 if not fast else 200
    raw = fetch_resolved_markets(target_count)
    if print_progress:
        log.info("Fetched %d markets", len(raw))

    # Filter
    now = datetime.now(timezone.utc)
    candidates = []
    for m in raw:
        if not is_binary_yes_no(m):
            continue
        # Gamma's closed-markets list doesn't include volume — fall back to a
        # real-trading proxy (lastTradePrice + spread) if the volume field is
        # missing or zero.
        vol = get_market_volume(m)
        if vol < MIN_VOLUME and not has_real_trading(m):
            continue
        resolution = get_resolution(m)
        if resolution is None:
            continue
        end_dt = get_close_dt(m)
        if end_dt is None:
            continue
        # Only last 6 months
        if (now - end_dt).days > 180:
            continue
        m["_resolution"] = resolution
        m["_end_dt"] = end_dt
        m["_volume"] = get_market_volume(m)
        candidates.append(m)

    if fast:
        # Top 50 by volume, last 2 months
        cutoff = now - timedelta(days=60)
        candidates = [m for m in candidates if m["_end_dt"] >= cutoff]
        candidates.sort(key=lambda x: -x["_volume"])
        candidates = candidates[:50]
    if print_progress:
        log.info("After filter: %d candidates", len(candidates))

    # Train/validate split based on end date
    opt_cutoff = now - timedelta(days=EDGE_OPT_MONTHS * 30)  # markets closed >4mo ago
    val_cutoff = now - timedelta(days=EDGE_VAL_MONTHS * 30)  # markets closed <2mo ago

    opt_results: list[dict] = []
    val_results: list[dict] = []
    opt_bankroll = STARTING_BANKROLL
    val_bankroll = STARTING_BANKROLL
    n_evaluated = 0
    n_skipped = 0

    for idx, market in enumerate(candidates):
        market_id = str(market.get("id") or market.get("conditionId") or "")
        if not market_id or market_id in completed_ids:
            # Restore from cached result
            cached = progress.get("completed", {}).get(market_id)
            if cached:
                bucket = "val" if cached.get("bucket") == "val" else "opt"
                if bucket == "opt":
                    opt_results.append(cached)
                else:
                    val_results.append(cached)
                if cached.get("bet_placed"):
                    if bucket == "opt":
                        opt_bankroll += cached.get("delta", 0)
                    else:
                        val_bankroll += cached.get("delta", 0)
                n_evaluated += 1
            continue

        end_dt = market["_end_dt"]
        resolution = market["_resolution"]
        question = market.get("question") or ""
        start_dt = parse_iso_dt(market.get("startDate")) or parse_iso_dt(market.get("createdAt"))
        if start_dt is None:
            n_skipped += 1
            continue

        bucket = "val" if end_dt >= val_cutoff else ("opt" if end_dt <= opt_cutoff else None)
        if bucket is None:
            # In the gap between opt and val — skip (avoids leakage)
            continue

        crowd_price = get_open_price(market)
        if crowd_price is None:
            n_skipped += 1
            continue

        # Call Claude
        estimate = estimate_probability(client, question, start_dt)
        time.sleep(RATE_LIMIT_SLEEP)

        record = {
            "id": market_id,
            "question": question,
            "end_date": end_dt.isoformat(),
            "open_date": start_dt.isoformat(),
            "crowd_price": crowd_price,
            "resolution": resolution,
            "bucket": bucket,
            "bet_placed": False,
            "delta": 0.0,
        }

        if estimate is None:
            record["skip_reason"] = "estimate_failed"
            n_skipped += 1
        else:
            record.update(estimate)
            if estimate["confidence"] == "low":
                record["skip_reason"] = "low_confidence"
                n_skipped += 1
            else:
                bankroll = opt_bankroll if bucket == "opt" else val_bankroll
                bet = kelly_bet(estimate["probability"], crowd_price, bankroll)
                if bet is None:
                    record["skip_reason"] = "edge_too_small"
                    n_skipped += 1
                else:
                    delta = settle_bet(bet, resolution, bankroll)
                    record["bet_placed"] = True
                    record["bet"] = bet
                    record["delta"] = delta
                    record["won"] = bet["side"] == resolution
                    if bucket == "opt":
                        opt_bankroll += delta
                        opt_results.append(record)
                    else:
                        val_bankroll += delta
                        val_results.append(record)

        if not record["bet_placed"]:
            if bucket == "opt":
                opt_results.append(record)
            else:
                val_results.append(record)

        progress.setdefault("completed", {})[market_id] = record
        n_evaluated += 1

        if n_evaluated % 10 == 0:
            save_progress(progress)
            if print_progress:
                log.info(
                    "Processed %d/%d  |  opt bets:%d val bets:%d  |  opt bankroll:$%.2f val bankroll:$%.2f",
                    n_evaluated, len(candidates),
                    sum(1 for r in opt_results if r.get("bet_placed")),
                    sum(1 for r in val_results if r.get("bet_placed")),
                    opt_bankroll, val_bankroll,
                )

    save_progress(progress)

    # ---------------- Report ----------------
    report = format_report(
        opt_results, val_results, opt_bankroll, val_bankroll,
        opt_cutoff, val_cutoff, now, n_evaluated, n_skipped,
    )
    if send_telegram_fn:
        send_telegram_fn(report)
    return report


def format_report(
    opt_results: list[dict],
    val_results: list[dict],
    opt_bankroll: float,
    val_bankroll: float,
    opt_cutoff: datetime,
    val_cutoff: datetime,
    now: datetime,
    n_evaluated: int,
    n_skipped: int,
) -> str:
    def summarize(results: list[dict], final_bankroll: float, label: str) -> str:
        bets = [r for r in results if r.get("bet_placed")]
        n = len(bets)
        if n == 0:
            return f"{label}\nMarkets bet: 0\nNo bets placed.\n"
        wins = sum(1 for r in bets if r.get("won"))
        wr = 100.0 * wins / n
        avg_edge = sum(r["bet"]["edge"] for r in bets) / n
        deltas = [r["delta"] for r in bets]
        ev_per_bet = sum(deltas) / n
        # Sharpe based on per-bet returns relative to stake
        returns = [r["delta"] / max(r["bet"]["stake"], 1e-9) for r in bets]
        if len(returns) > 1:
            mu = sum(returns) / len(returns)
            var = sum((x - mu) ** 2 for x in returns) / (len(returns) - 1)
            sharpe = mu / math.sqrt(var) * math.sqrt(len(returns)) if var > 0 else 0
        else:
            sharpe = 0
        return (
            f"{label}\n"
            f"Markets bet: {n}\n"
            f"Win rate: {wr:.1f}%\n"
            f"Avg edge per bet: {avg_edge*100:.1f}%\n"
            f"Expected value per bet: {'+' if ev_per_bet >= 0 else ''}{ev_per_bet:.3f}\n"
            f"Starting bankroll $100 → final ${final_bankroll:.2f}\n"
            f"Sharpe ratio: {sharpe:.2f}\n"
        )

    # Calibration
    val_bets = [r for r in val_results if r.get("bet_placed")]
    def bucket_wr(lo: float, hi: float) -> str:
        in_bucket = [
            r for r in val_bets
            if lo <= r.get("probability", 0) < hi
        ]
        if not in_bucket:
            return "(no bets in this band)"
        wins = sum(1 for r in in_bucket if r.get("won"))
        return f"actual win rate {100.0*wins/len(in_bucket):.0f}% (n={len(in_bucket)})"

    opt_window_str = f"closed before {opt_cutoff.date()}"
    val_window_str = f"closed since {val_cutoff.date()}"

    # Verdict
    val_bets_only = [r for r in val_results if r.get("bet_placed")]
    val_n = len(val_bets_only)
    val_wins = sum(1 for r in val_bets_only if r.get("won"))
    val_wr = (100.0 * val_wins / val_n) if val_n > 0 else 0
    val_ev = (sum(r["delta"] for r in val_bets_only) / val_n) if val_n > 0 else 0
    verdict = "PASS" if (val_wr > 55 and val_ev > 0 and val_n >= 30) else "FAIL"

    return (
        "POLYMARKET AI VERIFIER\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        f"Optimization window: {opt_window_str}\n"
        f"Validation window: {val_window_str}\n"
        f"Markets evaluated: {n_evaluated}\n"
        f"Markets skipped (low confidence or small edge): {n_skipped}\n\n"
        f"OPTIMIZATION RESULTS\n{summarize(opt_results, opt_bankroll, '')}\n"
        f"VALIDATION RESULTS\n{summarize(val_results, val_bankroll, '')}\n"
        f"CALIBRATION CHECK\n"
        f"When Claude says 70-80% probability: {bucket_wr(0.70, 0.80)}\n"
        f"When Claude says 60-70% probability: {bucket_wr(0.60, 0.70)}\n"
        f"When Claude says 50-60% probability: {bucket_wr(0.50, 0.60)}\n\n"
        f"VERDICT: {verdict}\n"
        f"Pass criteria: validation win rate above 55% AND positive expected value AND at least 30 markets bet in validation window\n"
        "━━━━━━━━━━━━━━━━━━━"
    )


# ---------------- Telegram integration entry points ----------------

_verifier_running = False
_verifier_lock = threading.Lock()


def kickoff_verifier_async(send_telegram_fn, fast: bool = False) -> str:
    """Start the verifier in a background thread. Returns the immediate ack message."""
    global _verifier_running
    with _verifier_lock:
        if _verifier_running:
            return ("/polyverify already running. Wait for it to finish before starting another.")
        _verifier_running = True

    eta = "10 minutes" if fast else "30 to 60 minutes"
    max_markets = 50 if fast else 500

    def _runner():
        global _verifier_running
        try:
            run_verifier(fast=fast, send_telegram_fn=send_telegram_fn, print_progress=True)
        except Exception as e:
            log.exception("polyverify run failed: %s", e)
            try:
                send_telegram_fn(f"⚠️ polyverify run failed: {e}")
            except Exception:
                pass
        finally:
            with _verifier_lock:
                _verifier_running = False

    threading.Thread(target=_runner, daemon=True).start()
    return (
        f"✅ polyverify started. Processing up to {max_markets} resolved markets. "
        f"Results in {eta}."
    )


if __name__ == "__main__":
    # CLI entry point: python3 polymarket_verifier.py [--fast]
    import sys
    fast_mode = "--fast" in sys.argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = run_verifier(fast=fast_mode, send_telegram_fn=None, print_progress=True)
    print(report)
