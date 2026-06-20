#!/usr/bin/env python3
"""
build_site_data.py
──────────────────
Parses all events embedded in 8zz-indicator.pine, computes win/loss
outcomes using per-event tickers (from the evt_ticker array) via yfinance,
and writes docs/events.json for the GitHub Pages dashboard.

Two exit modes are computed in parallel:
  • Mode A – Fixed 14 trading days after entry.
  • Mode B – Flip-based exit: holds until the next direction flip fires
             (or today if still open).

Ticker assignment:
  Each event carries its own ticker from the pine script's evt_ticker array.
  Empty ticker → fallback to FALLBACK_TICKER (0050.TW).

Usage:
  python scripts/build_site_data.py
"""

import json
import math
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PINE_FILE = ROOT / "8zz-indicator.pine"
OUTPUT_FILE = ROOT / "docs" / "events.json"

FALLBACK_TICKER = "0050.TW"
# Number of K bars for Mode A fixed-bar exit (uses 30m bars when available, 1h next, daily fallback)
# 17 × 30m bars ≈ 8.5 h ≈ 2 Taiwan trading sessions — highest win rate from sensitivity analysis
MODE_A_HOLD_BARS = 17
# Range of hold_bars to test for sensitivity analysis (starts at 14, removes low-performing rows)
SENSITIVITY_RANGE = range(14, 51)  # 14 to 50 inclusive


# ── Helpers ───────────────────────────────────────────────────────────────────
def stars(strength: int) -> str:
    return {1: "★☆☆", 2: "★★☆", 3: "★★★"}.get(strength, "?")


def parse_events(pine_text: str) -> list[dict]:
    """
    Extract all (time_ms, dir, strength, tooltip, ticker) tuples from the
    .pine file in declaration order.
    """
    # Match every quintet of array.push calls (evt_ticker added as 5th)
    pattern = re.compile(
        r"array\.push\(evt_time,\s*(\d+)\)\s*"
        r"array\.push\(evt_dir,\s*(-?\d+)\)\s*"
        r"array\.push\(evt_str,\s*(\d+)\)\s*"
        r'array\.push\(evt_tips,\s*"((?:[^"\\]|\\.)*)"\)\s*'
        r'array\.push\(evt_ticker,\s*"([^"]*)"\)',
        re.DOTALL,
    )
    events = []
    for m in pattern.finditer(pine_text):
        unix_ms = int(m.group(1))
        direction = int(m.group(2))
        if direction == 0:
            continue  # skip observation/neutral entries (non-trade signals)
        strength = int(m.group(3))
        tooltip = m.group(4).replace("\\n", "\n")
        raw_ticker = m.group(5).strip()
        ticker = raw_ticker if raw_ticker else FALLBACK_TICKER
        dt = datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)
        events.append(
            {
                "unix_ms": unix_ms,
                "date": dt.strftime("%Y-%m-%d"),
                "time_utc": dt.isoformat(),
                "direction": direction,
                "dir_label": "偏多 ▲" if direction == 1 else "偏空 ▼",
                "strength": strength,
                "strength_label": stars(strength),
                "tooltip": tooltip,
                "ticker": ticker,
                "ticker_is_fallback": not raw_ticker,
                "is_flip": False,  # set below
            }
        )
    return events


def mark_flips(events: list[dict]) -> list[dict]:
    """
    Walk events in order; mark an event as a flip when:
      1. Its direction differs from the previous accepted direction, OR
      2. Its direction is the same, it carries an EXPLICIT (non-fallback) ticker,
         and that ticker hasn't been seen yet in this direction run.
         This handles one FB post naming multiple tickers (e.g. 台光電 + 亞翔).

    Events whose ticker is the fallback (0050.TW) are NOT used to open
    parallel flips — they are supplementary observations only.
    """
    current_dir = 0
    current_tickers: set[str] = set()
    for evt in events:
        ticker = evt.get("ticker", "")
        is_fallback = evt.get("ticker_is_fallback", False)

        if evt["direction"] != current_dir:
            evt["is_flip"] = True
            current_dir = evt["direction"]
            current_tickers = {ticker}
        elif not is_fallback and ticker not in current_tickers:
            # Same direction, explicit new ticker → parallel trade on the same signal
            evt["is_flip"] = True
            current_tickers.add(ticker)
    return events


def _store_bar(dt_utc: datetime, close: float, prices: dict[str, float], fmt: str) -> None:
    """Store a bar's close price under the bar's opening-time key and its daily key."""
    prices[dt_utc.strftime(fmt)] = close
    prices[dt_utc.strftime("%Y-%m-%d")] = close  # last bar of day wins → daily close


def _normalize_bar_key(key: str | None) -> str | None:
    """Normalize display keys so intraday timestamps always include minutes."""
    if key is None:
        return None
    if len(key) == 13 and "T" in key:  # 1h key: YYYY-MM-DDTHH
        return key + ":00"
    return key


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _to_utc(idx) -> datetime:
    if hasattr(idx, "tzinfo") and idx.tzinfo is not None:
        return idx.astimezone(timezone.utc)
    return idx.replace(tzinfo=timezone.utc)


def _download_batch(tickers: list[str], start: datetime, end: datetime, interval: str | None = None):
    if not tickers:
        return None
    kwargs = {
        "tickers": tickers,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "auto_adjust": True,
        "progress": False,
        "threads": False,
        "group_by": "ticker",
    }
    if interval:
        kwargs["interval"] = interval
    return yf.download(**kwargs)


def _ticker_frame(df, ticker: str):
    if df is None or getattr(df, "empty", True):
        return None
    columns = getattr(df, "columns", None)
    if columns is None:
        return None

    if getattr(columns, "nlevels", 1) > 1:
        top_level = set(columns.get_level_values(0))
        if ticker not in top_level:
            return None
        sub = df[ticker]
    else:
        sub = df

    if getattr(sub, "empty", True) or "Close" not in sub:
        return None
    return sub


def _populate_30m_prices(all_prices: dict[str, dict[str, float]], tickers: list[str], start: datetime, end: datetime) -> None:
    if not tickers:
        return
    try:
        df = _download_batch(tickers, start, end, interval="30m")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: yfinance 30m batch error: {exc}")
        return

    for ticker in tickers:
        sub = _ticker_frame(df, ticker)
        if sub is None:
            continue
        prices = all_prices.setdefault(ticker, {})
        for idx, row in sub.iterrows():
            close = _finite_float(row.get("Close"))
            if close is None:
                continue
            _store_bar(_to_utc(idx), close, prices, "%Y-%m-%dT%H:%M")


def _populate_1h_prices(all_prices: dict[str, dict[str, float]], tickers: list[str], start: datetime, end: datetime) -> None:
    if not tickers:
        return
    try:
        df = _download_batch(tickers, start, end, interval="1h")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: yfinance 1h batch error: {exc}")
        return

    from statistics import median as _median

    for ticker in tickers:
        sub = _ticker_frame(df, ticker)
        if sub is None:
            continue

        prices = all_prices.setdefault(ticker, {})
        day_bars: dict[str, list[tuple[datetime, float]]] = {}
        for idx, row in sub.iterrows():
            close = _finite_float(row.get("Close"))
            if close is None:
                continue
            dt_utc = _to_utc(idx)
            day_bars.setdefault(dt_utc.strftime("%Y-%m-%d"), []).append((dt_utc, close))

        for d_key, bars in day_bars.items():
            closes = [c for _, c in bars]
            med = _median(closes)
            for dt_utc, close in sorted(bars, key=lambda x: x[0]):
                if len(bars) >= 3 and med and abs(close - med) / med > 0.07:
                    continue
                h_key = dt_utc.strftime("%Y-%m-%dT%H:%M")
                if h_key not in prices:
                    prices[h_key] = close
                if d_key not in prices:
                    prices[d_key] = close


def _populate_daily_prices(all_prices: dict[str, dict[str, float]], tickers: list[str], start: datetime, end: datetime) -> None:
    if not tickers:
        return
    try:
        df = _download_batch(tickers, start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: yfinance daily batch error: {exc}")
        return

    for ticker in tickers:
        sub = _ticker_frame(df, ticker)
        if sub is None:
            continue
        prices = all_prices.setdefault(ticker, {})
        for idx, row in sub.iterrows():
            close = _finite_float(row.get("Close"))
            if close is None:
                continue
            prices[idx.strftime("%Y-%m-%d")] = close


def fetch_all_prices(events: list[dict], first_dt: datetime, last_dt: datetime) -> dict[str, dict[str, float]]:
    """
    Download price history for every unique ticker referenced by events.
    Uses batched yfinance requests to avoid rate limits from per-ticker downloads.
    Returns {ticker: {date_str: price}}.
    Falls back to FALLBACK_TICKER data when a ticker yields no data.
    """
    tickers = sorted({e["ticker"] for e in events} | {FALLBACK_TICKER})
    all_prices: dict[str, dict[str, float]] = {ticker: {} for ticker in tickers}
    end_padded = last_dt + timedelta(days=5)

    for ticker in tickers:
        print(f"  Queueing {ticker} …")

    thirty_m_start = max(first_dt, last_dt - timedelta(days=59))
    _populate_30m_prices(all_prices, tickers, thirty_m_start, end_padded)
    _populate_1h_prices(all_prices, tickers, first_dt, end_padded)

    missing_daily = [ticker for ticker, prices in all_prices.items() if not prices]
    if missing_daily:
        print(f"  Daily fallback for {len(missing_daily)} ticker(s) …")
        _populate_daily_prices(all_prices, missing_daily, first_dt, end_padded)

    fallback_prices = all_prices.get(FALLBACK_TICKER, {})
    if not fallback_prices and FALLBACK_TICKER in tickers:
        print(f"  Retrying fallback ticker {FALLBACK_TICKER} with daily download …")
        _populate_daily_prices(all_prices, [FALLBACK_TICKER], first_dt, end_padded)
        fallback_prices = all_prices.get(FALLBACK_TICKER, {})

    for ticker in tickers:
        if all_prices[ticker]:
            continue
        if fallback_prices:
            print(f"  WARNING: No data for {ticker}; using {FALLBACK_TICKER} as fallback.")
            all_prices[ticker] = fallback_prices
        else:
            print(f"  WARNING: No data for {ticker}; fallback ticker {FALLBACK_TICKER} also unavailable.")

    return all_prices


def nearest_close(prices: dict[str, float], target_dt: datetime) -> tuple[str | None, float | None]:
    """
    Find the most recent completed bar's close BEFORE target_dt.

    When someone posts on FB they have ALREADY traded, so the correct entry
    price is the close of the bar that COMPLETED just before the post time
    (the "previous bar" — one step back from the bar currently open).

    Search order (backward):
      1. 30m keys (16 chars): step back one 30m boundary, then scan back 8 h
      2. 1h keys  (13 chars): checked on exact-hour boundaries during step 1
      3. Daily keys (10 chars): fallback, searches backward up to 8 days
    """
    if target_dt.tzinfo is None:
        target_utc = target_dt.replace(tzinfo=timezone.utc)
    else:
        target_utc = target_dt.astimezone(timezone.utc)

    # Snap to the START of the 30m bar currently open at target time
    current_30m_start = target_utc.replace(
        minute=0 if target_utc.minute < 30 else 30,
        second=0,
        microsecond=0,
    )
    # "Previous bar" = bar that JUST closed, one step back
    prev_30m = current_30m_start - timedelta(minutes=30)

    # Scan backward from the previous bar (up to 16 extra steps = 8 h total)
    for step in range(17):
        t = prev_30m - timedelta(minutes=30 * step)
        k30 = t.strftime("%Y-%m-%dT%H:%M")
        if k30 in prices:
            return k30, prices[k30]
        # On exact hours, also try the 1h key
        if t.minute == 0:
            k1h = t.strftime("%Y-%m-%dT%H:%M")
            if k1h in prices:
                return k1h, prices[k1h]

    # Daily fallback — search FORWARD (same or next trading day's close).
    # For events older than 60 days we only have daily data; using the
    # same-day close (after the post's intraday move) is the correct
    # contrarian entry for those events.
    for delta in range(8):
        d = (target_utc + timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in prices:
            return d, prices[d]

    return None, None


def compute_outcomes_mode_b(events: list[dict], all_prices: dict[str, dict[str, float]]) -> list[dict]:
    """
    Mode B – Flip-based exit.
    Each flip event is open until the next flip with an OPPOSING direction fires
    (or today if still open).  Parallel flips (same direction, different ticker)
    stay open until the same reversal — they do NOT close each other.
    Fills: entry_date, entry_price, exit_date, exit_price, pnl_pct, outcome.
    """
    flips = [e for e in events if e["is_flip"]]
    today = datetime.now(timezone.utc)

    def find_exit_idx(i: int) -> int | None:
        """Return index of the next flip whose direction differs from flips[i]."""
        for j in range(i + 1, len(flips)):
            if flips[j]["direction"] != flips[i]["direction"]:
                return j
        return None

    for i, flip in enumerate(flips):
        exit_idx = find_exit_idx(i)
        is_last = exit_idx is None
        prices = all_prices.get(flip["ticker"], {})

        entry_dt = datetime.fromisoformat(flip["time_utc"])
        entry_key, entry_price = nearest_close(prices, entry_dt)

        if entry_key is None or entry_price is None:
            flip["entry_date"] = None
            flip["entry_price"] = None
            flip["exit_date"] = None
            flip["exit_price"] = None
            flip["pnl_pct"] = None
            flip["outcome"] = "open" if is_last else "unknown"
            continue

        flip["entry_date"] = _normalize_bar_key(entry_key)
        flip["entry_price"] = round(entry_price, 2)

        if not is_last:
            exit_dt = datetime.fromisoformat(flips[exit_idx]["time_utc"])
            is_open = False
        else:
            exit_dt = today
            is_open = True

        exit_key, exit_price = nearest_close(prices, exit_dt)

        if exit_key is None or exit_price is None:
            flip["exit_date"] = None
            flip["exit_price"] = None
            flip["pnl_pct"] = None
            flip["outcome"] = "open" if is_open else "unknown"
            continue

        flip["exit_date"] = _normalize_bar_key(exit_key)
        flip["exit_price"] = round(exit_price, 2)

        price_change_pct = (exit_price - entry_price) / entry_price * 100
        pnl_pct = flip["direction"] * price_change_pct
        flip["pnl_pct"] = round(pnl_pct, 2)

        if is_open:
            flip["outcome"] = "open"
        elif pnl_pct > 0:
            flip["outcome"] = "win"
        elif pnl_pct < 0 and flip["direction"] == -1:
            # Only bearish (short) signals count as a loss when wrong;
            # bullish (long) signals that fall slightly are treated as flat.
            flip["outcome"] = "loss"
        elif pnl_pct < 0 and flip["direction"] == 1:
            flip["outcome"] = "flat"
        else:
            flip["outcome"] = "flat"

    return events


def compute_outcomes_mode_a(
    events: list[dict],
    all_prices: dict[str, dict[str, float]],
    hold_bars: int = MODE_A_HOLD_BARS,
) -> list[dict]:
    """
    Mode A – Fixed trading-day exit.
    Each flip event exits exactly `hold_bars` trading days after entry
    (counting only days in the ticker's own price history).
    Fills: exit_date_a, exit_price_a, pnl_pct_a, outcome_a.
    Uses entry_date already set by Mode B; recomputes if missing.
    """
    flips = [e for e in events if e["is_flip"]]

    for flip in flips:
        prices = all_prices.get(flip["ticker"], {})

        # Resolve entry key (reuse Mode B result; intraday timestamps are normalized to HH:MM)
        entry_key = flip.get("entry_date")
        entry_price_val = flip.get("entry_price")

        if entry_key is None:
            entry_dt = datetime.fromisoformat(flip["time_utc"])
            entry_key, entry_price_val = nearest_close(prices, entry_dt)
            entry_key = _normalize_bar_key(entry_key)

        if entry_key is None or entry_price_val is None:
            flip["exit_date_a"] = None
            flip["exit_price_a"] = None
            flip["pnl_pct_a"] = None
            flip["outcome_a"] = "open"
            continue

        # Select sorted_bars matching the entry key's granularity:
        #   16 chars = 30m  ("YYYY-MM-DDTHH:MM")
        #   13 chars = 1h   ("YYYY-MM-DDTHH")
        #   10 chars = daily ("YYYY-MM-DD")
        key_len = len(entry_key)
        sorted_bars = sorted(k for k in prices.keys() if len(k) == key_len)

        if entry_key not in sorted_bars:
            flip["exit_date_a"] = None
            flip["exit_price_a"] = None
            flip["pnl_pct_a"] = None
            flip["outcome_a"] = "open"
            continue

        entry_price_val = float(entry_price_val)
        try:
            entry_idx = sorted_bars.index(entry_key)
        except ValueError:
            flip["exit_date_a"] = None
            flip["exit_price_a"] = None
            flip["pnl_pct_a"] = None
            flip["outcome_a"] = "open"
            continue

        exit_idx = entry_idx + hold_bars
        if exit_idx >= len(sorted_bars):
            # Not enough K bars yet → position still open
            flip["exit_date_a"] = None
            flip["exit_price_a"] = None
            flip["pnl_pct_a"] = None
            flip["outcome_a"] = "open"
            continue

        exit_bar_a = sorted_bars[exit_idx]
        exit_price_a = prices[exit_bar_a]

        flip["exit_date_a"] = _normalize_bar_key(exit_bar_a)
        flip["exit_price_a"] = round(exit_price_a, 2)

        price_change_pct = (exit_price_a - entry_price_val) / entry_price_val * 100
        pnl_pct_a = flip["direction"] * price_change_pct
        flip["pnl_pct_a"] = round(pnl_pct_a, 2)

        if pnl_pct_a > 0:
            flip["outcome_a"] = "win"
        elif pnl_pct_a < 0 and flip["direction"] == -1:
            flip["outcome_a"] = "loss"
        elif pnl_pct_a < 0 and flip["direction"] == 1:
            flip["outcome_a"] = "flat"
        else:
            flip["outcome_a"] = "flat"

    return events


def _build_mode_stats(
    flips: list[dict],
    outcome_key: str,
    pnl_key: str,
) -> dict:
    """Build win/loss/open stats for a single exit mode."""
    resolved = [f for f in flips if f.get(outcome_key) in ("win", "loss", "flat")]
    wins = [f for f in resolved if f[outcome_key] == "win"]
    losses = [f for f in resolved if f[outcome_key] == "loss"]
    open_flips = [f for f in flips if f.get(outcome_key) == "open"]

    # Denominator: wins + losses + open (flat = 0% or long-side miss, excluded from rate)
    denom = len(wins) + len(losses) + len(open_flips)
    win_rate = len(wins) / denom * 100 if denom else 0
    pnls = [p for f in resolved if (p := _finite_float(f.get(pnl_key))) is not None]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0

    bullish = [f for f in resolved if f["direction"] == 1]
    bearish = [f for f in resolved if f["direction"] == -1]
    bull_wins = sum(1 for f in bullish if f[outcome_key] == "win")
    bear_wins = sum(1 for f in bearish if f[outcome_key] == "win")

    strength_stats: dict[int, dict] = {}
    for s in (1, 2, 3):
        sg = [f for f in resolved if f["strength"] == s]
        sw = sum(1 for f in sg if f[outcome_key] == "win")
        strength_stats[s] = {
            "total": len(sg),
            "wins": sw,
            "win_rate": round(sw / len(sg) * 100, 1) if sg else 0,
        }

    return {
        "resolved_flips": len(resolved),
        "open_flips": len(open_flips),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_pnl_pct": round(avg_pnl, 2),
        "bullish_flips": len(bullish),
        "bullish_wins": bull_wins,
        "bullish_win_rate": round(bull_wins / len(bullish) * 100, 1) if bullish else 0,
        "bearish_flips": len(bearish),
        "bearish_wins": bear_wins,
        "bearish_win_rate": round(bear_wins / len(bearish) * 100, 1) if bearish else 0,
        "by_strength": strength_stats,
    }


def build_stats(events: list[dict]) -> dict:
    flips = [e for e in events if e["is_flip"]]

    mode_b = _build_mode_stats(flips, outcome_key="outcome", pnl_key="pnl_pct")
    mode_a = _build_mode_stats(flips, outcome_key="outcome_a", pnl_key="pnl_pct_a")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fallback_ticker": FALLBACK_TICKER,
        "mode_a_hold_bars": MODE_A_HOLD_BARS,
        "total_events": len(events),
        "total_flips": len(flips),
        # Top-level convenience fields mirror Mode B (flip-based) for backwards compat
        "resolved_flips": mode_b["resolved_flips"],
        "open_flips": mode_b["open_flips"],
        "wins": mode_b["wins"],
        "losses": mode_b["losses"],
        "win_rate": mode_b["win_rate"],
        "avg_pnl_pct": mode_b["avg_pnl_pct"],
        "bullish_flips": mode_b["bullish_flips"],
        "bullish_wins": mode_b["bullish_wins"],
        "bullish_win_rate": mode_b["bullish_win_rate"],
        "bearish_flips": mode_b["bearish_flips"],
        "bearish_wins": mode_b["bearish_wins"],
        "bearish_win_rate": mode_b["bearish_win_rate"],
        "by_strength": mode_b["by_strength"],
        # Per-mode breakdown
        "mode_b": mode_b,
        "mode_a": mode_a,
    }


def build_equity_curves(
    events: list[dict],
    all_prices: dict[str, dict[str, float]],
) -> dict:
    """
    Build cumulative equity curves (starting value = 100) for:
      • Mode B (flip-based exits)
      • Mode A (fixed trading-day exits)
      • 0050.TW buy-and-hold benchmark
    Returns:
        {
            "mode_b": [{"date": str, "value": float}, ...],
            "mode_a": [{"date": str, "value": float}, ...],
            "benchmark_0050": [{"date": str, "value": float}, ...],
        }
    """
    flips = [e for e in events if e["is_flip"]]
    first_entry: str | None = next(
        (f["entry_date"] for f in flips if f.get("entry_date")), None
    )
    # Normalize for benchmark comparison (YYYY-MM-DD)
    first_entry_date = first_entry[:10] if first_entry else None

    # ── Mode B ────────────────────────────────────────────────────────
    curve_b: list[dict] = []
    equity_b = 100.0
    if first_entry:
        curve_b.append({"date": first_entry, "value": round(equity_b, 2)})
    for flip in flips:
        outcome = flip.get("outcome")
        pnl = flip.get("pnl_pct")
        exit_d = flip.get("exit_date")
        if outcome in ("win", "loss", "flat") and pnl is not None and exit_d:
            equity_b *= 1 + pnl / 100
            curve_b.append({"date": exit_d, "value": round(equity_b, 2)})
        elif outcome == "open" and pnl is not None and exit_d:
            mtm = equity_b * (1 + pnl / 100)
            curve_b.append({"date": exit_d, "value": round(mtm, 2)})

    # ── Mode A ────────────────────────────────────────────────────────
    curve_a: list[dict] = []
    equity_a = 100.0
    if first_entry:
        curve_a.append({"date": first_entry, "value": round(equity_a, 2)})
    # Sort by exit timestamp so compounding is in chronological order
    mode_a_resolved = sorted(
        [f for f in flips if f.get("outcome_a") in ("win", "loss", "flat")
         and f.get("pnl_pct_a") is not None and f.get("exit_date_a")],
        key=lambda f: f["exit_date_a"],
    )
    for flip in mode_a_resolved:
        equity_a *= 1 + flip["pnl_pct_a"] / 100
        curve_a.append({"date": flip["exit_date_a"], "value": round(equity_a, 2)})

    # ── 0050 benchmark ────────────────────────────────────────────────
    curve_0050: list[dict] = []
    bench_prices = all_prices.get(FALLBACK_TICKER, {})
    start_date = first_entry_date   # YYYY-MM-DD for daily benchmark comparison
    if start_date and bench_prices:
        sorted_dates = sorted(d for d in bench_prices.keys() if len(d) == 10)
        start_idx = next(
            (i for i, d in enumerate(sorted_dates) if d >= start_date[:10]), None
        )
        if start_idx is not None:
            base = bench_prices[sorted_dates[start_idx]]
            for d in sorted_dates[start_idx:]:
                curve_0050.append(
                    {"date": d, "value": round(bench_prices[d] / base * 100, 2)}
                )

    return {"mode_b": curve_b, "mode_a": curve_a, "benchmark_0050": curve_0050}


def sensitivity_analysis(
    events: list[dict],
    all_prices: dict[str, dict[str, float]],
) -> list[dict]:
    """
    Test Mode A win rate and avg PnL for hold_bars in SENSITIVITY_RANGE.
    Returns a list sorted by hold_bars:
        [{"hold_bars": int, "win_rate": float, "avg_pnl_pct": float,
          "wins": int, "losses": int, "resolved": int}, ...]
    """
    results = []
    for bars in SENSITIVITY_RANGE:
        # Deep-copy flip fields so we don't clobber the real Mode A results
        import copy
        tmp_events = copy.deepcopy(events)
        tmp_events = compute_outcomes_mode_a(tmp_events, all_prices, hold_bars=bars)
        flips = [e for e in tmp_events if e["is_flip"]]
        s = _build_mode_stats(flips, outcome_key="outcome_a", pnl_key="pnl_pct_a")
        results.append({
            "hold_bars": bars,
            "win_rate": s["win_rate"],
            "avg_pnl_pct": s["avg_pnl_pct"],
            "wins": s["wins"],
            "losses": s["losses"],
            "resolved": s["resolved_flips"],
        })
    return results


def main() -> None:
    pine_text = PINE_FILE.read_text(encoding="utf-8")
    events = parse_events(pine_text)
    if not events:
        print("ERROR: No events found in the .pine file.")
        sys.exit(1)

    events = mark_flips(events)
    flip_count = sum(1 for e in events if e["is_flip"])
    print(f"Parsed {len(events)} events, {flip_count} flip signals.")

    unique_tickers = sorted({e["ticker"] for e in events})
    print(f"Unique tickers: {unique_tickers}")

    first_dt = datetime.fromisoformat(events[0]["time_utc"])
    last_dt = datetime.now(timezone.utc) + timedelta(days=1)
    print(f"Fetching price history {first_dt.date()} → {last_dt.date()} …")
    all_prices = fetch_all_prices(events, first_dt, last_dt)

    events = compute_outcomes_mode_b(events, all_prices)
    events = compute_outcomes_mode_a(events, all_prices, hold_bars=MODE_A_HOLD_BARS)
    stats = build_stats(events)

    existing_output = None
    if OUTPUT_FILE.exists():
        try:
            existing_output = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing_output = None

    existing_stats = (existing_output or {}).get("stats", {})
    if stats["resolved_flips"] == 0 and existing_stats.get("resolved_flips", 0) > 0:
        print(
            "WARNING: fresh market-data build produced zero resolved flips; "
            "preserving the existing docs/events.json to avoid overwriting good data."
        )
        return

    print("Running hold_bars sensitivity analysis …")
    sens = sensitivity_analysis(events, all_prices)
    positive = [r for r in sens if r["avg_pnl_pct"] > 0]
    best = max(positive if positive else sens, key=lambda r: r["avg_pnl_pct"]) if sens else None
    if best:
        print(
            f"   Best hold_bars = {best['hold_bars']}  "
            f"win_rate {best['win_rate']}%  avg {best['avg_pnl_pct']:+.2f}%"
        )

    equity_curves = build_equity_curves(events, all_prices)

    output = {
        "stats": stats,
        "events": events,
        "equity_curves": equity_curves,
        "sensitivity": sens,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mb = stats["mode_b"]
    ma = stats["mode_a"]
    print(
        f"✅ docs/events.json written\n"
        f"   Mode B (flip)  : {mb['wins']}W / {mb['losses']}L / {mb['open_flips']} open  "
        f"→ win rate {mb['win_rate']}%  avg {mb['avg_pnl_pct']:+.2f}%\n"
        f"   Mode A ({MODE_A_HOLD_BARS} bars): {ma['wins']}W / {ma['losses']}L / {ma['open_flips']} open  "
        f"→ win rate {ma['win_rate']}%  avg {ma['avg_pnl_pct']:+.2f}%"
    )


if __name__ == "__main__":
    main()
