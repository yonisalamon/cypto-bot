#!/usr/bin/env python3
"""
Leveraged ETF Triple EMA Momentum Bot

Entry:  EMA8 > EMA21 > EMA50 (bullish triple alignment) → buy ETF directly
        EMA8 < EMA21 < EMA50 (bearish triple alignment) → buy paired inverse ETF
Signal: (EMA8 - EMA50) / price — largest absolute spread wins
Full equity, plain market order.

Exit (whichever triggers first):
  ema_reversal  — EMA8 crosses back below EMA21 on daily bars
  trailing_stop — price drops 5% from the high watermark since entry
  hard_stop     — price drops 10% from entry price (absolute safety net)
"""

import csv
import os
import sys
import time
import logging
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from alpaca.data.live import StockDataStream

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = os.getenv("BASE_URL") or None   # treat empty string same as unset

UNIVERSE = ["TQQQ", "SOXL", "UPRO", "TECL", "SQQQ", "SOXS", "SPXS", "TECS"]

PAIRS: dict[str, str] = {
    "TQQQ": "SQQQ", "SQQQ": "TQQQ",
    "SOXL": "SOXS", "SOXS": "SOXL",
    "UPRO": "SPXS", "SPXS": "UPRO",
    "TECL": "TECS", "TECS": "TECL",
}

EMA_FAST       = 8
EMA_MID        = 21
EMA_SLOW       = 50
LOOKBACK_DAYS  = 120   # ~84 trading days — EMA50 needs ≥ 55 bars to converge
HARD_STOP_PCT  = 0.10  # -10% from entry price
TRAIL_STOP_PCT = 0.05  # -5% from high watermark since entry
POLL_SECS      = 60
MIN_CLOSE_MINS = 15    # don't open a new position this close to market close

TRADE_LOG   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_log.csv")
CSV_HEADERS = [
    "timestamp", "symbol", "side", "entry_price", "exit_price",
    "pnl_dollar", "pnl_pct", "ema8", "ema21", "ema50", "ema_gap", "exit_reason",
]


@dataclass
class TradeRecord:
    symbol:         str
    order_id:       str
    entry_time:     datetime
    ema8:           float
    ema21:          float
    ema50:          float
    ema_gap:        float
    qty:            float           = 0.0
    entry_price:    float           = 0.0
    high_watermark: float           = 0.0
    exit_price:     float           = 0.0
    pnl_dollar:     float           = 0.0
    pnl_pct:        float           = 0.0
    exit_reason:    str             = ""
    resolved_time:  datetime | None = None


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("etf_bot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trades.log"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

logger = setup_logging()


# ── Client initialisation ─────────────────────────────────────────────────────

def init_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    paper   = BASE_URL is None or "paper" in BASE_URL.lower()
    trading = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=paper)
    data    = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
    return trading, data


# ── Market clock helpers ──────────────────────────────────────────────────────

def wait_for_open(trading_client: TradingClient) -> None:
    while True:
        clock = trading_client.get_clock()
        if clock.is_open:
            return
        now        = datetime.now(timezone.utc)
        sleep_secs = max(60, (clock.next_open - now).total_seconds() + 60)
        logger.info(
            f"Market closed — next open "
            f"{clock.next_open.strftime('%Y-%m-%d %H:%M %Z')} "
            f"({sleep_secs / 3600:.1f} h)"
        )
        time.sleep(sleep_secs)


def mins_to_close(trading_client: TradingClient) -> float:
    clock = trading_client.get_clock()
    return (clock.next_close - datetime.now(timezone.utc)).total_seconds() / 60


# ── Bar fetching ──────────────────────────────────────────────────────────────

def fetch_bars(data_client: StockHistoricalDataClient) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    req   = StockBarsRequest(
        symbol_or_symbols=UNIVERSE,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed="iex",
    )
    df = data_client.get_stock_bars(req).df
    # Drop today's incomplete bar — EMAs must be built on settled daily closes only.
    # Use positional level index to avoid SDK name dependency.
    # Match tz-awareness to avoid TypeError on comparison.
    ts    = df.index.get_level_values(1)
    today = pd.Timestamp.now(tz="UTC").normalize()
    if ts.tz is None:
        today = today.tz_localize(None)
    return df[ts < today]


# ── EMA scoring ───────────────────────────────────────────────────────────────

def compute_scores(
    data_client: StockHistoricalDataClient,
) -> tuple[dict[str, float], dict[str, dict]]:
    """
    Score ETFs by triple EMA alignment.

    Only symbols with full bullish (EMA8 > EMA21 > EMA50) or full bearish
    (EMA8 < EMA21 < EMA50) alignment are included in scores.

    scores      = {symbol: signed_gap}   (+ve = bullish, -ve = bearish)
    ema_details = {symbol: {ema8, ema21, ema50, ema_gap}}  (all symbols)
    """
    logger.info("── Scoring ──────────────────────────────────────────────")
    try:
        bars_df = fetch_bars(data_client)
    except Exception as exc:
        logger.error(f"Bar fetch failed: {exc}")
        return {}, {}

    scores:      dict[str, float] = {}
    ema_details: dict[str, dict]  = {}

    for symbol in UNIVERSE:
        try:
            lvl0 = bars_df.index.get_level_values(0)
            if symbol not in lvl0:
                logger.warning(f"SCORE  {symbol:<5}  no data")
                continue

            close = bars_df.loc[symbol].sort_index()["close"]
            if len(close) < EMA_SLOW + 5:
                logger.warning(
                    f"SCORE  {symbol:<5}  only {len(close)} bars "
                    f"(need ≥ {EMA_SLOW + 5})"
                )
                continue

            ema_fast = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
            ema_mid  = float(close.ewm(span=EMA_MID,  adjust=False).mean().iloc[-1])
            ema_slow = float(close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1])
            price    = float(close.iloc[-1])
            gap      = (ema_fast - ema_slow) / price  # composite spread across all three

            bullish = ema_fast > ema_mid > ema_slow
            bearish = ema_fast < ema_mid < ema_slow

            ema_details[symbol] = {
                "ema8": ema_fast, "ema21": ema_mid, "ema50": ema_slow, "ema_gap": gap,
            }

            if bullish or bearish:
                scores[symbol] = gap
                alignment = "▲ bullish" if bullish else "▼ bearish"
            else:
                alignment = "  neutral"

            logger.info(
                f"SCORE  {symbol:<5}  price=${price:>8,.2f}  "
                f"EMA{EMA_FAST}=${ema_fast:>8,.2f}  "
                f"EMA{EMA_MID}=${ema_mid:>8,.2f}  "
                f"EMA{EMA_SLOW}=${ema_slow:>8,.2f}  "
                f"gap={gap:+.4%}  {alignment}"
            )
        except Exception as exc:
            logger.error(f"SCORE  {symbol}: {exc}")

    return scores, ema_details


def pick_trade(scores: dict[str, float]) -> tuple[str, float, str] | None:
    """
    Choose the strongest aligned signal:
      - Largest positive gap → buy that ETF directly (bullish alignment)
      - Largest negative gap → buy its paired inverse ETF (bearish alignment)

    Returns (symbol_to_buy, abs_signal, reason) or None.
    """
    if not scores:
        return None

    best_pos_sym = max(scores, key=scores.__getitem__)
    best_pos_gap = scores[best_pos_sym]

    best_neg_sym = min(scores, key=scores.__getitem__)
    best_neg_gap = scores[best_neg_sym]

    candidates: list[tuple[str, float, str]] = []

    if best_pos_gap > 0:
        candidates.append((
            best_pos_sym,
            best_pos_gap,
            f"bullish triple alignment on {best_pos_sym} (gap {best_pos_gap:+.4%})",
        ))

    if best_neg_gap < 0:
        inverse = PAIRS[best_neg_sym]
        candidates.append((
            inverse,
            abs(best_neg_gap),
            f"bearish triple alignment on {best_neg_sym} (gap {best_neg_gap:+.4%}) → buy {inverse}",
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[0]


# ── Order helpers ─────────────────────────────────────────────────────────────

def get_current_price(data_client: StockHistoricalDataClient, symbol: str) -> float:
    try:
        quote = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        ).get(symbol)
        if quote is None:
            return 0.0
        try:
            return float(quote.ask_price or 0)
        except (TypeError, ValueError):
            return 0.0
    except Exception as exc:
        logger.warning(f"get_current_price({symbol}): {exc}")
        return 0.0


def place_entry(
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    symbol: str,
) -> tuple[str | None, float, float]:
    """
    Submit a plain market buy for `symbol` using full account equity.
    Returns (order_id, fill_price, fill_qty) or (None, 0.0, 0.0) on failure.
    """
    try:
        quote = data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        ).get(symbol)
        if quote is None:
            logger.error(f"No quote for {symbol}")
            return None, 0.0, 0.0

        ask = quote.ask_price
        try:
            ref_price = float(ask or 0)
        except (TypeError, ValueError):
            ref_price = 0.0
        if ref_price <= 0:
            logger.error(f"Invalid ask price for {symbol} ({ask!r})")
            return None, 0.0, 0.0

        equity = float(trading_client.get_account().equity)
        qty    = int(equity // ref_price)
        if qty < 1:
            logger.error(
                f"Insufficient equity (${equity:,.2f}) to buy 1 share of "
                f"{symbol} at ${ref_price:,.2f}"
            )
            return None, 0.0, 0.0

        logger.info(
            f"ENTRY    {symbol}  qty={qty}  ref=${ref_price:,.2f}  "
            f"hard_stop=${ref_price * (1 - HARD_STOP_PCT):,.2f} (-{HARD_STOP_PCT:.0%})  "
            f"trail={TRAIL_STOP_PCT:.0%} from watermark"
        )

        order    = trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        order_id = str(order.id)

        # Wait for fill confirmation (market orders on liquid ETFs fill in seconds)
        for attempt in range(10):
            filled = trading_client.get_order_by_id(order_id)
            if filled.status == OrderStatus.FILLED and filled.filled_avg_price:
                fill_price = float(filled.filled_avg_price)
                fill_qty   = float(filled.filled_qty or qty)
                logger.info(
                    f"ORDER FILLED  BUY {symbol}  qty={fill_qty:.0f}  "
                    f"fill=${fill_price:,.2f}"
                )
                return order_id, fill_price, fill_qty
            if attempt < 9:
                time.sleep(2)

        # Timed out waiting — use ref_price as best estimate
        logger.warning(f"Entry order {order_id} fill not confirmed in 20 s — using ref price")
        return order_id, ref_price, float(qty)

    except Exception as exc:
        logger.error(f"place_entry {symbol}: {exc}")
        return None, 0.0, 0.0


def close_position(trading_client: TradingClient, symbol: str) -> float:
    """
    Market-sell the entire position in `symbol`.
    Returns fill price, or 0.0 if fill not confirmed within 30 s.
    """
    try:
        order    = trading_client.close_position(symbol)
        order_id = str(order.id)
        for attempt in range(15):
            filled = trading_client.get_order_by_id(order_id)
            if filled.status == OrderStatus.FILLED and filled.filled_avg_price:
                fill_price = float(filled.filled_avg_price)
                logger.info(f"CLOSED  {symbol}  fill=${fill_price:,.2f}")
                return fill_price
            if attempt < 14:
                time.sleep(2)
        logger.warning(f"Close order {order_id} fill not confirmed in 30 s")
        return 0.0
    except Exception as exc:
        logger.error(f"close_position {symbol}: {exc}")
        return 0.0


# ── Stream exit handler ───────────────────────────────────────────────────────

def make_bar_handler(
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    shared: dict,
    lock: threading.Lock,
):
    """
    Returns an async 1-minute bar handler for Alpaca's real-time stream.
    Checks hard stop, trailing stop, and EMA reversal in priority order.
    All shared state access is protected by `lock`.
    """
    async def bar_handler(bar) -> None:
        with lock:
            if not shared["in_trade"] or shared["pending"] is None:
                return
            rec = shared["pending"]
            if bar.symbol != rec.symbol:
                return

            price = float(bar.close)

            # 1. Hard stop — absolute floor
            hard_floor = rec.entry_price * (1 - HARD_STOP_PCT)
            if price <= hard_floor:
                logger.warning(
                    f"HARD STOP  {rec.symbol}  price=${price:,.2f}  "
                    f"floor=${hard_floor:,.2f}  entry=${rec.entry_price:,.2f}"
                )
                exit_reason = "hard_stop"
            else:
                # 2. Trailing stop — locks in gains
                trail_floor = rec.high_watermark * (1 - TRAIL_STOP_PCT)
                if price <= trail_floor:
                    logger.warning(
                        f"TRAILING STOP  {rec.symbol}  price=${price:,.2f}  "
                        f"trail_floor=${trail_floor:,.2f}  watermark=${rec.high_watermark:,.2f}"
                    )
                    exit_reason = "trailing_stop"
                else:
                    exit_reason = None

            # 3. EMA reversal — only if stops not triggered
            if exit_reason is None:
                try:
                    end   = datetime.now(timezone.utc)
                    start = end - timedelta(hours=2)
                    req   = StockBarsRequest(
                        symbol_or_symbols=[rec.symbol],
                        timeframe=TimeFrame.Minute,
                        start=start,
                        end=end,
                        feed="iex",
                        limit=60,
                    )
                    bars_df  = data_client.get_stock_bars(req).df
                    lvl0     = bars_df.index.get_level_values(0)
                    if rec.symbol in lvl0:
                        close    = bars_df.loc[rec.symbol].sort_index()["close"]
                        ema_fast = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
                        ema_mid  = float(close.ewm(span=EMA_MID,  adjust=False).mean().iloc[-1])
                        crossed  = ema_fast < ema_mid
                        logger.info(
                            f"EMA check — EMA8={ema_fast:.2f}  EMA21={ema_mid:.2f}  crossed={crossed}"
                        )
                        if crossed:
                            logger.info(
                                f"EMA REVERSAL  {rec.symbol}  "
                                f"EMA{EMA_FAST}=${ema_fast:,.2f} < EMA{EMA_MID}=${ema_mid:,.2f}"
                            )
                            exit_reason = "ema_reversal"
                except Exception as exc:
                    logger.error(f"bar_handler EMA reversal failed: {exc}", exc_info=True)

            if exit_reason is not None:
                logger.info(f"EXIT TRIGGERED: {exit_reason}  {rec.symbol}")
                fill_price = close_position(trading_client, rec.symbol)
                if fill_price > 0:
                    rec.exit_price    = fill_price
                    rec.exit_reason   = exit_reason
                    rec.pnl_dollar    = (fill_price - rec.entry_price) * rec.qty
                    rec.pnl_pct       = (fill_price - rec.entry_price) / rec.entry_price
                    rec.resolved_time = datetime.now(timezone.utc)
                    log_trade_to_csv(rec)
                else:
                    logger.error(
                        "close_position returned 0 — fill unconfirmed; "
                        "manual check required"
                    )
                shared["in_trade"] = False
                shared["pending"]  = None

    return bar_handler


# ── Portfolio snapshot ────────────────────────────────────────────────────────

def log_portfolio(trading_client: TradingClient, rec: TradeRecord | None = None) -> None:
    try:
        account   = trading_client.get_account()
        positions = trading_client.get_all_positions()
        logger.info(
            f"PORTFOLIO  equity=${float(account.equity):,.2f}  "
            f"cash=${float(account.cash):,.2f}"
        )
        for p in positions:
            logger.info(
                f"  {p.symbol:<5}  qty={float(p.qty):.0f}  "
                f"value=${float(p.market_value):>10,.2f}  "
                f"unrealised_PnL=${float(p.unrealized_pl):>8,.2f} "
                f"({float(p.unrealized_plpc):.2%})"
            )
        if rec and rec.high_watermark > 0:
            logger.info(
                f"  watermark=${rec.high_watermark:,.2f}  "
                f"trail_floor=${rec.high_watermark * (1 - TRAIL_STOP_PCT):,.2f}  "
                f"hard_floor=${rec.entry_price * (1 - HARD_STOP_PCT):,.2f}"
            )
    except Exception as exc:
        logger.error(f"log_portfolio error: {exc}")


# ── Trade CSV logging ─────────────────────────────────────────────────────────

def log_trade_to_csv(rec: TradeRecord) -> None:
    write_header = not os.path.isfile(TRADE_LOG)
    try:
        with open(TRADE_LOG, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp":   rec.resolved_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol":      rec.symbol,
                "side":        "buy",
                "entry_price": round(rec.entry_price, 4),
                "exit_price":  round(rec.exit_price,  4),
                "pnl_dollar":  round(rec.pnl_dollar,  2),
                "pnl_pct":     round(rec.pnl_pct,     6),
                "ema8":        round(rec.ema8,         4),
                "ema21":       round(rec.ema21,        4),
                "ema50":       round(rec.ema50,        4),
                "ema_gap":     round(rec.ema_gap,      6),
                "exit_reason": rec.exit_reason,
            })
        logger.info(
            f"TRADE LOG  {rec.symbol}  {rec.exit_reason}  "
            f"entry=${rec.entry_price:.4f}  exit=${rec.exit_price:.4f}  "
            f"PnL=${rec.pnl_dollar:+.2f} ({rec.pnl_pct:+.2%})"
        )
    except Exception as exc:
        logger.error(f"log_trade_to_csv error: {exc}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info("=" * 60)
    logger.info("Leveraged ETF Triple EMA Momentum Bot  —  starting")
    logger.info(f"Universe      : {UNIVERSE}")
    logger.info(f"Signal        : EMA{EMA_FAST}/EMA{EMA_MID}/EMA{EMA_SLOW} triple alignment")
    logger.info(f"Hard stop     : -{HARD_STOP_PCT:.0%} from entry price")
    logger.info(f"Trailing stop : -{TRAIL_STOP_PCT:.0%} from high watermark")
    logger.info(f"EMA reversal  : EMA{EMA_FAST} crosses below EMA{EMA_MID} (stream)")
    logger.info("=" * 60)

    trading_client, data_client = init_clients()
    logger.info(f"Connected — equity=${float(trading_client.get_account().equity):,.2f}")

    state_lock = threading.Lock()
    shared: dict = {"in_trade": False, "pending": None}

    existing = trading_client.get_all_positions()
    if existing:
        p = existing[0]
        shared["in_trade"] = True
        logger.info(
            f"Existing position: {p.symbol}  qty={float(p.qty):.0f}  "
            f"value=${float(p.market_value):,.2f} — monitoring"
        )
    else:
        logger.info("No open position — will score and enter on next market open")

    stream = StockDataStream(API_KEY, SECRET_KEY, feed="iex")
    stream.subscribe_bars(
        make_bar_handler(trading_client, data_client, shared, state_lock),
        *UNIVERSE,
    )
    stream_thread = threading.Thread(target=stream.run, daemon=True)
    stream_thread.start()
    logger.info(f"Real-time stream started — subscribed to 1-min bars for {UNIVERSE}")

    while True:
        try:
            wait_for_open(trading_client)
            now       = datetime.now(timezone.utc)
            positions = trading_client.get_all_positions()

            with state_lock:
                in_trade = shared["in_trade"]
                pending  = shared["pending"]

            # Guard: place_entry can succeed but raise before in_trade=True is set.
            # If a position exists but in_trade is False, sync state to avoid double-entry.
            if not in_trade and len(positions) > 0:
                logger.warning("Open position found with in_trade=False — syncing state")
                with state_lock:
                    shared["in_trade"] = True
                    shared["pending"]  = None
                time.sleep(POLL_SECS)
                continue

            # ── Monitoring an open trade ──────────────────────────────────────
            if in_trade:
                if len(positions) == 0:
                    # Position closed by stream handler or externally
                    logger.info("Position no longer open — resetting to cash")
                    with state_lock:
                        shared["in_trade"] = False
                        shared["pending"]  = None
                    continue

                symbol        = pending.symbol if pending else positions[0].symbol
                current_price = get_current_price(data_client, symbol)

                if current_price > 0 and pending is not None:
                    with state_lock:
                        pending.high_watermark = max(pending.high_watermark, current_price)

                logger.info(
                    f"── Poll {now:%H:%M:%S} UTC  {symbol}  "
                    f"price=${current_price:,.2f} ──"
                )
                log_portfolio(trading_client, pending)

                time.sleep(POLL_SECS)
                continue

            # ── Looking for a new entry ───────────────────────────────────────

            remaining = mins_to_close(trading_client)
            if remaining < MIN_CLOSE_MINS:
                logger.info(
                    f"Only {remaining:.0f} min until close "
                    f"(< {MIN_CLOSE_MINS} min buffer) — waiting for next open"
                )
                time.sleep(int(remaining * 60) + 120)
                continue

            scores, ema_details = compute_scores(data_client)
            trade = pick_trade(scores)

            if trade is None:
                logger.info("No triple EMA alignment — staying in cash, retrying in 60 s")
                time.sleep(POLL_SECS)
                continue

            symbol, strength, reason = trade
            logger.info(f"WINNER: {symbol}  strength={strength:.4%}  reason={reason}")

            if not trading_client.get_clock().is_open:
                logger.info("Market just closed — waiting for next open")
                continue

            order_id, fill_price, fill_qty = place_entry(trading_client, data_client, symbol)
            if order_id and fill_price > 0:
                det         = ema_details.get(symbol, {})
                new_pending = TradeRecord(
                    symbol         = symbol,
                    order_id       = order_id,
                    entry_time     = datetime.now(timezone.utc),
                    ema8           = det.get("ema8",    0.0),
                    ema21          = det.get("ema21",   0.0),
                    ema50          = det.get("ema50",   0.0),
                    ema_gap        = det.get("ema_gap", 0.0),
                    qty            = fill_qty,
                    entry_price    = fill_price,
                    high_watermark = fill_price,
                )
                with state_lock:
                    shared["pending"]  = new_pending
                    shared["in_trade"] = True
                logger.info(
                    f"IN TRADE: {symbol}  entry=${fill_price:,.2f}  qty={fill_qty:.0f}  "
                    f"hard_floor=${fill_price * (1 - HARD_STOP_PCT):,.2f}  "
                    f"initial_trail=${fill_price * (1 - TRAIL_STOP_PCT):,.2f}"
                )
            else:
                logger.warning("Entry failed — retrying in 60 s")
                time.sleep(60)

        except Exception:
            logger.error("Unhandled exception in main loop:")
            logger.error(traceback.format_exc())
            logger.info("Restarting in 60 seconds…")
            time.sleep(60)


if __name__ == "__main__":
    run()
