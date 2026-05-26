#!/usr/bin/env python3
"""
Leveraged ETF EMA Momentum Bracket Bot

Scores TQQQ/SOXL/UPRO/TECL and their inverses using the gap between
the 8-day EMA and 21-day EMA, normalised by price.

  - Largest POSITIVE gap  → buy that ETF directly (trending up)
  - Largest NEGATIVE gap  → buy the paired inverse ETF (underlying trending down)

Whichever signal has the greater absolute magnitude wins.
Every entry is a bracket order (SL -10%, TP +25%).
The instant the bracket resolves, the bot rescores and enters the next trade.
One position at a time. NYSE market hours only.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = os.getenv("BASE_URL")

UNIVERSE = ["TQQQ", "SOXL", "UPRO", "TECL", "SQQQ", "SOXS", "SPXS", "TECS"]

# Each ETF paired with its inverse; used to find the buy target for a negative gap
PAIRS: dict[str, str] = {
    "TQQQ": "SQQQ", "SQQQ": "TQQQ",
    "SOXL": "SOXS", "SOXS": "SOXL",
    "UPRO": "SPXS", "SPXS": "UPRO",
    "TECL": "TECS", "TECS": "TECL",
}

EMA_FAST         = 8
EMA_SLOW         = 21
LOOKBACK_DAYS    = 90    # enough history for EMAs to converge
STOP_LOSS_PCT    = 0.10
TAKE_PROFIT_PCT  = 0.25
POLL_SECS        = 60    # seconds between position checks
FILL_WAIT_SECS   = 10    # wait after placing before first position check
MIN_CLOSE_MINS   = 15    # don't open a new position this close to market close

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
    if BASE_URL:
        trading = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, url_override=BASE_URL)
    else:
        trading = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
    data = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
    return trading, data

# ── Market clock helpers ──────────────────────────────────────────────────────

def wait_for_open(trading_client: TradingClient) -> None:
    """Block until NYSE is open, sleeping precisely until next_open."""
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

# ── EMA scoring ───────────────────────────────────────────────────────────────

def fetch_bars(data_client: StockHistoricalDataClient) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    req   = StockBarsRequest(
        symbol_or_symbols=UNIVERSE,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
    )
    return data_client.get_stock_bars(req).df


def compute_scores(data_client: StockHistoricalDataClient) -> dict[str, float]:
    """
    Return {symbol: ema_gap} for every symbol in UNIVERSE.

    ema_gap = (EMA_FAST - EMA_SLOW) / current_price

    Positive gap  → ETF is trending up relative to its slower average.
    Negative gap  → ETF is trending down.
    """
    logger.info("── Scoring ──────────────────────────────────────────────")
    try:
        bars_df = fetch_bars(data_client)
    except Exception as exc:
        logger.error(f"Bar fetch failed: {exc}")
        return {}

    scores: dict[str, float] = {}
    for symbol in UNIVERSE:
        try:
            lvl0 = bars_df.index.get_level_values(0)
            if symbol not in lvl0:
                logger.warning(f"SCORE  {symbol:<5}  no data returned")
                continue

            close = bars_df.loc[symbol].sort_index()["close"]
            if len(close) < EMA_SLOW + 5:
                logger.warning(
                    f"SCORE  {symbol:<5}  only {len(close)} bars "
                    f"(need ≥ {EMA_SLOW + 5})"
                )
                continue

            ema_fast  = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
            ema_slow  = float(close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1])
            price     = float(close.iloc[-1])
            gap       = (ema_fast - ema_slow) / price

            scores[symbol] = gap

            logger.info(
                f"SCORE  {symbol:<5}  price=${price:>8,.2f}  "
                f"EMA{EMA_FAST}=${ema_fast:>8,.2f}  EMA{EMA_SLOW}=${ema_slow:>8,.2f}  "
                f"gap={gap:+.4%}"
            )
        except Exception as exc:
            logger.error(f"SCORE  {symbol}: {exc}")

    return scores


def pick_trade(scores: dict[str, float]) -> tuple[str, float, str] | None:
    """
    Choose the single best trade from the scored ETFs.

    Two candidates are evaluated:
      1. ETF with the largest POSITIVE gap  → buy it directly.
      2. ETF with the largest NEGATIVE gap  → buy its paired inverse ETF.

    The candidate with the greater absolute signal wins.
    Returns (symbol_to_buy, abs_signal, reason) or None if no actionable signal.
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
            f"direct long on {best_pos_sym} (gap {best_pos_gap:+.4%})",
        ))

    if best_neg_gap < 0:
        inverse = PAIRS[best_neg_sym]
        candidates.append((
            inverse,
            abs(best_neg_gap),
            f"inverse play: {best_neg_sym} gap {best_neg_gap:+.4%} → buy {inverse}",
        ))

    if not candidates:
        return None

    # Strongest absolute signal wins
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[0]

# ── Bracket order ─────────────────────────────────────────────────────────────

def place_bracket(
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    symbol: str,
) -> bool:
    """Fetch live ask price, compute bracket levels, and submit. Returns True on success."""
    try:
        quote_req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quote     = data_client.get_stock_latest_quote(quote_req).get(symbol)
        if quote is None:
            logger.error(f"No quote available for {symbol}")
            return False

        ref_price    = float(quote.ask_price)
        stop_price   = round(ref_price * (1 - STOP_LOSS_PCT), 2)
        target_price = round(ref_price * (1 + TAKE_PROFIT_PCT), 2)

        equity = float(trading_client.get_account().equity)
        qty    = int(equity // ref_price)
        if qty < 1:
            logger.error(
                f"Insufficient equity (${equity:,.2f}) to buy 1 share of "
                f"{symbol} at ${ref_price:,.2f}"
            )
            return False

        logger.info(
            f"ENTRY    {symbol}  qty={qty}  ref=${ref_price:,.2f}  "
            f"stop=${stop_price:,.2f} (-{STOP_LOSS_PCT:.0%})  "
            f"target=${target_price:,.2f} (+{TAKE_PROFIT_PCT:.0%})"
        )

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=stop_price),
                take_profit=TakeProfitRequest(limit_price=target_price),
            )
        )

        logger.info(
            f"ORDER SUBMITTED  BUY {symbol}  qty={qty}  "
            f"SL=${stop_price:,.2f}  TP=${target_price:,.2f}"
        )
        return True

    except Exception as exc:
        logger.error(f"ORDER FAILED  {symbol}  —  {exc}")
        return False

# ── Pending order check ───────────────────────────────────────────────────────

def has_pending_orders(trading_client: TradingClient) -> bool:
    """Return True if any open orders exist (unfilled entry or live bracket legs)."""
    try:
        orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        return len(orders) > 0
    except Exception as exc:
        logger.warning(f"Could not check pending orders: {exc}")
        return False

# ── Portfolio snapshot ────────────────────────────────────────────────────────

def log_portfolio(trading_client: TradingClient) -> None:
    try:
        account   = trading_client.get_account()
        positions = trading_client.get_all_positions()
        logger.info(
            f"PORTFOLIO  equity=${float(account.equity):,.2f}  "
            f"cash=${float(account.cash):,.2f}"
        )
        if not positions:
            logger.info("  (no open positions)")
            return
        for p in positions:
            logger.info(
                f"  {p.symbol:<5}  qty={float(p.qty):.0f}  "
                f"value=${float(p.market_value):>10,.2f}  "
                f"unrealised_PnL=${float(p.unrealized_pl):>8,.2f} "
                f"({float(p.unrealized_plpc):.2%})"
            )
    except Exception as exc:
        logger.error(f"log_portfolio error: {exc}")

# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info("=" * 60)
    logger.info("Leveraged ETF EMA Momentum Bracket Bot  —  starting")
    logger.info(f"Universe        : {UNIVERSE}")
    logger.info(f"Signal          : (EMA{EMA_FAST} - EMA{EMA_SLOW}) / price")
    logger.info(f"Stop loss       : -{STOP_LOSS_PCT:.0%} from entry")
    logger.info(f"Take profit     : +{TAKE_PROFIT_PCT:.0%} from entry")
    logger.info(f"Position sizing : 1 position, fully allocated")
    logger.info("=" * 60)

    trading_client, data_client = init_clients()
    logger.info(f"Connected — equity=${float(trading_client.get_account().equity):,.2f}")

    # Detect a position left over from a previous session
    existing = trading_client.get_all_positions()
    in_trade  = len(existing) > 0
    if in_trade:
        p = existing[0]
        logger.info(
            f"Existing position: {p.symbol}  qty={float(p.qty):.0f}  "
            f"value=${float(p.market_value):,.2f} — monitoring bracket"
        )
    else:
        logger.info("No open position — will score and enter on next market open")

    while True:
        try:
            wait_for_open(trading_client)
            now       = datetime.now(timezone.utc)
            positions = trading_client.get_all_positions()

            # ── Monitoring an open trade ──────────────────────────────────────
            if in_trade:
                if len(positions) > 0:
                    logger.info(f"── Poll {now:%H:%M:%S} UTC ──")
                    log_portfolio(trading_client)
                    time.sleep(POLL_SECS)
                    continue

                # Position gone — confirm bracket legs have settled too
                if has_pending_orders(trading_client):
                    logger.info("Orders still settling — rechecking in 5 s…")
                    time.sleep(5)
                    continue

                logger.info("★ BRACKET RESOLVED — position closed, rescoring immediately")
                in_trade = False
                # fall through to scoring with no sleep

            # ── Looking for a new entry ───────────────────────────────────────

            # Refuse to open a new bracket close to the close bell
            remaining = mins_to_close(trading_client)
            if remaining < MIN_CLOSE_MINS:
                logger.info(
                    f"Only {remaining:.0f} min until close "
                    f"(< {MIN_CLOSE_MINS} min buffer) — skipping, waiting for next open"
                )
                time.sleep(int(remaining * 60) + 120)
                continue

            # Score and select
            scores = compute_scores(data_client)
            trade  = pick_trade(scores)

            if trade is None:
                logger.info("No actionable EMA signal — staying in cash, polling in 60 s")
                time.sleep(POLL_SECS)
                continue

            symbol, strength, reason = trade
            logger.info(f"WINNER: {symbol}  signal_strength={strength:.4%}  reason={reason}")

            # Final market-open check before submitting
            if not trading_client.get_clock().is_open:
                logger.info("Market just closed — will retry at next open")
                continue

            placed = place_bracket(trading_client, data_client, symbol)
            if placed:
                in_trade = True
                logger.info(f"Waiting {FILL_WAIT_SECS} s for entry fill…")
                time.sleep(FILL_WAIT_SECS)
            else:
                logger.warning("Placement failed — retrying in 60 s")
                time.sleep(60)

        except Exception:
            logger.error("Unhandled exception in main loop:")
            logger.error(traceback.format_exc())
            logger.info("Restarting in 60 seconds…")
            time.sleep(60)


if __name__ == "__main__":
    run()
