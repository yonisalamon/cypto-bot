#!/usr/bin/env python3
"""
Stock Momentum Rotation Bot
Scores 12 large-cap stocks by 28-day momentum every 7 days,
holds the top 3 equally weighted. Every buy is a bracket order
with stop loss at -10% and take profit at +25% from entry.
Only trades during NYSE market hours.
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
    MarketOrderRequest, StopLossRequest, TakeProfitRequest,
    GetOrdersRequest,
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

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "V", "UNH", "XOM", "LLY",
]

REBALANCE_EVERY_DAYS = 7
MOMENTUM_DAYS        = 28
TOP_N                = 3
MIN_NOTIONAL         = 1.0
STOP_LOSS_PCT        = 0.10
TAKE_PROFIT_PCT      = 0.25

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("stock_bot")
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

def init_clients():
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    if BASE_URL:
        trading = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, url_override=BASE_URL)
    else:
        trading = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
    data = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
    return trading, data

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_bars(data_client: StockHistoricalDataClient) -> pd.DataFrame:
    """Return a MultiIndex (symbol, timestamp) DataFrame of daily closes."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=MOMENTUM_DAYS + 10)
    req   = StockBarsRequest(
        symbol_or_symbols=UNIVERSE,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="all",
    )
    return data_client.get_stock_bars(req).df

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_assets(bars_df: pd.DataFrame) -> dict[str, float]:
    """Return {symbol: 28-day momentum} for every symbol with enough history."""
    scores: dict[str, float] = {}
    for symbol in UNIVERSE:
        try:
            level0 = bars_df.index.get_level_values(0)
            if symbol not in level0:
                logger.warning(f"{symbol}: no data returned — skipping")
                continue
            df = bars_df.loc[symbol].sort_index()
            if len(df) < MOMENTUM_DAYS:
                logger.warning(f"{symbol}: only {len(df)} bars (need {MOMENTUM_DAYS}) — skipping")
                continue
            close         = df["close"]
            current_price = float(close.iloc[-1])
            past_price    = float(close.iloc[-MOMENTUM_DAYS])
            momentum      = (current_price - past_price) / past_price
            logger.info(f"SCORE  {symbol:<5}  price=${current_price:>9,.2f}  28d_mom={momentum:+.2%}")
            scores[symbol] = momentum
        except Exception as exc:
            logger.error(f"Error scoring {symbol}: {exc}")
    return scores

# ── Order helpers ─────────────────────────────────────────────────────────────

def _submit(trading_client: TradingClient, req: MarketOrderRequest, label: str):
    try:
        trading_client.submit_order(req)
        logger.info(f"ORDER SUBMITTED  {label}")
    except Exception as exc:
        logger.error(f"ORDER FAILED     {label}  —  {exc}")


def _cancel_open_orders(trading_client: TradingClient, symbol: str):
    """Cancel any open orders for a symbol (e.g. stale bracket legs)."""
    try:
        req    = GetOrdersRequest(symbols=[symbol], status=QueryOrderStatus.OPEN)
        orders = trading_client.get_orders(filter=req)
        for order in orders:
            trading_client.cancel_order_by_id(order.id)
            logger.info(f"CANCELLED order {order.id}  ({symbol})")
        if orders:
            time.sleep(2)
    except Exception as exc:
        logger.warning(f"Could not cancel orders for {symbol}: {exc}")

# ── Rebalancing ───────────────────────────────────────────────────────────────

def execute_rebalance(
    trading_client: TradingClient,
    data_client: StockHistoricalDataClient,
    top_symbols: list[str],
):
    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    exiting   = [sym for sym in positions if sym not in top_symbols]

    # ── Step 1: exit positions dropped from the top N ─────────────────────────
    for sym in exiting:
        _cancel_open_orders(trading_client, sym)
        pos = positions[sym]
        qty = float(pos.qty_available)
        if qty <= 0:
            logger.warning(f"SKIP SELL {sym}: no available qty")
            continue
        _submit(
            trading_client,
            MarketOrderRequest(
                symbol=sym, qty=qty,
                side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            ),
            f"SELL {sym}  qty={qty:.0f}  (dropped from top {TOP_N})",
        )

    if exiting:
        logger.info("Waiting 15 s for sell orders to settle…")
        time.sleep(15)

    # ── Step 2: target allocation ─────────────────────────────────────────────
    account     = trading_client.get_account()
    equity      = float(account.equity)
    target_each = equity / TOP_N
    logger.info(f"REBALANCE  equity=${equity:,.2f}  target_per_asset=${target_each:,.2f}")

    # ── Step 3: fetch live ask prices for bracket calculation ─────────────────
    quote_req     = StockLatestQuoteRequest(symbol_or_symbols=top_symbols)
    latest_quotes = data_client.get_stock_latest_quote(quote_req)

    # ── Step 4: buy / trim each target asset ─────────────────────────────────
    positions = {p.symbol: p for p in trading_client.get_all_positions()}

    for symbol in top_symbols:
        current_value = float(positions[symbol].market_value) if symbol in positions else 0.0
        delta         = target_each - current_value

        if delta >= MIN_NOTIONAL:
            quote = latest_quotes.get(symbol)
            if quote is None:
                logger.error(f"No quote for {symbol} — skipping bracket buy")
                continue
            ref_price    = float(quote.ask_price)
            qty          = int(delta // ref_price)   # whole shares only
            if qty < 1:
                logger.info(
                    f"SKIP {symbol}: ${delta:,.2f} buys < 1 share at ${ref_price:,.2f}"
                )
                continue
            stop_price   = round(ref_price * (1 - STOP_LOSS_PCT), 2)
            target_price = round(ref_price * (1 + TAKE_PROFIT_PCT), 2)
            logger.info(
                f"BRACKET  {symbol}  ref=${ref_price:,.2f}  "
                f"stop=${stop_price:,.2f} (-{STOP_LOSS_PCT:.0%})  "
                f"target=${target_price:,.2f} (+{TAKE_PROFIT_PCT:.0%})"
            )
            _submit(
                trading_client,
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(stop_price=stop_price),
                    take_profit=TakeProfitRequest(limit_price=target_price),
                ),
                f"BUY {symbol}  qty={qty}  ≈${qty * ref_price:,.2f}  "
                f"SL=${stop_price:,.2f}  TP=${target_price:,.2f}",
            )

        elif delta <= -MIN_NOTIONAL:
            pos           = positions[symbol]
            current_price = float(pos.current_price)
            qty_to_sell   = int(abs(delta) // current_price)
            if qty_to_sell < 1:
                logger.info(
                    f"HOLD {symbol}  overweight ${-delta:,.2f} but < 1 share — skipping trim"
                )
                continue
            _submit(
                trading_client,
                MarketOrderRequest(
                    symbol=symbol, qty=qty_to_sell,
                    side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                ),
                f"TRIM {symbol}  qty={qty_to_sell}  (reduce overweight ${-delta:,.2f})",
            )

        else:
            logger.info(f"HOLD {symbol}  current=${current_value:,.2f}  already at target")

# ── Portfolio snapshot ────────────────────────────────────────────────────────

def log_portfolio(trading_client: TradingClient):
    account   = trading_client.get_account()
    positions = trading_client.get_all_positions()
    logger.info(
        f"PORTFOLIO  equity=${float(account.equity):,.2f}  "
        f"cash=${float(account.cash):,.2f}"
    )
    if not positions:
        logger.info("  (no open positions — fully in cash)")
        return
    for p in positions:
        logger.info(
            f"  {p.symbol:<5}  qty={float(p.qty):.0f}  "
            f"value=${float(p.market_value):>10,.2f}  "
            f"unrealised_PnL=${float(p.unrealized_pl):>8,.2f} "
            f"({float(p.unrealized_plpc):.2%})"
        )

# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 60)
    logger.info("Stock Momentum Rotation Bot  —  starting")
    logger.info(f"Universe         : {UNIVERSE}")
    logger.info(f"Rebalance every  : {REBALANCE_EVERY_DAYS} days")
    logger.info(f"Momentum window  : {MOMENTUM_DAYS} days")
    logger.info(f"Positions held   : top {TOP_N}")
    logger.info(f"Stop loss        : -{STOP_LOSS_PCT:.0%} from entry (bracket order)")
    logger.info(f"Take profit      : +{TAKE_PROFIT_PCT:.0%} from entry (bracket order)")
    logger.info("=" * 60)

    trading_client, data_client = init_clients()
    account = trading_client.get_account()
    logger.info(f"Connected  —  account equity=${float(account.equity):,.2f}")

    last_rebalance: datetime | None = None

    while True:
        try:
            now   = datetime.now(timezone.utc)
            clock = trading_client.get_clock()

            if not clock.is_open:
                next_open  = clock.next_open
                sleep_secs = max(60, (next_open - now).total_seconds() + 60)
                logger.info(
                    f"Market closed — sleeping until "
                    f"{next_open.strftime('%Y-%m-%d %H:%M %Z')} "
                    f"({sleep_secs / 3600:.1f} h)"
                )
                time.sleep(sleep_secs)
                continue

            logger.info(f"── Check at {now:%Y-%m-%d %H:%M:%S} UTC ──")

            days_since = (now - last_rebalance).days if last_rebalance else REBALANCE_EVERY_DAYS
            due        = days_since >= REBALANCE_EVERY_DAYS

            if due:
                logger.info("Rebalance due — fetching market data…")
                bars_df = fetch_bars(data_client)
                scores  = score_assets(bars_df)

                if not scores:
                    logger.warning("No scored assets — holding current positions, skipping rebalance")
                else:
                    ranked      = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_symbols = [sym for sym, _ in ranked[:TOP_N]]

                    logger.info("Momentum rank:")
                    for i, (sym, mom) in enumerate(ranked, 1):
                        tag = " ← SELECTED" if sym in top_symbols else ""
                        logger.info(f"  {i:2d}. {sym:<5}  {mom:+.2%}{tag}")

                    execute_rebalance(trading_client, data_client, top_symbols)
                    last_rebalance = now
                    logger.info(f"Rebalance complete — next in {REBALANCE_EVERY_DAYS} days")
            else:
                days_left = REBALANCE_EVERY_DAYS - days_since
                logger.info(f"No rebalance needed — next in {days_left} day(s)")

            log_portfolio(trading_client)

        except Exception:
            logger.error("Unhandled exception in main loop:")
            logger.error(traceback.format_exc())
            logger.info("Restarting loop in 60 seconds…")
            time.sleep(60)
            continue

        logger.info("Sleeping 1 h until next check…")
        time.sleep(3600)


if __name__ == "__main__":
    run()
