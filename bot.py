#!/usr/bin/env python3
"""
Crypto Momentum Rotation Bot
Scores BTC, ETH, SOL, LINK, UNI by 28-day momentum every 7 days,
holds the top 2 with positive momentum equally weighted.
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
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD", "UNI/USD"]
# Alpaca trading orders use no-slash symbols (BTCUSD, ETHUSD, …)
TRADE_SYM = {s: s.replace("/", "") for s in UNIVERSE}

REBALANCE_EVERY_DAYS = 7
CHECK_EVERY_HOURS    = 1
MOMENTUM_DAYS        = 28
MA_PERIOD            = 20
TOP_N                = 2
MIN_NOTIONAL         = 1.0   # dollars — Alpaca minimum order size

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("crypto_bot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler("trades.log")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

logger = setup_logging()

# ── Client initialisation ─────────────────────────────────────────────────────

def init_clients():
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
    trading = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)
    data    = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
    return trading, data

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_bars(data_client: CryptoHistoricalDataClient) -> pd.DataFrame:
    """Return a multi-index DataFrame (symbol, timestamp) of daily closes."""
    end   = datetime.now(timezone.utc)
    # Extra buffer so we always have enough bars after any gaps
    start = end - timedelta(days=MOMENTUM_DAYS + MA_PERIOD + 10)

    req  = CryptoBarsRequest(
        symbol_or_symbols=list(UNIVERSE),
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = data_client.get_crypto_bars(req)
    return bars.df  # MultiIndex: (symbol, timestamp)

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_assets(bars_df: pd.DataFrame) -> dict[str, float]:
    """Returns {symbol: momentum} for assets with positive momentum and enough history."""
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

            logger.info(
                f"SCORE  {symbol}: price=${current_price:,.4f}  "
                f"28d_mom={momentum:+.2%}"
            )

            if momentum > 0:
                scores[symbol] = momentum

        except Exception as exc:
            logger.error(f"Error scoring {symbol}: {exc}")

    return scores

# ── Rebalancing ───────────────────────────────────────────────────────────────

def _submit(trading_client: TradingClient, req: MarketOrderRequest, label: str):
    try:
        trading_client.submit_order(req)
        logger.info(f"ORDER SUBMITTED  {label}")
    except Exception as exc:
        logger.error(f"ORDER FAILED     {label}  —  {exc}")


def execute_rebalance(trading_client: TradingClient, top_symbols: list[str]):
    top_trade_syms = {TRADE_SYM[s] for s in top_symbols}

    # ── Step 1: exit positions not in target ─────────────────────────────────
    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    exiting   = [sym for sym in positions if sym not in top_trade_syms]

    for sym in exiting:
        pos = positions[sym]
        qty = float(pos.qty_available)
        if qty <= 0:
            logger.warning(f"SKIP SELL {sym}: no available qty")
            continue
        _submit(
            trading_client,
            MarketOrderRequest(symbol=sym, qty=qty,
                               side=OrderSide.SELL, time_in_force=TimeInForce.GTC),
            f"SELL {sym}  qty={qty:.8f}  (exiting — not in top {TOP_N})",
        )

    if exiting:
        logger.info("Waiting 15 s for sell orders to settle…")
        time.sleep(15)

    # ── Step 2: calculate target allocation ──────────────────────────────────
    account        = trading_client.get_account()
    equity         = float(account.equity)
    target_each    = equity / TOP_N

    logger.info(f"REBALANCE  equity=${equity:,.2f}  target_per_asset=${target_each:,.2f}")

    # ── Step 3: buy / trim each target asset ─────────────────────────────────
    positions = {p.symbol: p for p in trading_client.get_all_positions()}

    for symbol in top_symbols:
        tsym          = TRADE_SYM[symbol]
        current_value = float(positions[tsym].market_value) if tsym in positions else 0.0
        delta         = target_each - current_value

        if delta >= MIN_NOTIONAL:
            _submit(
                trading_client,
                MarketOrderRequest(symbol=tsym, notional=round(delta, 2),
                                   side=OrderSide.BUY, time_in_force=TimeInForce.GTC),
                f"BUY  {tsym}  notional=${delta:,.2f}",
            )
        elif delta <= -MIN_NOTIONAL:
            # Overweight — trim by selling the excess qty
            pos           = positions[tsym]
            current_price = float(pos.current_price)
            qty_to_sell   = abs(delta) / current_price
            _submit(
                trading_client,
                MarketOrderRequest(symbol=tsym, qty=round(qty_to_sell, 8),
                                   side=OrderSide.SELL, time_in_force=TimeInForce.GTC),
                f"TRIM {tsym}  qty={qty_to_sell:.8f}  (reduce overweight ${-delta:,.2f})",
            )
        else:
            logger.info(f"HOLD {tsym}  current=${current_value:,.2f}  already at target")

# ── Portfolio snapshot ────────────────────────────────────────────────────────

def log_portfolio(trading_client: TradingClient):
    account   = trading_client.get_account()
    positions = trading_client.get_all_positions()

    logger.info(f"PORTFOLIO  equity=${float(account.equity):,.2f}  "
                f"cash=${float(account.cash):,.2f}")

    if not positions:
        logger.info("  (no open positions — fully in cash)")
        return

    for p in positions:
        logger.info(
            f"  {p.symbol}  qty={float(p.qty):.6f}  "
            f"value=${float(p.market_value):,.2f}  "
            f"unrealised_PnL=${float(p.unrealized_pl):,.2f} "
            f"({float(p.unrealized_plpc):.2%})"
        )

# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 60)
    logger.info("Crypto Momentum Rotation Bot  —  starting")
    logger.info(f"Universe         : {UNIVERSE}")
    logger.info(f"Rebalance every  : {REBALANCE_EVERY_DAYS} days")
    logger.info(f"Momentum window  : {MOMENTUM_DAYS} days")
    logger.info(f"Positions held   : top {TOP_N}")
    logger.info("=" * 60)

    trading_client, data_client = init_clients()

    account = trading_client.get_account()
    logger.info(f"Connected  —  account equity=${float(account.equity):,.2f}")

    last_rebalance: datetime | None = None

    while True:
        try:
            now = datetime.now(timezone.utc)
            logger.info(f"── Check at {now:%Y-%m-%d %H:%M:%S} UTC ──")

            days_since = (now - last_rebalance).days if last_rebalance else REBALANCE_EVERY_DAYS
            due        = days_since >= REBALANCE_EVERY_DAYS

            if due:
                logger.info("Rebalance due — fetching market data…")
                bars_df = fetch_bars(data_client)
                scores  = score_assets(bars_df)

                if not scores:
                    logger.warning("No assets with positive momentum — holding cash, skipping rebalance")
                else:
                    ranked       = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_symbols  = [sym for sym, _ in ranked[:TOP_N]]

                    logger.info("Momentum rank:")
                    for i, (sym, mom) in enumerate(ranked, 1):
                        tag = " ← SELECTED" if sym in top_symbols else ""
                        logger.info(f"  {i}. {sym}  {mom:+.2%}{tag}")

                    execute_rebalance(trading_client, top_symbols)
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

        logger.info(f"Sleeping {CHECK_EVERY_HOURS}h until next check…")
        time.sleep(CHECK_EVERY_HOURS * 3600)


if __name__ == "__main__":
    run()
