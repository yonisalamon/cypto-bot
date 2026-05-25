# Crypto Momentum Rotation Bot

Rotates into the top-2 momentum crypto assets using Alpaca paper trading.

## Strategy

| Parameter | Value |
|-----------|-------|
| Universe | BTC/USD, ETH/USD, SOL/USD, LINK/USD, UNI/USD |
| Scoring | 28-day price momentum |
| Filter | Asset must be above its 20-day SMA |
| Positions | Top 2 assets, equally weighted |
| Rebalance | Every 7 days |
| Check cadence | Every 24 hours |

Every 7 days the bot scores each asset by `(price_now − price_28d_ago) / price_28d_ago`, filters out anything trading below its 20-day moving average, selects the top 2, sells anything not in the selection, and buys/rebalances into the winners with equal capital allocation.

## Setup

### 1. Get Alpaca paper-trading keys

1. Sign up at <https://alpaca.markets>
2. In the dashboard switch to **Paper Trading**
3. Copy your **API Key** and **Secret Key**

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in ALPACA_API_KEY and ALPACA_SECRET_KEY
```

`.env` contents:
```
ALPACA_API_KEY=PKxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BASE_URL=https://paper-api.alpaca.markets
```

### 4. Run the bot

```bash
python bot.py
```

The bot runs in an infinite loop. Press `Ctrl+C` to stop.

To run it in the background and keep it alive across sessions:

```bash
# Using nohup
nohup python bot.py &

# Or with screen
screen -S crypto-bot
python bot.py
# Ctrl+A, D to detach
```

## Logs

All decisions and trades are written to **`trades.log`** in the project directory, and mirrored to stdout.

Example log lines:
```
2024-01-15 09:00:01 [INFO] SCORE  BTC/USD: price=$42,000.0000  28d_mom=+12.34%  20d_MA=$40,500.0000  above_MA=True
2024-01-15 09:00:01 [INFO] SCORE  ETH/USD: price=$2,200.0000  28d_mom=+8.10%  20d_MA=$2,100.0000  above_MA=True
2024-01-15 09:00:02 [INFO] ORDER SUBMITTED  BUY  BTCUSD  notional=$5,000.00
2024-01-15 09:00:02 [INFO] ORDER SUBMITTED  BUY  ETHUSD  notional=$5,000.00
```

## Files

```
crypto-bot/
├── bot.py            # Main bot
├── trades.log        # Generated at runtime
├── requirements.txt
├── .env              # Your credentials (never commit this)
└── .env.example      # Template
```

## Disclaimer

This is a paper-trading bot for educational purposes. It does not use real money. Past momentum is not a guarantee of future returns.
