import ccxt
import pandas as pd
import numpy as np
import os
import time

from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

print("BOT QUANTITATIVO AVVIATO")
print("CONNESSIONE A KRAKEN...")

exchange = ccxt.kraken({
    'apiKey': os.getenv("KRAKEN_API_KEY"),
    'secret': os.getenv("KRAKEN_SECRET"),
    'enableRateLimit': True
})

SYMBOL = 'BTC/EUR'

IN_POSITION = False
ENTRY_PRICE = 0
HIGHEST_PRICE = 0

TIMEFRAME = '15m'

TRAILING_STOP = 0.025
STOP_LOSS = 0.03

CHECK_INTERVAL = 30

def get_data():

    ohlcv = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe=TIMEFRAME,
        limit=300
    )

    df = pd.DataFrame(
        ohlcv,
        columns=[
            'time',
            'open',
            'high',
            'low',
            'close',
            'volume'
        ]
    )

    df['ema20'] = EMAIndicator(
        df['close'],
        window=20
    ).ema_indicator()

    df['ema50'] = EMAIndicator(
        df['close'],
        window=50
    ).ema_indicator()

    df['ema200'] = EMAIndicator(
        df['close'],
        window=200
    ).ema_indicator()

    df['rsi'] = RSIIndicator(
        df['close'],
        window=14
    ).rsi()

    atr = AverageTrueRange(
        df['high'],
        df['low'],
        df['close'],
        window=14
    )

    df['atr'] = atr.average_true_range()

    return df

def get_balance():

    balance = exchange.fetch_balance()

    eur = balance['total'].get('EUR', 0)
    btc = balance['total'].get('BTC', 0)

    return eur, btc

def calculate_edge(df):

    last = df.iloc[-1]

    edge = 0

    if last['ema20'] > last['ema50']:
        edge += 20

    if last['ema50'] > last['ema200']:
        edge += 30

    if 55 < last['rsi'] < 72:
        edge += 20

    volatility_score = min(last['atr'] * 1000, 30)

    edge += volatility_score

    return min(edge, 100)

def calculate_position_size(edge, balance):

    if edge > 85:
        return balance * 1.0

    elif edge > 70:
        return balance * 0.7

    elif edge > 60:
        return balance * 0.4

    elif edge > 50:
        return balance * 0.2

    return 0

def buy(price, eur_balance, edge):

    global IN_POSITION
    global ENTRY_PRICE
    global HIGHEST_PRICE

    amount_eur = calculate_position_size(
        edge,
        eur_balance
    )

    if amount_eur < 5:
        return

    btc_amount = amount_eur / price

    print("================================")
    print("ACQUISTO BTC")
    print(f"EDGE: {edge}")
    print(f"PREZZO: {price}")
    print(f"EURO INVESTITI: {amount_eur}")

    exchange.create_market_buy_order(
        SYMBOL,
        btc_amount
    )

    IN_POSITION = True
    ENTRY_PRICE = price
    HIGHEST_PRICE = price

def sell(price, btc_balance, reason):

    global IN_POSITION

    print("================================")
    print("VENDITA BTC")
    print(f"MOTIVO: {reason}")
    print(f"PREZZO: {price}")

    exchange.create_market_sell_order(
        SYMBOL,
        btc_balance
    )

    IN_POSITION = False

def strategy():

    global IN_POSITION
    global ENTRY_PRICE
    global HIGHEST_PRICE

    df = get_data()

    last = df.iloc[-1]

    price = last['close']

    eur_balance, btc_balance = get_balance()

    edge = calculate_edge(df)

    print("================================")
    print(f"PREZZO BTC: {price}")
    print(f"EDGE SCORE: {edge}")
    print(f"RSI: {last['rsi']}")
    print(f"EUR: {eur_balance}")
    print(f"BTC: {btc_balance}")

    if not IN_POSITION:

        if edge > 60:

            buy(
                price,
                eur_balance,
                edge
            )

    else:

        pnl = (
            price - ENTRY_PRICE
        ) / ENTRY_PRICE

        if price > HIGHEST_PRICE:
            HIGHEST_PRICE = price

        trailing_price = HIGHEST_PRICE * (
            1 - TRAILING_STOP
        )

        print(f"PNL: {pnl * 100:.2f}%")
        print(f"HIGHEST: {HIGHEST_PRICE}")
        print(f"TRAILING: {trailing_price}")

        if pnl <= -STOP_LOSS:

            sell(
                price,
                btc_balance,
                "STOP LOSS"
            )

        elif price < trailing_price:

            sell(
                price,
                btc_balance,
                "TRAILING STOP"
            )

while True:

    try:

        strategy()

    except Exception as e:

        print("ERRORE:")
        print(str(e))

    time.sleep(CHECK_INTERVAL)
