print("BOT PARTITO")
print("CONNESSIONE A KRAKEN...")
print("BOT IN ESECUZIONE...")
import os
import time
import ccxt
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_SECRET = os.getenv("KRAKEN_SECRET")

exchange = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_SECRET,
    'enableRateLimit': True,
})
balance = exchange.fethBalance()
print (balance)

SYMBOL = 'BTC/EUR'

IN_POSITION = False
ENTRY_PRICE = 0

def get_data():

    ohlcv = exchange.fetch_ohlcv(
        SYMBOL,
        timeframe='15m',
        limit=250
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

    df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['ema200'] = EMAIndicator(df['close'], window=200).ema_indicator()

    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

    atr = AverageTrueRange(
        df['high'],
        df['low'],
        df['close'],
        window=14
    )

    df['atr'] = atr.average_true_range()

    return df

def get_eur_balance():

    balance = exchange.fetch_balance()

    return balance['total'].get('EUR', 0)

def get_btc_balance():

    balance = exchange.fetch_balance()

    return balance['total'].get('BTC', 0)

def long_signal(df):

    last = df.iloc[-1]

    trend = (
        last['ema20'] >
        last['ema50'] >
        last['ema200']
    )

    momentum = (
        last['rsi'] > 58 and
        last['rsi'] < 72
    )

    return trend and momentum

def calculate_position_size(df):

    last = df.iloc[-1]

    rsi = last['rsi']

    balance = get_eur_balance()

    if rsi > 68:
        risk = 1.0

    elif rsi > 62:
        risk = 0.5

    else:
        risk = 0.25

    return balance * risk

def buy(df):

    global IN_POSITION
    global ENTRY_PRICE

    eur_amount = calculate_position_size(df)

    if eur_amount < 5:
        return

    ticker = exchange.fetch_ticker(SYMBOL)

    price = ticker['last']

    btc_amount = eur_amount / price

    exchange.create_market_buy_order(
        SYMBOL,
        btc_amount
    )

    ENTRY_PRICE = price
    IN_POSITION = True

    print(f'BUY at {price}')

def sell():

    global IN_POSITION

    btc_balance = get_btc_balance()

    if btc_balance > 0:

        exchange.create_market_sell_order(
            SYMBOL,
            btc_balance
        )

    IN_POSITION = False

    print('SELL')

def strategy():

    global IN_POSITION
    global ENTRY_PRICE

    df = get_data()

    last = df.iloc[-1]

    price = last['close']

    if not IN_POSITION:

        if long_signal(df):
            buy(df)

    else:

        pnl = (
            price - ENTRY_PRICE
        ) / ENTRY_PRICE

        if pnl < -0.03:
            sell()

        elif pnl > 0.05:
            sell()

while True:

    try:

        strategy()

    except Exception as e:

        print(e)

    time.sleep(60)
