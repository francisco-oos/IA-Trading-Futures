# main.py
import pandas as pd
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, USE_TESTNET

def init_binance_client():
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    if USE_TESTNET:
        client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    return client

def get_ohlcv(symbol='BTCUSDT', interval='15m', limit=500):
    client = init_binance_client()
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

if __name__ == "__main__":
    df = get_ohlcv()
    print(df.tail())
    df.to_csv("data/btc_futures_ohlcv.csv")
