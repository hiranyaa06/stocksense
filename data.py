# data.py — fetches real stock data from Yahoo Finance

import yfinance as yf
import pandas as pd

def get_stock_data(ticker, period="2y"):

    df = yf.download(ticker, period=period, auto_adjust=False)

    # Handle MultiIndex columns returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # keeps only close values as tht is only impt n req
    df = df[['Close']].copy()

    # Fix missing values- with last known value
    df['Close'] = df['Close'].ffill()

    # Sort chronologically just in case(yfinance already does it,but jic)
    df = df.sort_index()

    print(f"Fetched {len(df)} days of data for {ticker}")
    print(df.tail(5))  # show last 5 rows(so tht we can get insights abt the newest dates)

    return df


# testing
if __name__ == "__main__":
    df = get_stock_data("AAPL")