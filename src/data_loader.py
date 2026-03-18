import os
import pandas as pd
import yfinance as yf


def download_data(tickers, start="2020-01-01", end="2024-01-01"):
    """Download adjusted close prices for given tickers."""
    print(f"Downloading data for {tickers} from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end)

    # Handle different yfinance output formats
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        print("Warning: 'Adj Close' not found, using 'Close' instead.")
        prices = data["Close"]
    else:
        # Multi-level columns from newer yfinance
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"]
            else:
                prices = data["Close"]
        else:
            prices = data

    prices = prices.dropna()
    print(f"Downloaded {len(prices)} trading days of data.")
    return prices


def split_data(prices, train_end="2022-12-31"):
    """Split prices chronologically into train and test sets."""
    prices_train = prices.loc[:train_end]
    prices_test = prices.loc[train_end:].iloc[1:]  # exclude the split date itself

    print(f"Train set: {prices_train.index[0].date()} to {prices_train.index[-1].date()} ({len(prices_train)} days)")
    print(f"Test set:  {prices_test.index[0].date()} to {prices_test.index[-1].date()} ({len(prices_test)} days)")
    return prices_train, prices_test


def save_data(df, path="data/prices.csv"):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Data saved to {path}")


def load_data(path="data/prices.csv"):
    """Load DataFrame from CSV."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Data loaded from {path} ({len(df)} rows)")
    return df
