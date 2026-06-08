"""
data_fetcher.py
---------------
Fetches real OHLCV market data via yfinance for any ticker or portfolio.
Handles caching, validation, and returns clean DataFrames ready for risk engine.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Default Universe ─────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GS", "BAC", "SPY", "QQQ"]
BENCHMARK       = "SPY"


def fetch_prices(tickers: list[str], start: str, end: str = None,
                 interval: str = "1d") -> pd.DataFrame:
    """
    Fetch adjusted closing prices for a list of tickers.

    Returns:
        DataFrame with dates as index, tickers as columns.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(tickers, start=start, end=end,
                      interval=interval, auto_adjust=True,
                      progress=False, show_errors=False)

    if isinstance(tickers, str) or len(tickers) == 1:
        prices = raw[["Close"]].rename(
            columns={"Close": tickers if isinstance(tickers, str) else tickers[0]}
        )
    else:
        prices = raw["Close"]

    prices.dropna(how="all", inplace=True)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)

    missing = [t for t in (tickers if isinstance(tickers, list) else [tickers])
               if t not in prices.columns]
    if missing:
        print(f"Warning: Could not fetch data for: {missing}")

    return prices


def fetch_returns(tickers: list[str], start: str, end: str = None,
                  log_returns: bool = True) -> pd.DataFrame:
    """
    Compute daily log or simple returns from price data.
    """
    prices  = fetch_prices(tickers, start, end)
    if log_returns:
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns


def fetch_ticker_info(ticker: str) -> dict:
    """Fetch basic company info for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name"       : info.get("longName", ticker),
            "sector"     : info.get("sector", "N/A"),
            "market_cap" : info.get("marketCap", 0),
            "beta"       : info.get("beta", None),
            "pe_ratio"   : info.get("trailingPE", None),
            "52w_high"   : info.get("fiftyTwoWeekHigh", None),
            "52w_low"    : info.get("fiftyTwoWeekLow", None),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A"}


def fetch_benchmark(start: str, end: str = None) -> pd.Series:
    """Fetch SPY as market benchmark."""
    prices = fetch_prices([BENCHMARK], start, end)
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns[BENCHMARK]


def get_date_range(period: str = "1y") -> tuple[str, str]:
    """Convert period string to start/end dates."""
    end   = datetime.today()
    periods = {
        "3m" : timedelta(days=90),
        "6m" : timedelta(days=180),
        "1y" : timedelta(days=365),
        "2y" : timedelta(days=730),
        "3y" : timedelta(days=1095),
        "5y" : timedelta(days=1825),
    }
    delta = periods.get(period, timedelta(days=365))
    start = end - delta
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


if __name__ == "__main__":
    start, end = get_date_range("1y")
    tickers    = ["AAPL", "JPM", "SPY"]

    print(f"Fetching {tickers} from {start} to {end}...")
    returns = fetch_returns(tickers, start, end)
    print(f"\nReturns shape: {returns.shape}")
    print(returns.tail(5))

    print("\nAAPL info:")
    info = fetch_ticker_info("AAPL")
    for k, v in info.items():
        print(f"  {k}: {v}")
