import yfinance as yf
from pathlib import Path

#creating data directory if it doesn't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)

#yahoo finance tickers for assets
#yfinance does not support company names
ASSETS = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "META": "META",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD"
}

def download_asset(name, ticker, start="2015-01-01"):
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=start)

    if df.empty:
        print(f"Failed to download data for {ticker}")
        return
    
    df.reset_index(inplace=True)
    filePath = f"data/raw/{name}.csv"
    df.to_csv(filePath, index=False)
    print(f"Saved: {filePath}")

if __name__ == "__main__":
    for name, ticker in ASSETS.items():
        download_asset(name, ticker)