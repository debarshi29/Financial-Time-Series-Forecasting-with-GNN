# A minimal script to test the core download functionality.
# Replace the contents of download_market_data.py with this.

import yfinance as yf
import pandas as pd

# Use the exact same tickers and dates from your test
ticker = "RELIANCE.NS"
start_date = "2020-01-01"
end_date = "2024-01-01"

print(f"--- Running Minimal Debug Test ---")
print(f"Attempting to download: {ticker} from {start_date} to {end_date}")

try:
    # This is the most basic download call, similar to your working script
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False
    )

    if data.empty:
        print("\n>>> RESULT: FAILED. yfinance returned an empty DataFrame.")
    else:
        print("\n>>> RESULT: SUCCESS! Data was downloaded.")
        print("Data head:")
        print(data.head())

except Exception as e:
    print(f"\n>>> RESULT: FAILED with an unexpected exception: {e}")
