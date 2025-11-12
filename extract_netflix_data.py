import yfinance as yf
import pandas as pd
from datetime import datetime

def download_netflix_data():
    # Create a Ticker object for Netflix
    nflx = yf.Ticker("NFLX")
    
    # Download data for the specified period
    data = nflx.history(
        start="2022-01-01",
        end="2023-12-31",
        interval="1d"
    )
    
    # Reset index to make Date a column
    data.reset_index(inplace=True)
    
    # Save to CSV
    csv_path = "netflix_2022_2023.csv"
    data.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    return data

if __name__ == "__main__":
    data = download_netflix_data()
    print(f"Downloaded {len(data)} rows of data")
