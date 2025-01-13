import pandas as pd
import numpy as np

def generate_stock_data():
    companies = ["AAPL", "META", "NVDA", "GOOGL", "MSFT", "AMZN", "TSLA"]
    dates = pd.date_range(start="2020-01-01", end="2022-12-31")
    
    data = {}
    for company in companies:
        # Generate random price data with trends
        trend = np.cumsum(np.random.normal(0, 1, len(dates))) * 2 + 100
        prices = trend + np.random.normal(0, 5, len(dates))  # Add noise
        data[company] = prices

    df = pd.DataFrame(data, index=dates)
    df.to_csv("data/stock_data.csv")
    print("Stock data saved to data/stock_data.csv")

generate_stock_data()
