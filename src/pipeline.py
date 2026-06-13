import pandas as pd
import sqlite3
import os
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data():
    np.random.seed(42)
    start = datetime(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(730)]
    rows = []
    products = [
        ("Laptop Pro", 85000), ("Wireless Mouse", 1200),
        ("Standing Desk", 22000), ("Office Chair", 15000),
        ("Monitor 27in", 28000), ("Webcam HD", 4500),
        ("Keyboard Mech", 6500), ("Desk Lamp", 1800)
    ]
    for i, date in enumerate(dates):
        trend = 1 + (i / 730) * 0.3
        month = date.month
        seasonality = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
        is_weekend = 1.3 if date.weekday() >= 5 else 1.0
        num_orders = max(1, int(np.random.poisson(8) * trend * seasonality))
        for _ in range(num_orders):
            product, base_price = products[np.random.randint(len(products))]
            quantity = np.random.randint(1, 10)
            price = base_price * np.random.uniform(0.9, 1.1) * is_weekend
            revenue = round(quantity * price, 2)
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product": product,
                "quantity": quantity,
                "unit_price": round(price, 2),
                "revenue": revenue
            })
    return pd.DataFrame(rows)

def load_to_sqlite(df, db_path="data/sales.db"):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} rows into {db_path}")

def load_from_sqlite(db_path="data/sales.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM sales", conn)
    conn.close()
    return df

if __name__ == "__main__":
    df = generate_sales_data()
    load_to_sqlite(df)
    df.to_csv("data/sales_raw.csv", index=False)
    print(df.head())
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total revenue: {df['revenue'].sum():,.2f}")
