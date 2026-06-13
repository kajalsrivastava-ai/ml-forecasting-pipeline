import pandas as pd
import sqlite3
import os
import random
from datetime import datetime, timedelta

def generate_sales_data():
    random.seed(42)
    products = ["Laptop Pro", "Wireless Mouse", "Standing Desk",
                "Office Chair", "Monitor 27in", "Webcam HD",
                "Keyboard Mech", "Desk Lamp"]
    regions = ["North", "South", "East", "West", "Central"]
    rows = []
    start = datetime(2022, 1, 1)
    for i in range(1000):
        product = random.choice(products)
        region = random.choice(regions)
        date = start + timedelta(days=random.randint(0, 729))
        quantity = random.randint(1, 20)
        price = random.uniform(500, 90000)
        revenue = round(quantity * price, 2)
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "product": product,
            "region": region,
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