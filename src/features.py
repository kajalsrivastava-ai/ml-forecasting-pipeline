import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["revenue"].sum().reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["quarter"] = daily["date"].dt.quarter
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["lag_1"] = daily["revenue"].shift(1)
    daily["lag_7"] = daily["revenue"].shift(7)
    daily["lag_14"] = daily["revenue"].shift(14)
    daily["lag_30"] = daily["revenue"].shift(30)
    daily["rolling_7"] = daily["revenue"].shift(1).rolling(7).mean()
    daily["rolling_14"] = daily["revenue"].shift(1).rolling(14).mean()
    daily["rolling_30"] = daily["revenue"].shift(1).rolling(30).mean()
    daily = daily.dropna().reset_index(drop=True)
    return daily

def get_feature_columns():
    return [
        "day_of_week", "month", "quarter", "day_of_year",
        "is_weekend", "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_7", "rolling_14", "rolling_30"
    ]
