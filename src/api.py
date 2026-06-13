from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from src.features import get_feature_columns
from src.pipeline import load_from_sqlite, generate_sales_data, load_to_sqlite

app = FastAPI(
    title="Sales Forecasting API",
    description="ML-powered sales revenue forecasting endpoint",
    version="1.0.0"
)

def load_model():
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run src/train.py first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open("models/model_name.txt") as f:
        model_name = f.read().strip()
    return model, scaler, model_name

def get_recent_stats():
    if not os.path.exists("data/sales.db"):
        df = generate_sales_data()
        load_to_sqlite(df)
    raw = load_from_sqlite()
    raw["date"] = pd.to_datetime(raw["date"])
    daily = raw.groupby("date")["revenue"].sum().reset_index()
    daily = daily.sort_values("date")
    recent = daily["revenue"].tail(30)
    return {
        "lag_1": recent.iloc[-1],
        "lag_7": recent.iloc[-7] if len(recent) >= 7 else recent.mean(),
        "lag_14": recent.iloc[-14] if len(recent) >= 14 else recent.mean(),
        "lag_30": recent.iloc[-30] if len(recent) >= 30 else recent.mean(),
        "rolling_7": recent.tail(7).mean(),
        "rolling_14": recent.tail(14).mean(),
        "rolling_30": recent.tail(30).mean(),
    }

class ForecastRequest(BaseModel):
    days: int = 7

class ForecastResponse(BaseModel):
    model_used: str
    forecasts: list
    total_forecast: float
    average_daily: float

@app.get("/")
def root():
    return {
        "message": "Sales Forecasting API",
        "version": "1.0.0",
        "endpoints": ["/forecast", "/health", "/model-info", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model-info")
def model_info():
    try:
        _, _, model_name = load_model()
        return {
            "model": model_name,
            "features": get_feature_columns(),
            "feature_count": len(get_feature_columns())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    if request.days < 1 or request.days > 90:
        raise HTTPException(
            status_code=400,
            detail="days must be between 1 and 90"
        )
    try:
        model, scaler, model_name = load_model()
        stats = get_recent_stats()
        forecasts = []
        rolling_window = []
        for i in range(request.days):
            target_date = datetime.now() + timedelta(days=i+1)
            features = {
                "day_of_week": target_date.weekday(),
                "month": target_date.month,
                "quarter": (target_date.month - 1) // 3 + 1,
                "day_of_year": target_date.timetuple().tm_yday,
                "is_weekend": 1 if target_date.weekday() >= 5 else 0,
                "lag_1": stats["lag_1"] if i == 0 else forecasts[-1]["revenue"],
                "lag_7": stats["lag_7"] if i < 7 else forecasts[-7]["revenue"],
                "lag_14": stats["lag_14"] if i < 14 else forecasts[-14]["revenue"],
                "lag_30": stats["lag_30"] if i < 30 else forecasts[-30]["revenue"],
                "rolling_7": stats["rolling_7"] if i < 7 else np.mean([f["revenue"] for f in forecasts[-7:]]),
                "rolling_14": stats["rolling_14"] if i < 14 else np.mean([f["revenue"] for f in forecasts[-14:]]),
                "rolling_30": stats["rolling_30"] if i < 30 else np.mean([f["revenue"] for f in forecasts[-30:]]),
            }
            X = np.array([[features[c] for c in get_feature_columns()]])
            X_scaled = scaler.transform(X)
            predicted = float(model.predict(X_scaled)[0])
            predicted = max(0, predicted)
            forecasts.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "day": target_date.strftime("%A"),
                "revenue": round(predicted, 2)
            })
        total = sum(f["revenue"] for f in forecasts)
        return ForecastResponse(
            model_used=model_name,
            forecasts=forecasts,
            total_forecast=round(total, 2),
            average_daily=round(total / len(forecasts), 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))