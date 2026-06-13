import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.pipeline import generate_sales_data, load_to_sqlite, load_from_sqlite
from src.features import engineer_features, get_feature_columns

def load_data():
    if not os.path.exists("data/sales.db"):
        df = generate_sales_data()
        load_to_sqlite(df)
    raw = load_from_sqlite()
    features = engineer_features(raw)
    return features

def split_data(df):
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["revenue"]
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

def train_all_models():
    mlflow.set_experiment("sales_forecasting")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=3, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate(y_test, y_pred)
            mlflow.log_params({"model": name, "features": len(get_feature_columns())})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name)
            results[name] = {"model": model, "metrics": metrics}
            print(f"  MAE:  {metrics['mae']:,.0f}")
            print(f"  RMSE: {metrics['rmse']:,.0f}")
            print(f"  R2:   {metrics['r2']}")

    best_name = max(results, key=lambda x: results[x]["metrics"]["r2"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")

    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/model_name.txt", "w") as f:
        f.write(best_name)

    print(f"Saved best model to models/best_model.pkl")
    return results, best_name

if __name__ == "__main__":
    results, best = train_all_models()
    print(f"\nFinal results:")
    for name, data in results.items():
        marker = " <- best" if name == best else ""