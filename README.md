# 📈 ML Sales Forecasting Pipeline

An end-to-end machine learning pipeline that ingests raw sales data, engineers features, trains and compares multiple models with experiment tracking, serves predictions via a REST API, and visualizes forecasts on a monitoring dashboard.

> **Status:** ✅ Complete

---

## 🏗️ Architecture

Raw Sales Data → Feature Engineering → Model Training → MLflow Tracking
↓
Best Model → FastAPI REST API
↓
Streamlit Monitoring Dashboard

---

## 💡 What it does

- Ingests raw sales transactions and engineers 12 time-series features
- Trains and compares Linear Regression, Random Forest, and XGBoost models
- Tracks all experiments with MLflow — metrics, parameters, and artifacts
- Serves the best model via a FastAPI REST endpoint
- Visualizes 30-day forecasts on a Streamlit monitoring dashboard

---

## 🔧 Tech Stack

`Python` `scikit-learn` `XGBoost` `MLflow` `FastAPI` `Streamlit` `Pandas` `SQLite` `Plotly`

---

## 📁 Project Structure
```
ml-forecasting-pipeline/
├── data/              # Raw and processed sales data
├── models/            # Trained model artifacts
├── src/
│   ├── pipeline.py    # Data ingestion and generation
│   ├── features.py    # Feature engineering
│   ├── train.py       # Model training + MLflow tracking
│   ├── api.py         # FastAPI REST endpoint
│   └── dashboard.py   # Streamlit monitoring dashboard
└── requirements.txt
```
---

## 🚀 How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models
python -m src.train

# 3. Start the API
uvicorn src.api:app --reload

# 4. Start the dashboard
streamlit run src/dashboard.py
```

---

## 🌐 API Usage

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'
```

---

## 📊 MLflow Experiment Tracking

All model runs tracked with MAE, RMSE, and R2 metrics across Linear Regression, Random Forest, and XGBoost.
