import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import pickle
import os
import numpy as np
from src.pipeline import load_from_sqlite, generate_sales_data, load_to_sqlite
from src.features import engineer_features

st.set_page_config(
    page_title="ML Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

if not os.path.exists("data/sales.db"):
    df = generate_sales_data()
    load_to_sqlite(df)

def load_model_name():
    try:
        with open("models/model_name.txt") as f:
            return f.read().strip()
    except:
        return "Unknown"

def get_actual_data():
    raw = load_from_sqlite()
    raw["date"] = pd.to_datetime(raw["date"])
    daily = raw.groupby("date")["revenue"].sum().reset_index()
    return daily.sort_values("date")

def get_forecasts(days=30):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/forecast",
            json={"days": days},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["forecasts"])
            df["date"] = pd.to_datetime(df["date"])
            return df, data["model_used"]
    except:
        pass
    return None, None

st.title("📈 ML Sales Forecasting Dashboard")
st.markdown("Real-time sales forecasting powered by machine learning.")
st.divider()

actual = get_actual_data()
model_name = load_model_name()
features = engineer_features(load_from_sqlite())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(load_from_sqlite()):,}")
with col2:
    last_30 = actual.tail(30)["revenue"].sum()
    st.metric("Last 30 Days Revenue", f"₹{last_30:,.0f}")
with col3:
    avg_daily = actual["revenue"].mean()
    st.metric("Avg Daily Revenue", f"₹{avg_daily:,.0f}")
with col4:
    st.metric("Best Model", model_name)

st.divider()

st.subheader("Historical Sales Trend")
fig_actual = px.line(
    actual.tail(180),
    x="date", y="revenue",
    title="Daily Revenue — Last 6 Months",
    color_discrete_sequence=["#378ADD"]
)
fig_actual.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title=None,
    yaxis_title="Revenue (₹)",
    showlegend=False
)
st.plotly_chart(fig_actual, use_container_width=True)

st.divider()
st.subheader("Revenue Forecast — Next 30 Days")

forecast_df, used_model = get_forecasts(30)

if forecast_df is not None:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total 30-Day Forecast",
                  f"₹{forecast_df['revenue'].sum():,.0f}")
    with col_b:
        st.metric("Avg Daily Forecast",
                  f"₹{forecast_df['revenue'].mean():,.0f}")
    with col_c:
        st.metric("Model Used", used_model or model_name)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=actual.tail(30)["date"],
        y=actual.tail(30)["revenue"],
        name="Actual",
        line=dict(color="#378ADD", width=2)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["revenue"],
        name="Forecast",
        line=dict(color="#E85D4A", width=2, dash="dash")
    ))
    fig_forecast.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title="Revenue (₹)",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Forecast Table")
    forecast_display = forecast_df.copy()
    forecast_display["revenue"] = forecast_display["revenue"].apply(
        lambda x: f"₹{x:,.0f}")
    st.dataframe(forecast_display, use_container_width=True)
else:
    st.warning("API not running. Start the FastAPI server to see forecasts.")
    st.code("uvicorn src.api:app --reload", language="bash")

st.divider()
st.subheader("Revenue by Day of Week")
actual["day_of_week"] = actual["date"].dt.day_name()
dow = actual.groupby("day_of_week")["revenue"].mean().reset_index()
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=day_order, ordered=True)
dow = dow.sort_values("day_of_week")
fig_dow = px.bar(dow, x="day_of_week", y="revenue",
                 color_discrete_sequence=["#378ADD"])
fig_dow.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title=None, yaxis_title="Avg Revenue (₹)")
st.plotly_chart(fig_dow, use_container_width=True)