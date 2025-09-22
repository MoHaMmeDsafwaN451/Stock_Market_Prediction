# app.py
import os
import pickle
import requests
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try Prophet (long-term model)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# =============================
# App config
# =============================
st.set_page_config(page_title="Stock Predictor + News", layout="wide")


# =============================
# Constants / API
# =============================
# Finnhub – use your real key here
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d360nbhr01qumnp3lp3gd360nbhr01qumnp3lp40")
FINNHUB_NEWS_URL = f"https://finnhub.io/api/v1/business-insider-news?token={FINNHUB_API_KEY}"


# =============================
# Utilities
# =============================
def normalize_dates_utc_naive(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """
    Ensure a consistent, tz-naive datetime column in UTC (dropping tz).
    Fixes Prophet/pandas 'tz-aware cannot be converted' errors.
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df[col] = df[col].dt.tz_localize(None)
    df = df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    return df


@st.cache_data
def list_csv_files(data_dir: str = "data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]


@st.cache_data
def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV and return a standardized dataframe with columns: Date, Close, (others optional)
    Date becomes tz-naive UTC, sorted ascending.
    """
    df = pd.read_csv(filepath)
    # detect date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        # fabricate if not present
        df["Date"] = pd.date_range(end=pd.Timestamp.utcnow().tz_localize(None), periods=len(df))
    else:
        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})
    if "Close" not in df.columns:
        raise ValueError("CSV must contain a 'Close' column")
    df = normalize_dates_utc_naive(df, "Date")
    return df


@st.cache_data
def auto_fetch_and_save_stock_data(ticker: str, data_dir: str = "data", period: str = "5y"):
    """
    If a CSV for the ticker doesn't exist (or fails to load), fetch via yfinance and store CSV.
    Always returns a standardized dataframe with Date and Close at minimum.
    """
    ticker_upper = ticker.upper()
    csv_filename = f"{ticker_upper.lower()}.csv"
    csv_path = os.path.join(data_dir, csv_filename)

    # try load existing
    if os.path.exists(csv_path):
        try:
            df = load_csv(csv_path)
            st.success(f"✅ Loaded existing {ticker_upper} from CSV")
            return df, csv_filename
        except Exception as e:
            st.warning(f"⚠️ Error loading CSV, fetching fresh data: {e}")

    # fetch fresh
    try:
        st.info(f"📥 Fetching {ticker_upper} from Yahoo Finance...")
        p = st.progress(0)
        stock = yf.Ticker(ticker_upper)
        p.progress(25)
        df = stock.history(period=period)  # DatetimeIndex (tz-aware)
        p.progress(50)
        if df.empty:
            st.error(f"❌ No data found for {ticker_upper}")
            p.empty()
            return None, None

        df = df.reset_index()
        # Ensure we keep consistent columns
        keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
        # Some variants may use lowercase or different order; normalize
        df.columns = [str(c) for c in df.columns]
        if "Date" not in df.columns:
            # yfinance uses 'Date' in reset_index by default
            # but fallback if different
            date_like = [c for c in df.columns if c.lower() in ("date", "datetime", "time")]
            if date_like:
                df = df.rename(columns={date_like[0]: "Date"})
            else:
                # create one if absolutely missing
                df["Date"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df))
        # Keep only needed columns if present
        subset = [c for c in keep if c in df.columns]
        df = df[subset].copy()
        # Normalize dates
        df = normalize_dates_utc_naive(df, "Date")

        # save
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        df.to_csv(csv_path, index=False)
        p.progress(100)
        st.success(f"✅ Saved {ticker_upper} to {csv_filename}")
        st.info(f"📊 {len(df)} rows ({df['Date'].min().date()} → {df['Date'].max().date()})")
        p.empty()
        return df, csv_filename
    except Exception as e:
        st.error(f"❌ Fetch failed for {ticker_upper}: {e}")
        return None, None


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def prepare_features(df, target_col="Close", lags=5, rolling_windows=(3, 7, 14), add_indicators=True):
    df2 = df.copy().sort_values("Date").reset_index(drop=True)
    if add_indicators:
        df2["EMA_12"] = ema(df2[target_col], 12)
        df2["EMA_26"] = ema(df2[target_col], 26)
        macd_line, macd_sig, macd_hist = macd(df2[target_col])
        df2["MACD"] = macd_line
        df2["MACD_signal"] = macd_sig
        df2["MACD_hist"] = macd_hist
        df2["RSI_14"] = rsi(df2[target_col], 14)

    X = pd.DataFrame()
    for lag in range(1, lags + 1):
        X[f"lag_{lag}"] = df2[target_col].shift(lag)
    for w in rolling_windows:
        X[f"roll_mean_{w}"] = df2[target_col].rolling(window=w).mean().shift(1)
        X[f"roll_std_{w}"] = df2[target_col].rolling(window=w).std().shift(1)

    # include other numeric cols if present
    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    for c in numeric_cols:
        if c != target_col:
            X[c] = df2[c]

    y = df2[target_col]
    X = X.dropna()
    y = y.loc[X.index]
    X["Date"] = df2.loc[X.index, "Date"].values
    return X.reset_index(drop=True), y.reset_index(drop=True)


def train_random_forest(X, y):
    feat_cols = [c for c in X.columns if c not in ("Date",)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feat_cols].values)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y.values)
    return model, scaler, feat_cols


def iterative_forecast_rf(last_df, model, scaler, feat_cols, horizon, lags=5, rolling_windows=(3, 7, 14), add_indicators=True):
    """
    Iterative day-ahead forecasting with RF (short-term friendly).
    """
    df = last_df.copy().sort_values("Date").reset_index(drop=True)
    preds = []
    for _ in range(horizon):
        X_all, _ = prepare_features(df, lags=lags, rolling_windows=rolling_windows, add_indicators=add_indicators)
        if X_all.empty:
            raise ValueError("Not enough history to build features for iterative forecasting.")
        X_next = X_all[feat_cols].iloc[-1:].values
        X_next_scaled = scaler.transform(X_next)
        y_pred = model.predict(X_next_scaled)[0]
        next_date = df["Date"].iloc[-1] + pd.Timedelta(days=1)
        new_row = {"Date": next_date, "Close": y_pred}
        # carry forward other numeric cols if any
        for col in df.columns:
            if col not in ("Date", "Close") and pd.api.types.is_numeric_dtype(df[col]):
                new_row[col] = df[col].iloc[-1]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        preds.append((next_date, y_pred))
    pred_df = pd.DataFrame(preds, columns=["ds", "yhat"])
    return pred_df


def prophet_forecast(df, horizon):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed. pip install prophet")
    df_prop = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"}).copy()
    # Defensive: ensure tz-naive
    df_prop["ds"] = pd.to_datetime(df_prop["ds"], utc=True).dt.tz_localize(None)
    m = Prophet(daily_seasonality=True)
    m.fit(df_prop)
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)
    return forecast


def suggestion_from_prices(current_price, predicted_price, investor_type="long"):
    expected_return = (predicted_price - current_price) / max(current_price, 1e-9)
    if investor_type == "long":
        if expected_return > 0.25:
            return "Buy (Long-term) ✅ — expects >25% gain"
        elif expected_return > 0.05:
            return "Hold (Long-term) 🟡 — moderate expected gain"
        else:
            return "Sell (Long-term) ❌ — little or negative expected gain"
    else:
        if expected_return > 0.05:
            return "Buy (Short-term) ✅ — good for trading"
        elif expected_return > 0.01:
            return "Hold (Short-term) 🟡 — weak short-term move"
        else:
            return "Sell (Short-term) ❌ — not attractive for trading"


def fetch_finnhub_news(limit=30):
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY":
        # fallback sample
        return [{
            "headline": "Sample market news headline",
            "summary": "Sample description. Place your Finnhub API key into the code to fetch live news.",
            "url": "https://example.com",
            "datetime": pd.Timestamp.now().isoformat()
        }]
    try:
        r = requests.get(FINNHUB_NEWS_URL, timeout=10)
        if r.status_code == 200:
            return r.json()[:limit]
        else:
            return [{"headline": f"News fetch failed: {r.status_code}", "summary": "", "url": "", "datetime": ""}]
    except Exception as e:
        return [{"headline": f"News fetch error: {e}", "summary": "", "url": "", "datetime": ""}]


# =============================
# Sidebar
# =============================
st.sidebar.title("Settings")

input_method = st.sidebar.radio(
    "Select Stock Method",
    ["Choose from existing CSV files", "Enter ticker symbol"]
)

if input_method == "Choose from existing CSV files":
    csv_files = list_csv_files("data")
    if not csv_files:
        st.sidebar.warning("No CSV files found in data/ folder")
    selected_csv = st.sidebar.selectbox("Select Stocks", csv_files) if csv_files else None
    selected_ticker = None
else:
    selected_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AMZN, MSFT, NVDA)", value="").upper()
    selected_csv = None

model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Prophet (time-series)"])
timeframe = st.sidebar.selectbox("Forecast timeframe", [
    "1 Week", "1 Month", "6 Months", "1 Year", "3 Years", "10 Years", "10+ Years"
])
timeframe_map = {
    "1 Week": 7,
    "1 Month": 30,
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 365 * 3,
    "10 Years": 365 * 10,
    "10+ Years": 365 * 15
}
forecast_horizon = timeframe_map[timeframe]

add_indicators = st.sidebar.checkbox("Add technical indicators for RF (EMA/MACD/RSI)", value=True)
use_scaler = st.sidebar.checkbox("Scale features (RF)", value=True)


# =============================
# Main
# =============================
st.title("📈 Stock Prediction Dashboard — Home")

df_raw = None
display_name = ""
ticker_guess = ""

if input_method == "Choose from existing CSV files":
    if selected_csv is None:
        st.warning("No CSV selected. Select a file or switch to ticker input mode.")
        st.stop()
    data_path = os.path.join("data", selected_csv)
    try:
        df_raw = load_csv(data_path)
        display_name = selected_csv
        ticker_guess = os.path.splitext(selected_csv)[0].upper().split("_")[0]
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        st.stop()
else:
    if not selected_ticker:
        st.warning("Please enter a stock ticker symbol (e.g., AMZN, MSFT, NVDA)")
        st.stop()
    df_raw, csv_filename = auto_fetch_and_save_stock_data(selected_ticker, period="5y")
    if df_raw is None:
        st.error("Failed to fetch stock data. Please check the ticker symbol and try again.")
        st.stop()
    display_name = f"{selected_ticker} (auto-fetched)"
    ticker_guess = selected_ticker

# Ensure dates always normalized (defensive)
df_raw = normalize_dates_utc_naive(df_raw, "Date")

st.subheader(f"Loaded: {display_name}")
st.dataframe(df_raw.tail(6))

# Live price metric
live_price = None
try:
    t = yf.Ticker(ticker_guess)
    hist = t.history(period="1d", interval="1m")
    if not hist.empty:
        live_price = hist["Close"].iloc[-1]
except Exception:
    live_price = None

if live_price is None:
    live_price = float(df_raw["Close"].iloc[-1])

st.metric(label=f"{ticker_guess} — Current price (realtime via yfinance or CSV last)", value=f"{live_price:.2f}")

# Historical chart
st.subheader("Historical Price (interactive)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df_raw["Date"], y=df_raw["Close"], mode="lines", name="Historical Close"))
fig_hist.update_layout(height=420, xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_hist, use_container_width=True)

# Forecast section
st.subheader(f"Forecast ({model_choice}) — {timeframe}")

if model_choice == "Random Forest":
    with st.spinner("Preparing features and training Random Forest..."):
        X, y = prepare_features(df_raw, target_col="Close", lags=7, rolling_windows=(7, 14, 30), add_indicators=add_indicators)
        if X.empty or len(X) < 50:
            st.error("Not enough data after feature creation. Try a different CSV or reduce lags/windows.")
        else:
            feat_cols = [c for c in X.columns if c not in ("Date",)]
            model, scaler, trained_feat_cols = train_random_forest(X, y)
            with st.spinner(f"Generating {forecast_horizon}-day forecast with Random Forest..."):
                try:
                    preds_rf = iterative_forecast_rf(
                        df_raw, model, scaler, trained_feat_cols,
                        horizon=forecast_horizon, lags=7, rolling_windows=(7, 14, 30),
                        add_indicators=add_indicators
                    )
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    preds_rf = pd.DataFrame(columns=["ds", "yhat"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_raw["Date"], y=df_raw["Close"], mode="lines", name="Historical"))
            if not preds_rf.empty:
                fig.add_trace(go.Scatter(x=preds_rf["ds"], y=preds_rf["yhat"], mode="lines+markers", name="RF Forecast"))
            fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            if not preds_rf.empty:
                predicted_price = float(preds_rf["yhat"].iloc[-1])
                st.metric("Predicted price (end of horizon)", f"{predicted_price:.2f}")
                st.write("**Suggestions (heuristic)**")
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="long"))
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="short"))
                # quick holdout
                try:
                    split_idx = int(len(y) * 0.9)
                    X_train_vals = X[trained_feat_cols].values[:split_idx]
                    X_test_vals = X[trained_feat_cols].values[split_idx:]
                    if use_scaler:
                        X_train_vals = scaler.transform(X_train_vals)
                        X_test_vals = scaler.transform(X_test_vals)
                    y_train_vals = y.values[:split_idx]
                    y_test_vals = y.values[split_idx:]
                    y_pred_test = model.predict(X_test_vals)
                    rmse = mean_squared_error(y_test_vals, y_pred_test) ** 0.5
                    mae = mean_absolute_error(y_test_vals, y_pred_test)
                    st.write(f"Model test RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                except Exception:
                    pass
            try:
                with open("rf_model.pkl", "wb") as f:
                    pickle.dump({"model": model, "scaler": scaler, "features": trained_feat_cols}, f)
                st.download_button("Download Random Forest model (pickle)", data=open("rf_model.pkl", "rb"), file_name="rf_model.pkl")
            except Exception:
                pass

elif model_choice == "Prophet (time-series)":
    if not PROPHET_AVAILABLE:
        st.error("Prophet library not available. Install with: pip install prophet")
    else:
        with st.spinner("Fitting Prophet and forecasting..."):
            try:
                forecast = prophet_forecast(df_raw, forecast_horizon)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_raw["Date"], y=df_raw["Close"], mode="lines", name="Historical"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Prophet Forecast"))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill="tonexty", line=dict(width=0), name="Uncertainty"))
                fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                predicted_price = float(forecast["yhat"].iloc[-1])
                st.metric("Predicted price (end of horizon)", f"{predicted_price:.2f}")
                st.write("**Suggestions (heuristic)**")
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="long"))
                st.write(suggestion_from_prices(live_price, predicted_price, investor_type="short"))
                try:
                    with open("prophet_model.pkl", "wb") as f:
                        pickle.dump(forecast, f)
                    st.download_button("Download Prophet forecast (pickle)", data=open("prophet_model.pkl", "rb"), file_name="prophet_forecast.pkl")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Prophet forecast failed: {e}")

# Footer
st.markdown("""
### ⚠️ Important Notice — Read carefully
- **Predictions are not guaranteed.** They are based on historical patterns and statistical models.
- **Real-world events** (new tax rules, regulatory changes, company earnings surprises, geopolitical events, or market breakdowns) can change prices quickly and invalidate predictions.
- **Do not rely solely on these predictions** for investment decisions. Always consult up-to-date news, company fundamentals, and financial professionals.
- **Use this app as one tool among many** — combine technical forecasts, news, risk management, and your own research.
- For serious backtesting / trading, include transaction costs, slippage, taxes, and a proper risk model.
""")
