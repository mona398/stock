import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Prediction App")
st.markdown("""
Upload a CSV file containing stock prices (with a `Date` and `Close` column), and this app will:
- Train a Linear Regression model
- Predict the **next day's closing price** based on your input
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload your stock price CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check for required columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain at least 'Date' and 'Close' columns.")
        st.stop()

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.ffill(inplace=True)

    # Feature engineering
    df['Prev_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    # Show raw data
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.tail())

    # Split
    X = df[['Prev_Close']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model (optional)
    joblib.dump(model, 'linear_model.pkl')

    # Predict test set
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("📊 Model Performance")
    st.write(f"R² Score: **{r2_score(y_test, y_pred):.4f}**")
    st.write(f"RMSE: **{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}**")

    # Plot
    st.subheader("🔍 Actual vs Predicted Close")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Close")
    ax.set_ylabel("Predicted Close")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # Prediction input
    st.subheader("🔮 Predict Next Closing Price")
    user_input = st.number_input("Enter Previous Day's Closing Price", min_value=0.0, value=float(X.iloc[-1][0]))

    if st.button("Predict"):
        pred = model.predict(np.array([[user_input]]))[0]
        st.success(f"📌 Predicted Next Close: ₹{pred:.2f}")
