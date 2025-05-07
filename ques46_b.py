# 📦 Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# 📅 Generate Synthetic Electricity Consumption Data
n_hours = 17520
date_range = pd.date_range(start='1/1/2020', periods=n_hours, freq='H')
np.random.seed(42)
seasonality = np.sin(np.linspace(0, 2 * np.pi, 24))
daily_cycle = np.tile(seasonality, int(n_hours / 24))
noise = np.random.normal(0, 0.2, n_hours)
consumption = 1 + daily_cycle + noise
data = pd.DataFrame({'Hour': date_range, 'Consumption': consumption})

# 📊 Visualize Time Series (First 1 Week)
plt.figure(figsize=(14, 4))
plt.plot(data['Hour'][:24*7], data['Consumption'][:24*7])
plt.title("Electricity Consumption (1 Week Sample)")
plt.xlabel("Hour")
plt.ylabel("Consumption (kWh)")
plt.grid(True)
plt.show()

# 🔃 Normalize the Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Consumption']])

# 🔄 Create Sequences for LSTM
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 24
X, y = create_sequences(scaled_data, seq_len)

# ✂️ Train-Test Split
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 🤖 Build and Train LSTM Model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(seq_len, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 🔮 Forecast with LSTM
predictions = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
pred_inv = scaler.inverse_transform(predictions)

plt.figure(figsize=(12,4))
plt.plot(y_test_inv[:100], label='Actual')
plt.plot(pred_inv[:100], label='LSTM Prediction')
plt.title("LSTM Forecast vs Actual")
plt.legend()
plt.show()

mse_lstm = mean_squared_error(y_test_inv, pred_inv)
print("✅ LSTM MSE:", round(mse_lstm, 4))

# 📈 ARIMA Forecasting
arima_model = ARIMA(data['Consumption'], order=(5,1,0)).fit()
forecast_arima = arima_model.predict(start=len(data)-len(y_test), end=len(data)-1)

plt.figure(figsize=(12,4))
plt.plot(data['Consumption'].values[-len(y_test):], label='Actual')
plt.plot(forecast_arima, label='ARIMA Prediction')
plt.title("ARIMA Forecast vs Actual")
plt.legend()
plt.show()

mse_arima = mean_squared_error(data['Consumption'].values[-len(y_test):], forecast_arima)
print("📊 ARIMA MSE:", round(mse_arima, 4))

# ✅ Compare Models
if mse_lstm < mse_arima:
    print("🏆 LSTM performs better (lower MSE) and captures complex daily patterns.")
else:
    print("🏆 ARIMA performs better (lower MSE). It works well for stationary trends.")
