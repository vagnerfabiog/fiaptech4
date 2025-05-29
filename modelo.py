import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json

# 1. Parâmetros do modelo/dataset
symbol = 'AAPL'
start_date = '2018-01-01'
end_date = '2025-05-20'
look_back = 60
EPOCHS = 20
BATCH_SIZE = 32

# 2. Baixar dados
df = yf.download(symbol, start=start_date, end=end_date)
df = df[['Close']].dropna()

# 3. Normalização
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# 4. Criar dataset para LSTM
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 5. Divisão em treino e teste
split = int(0.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Modelo LSTM
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Treinamento
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# 8. Avaliação
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(real_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
mape = np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")

# 9. Salvar modelo, scaler e metadados
import os
os.makedirs("model_artifacts", exist_ok=True)
model.save('model_artifacts/lstm_model.h5')
joblib.dump(scaler, 'model_artifacts/scaler.pkl')

# Salva os metadados
metadata = {
    "stock_symbol": symbol,
    "start_date": start_date,
    "end_date": end_date,
    "look_back": look_back,
    "metrics": {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2)
    }
}
with open("model_artifacts/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Modelo, scaler e metadados salvos em 'model_artifacts/'.")
