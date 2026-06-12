import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage

# ========== 設定周り ==========
BUCKET_NAME = "household-electic-data-20260501"
HISTORY_FILE_PATH = f"gs://{BUCKET_NAME}/database/history.csv"
MODEL_WEIGHTS_BLOB = "models/transformer_weights.pth" # GCS上の保存パス
LOCAL_WEIGHTS_PATH = "/tmp/transformer_weights.pth" # 一時保存用

# ========== モデルの定義 ==========
class TransformerForecaster(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dim_feedforward=2048, dropout=0.2):
        super(TransformerForecaster, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        last_out = x[:, -1, :]
        return self.fc(last_out)

def create_window(data, window_size=72, forecast_size=24, target_col_index=0):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_size + 1):
        X.append(data[i:i + window_size, :])
        y.append(data[i + window_size : i + window_size + forecast_size, target_col_index])
    return np.array(X), np.array(y)

# ========== メイン処理 ==========
def run_training_pipeline():
    print("--- 1. データの取得と前処理を開始 ---")
    try:
        df = pd.read_csv(HISTORY_FILE_PATH)
    except Exception as e:
        print(f"履歴データの読み込みに失敗しました: {e}")
        return

    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df_cleaned = df.interpolate(method='linear', limit_direction='both')

    df_hourly = df_cleaned.set_index('DateTime').resample('h').agg({
        'Global_active_power':['sum', 'mean', 'max', 'std'],
        'Global_reactive_power': ['mean'],
        'Voltage':['mean']
    })
    df_hourly.columns = [f"{col[0]}_{col[1]}" for col in df_hourly.columns]

    df_hourly['gap_mean_lag1'] = df_hourly['Global_active_power_mean'].shift(1)
    df_hourly['gap_mean_lag24'] = df_hourly['Global_active_power_mean'].shift(24)
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['dayofweek'] = df_hourly.index.dayofweek
    df_hourly['is_weekend'] = df_hourly.index.dayofweek.isin([5,6]).astype(int)

    df_final = df_hourly.dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_final)

    col_to_idx = {col: c for c, col in enumerate(df_final.columns)}
    target_idx = col_to_idx['Global_active_power_mean']
    X, y = create_window(scaled_data, window_size=72, forecast_size=24, target_col_index=target_idx)

    print("--- 2. モデルの学習を開始 ---")
    input_features = len(df_final.columns)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerForecaster(input_features, d_model=256, nhead=8, num_layers=3, output_size=24).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    num_epochs = 200 # 本番用の十分な学習回数
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.6f}')

    print("--- 3. 学習済みモデルをGCSへ保存 ---")
    # ローカル(一時領域)にモデルを保存
    torch.save(model.state_dict(), LOCAL_WEIGHTS_PATH)

    # GCSへアップロード
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_WEIGHTS_BLOB)
    blob.upload_from_filename(LOCAL_WEIGHTS_PATH)
    print(f"モデルの重みを GCS ({MODEL_WEIGHTS_BLOB}) に保存しました。")

if __name__ == "__main__":
    run_training_pipeline()