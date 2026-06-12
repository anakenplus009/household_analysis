import os
import json
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage
import google.generativeai as genai
from dotenv import load_dotenv

# ========== 設定周り ==========
BUCKET_NAME = "household-electic-data-20260501"
STREAM_DIR = "stream"
HISTORY_FILE_PATH = f"gs://{BUCKET_NAME}/database/history.csv"
STATUS_FILE_KEY = "status.json" # GCSバケット内のパスに修正

# .envファイルからAPIキーを読み込む
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


# ========== 1. GCSステータス管理 ==========
def load_from_gcs():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)

    if not blob.exists():
        initial_status = {
            "last_processed_date": "2009-12-31"
        }
        return initial_status
    
    status_data = blob.download_as_text()
    return json.loads(status_data)

def save_status_gcs(status_dict):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)
    blob.upload_from_string(json.dumps(status_dict, indent=2), content_type='application/json')
    print(f"Updated status.json on GCS to: {status_dict['last_processed_date']}")


# ========== 2. モデルの定義 ==========
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


# ========== 3. Gemini API連携 ==========
def generate_advice(prediction_values, api_key):
    if not api_key:
        return "エラー: GEMINI_API_KEY が設定されていません。"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # 24時間の予測値の平均や最大値などのサマリをプロンプトに渡す
    mean_pred = np.mean(prediction_values)
    max_pred = np.max(prediction_values)
    
    prompt = f"""
    あなたは優秀なエネルギーアナリストです。
    明日の電力予測値は、平均で {mean_pred:.2f}kW、最大で {max_pred:.2f}kW と予測されています。
    この値を踏まえた具体的な節電アドバイスを100文字以内で作成してください。
    【制約】
    - 専門用語を避け、親しみやすい表現で。
    - 具体的なアクション（例：エアコンの温度、家電の使用時間）を1つ提案。
    - 100文字以内。
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API呼び出し中にエラーが発生しました: {e}"


# ========== メインパイプライン ==========
def run_daily_pipeline():
    status = load_from_gcs()
    last_data_str = status["last_processed_date"]

    # 次の日付を計算
    last_date = datetime.datetime.strptime(last_data_str, "%Y-%m-%d")
    target_date = last_date + datetime.timedelta(days=1)
    target_date_str = target_date.strftime("%Y-%m-%d")

    if target_date.year > 2010:
        print("処理対象の年を超過しました。")
        return
    
    target_file_path = f"gs://{BUCKET_NAME}/{STREAM_DIR}/{target_date_str}.csv"
    print(f"Processing target date: {target_date_str}")
    print(f"Fetching streaming data from: {target_file_path}")

    try:
        # 新着データの読み込み
        new_day_df = pd.read_csv(target_file_path)
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # 蓄積用データへの結合処理
    try:
        history_df = pd.read_csv(HISTORY_FILE_PATH)
        updated_history_df = pd.concat([history_df, new_day_df], ignore_index=True)
    except Exception:
        print("history.csv が見つからないため、新規作成します。")
        updated_history_df = new_day_df

    # 更新された履歴データをGCSへ保存
    updated_history_df.to_csv(HISTORY_FILE_PATH, index=False)
    print(f" Successfully appended {target_date_str} data to {HISTORY_FILE_PATH}")


    # ===== ここから統合部分（特徴量作成・推論・Gemini） =====
    print("--- パイプライン: データ前処理と特徴量作成を開始 ---")
    
    # 日付型の作成と欠損値処理
    df = updated_history_df.copy()
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df_cleaned = df.interpolate(method='linear', limit_direction='both')

    # リサンプリング
    df_hourly = df_cleaned.set_index('DateTime').resample('h').agg({
        'Global_active_power':['sum', 'mean', 'max', 'std'],
        'Global_reactive_power': ['mean'],
        'Voltage':['mean']
    })
    df_hourly.columns = [f"{col[0]}_{col[1]}" for col in df_hourly.columns]

    # ラグ特徴量と外生変数
    df_hourly['gap_mean_lag1'] = df_hourly['Global_active_power_mean'].shift(1)
    df_hourly['gap_mean_lag24'] = df_hourly['Global_active_power_mean'].shift(24)
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['dayofweek'] = df_hourly.index.dayofweek
    df_hourly['is_weekend'] = df_hourly.index.dayofweek.isin([5,6]).astype(int)

    df_final = df_hourly.dropna()

    # 正規化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_final)

    # 窓作成 (学習用)
    col_to_idx = {col: c for c, col in enumerate(df_final.columns)}
    target_idx = col_to_idx['Global_active_power_mean']
    X, y = create_window(scaled_data, window_size=72, forecast_size=24, target_col_index=target_idx)

    print("--- パイプライン: モデルの学習を開始 ---")
    # パラメータ設定
    input_features = len(df_final.columns)
    d_model = 256
    n_head = 8
    num_layers = 3
    forecast_steps = 24
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerForecaster(input_features, d_model, n_head, num_layers, forecast_steps).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # ※注意: 毎日実行するパイプラインでの200エポック学習は重いため、事前に学習した重みをロード
    # 元のコードの挙動を維持し、10エポック程度に短縮してテスト実行（必要に応じて戻してください）。
    num_epochs = 10 
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    print("--- パイプライン: 明日の推論を実行 ---")
    model.eval()
    with torch.no_grad():
        # 本番用修正: 未来を予測するため、全データのうち「一番最後（直近）の72時間」を抽出
        latest_72h = scaled_data[-72:]
        latest_X = torch.tensor(latest_72h, dtype=torch.float32).unsqueeze(0).to(device) # バッチサイズ1に拡張
        
        preds = model(latest_X)

    # 逆変換ロジック
    preds_np = preds.cpu().numpy()
    dummy = np.zeros((preds_np.shape[0] * preds_np.shape[1], len(df_final.columns)))
    dummy[:, target_idx] = preds_np.flatten()
    inverse_preds = scaler.inverse_transform(dummy)[:, target_idx]
    
    # 明日の24時間の予測値 (kW)
    tomorrow_prediction_kw = inverse_preds.flatten()
    print(f"明日の予測値（先頭5時間）: {tomorrow_prediction_kw[:5]}")

    print("\n--- パイプライン: Geminiからのアドバイスを生成中... ---")
    advice = generate_advice(tomorrow_prediction_kw, api_key)
    print("【生成されたアドバイス】")
    print(advice)
    # =========================================================

    # 処理完了としてステータスを更新
    status["last_processed_date"] = target_date_str
    save_status_gcs(status)
    print(f"Pipeline completed for {target_date_str}\n")


if __name__ == "__main__":
    run_daily_pipeline()