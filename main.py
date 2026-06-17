import os
import json
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage
from google import genai

# ========== 設定周り ==========
BUCKET_NAME = "household-electic-data-20260501"
STREAM_DIR = "stream"
HISTORY_FILE_PATH = f"gs://{BUCKET_NAME}/database/history.csv"
STATUS_FILE_KEY = "database/status.json"
MODEL_WEIGHTS_BLOB = "models/transformer_weights.pth"
LOCAL_WEIGHTS_PATH = "/tmp/transformer_weights.pth"


api_key = os.getenv("GEMINI_API_KEY")

# ========== クラス定義 (推論のために構造のみ必要) ==========
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

# ========== ヘルパー関数 ==========
def load_from_gcs():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)
    if not blob.exists():
        return {"last_processed_date": "2009-12-31"}
    return json.loads(blob.download_as_text())

def save_status_gcs(status_dict):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)
    blob.upload_from_string(json.dumps(status_dict, indent=2), content_type='application/json')

def download_model_weights():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_WEIGHTS_BLOB)
    if not blob.exists():
        raise FileNotFoundError("GCS上に学習済みモデルが見つかりません。先に学習スクリプトを実行してください。")
    blob.download_to_filename(LOCAL_WEIGHTS_PATH)

def generate_advice(prediction_values, api_key):
    if not api_key: return "エラー: GEMINI_API_KEY が設定されていません。"
    client = genai.Client(api_key=api_key)
    
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
        response = client.models.generate_content(
            model='gemini-3.1-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"APIエラー: {e}"

# ========== メインパイプライン ==========
def run_daily_inference():
    status = load_from_gcs()
    last_date = datetime.datetime.strptime(status["last_processed_date"], "%Y-%m-%d")
    target_date = last_date + datetime.timedelta(days=1)
    target_date_str = target_date.strftime("%Y-%m-%d")

    if target_date.year > 2010:
        print("処理対象の年を超過しました。")
        return
    
    # 1. データの更新処理
    target_file_path = f"gs://{BUCKET_NAME}/{STREAM_DIR}/{target_date_str}.csv"
    try:
        new_day_df = pd.read_csv(target_file_path)
        history_df = pd.read_csv(HISTORY_FILE_PATH)
        updated_history_df = pd.concat([history_df, new_day_df], ignore_index=True)
        updated_history_df.to_csv(HISTORY_FILE_PATH, index=False)
        print(f"データ更新完了: {target_date_str}")
    except Exception as e:
        print(f"データ結合エラー: {e}")
        return

    # 2. 直近データの前処理
    df = updated_history_df.copy()
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    # 不要になった文字列の列削除
    df = df.drop(columns=['Date', 'Time'])

    # 文字列('?'など)が存在している対策
    #　DateTime以外を数値型に変換(エラー時はNaN)
    numeric_cols = df.columns.drop('DateTime')
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
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

    # --- 追加: データ行数チェック ---
    if len(df_final) == 0:
        print(f"スキップ: 予測に必要なデータが蓄積されていません（現在の有効な時間数: {len(df_hourly)}h）")
        print("最低でも数日分のデータがhistory.csvに蓄積されるまで推論を待機します。")
        return # ここで処理を安全に終了させる
    # --------------------------------

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_final) # ※厳密には学習時のscalerの再利用が推奨されます

    col_to_idx = {col: c for c, col in enumerate(df_final.columns)}
    target_idx = col_to_idx['Global_active_power_mean']

    # 3. モデルのロードと推論
    print("--- GCSからモデルをダウンロードし、推論を実行 ---")
    download_model_weights()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_features = len(df_final.columns)
    model = TransformerForecaster(input_features, d_model=256, nhead=8, num_layers=3, output_size=24)
    
    # 保存した重みをモデルに適用
    model.load_state_dict(torch.load(LOCAL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 推論には直近72時間のデータのみ使用
        latest_72h = scaled_data[-72:]
        latest_X = torch.tensor(latest_72h, dtype=torch.float32).unsqueeze(0).to(device)
        preds = model(latest_X)

    preds_np = preds.cpu().numpy()
    dummy = np.zeros((preds_np.shape[0] * preds_np.shape[1], len(df_final.columns)))
    dummy[:, target_idx] = preds_np.flatten()
    tomorrow_prediction_kw = scaler.inverse_transform(dummy)[:, target_idx].flatten()

    print(f"明日の予測値（先頭5時間）: {tomorrow_prediction_kw[:5]}")

    # 4. Gemini連携とステータス更新
    advice = generate_advice(tomorrow_prediction_kw, api_key)
    print("\n【生成されたアドバイス】\n", advice)

    status["last_processed_date"] = target_date_str
    save_status_gcs(status)
    print(f"パイプライン完了: {target_date_str}")

if __name__ == "__main__":
    print("--- Cloud Run Job: 日次推論パイプラインを開始します ---")
    run_daily_inference()
    print("--- Cloud Run Job: 全ての処理が完了しました ---")
