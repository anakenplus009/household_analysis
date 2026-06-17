import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv



"""
データの読み込み、整合、EDA
"""


# データの読み込み
df = pd.read_csv('gs://household-electic-data-20260501/raw_data/power_usage.csv')
# Datetime作成
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

# 欠損値の補完
df_cleaned = df.interpolate(method='linear', limit_direction='both')

#リサンプリング
df_hourly = df_cleaned.set_index('DateTime').resample('h').agg({
    'Global_active_power':['sum', 'mean', 'max', 'std'],
    'Global_reactive_power': ['mean'],
    'Voltage':['mean']
})

# カラム名の階層をフラットにする(ex:'Global_active_power_sum')
df_hourly.columns = [f"{col[0]}_{col[1]}" for col in df_hourly.columns]

# 多段階予測に向けた「ラグ特徴量」の作成
# 1時間前と1日前（24時間前）の平均消費量を追加する
df_hourly['gap_mean_lag1'] = df_hourly['Global_active_power_mean'].shift(1)
df_hourly['gap_mean_lag24'] = df_hourly['Global_active_power_mean'].shift(24)

# 外生変数の追加（カレンダー情報）
df_hourly['hour'] = df_hourly.index.hour
df_hourly['dayofweek'] = df_hourly.index.dayofweek
df_hourly['is_weekend'] = df_hourly.index.dayofweek.isin([5,6]).astype(int)


# 窓関数の作成
def create_window(data, window_size=72, forecast_size=24, target_col_index=0):
    """
    data: 特徴量を含むnumpy配列 (N, features)
    window_size: 入力として使う過去のステップ数 (72)
    forecast_size: 予測する未来のステップ数 (24)
    target_col_idx: 予測対象（y）とするカラムのインデックス
    """

    X, y = [], []
    
    # 全データから (入力窓 + 予測窓) 分を確保できる範囲でループ
    for i in range(len(data) - window_size - forecast_size +1):
        # 72hの全特徴量
        X.append(data[i:i + window_size, :])
        # 未来24時間のターゲット変数
        y.append(data[i + window_size : i + window_size + forecast_size, target_col_index])

    return np.array(X), np.array(y)

# 手順
# 1.欠損値の削除
df_final = df_hourly.dropna()

# 2.データの正規化(PyTorch採用を見据えて)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_final)

# 3.窓の作成
col_to_idx = {col: c for c, col in enumerate(df_final.columns)}
print(col_to_idx)

target_idx = col_to_idx['Global_active_power_mean']
X, y = create_window(scaled_data, window_size=72, forecast_size=24, target_col_index=target_idx)



"""
多段階モデルの実装
使用するモデル：Transformer
"""


# PyTorchによる多段階予測モデルの実装
class TransformerForecaster(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dim_feedforward=2048, dropout=0.2):
        super(TransformerForecaster, self).__init__()

        # 1. 入力層（特徴量をd_modelの次元へ投影）
        self.embedding = nn.Linear(input_size, d_model)

        # 2. 位置符号化 (Positional Encoding)
        # 時系列の「順番」を教えるための固定ベクトル
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. 出力層（未来24時間を予測）
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # x: [batch, 72, input_size]
        x = self.embedding(x) # [batch, 72, d_model]

        # 位置情報の付加
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformerによる特徴抽出
        x = self.transformer_encoder(x)

        # 最後のタイムステップ、あるいは全体の平均を使用して出力
        last_out = x[:, -1, :]
        return self.fc(last_out) # [batch, 24]
    
# パラメータ設定
input_features = len(df_final.columns)
d_model = 256
n_head = 8
num_layers = 3
forecast_steps = 24 #24h

model = TransformerForecaster(input_features, d_model, n_head, num_layers, forecast_steps)


# 学習のためのDataLoader作成
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# データセットの作成
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 損失関数と最適化手法
criterion = nn.MSELoss() #平均二乗誤差
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


"""
作成したモデルの推論・評価
"""

# 推論フェーズ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_losses = []
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 勾配の初期化
        optimizer.zero_grad()
        # 予測
        outputs = model(batch_X)
        # 損失計算 (outputs: [32, 24], batch_y: [32, 24])
        loss = criterion(outputs, batch_y)

        # 逆伝播と最適化
        loss.backward()
        # 勾配が大きくなりすぎないように制限（Transformerでは重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# matplotlibで可視化
plt.figure(figsize=(20, 5))
plt.plot(train_losses, label='training loss')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.show()


# 評価フェーズ
# 1. 評価モードに切り替え、テストデータを1つ取り出す
model.eval()
with torch.no_grad():
    test_X, test_y = next(iter(train_loader))
    test_X, test_y = test_X.to(device), test_y.to(device)

    preds = model(test_X)

# 2. 元のスケールに戻すための準備
# 予測値（preds）と正解値（test_y）をCPUへ戻してnumpy化
preds_np = preds.cpu().numpy()
actual = test_y.cpu().numpy()

# スケーラーを逆適用するために、ダミーの配列（カラム数分）を作成
def inverse_transform_target(data, scaler, target_idx, num_features):
    """
    特定の一列（ターゲット）だけを逆変換するための補助関数
    data: [Batch, 24] の予測値または正解値
    """

    dummy = np.zeros((data.shape[0] * data.shape[1], num_features))
    dummy[:, target_idx] = data.flatten()
    inverse = scaler.inverse_transform(dummy)
    return inverse[:, target_idx].reshape(data.shape[0], data.shape[1])

# カラム数とターゲット列のインデックスを取得
num_features = len(df_final.columns)
target_idx = col_to_idx['Global_active_power_mean']

# 逆変換の実行
preds_kw = inverse_transform_target(preds_np, scaler, target_idx, num_features)
actual_kw = inverse_transform_target(actual, scaler, target_idx, num_features)

# 3. 最初のサンプル（24時間分）をプロット
plt.figure(figsize=(12,5))
plt.plot(actual_kw[0], label='actual(kw)', color='Blue', marker='o')
plt.plot(preds_kw[0], label='predicted(kw)', color='Red', linestyle='--', marker='x')
plt.xlabel('Hours')
plt.ylabel('Global active Power(kW)')
plt.legend()
plt.grid(True)
plt.show()


"""
GeminiAPIを利用したアドバイス機能の追加
・ビジネスとしての付加価値を高める
・予測値に対して今後のアクションをアドバイス
"""

# 1. .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("エラー: GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")

# 2. Gemini クライアントの初期化
# 引数を空にすると、自動的に環境変数「GEMINI_API_KEY」を参照します
# client = genai.Client()


def generate_advice(prediction_value):
    # APIキーの設定（環境変数から取得するのが安全）
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    "あなたは優秀なエネルギーアナリストです。明日の電力予測値は {prediction_value}kW です。この値を踏まえた具体的な節電アドバイスを100文字以内で作成してください"
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

# 予測直後に実行
print("\n--- Geminiからのアドバイスを生成中... ---")
advice = generate_advice(preds_np)

print("\n--- 生成されたアドバイス ---")
print(advice)

