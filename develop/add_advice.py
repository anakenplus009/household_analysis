import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

# 1. .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からAPIキーを取得
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("エラー: GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")

# 2. Gemini クライアントの初期化
# 引数を空にすると、自動的に環境変数「GEMINI_API_KEY」を参照します
client = genai.Client()

def generate_power_advice(forecast_df):
    """
    予測データを基にGemini APIでアドバイスを生成する関数
    """
    # 予測データをテキスト形式に変換してプロンプトに組み込む
    data_summary = forecast_df.to_string(index=False)
    
    prompt = f"""
    あなたは優秀なエネルギーアナリストです。
    以下に示す「翌週の予測電力消費量データ」を分析し、一般家庭または管理者向けに以下の3点を含んだ具体的なアドバイスをそれぞれ３００字程度で生成してください。

    1. 全体的な消費傾向と、最も注意すべき「ピーク時間帯（曜日・時間）」
    2. ピーク時における具体的な省エネ・節電アクションプラン
    3. 前週比やベースライン（想定）と比較した際の特徴的な動き（もしあれば）

    # 予測電力消費量データ（kW）:
    {data_summary}
    """

    try:
        # Gemini 2.5 Flash などの軽量・高速モデルを使用
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"API呼び出し中にエラーが発生しました: {e}"

# --- 動作確認用のダミーデータ (3週目で出力した予測結果のイメージ) ---
if __name__ == "__main__":
    # 例: 翌日の12時間分の予測データ
    dummy_forecast = pd.DataFrame({
        "Timestamp": [f"2026-05-19 {hour:02d}:00" for hour in range(8, 20)],
        "Predicted_Consumption_kW": [1.2, 1.5, 2.3, 2.8, 3.5, 3.8, 2.9, 2.1, 1.8, 2.5, 3.2, 1.5]
    })
    
    print("--- 予測データ ---")
    print(dummy_forecast)
    print("\n--- Geminiからのアドバイスを生成中... ---")
    
    advice = generate_power_advice(dummy_forecast)
    
    print("\n--- 生成されたアドバイス ---")
    print(advice)