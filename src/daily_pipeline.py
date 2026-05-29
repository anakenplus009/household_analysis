import os
import json
import datetime
import pandas as pd
from google.cloud import storage

# --- 設定値 ---
BUCKET_NAME = "your-project-bucket"
STREAM_DIR = "stream_simulated"         # 日別CSVが格納されているフォルダ
HISTORY_FILE_PATH = "gs://your-project-bucket/data_base/history.csv"
STATUS_FILE_KEY = "data_base/status.json"  # 状態管理ファイル（バケット内のパス）


def load_status_from_gcs():
    """GCSから現在のステート(json)を読み込む"""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)
    
    # 完全に新規で、status.jsonがまだGCSにない場合の初期値
    if not blob.exists():
        initial_status = {
            "last_processed_date": "2009-12-31"  # 2010年の前日からスタート
        }
        return initial_status
        
    status_data = blob.download_as_text()
    return json.loads(status_data)


def save_status_to_gcs(status_dict):
    """更新したステート(json)をGCSに保存する"""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)
    blob.upload_from_string(json.dumps(status_dict, indent=2), content_type='application/json')
    print(f"Updated status.json on GCS to: {status_dict['last_processed_date']}")


def run_daily_pipeline():
    # 1. 前回のステートを読み込み
    status = load_status_from_gcs()
    last_date_str = status["last_processed_date"]
    
    # 2. 次に処理すべき「明日」の日付を計算
    last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
    target_date = last_date + datetime.timedelta(days=1)
    target_date_str = target_date.strftime("%Y-%m-%d")
    
    # 2010年をオーバーした場合のストッパー（運用中のバグ防止）
    if target_date.year > 2010:
        print("💡 2010年のデータはすべて処理が完了しています。処理をスキップします。")
        return

    # 3. 対象日の新着CSVのパスを指定して読み込み
    target_file_path = f"gs://{BUCKET_NAME}/{STREAM_DIR}/{target_date_str}.csv"
    print(f"Processing target date: {target_date_str}")
    print(f"Fetching streaming data from: {target_file_path}")
    
    try:
        # 新着データの読み込み
        new_day_df = pd.read_csv(target_file_path)
    except Exception as e:
        print(f"❌ ファイルの読み込みに失敗しました (存在しない可能性があります): {e}")
        return

    # 4. 蓄積用データ（history.csv）へのアペンド処理
    # ※ 初回などでhistory.csvが存在しない場合は、新着データだけで新規作成する
    try:
        history_df = pd.read_csv(HISTORY_FILE_PATH)
        # 結合
        updated_history_df = pd.concat([history_df, new_day_df], ignore_index=True)
    except Exception:
        print("⚠️ history.csv が見つからないため、新規作成します。")
        updated_history_df = new_day_df

    # 更新された履歴データをGCSへ保存
    updated_history_df.to_csv(HISTORY_FILE_PATH, index=False)
    print(f" Successfully appended {target_date_str} data to {HISTORY_FILE_PATH}")

    # ===== ここに既存の「5. 特徴量作成」「6. PyTorch推論」「7. Gemini連携」を差し込む =====
    # 例: 
    # features = generate_lag_features(updated_history_df)
    # prediction = model.predict(features.tail(24)) # 直近データを使って未来予測
    # advice = call_gemini_api(prediction)
    # =========================================================================

    # 8. すべて成功したらステートを更新してGCSへ書き戻す
    status["last_processed_date"] = target_date_str
    save_status_to_gcs(status)


if __name__ == "__main__":
    run_daily_pipeline()