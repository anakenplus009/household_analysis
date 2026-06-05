import os
import json
import datetime
import pandas as pd
from google.cloud import storage


BUCKET_NAME = "household-electic-data-20260501"
STREAM_DIR = "stream"
HISTORY_FILE_PATH = "gs://household-electic-data-20260501/database/history.csv"
STATUS_FILE_KEY = "gs://household-electic-data-20260501/database/status.json"

# GCSから現在のステート(json)を読み込む
def load_from_gcs():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)

    #
    if not blob.exists():
        initial_status = {
            "last_processed_date": "2009-12-31"
        }
        return initial_status
    
    status_data = blob.download_as_text()
    return json.loads(status_data)


# 更新したステート(json)をGCSに保存する
def save_status_gcs(status_dict):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STATUS_FILE_KEY)
    blob.upload_from_string(json.dump(status_dict, indent=2), content_type='application/json')
    print(f"Updated status.json on GCS to: {status_dict['last_processed_date']}")


def run_daily_pipeline():
    status = load_from_gcs()
    last_data_str = status["last_processed_date"]

    # 次の日付を計算
    last_date = datetime.datetime.strptime(last_data_str, "%Y-%m-%d")
    target_date = last_date + datetime.timedelta(days=1)
    target_date_str = target_date.strptime("%Y-%m-%d")

    #
    if target_date.year > 2010:
        return
    
    #
    target_file_path = f"gs://{BUCKET_NAME}/{STREAM_DIR}/{target_date_str}.csv"
    print(f"Processing target date: {target_date_str}")
    print(f"Fetching streaming data from: {target_file_path}")

    try:
        # 新着データの読み込み
        new_day_df = pd.read_csv(target_file_path)
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました (存在しない可能性があります): {e}")


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


    # ===== ここに既存の「5. 特徴量作成」「6. PyTorch推論」「7. Gemini連携」を差し込む =====
    # 例: 
    # features = generate_lag_features(updated_history_df)
    # prediction = model.predict(features.tail(24)) # 直近データを使って未来予測
    # advice = call_gemini_api(prediction)
    # =========================================================================


    #
    status["last_processed_date"] = target_date_str
    save_status_gcs(status)


if __name__ == "__main__":
    run_daily_pipeline()
    