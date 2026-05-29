import os
import pandas as pd

def split_gcs_data_by_day(input_path, output_path):
    """
    GCS上の巨大なCSVファイルを読み込み、日ごとに分割してGCSに保存する
    
    :param input_path: 入力元のGCSパス (ex. gs://my-bucket/raw/huge_data.csv)
    :param output_path: 出力先のGCSディレクトリパス (ex. gs://my-bucket/stream/)
    """
    print(f"Loading data from: {input_path}")

    daily_buffers = {}
    
    # 1. データの読み込み
    chunk_size = 250000
    df = pd.read_csv(
        input_path,
        sep=';',
        na_values='?',
        chunk_size=chunk_size
    )
    
    for i, chunk in enumerate(df):
        # 2. 日付列を datetime 型に変換
        # UCIデータの場合、日と時間が別々の列になっていることがあるので、適宜結合してください
        df['parse'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        condition = chunk['parse'].dt.year == 2010
        filtered_chunk = chunk[condition].copy()

        if filtered_chunk.empty():
            continue

        filtered_chunk['date_str'] =filtered_chunk['parse'].dt.strftime('%Y-%m-%d')

        for date_str, group in filtered_chunk.groupby('date_str'):
            clean_group = group.drop(columns=['parse', 'date_str'])
            if date_str not in daily_buffers:
                daily_buffers['date_str'] = []
            
            daily_buffers['date_str'].append(clean_group)
        print(f" Chunk {i+1} processed.")

    if not daily_buffers:
            return
    
    for date_str, df_list in daily_buffers.items():
        final_day_df = pd.concat(df_list, ignore_index=True)
        # 出力先のファイルパスを作成 (例: gs://my-bucket/stream/2010-01-01.csv)
        # フォルダの末尾のスラッシュの有無を考慮
        output_file_path = os.path.join(output_path, f"{date_str}.csv").replace("\\", "/")
            
        # 4. GCSへ直接書き込み（gcsfsが裏で動く）
        final_day_df.to_csv(output_file_path, index=False)
        print(f" Successfully saved: {output_file_path}")

if __name__ == "__main__":
    # パスの設定例（ご自身の環境に合わせて変更してください）
    INPUT_PATH = "gs://your-project-bucket/raw_data/household_power_consumption.csv"
    OUTPUT_DIR = "gs://your-project-bucket/stream_simulated/"
    
    
    split_gcs_data_by_day(INPUT_PATH, OUTPUT_DIR)