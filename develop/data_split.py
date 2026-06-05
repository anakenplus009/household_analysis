import os
import pandas as pd
import numpy as np


#
def split_data_day(input_path, output_path):
    """
        GCS上の巨大なCSVファイルを読み込み、日ごとに分割してGCSに保存する
        
        :param input_path: 入力元のGCSパス (ex. gs://my-bucket/raw/huge_data.csv)
        :param output_path: 出力先のGCSディレクトリパス (ex. gs://my-bucket/stream/)
        :param TARGET_MONTH: 出力する月次
        """
    print(f"Loading data from: {input_path}")

    daily_buffers = {}

    # 1. データの読み込み
    chunk_size = 250000
    chunks = pd.read_csv(
        input_path,
        sep=';',
        na_values='?',
        low_memory=False,
        chunksize=chunk_size
    )

    for i, chunk in enumerate(chunks):
        # 日付列を datetime 型に変換
        chunk['parse'] = pd.to_datetime(chunk['Date'], format='%d/%m/%Y', errors='coerce')
        # 2010年を分割する対象のフィルタリング
        condition = chunk['parse'].dt.year == 2010
        filtered_chunk = chunk[condition].copy()

        if filtered_chunk.empty:
            continue

        filtered_chunk['date_str'] = filtered_chunk['parse'].dt.strftime('%Y-%m-%d')
        # メモリ上のバッファに日付ごとに格納
        for date_str, group in filtered_chunk.groupby('date_str'):
            clean_group = group.drop(columns=['parse', 'date_str'])
            if date_str not in daily_buffers:
                daily_buffers[date_str] = []
            daily_buffers[date_str].append(clean_group)
        
        print(f" Chunk {i+1} processed.")

    if not daily_buffers:
            return
        
    for date_str, df_list in daily_buffers.items():
        final_day_df = pd.concat(df_list, ignore_index=True)
        output_file_path = os.path.join(output_path, f"{date_str}.csv").replace("\\", "/")
        #
        final_day_df.to_csv(output_file_path, index=False)
        print(f"Successfully saved:{output_file_path}")


if __name__ == "__main__":
    INPUT_PATH = "gs://household-electic-data-20260501/household_power_consumption.txt"
    OUTPUT_DIR = "gs://household-electic-data-20260501/stream"
    
    split_data_day(INPUT_PATH, OUTPUT_DIR)




