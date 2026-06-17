# 安定していて軽量なPythonベースイメージを使用
FROM python:3.12-slim

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 依存関係のリストをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# スクリプト本体をコピー
COPY main.py .

# コンテナ起動時に実行するコマンド
CMD ["python", "main.py"]