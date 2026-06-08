# household_analysis
過去の消費電力と気象データから、「翌24時間の消費電力」を予測します。

# Household Electricity Consumption Multi-Step Forecasting System

## 1. プロジェクト概要
- **目的:** 過去の電力消費量データから、未来の複数コマ（例: 24時間先まで）の消費量を予測し、Gemini APIを用いて最適な省エネアクションを提案する。
- **背景:** 伝統的な時系列モデル（ARIMA等）では捉えきれない長期依存関係をLSTM/Transformerでモデル化。

## 2. システム構成図（Architecture）
```mermaid
flowchart TD
    %% スタイル定義
    classDef source fill:#ECEFF1,stroke:#37474F,stroke-width:2px;
    classDef de fill:#E3F2FD,stroke:#0D47A1,stroke-width:2px;
    classDef ml fill:#F3E5F5,stroke:#4A148C,stroke-width:2px;
    classDef cicd fill:#FFFDE7,stroke:#F57F17,stroke-width:2px;
    classDef insight fill:#E0F7FA,stroke:#006064,stroke-width:2px;

    subgraph Data_Sources [1. Data Sources]
        UCI[UCI Web Repository<br>household_power_consumption.txt]:::source
    end

    subgraph Data_Pipeline ["2. Data Engineering Pipeline (GCP)"]
        GCS[(Cloud Storage<br>Raw Data Bucket)]:::de
        CloudRun[Cloud Run<br>Pre-processing & Clean<br>1-hour Resampling & Imputation]:::de
        BQ[(BigQuery<br>Cleaned Feature Store)]:::de
    end

    subgraph CICD_Automation [CI/CD & MLOps Automation]
        GHA[GitHub Actions<br>CI/CD Trigger & Push]:::cicd
    end

    subgraph ML_AI_Phase ["3. Machine Learning & AI (Vertex AI)"]
        VAP[Vertex AI Pipelines<br>Orchestration]:::ml
        VAW[Vertex AI Workbench<br>JupyterLab / EDA & Prototyping]:::ml
        VAT[Vertex AI Training<br>Python Scripts: train.py<br>Model: Transformer]:::ml
        VAMR[Vertex AI Model Registry<br>Version Control]:::ml
        VAE[Vertex AI Endpoint<br>Online Inference / predict.py]:::ml
    end

    subgraph Insights [4. Insights & Consumption]
        Gemini["Gemini API (LLM)"<br>Generative Text:<br>Actionable Energy Advice]:::insight
        Dashboard[Looker Studio / Streamlit<br>Interactive Dashboard]:::insight
    end

    %% データと処理の流れ
    UCI -->|Upload| GCS
    GCS -->|Trigger| CloudRun
    CloudRun -->|Load| BQ
    
    %% CI/CDと学習の自動化フロー
    GHA -->|Deploy Pipeline| VAP
    VAP -->|Execute Script| VAT
    BQ -->|Fetch Training Data| VAT
    VAW -.->|Push Code to GitHub| GHA
    
    VAT -->|Register Model| VAMR
    VAMR -->|Deploy| VAE
    
    %% 推論と可視化の流れ
    VAE -->|Inference Output| Gemini
    VAE -->|Predictions| Dashboard
    Gemini -->|Natural Language Reports| Dashboard
```
- データの収集(Ingest) -> 蓄積(Store) -> 変換(Transform) -> 学習/予測(ML) -> 意思決定(LLM)の流れを説明。

## 3. 技術スタック（Tech Stack）
- **言語:** Python 3.12+
- **データ基盤・加工:** Pandas / BigQuery
- **モデリング:** PyTorch (LSTM, Transformer)
- **LLM統合:** Gemini API (google-genai)
- **環境構築:** Docker / Docker Compose (予定)

## 4. データパイプライン & 特徴量エンジニアリング
- 使用データ: UCI Individual household electric power consumption dataset
- 実装した特徴量:
  - 時間ベース: 曜日、時間、週末
  - ラグ変数: $t-1$, $t-24$ などの過去の消費量
  - ローリング特徴量: 過去24時間の移動平均、移動標準偏差

## 5. モデル評価（Evaluation）
- 予測ホライズン: 未来Nステップ（例: 24時間）
- 評価指標の比較テーブル:
  | Model | MSE |
  | :--- | :--- |
  | Baseline (Last Value) | 0.XX |
  | LSTM | 0.XX |
  | Transformer | **0.XX** |

## 6. Gemini APIによるアドバイス生成の例
- 予測された値をもとに、Geminiがどのような「インサイト」や「アクションプラン」を出力したかのサンプルログを掲載。
