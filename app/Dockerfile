# ベースイメージ
FROM python:3.12-slim

# Python 実行環境の定番設定
# PYTHONUNBUFFERED=1      : 標準出力をバッファせずログを即時表示（コンテナ運用の基本）
# PYTHONDONTWRITEBYTECODE=1: .pyc を生成しない（コンテナでは不要・イメージを汚さない）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 作業ディレクトリ
WORKDIR /app

# PyTorch (CPU版) を先にインストール
# CPU専用インデックスから入れる。requirements.txt に torch を書かないことで
# CUDA同梱版での上書き（巨大化）を防ぐ。本番では torch==2.x.x のようにバージョン固定推奨
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# 残りの依存をインストール
# torch/torchvision は requirements.txt に含めない（上で入れたCPU版を維持するため）
# アプリコードより先に入れることで、コード変更時もこの重いレイヤをキャッシュ再利用できる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY model.py inference.py schemas.py ui.py main.py ./
COPY model.pth .

# 非rootユーザーで実行（本番デプロイ向けの基本的なセキュリティ対策）
# コードとモデルは読み取りのみなので、root所有のままでも appuser から参照可能
RUN useradd --create-home appuser
USER appuser

# ネットワーク設定
# コンテナの 8000 番で待ち受けることをドキュメント化
EXPOSE 8000

# 起動コマンド
# 0.0.0.0 にバインドしないとコンテナ外から到達できない
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
