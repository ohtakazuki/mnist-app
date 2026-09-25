"""MNIST 分類アプリ（FastAPI + Gradio UI）

エンドポイント:
- GET  /health          稼働確認
- POST /predict         画像ファイル（multipart/form-data）を受けて分類
- POST /predict_json    28x28 の画素配列（JSON）を受けて分類
- GET  /docs            Swagger UI（FastAPI 自動生成）
- GET  /ui              Gradio UI。★UI追記

起動: `uvicorn main:app --host 0.0.0.0 --port 8000`
"""

import io
import logging

import gradio as gr 
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.responses import RedirectResponse
from PIL import Image

from inference import Predictor, get_predictor
from schemas import ImageJsonRequest, PredictionResponse
from ui import demo as gradio_demo


logger = logging.getLogger(__name__)

app = FastAPI(
    title="MNIST Classifier App",
    description="第8回 MLOps応用の題材。PyTorchの軽量CNNを使った手書き数字分類",
    version="1.0.0",
)


# 各エンドポイントに Depends(...) で Predictor を渡すための関数（依存性注入）。
# テストでは dependency_overrides でここを偽物に差し替えられる。
# モデルのロードに失敗した場合は 503（サービス利用不可）を返す。
def get_predictor_dep() -> Predictor:
    try:
        return get_predictor()
    except Exception:
        logger.exception("モデルのロードに失敗しました")
        raise HTTPException(status_code=503, detail="モデルが読み込めませんでした")


# 稼働確認。モデルが読めない場合もエラーにせず "degraded" を返す
# （docker-compose の healthcheck と K8s の readiness/liveness Probe がここを叩く）
@app.get("/health")
def health():
    try:
        get_predictor()
        ok = True
    except Exception:
        logger.warning("ヘルスチェックでモデル未ロード")
        ok = False
    return {"status": "ok" if ok else "degraded", "model_loaded": ok}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), predictor: Predictor = Depends(get_predictor_dep)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="画像のデコードに失敗しました")
    # 注: MNIST は「黒地に白文字」。白地に黒で書いた画像は階調が逆になり精度が落ちる。
    #     反転処理はあえて入れていない（入力前提を受講者に意識させるため）。
    arr = np.asarray(image.resize((28, 28)), dtype=np.float32) / 255.0
    return predictor.predict(arr)


@app.post("/predict_json", response_model=PredictionResponse)
def predict_json(req: ImageJsonRequest, predictor: Predictor = Depends(get_predictor_dep)):
    # 形状検証は ImageJsonRequest（Pydantic）に集約済み。ここでは変換のみ
    arr = np.array(req.pixels, dtype=np.float32)
    return predictor.predict(arr)


# ★UI追記（ここから）
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")

# Gradio UI を /ui にマウント
# これで FastAPI の API と Gradio の UI が同じプロセス・同じポートで配信できる
app = gr.mount_gradio_app(app, gradio_demo, path="/ui")
# ★UI追記（ここまで）