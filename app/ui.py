"""Gradio UI（手書き数字を画像でアップロード→予測クラスと確率を表示）。

このファイルが「フロントエンド」担当
FastAPI（main.py）の/predict相当の処理をGradioのInterfaceでラップし、
gr.mount_gradio_app(app, demo, path="/ui")でFastAPI に同居させる構成。

ポイント:
- 単独でも起動できる python ui.py
- ただし本研修ではFastAPIと同居
- Gradio は内部で関数を呼ぶだけなので、API ロジック（Predictor）を共有できる。
"""

import gradio as gr
import numpy as np
from PIL import Image
# Predictorを直接インポートせず、共有のget_predictor()を呼ぶことでmain.py側とモデルを共有
from inference import get_predictor

def _classify(image: Image.Image, predictor) -> tuple[str, dict]:
    """推論の本体（テスト用に predictor を引数で受け取る純粋関数）。

    Gradio からは入口関数 classify() 経由で呼ばれる。

    Args:
        image: アップロードされた PIL Image（RGB or L）。
        predictor: 共有される Predictor インスタンス。

    Returns:
        (見出し文字列, クラスごとの確率辞書)。
        Gradio の `gr.Label` は dict[str, float] を渡すと棒グラフで表示してくれる。
    """
    if image is None:
        return "画像がアップロードされていません", {}
    # 注: MNIST は黒地に白。白地に黒の画像は反転が必要になる点に注意（main.py の /predict と同じ前提）
    gray = image.convert("L").resize((28, 28))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    result = predictor.predict(arr)
    headline = f"予測: {result['class_id']}（確信度 {result['confidence']:.1%}）"
    probs = {str(i): float(p) for i, p in enumerate(result["probabilities"])}
    return headline, probs

def classify(image: Image.Image) -> tuple[str, dict]:
    # Gradio から呼ばれる入口。本番では共有 Predictor を渡す
    return _classify(image, get_predictor())

def build_demo() -> gr.Blocks:
    """Gradio UI を組み立てて返す。"""
    with gr.Blocks(title="MNIST 手書き数字分類") as demo:
        gr.Markdown(
            """
            # MNIST 手書き数字分類アプリ
            手書き数字（0〜9）の画像をアップロードすると、CNN が分類します。

            - 内部では `inference.Predictor` を呼ぶ（`/predict` API と同じロジック）
            - グレースケール 28x28 にリサイズしてからモデルに入力
            """
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="画像をアップロード")
                submit = gr.Button("予測する", variant="primary")
            with gr.Column():
                headline = gr.Textbox(label="予測結果")
                probs = gr.Label(label="クラスごとの確率（上位3件を棒グラフ表示）", num_top_classes=3)
        submit.click(fn=classify, inputs=image_input, outputs=[headline, probs])
    return demo

demo = build_demo()

if __name__ == "__main__":
    # 単独起動用（http://127.0.0.1:7860/）
    demo.launch()