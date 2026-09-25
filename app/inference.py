"""学習済みモデルの読み込みと推論を担う薄いラッパ。

ポイント：
- 起動時に1回だけモデルをロード（リクエストのたびに読み直さない）。
- `model.eval()` で Dropout / BatchNorm を推論モードに切り替える。
- `torch.no_grad()` で計算グラフを構築しないようにする（メモリ・速度・安全性の利得）。
"""

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import torch

from model import MnistCNN

# モデルの重みパスをここで一元管理
# 既定はリポジトリ同梱の model.pth。テスト/CI では MNIST_WEIGHTS で差し替え可能
WEIGHTS_PATH = Path(os.getenv("MNIST_WEIGHTS", str(Path(__file__).parent / "model.pth")))

class Predictor:
    def __init__(self, model):
        # モデルオブジェクトを受け取るだけ（テストではモックを注入できる）
        self.model = model
        self.model.eval()

    @classmethod
    def from_weights(cls, weights_path: str | Path) -> "Predictor":
        # ディスクからの読み込みはこちらに分離
        model = MnistCNN()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(model)

    def predict(self, image_2d: np.ndarray) -> dict:
        # (28, 28) を厳格に要求。リサイズは呼び出し側（前処理）の責務とし、
        # ここに不正形状が来たら黙って誤推論せず即エラーにする
        if image_2d.shape != (28, 28):
            raise ValueError(f"image_2d は (28, 28) を期待。received shape={image_2d.shape}")
        # (28, 28) -> (1, 1, 28, 28)
        x = torch.from_numpy(image_2d).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        cls_id = int(probs.argmax())
        return {
            "class_id": cls_id,
            "class_name": str(cls_id),
            "confidence": float(probs[cls_id]),
            "probabilities": [float(p) for p in probs],
        }

# プロセス内で1個だけ Predictor を生成・共有する
@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """プロセス内で唯一の Predictor を返す（モデルのロードは初回のみ）。

    `@lru_cache(maxsize=1)` により、何度呼んでも同じインスタンスが返る。
    （例外はキャッシュされないため、初回ロード失敗時は次回呼び出しで再試行される）
    main.py と ui.py の双方がこの関数を呼ぶことで、
    モデルのロードがプロセス全体で1回だけになる。
    """
    return Predictor.from_weights(WEIGHTS_PATH)
