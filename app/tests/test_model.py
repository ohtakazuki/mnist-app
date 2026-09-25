"""model.py の単体テスト。

モデル本体の精度ではなく「契約（入出力の形）」だけを検証する。
Predictor 以降のコードは『model(x) が (B, 10) を返す』前提で書かれているので、
その前提が崩れていないことをここで担保する。
"""

import torch

from model import MnistCNN


def test_forward_output_shape():
    """(B,1,28,28) を入れたら (B,10) が返る、という契約を固定する。"""
    model = MnistCNN()
    model.eval()
    x = torch.zeros(4, 1, 28, 28)  # バッチ4の擬似入力
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_forward_single_sample():
    """バッチ=1（推論時の実際の形）でも (1,10) が返る。"""
    model = MnistCNN()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 28, 28))
    assert out.shape == (1, 10)
