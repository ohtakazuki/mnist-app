"""schemas.py の単体テスト。

入力検証（28x28 のガード）と出力スキーマの制約（confidence の範囲）が
意図どおり効いているかを、HTTP を介さずモデル単体で確認する。
"""

import pytest
from pydantic import ValidationError

from schemas import ImageJsonRequest, PredictionResponse


def _grid(rows: int, cols: int) -> list[list[float]]:
    """rows x cols の 0.0 埋めグリッドを作るヘルパ。"""
    return [[0.0] * cols for _ in range(rows)]


def test_accepts_valid_28x28():
    req = ImageJsonRequest(pixels=_grid(28, 28))
    assert len(req.pixels) == 28
    assert len(req.pixels[0]) == 28


@pytest.mark.parametrize("rows,cols", [(27, 28), (29, 28), (28, 27), (28, 29), (0, 0)])
def test_rejects_wrong_shape(rows, cols):
    """28x28 以外は ValidationError（FastAPI 経由なら 422 になる）。"""
    with pytest.raises(ValidationError):
        ImageJsonRequest(pixels=_grid(rows, cols))


def test_rejects_ragged_rows():
    """行ごとに長さが違う（ギザギザ）配列も弾く。"""
    pixels = _grid(28, 28)
    pixels[0] = [0.0] * 27  # 1行だけ短くする
    with pytest.raises(ValidationError):
        ImageJsonRequest(pixels=pixels)


def test_prediction_response_confidence_must_be_0_1():
    """confidence は ge=0/le=1 制約。範囲外は弾く。"""
    with pytest.raises(ValidationError):
        PredictionResponse(
            class_id=1, class_name="1", confidence=1.5, probabilities=[0.0] * 10
        )


def test_prediction_response_valid():
    resp = PredictionResponse(
        class_id=3, class_name="3", confidence=0.9, probabilities=[0.1] * 10
    )
    assert resp.class_id == 3
    assert resp.confidence == 0.9
