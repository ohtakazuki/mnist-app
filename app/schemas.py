"""FastAPI が入出力を検証・ドキュメント化するための Pydantic スキーマ。

これらのクラスを書いておくと、
- リクエストボディの型チェックが自動で走る
- /docs（Swagger UI）に正確な JSON Schema が表示される
- レスポンスも `response_model` で型に揃えて返せる
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class PredictionResponse(BaseModel):
    class_id: int = Field(..., description="予測クラスID（0〜9）")
    class_name: str = Field(..., description="クラス名（文字列表現）")
    confidence: float = Field(..., ge=0.0, le=1.0, description="予測確率の最大値")
    probabilities: List[float] = Field(..., description="クラス0〜9の確率（合計1.0）")


class ImageJsonRequest(BaseModel):
    pixels: List[List[float]] = Field(
        ...,
        description="28x28 のグレースケール画素値。値は 0.0〜1.0 を想定。",
    )

    @field_validator("pixels")
    @classmethod
    def _check_shape(cls, v: List[List[float]]) -> List[List[float]]:
        # 行数・各行の長さを検証。np.array に渡す前に弾くことで 500 ではなく 422 を返す
        if len(v) != 28 or any(len(row) != 28 for row in v):
            raise ValueError("pixels は 28x28 である必要があります")
        return v