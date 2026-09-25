"""inference.py の単体テスト。

ここがアプリの心臓部。重み（model.pth）に依存せず、
偽モデル注入で「predict の変換ロジック」と「get_predictor のキャッシュ挙動」を検証する。
"""

import numpy as np
import pytest

from conftest import FakeModel
from inference import Predictor, get_predictor


# ---- Predictor.__init__ / predict ----

def test_init_calls_eval(fake_model):
    """Predictor 生成時にモデルを推論モードへ切り替える（eval を呼ぶ）。"""
    Predictor(fake_model)
    assert fake_model.eval_called is True


def test_predict_returns_injected_class(fake_predictor):
    """注入した FakeModel(target=7) のとおり class_id=7 を返す（決定的）。"""
    out = fake_predictor.predict(np.zeros((28, 28), dtype=np.float32))
    assert out["class_id"] == 7
    assert out["class_name"] == "7"


def test_predict_output_contract(fake_predictor):
    """戻り値の辞書が schema と同じ形・性質を満たす。"""
    out = fake_predictor.predict(np.zeros((28, 28), dtype=np.float32))
    assert set(out) == {"class_id", "class_name", "confidence", "probabilities"}
    assert len(out["probabilities"]) == 10
    # softmax の合計は 1.0
    assert sum(out["probabilities"]) == pytest.approx(1.0, abs=1e-5)
    # confidence は最大確率と一致し、0〜1 に収まる
    assert out["confidence"] == pytest.approx(max(out["probabilities"]))
    assert 0.0 <= out["confidence"] <= 1.0
    # 型も確認（JSON 化される値なので int / float）
    assert isinstance(out["class_id"], int)
    assert all(isinstance(p, float) for p in out["probabilities"])


@pytest.mark.parametrize("bad_shape", [(27, 28), (28, 27), (10, 10), (1, 784)])
def test_predict_rejects_wrong_shape(fake_predictor, bad_shape):
    """(28,28) 以外は黙って誤推論せず ValueError を投げる（前段で入れたガード）。"""
    with pytest.raises(ValueError):
        fake_predictor.predict(np.zeros(bad_shape, dtype=np.float32))


# ---- get_predictor（lru_cache）----

def test_get_predictor_loads_once(monkeypatch):
    """lru_cache により from_weights は一度しか呼ばれず、同一インスタンスを返す。"""
    calls = {"n": 0}

    def fake_from_weights(weights_path):
        calls["n"] += 1
        return Predictor(FakeModel())

    # ディスク読み込みを偽装（model.pth 不要）
    monkeypatch.setattr(Predictor, "from_weights", fake_from_weights)

    p1 = get_predictor()
    p2 = get_predictor()

    assert p1 is p2          # キャッシュにより同じオブジェクト
    assert calls["n"] == 1   # ロードは初回のみ


def test_get_predictor_retries_after_failure(monkeypatch):
    """ロード失敗は lru_cache に乗らないので、次回呼び出しで再試行できる。"""
    attempts = {"n": 0}

    def flaky_from_weights(weights_path):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("初回はわざと失敗させる")
        return Predictor(FakeModel())

    monkeypatch.setattr(Predictor, "from_weights", flaky_from_weights)

    with pytest.raises(RuntimeError):
        get_predictor()              # 1回目: 失敗（キャッシュされない）
    p = get_predictor()              # 2回目: 成功
    assert isinstance(p, Predictor)
    assert attempts["n"] == 2
