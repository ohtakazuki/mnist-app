"""ui.py の単体テスト。

Gradio のサーバを起動せずに、UI ロジックだけを検証する。
- _classify(image, predictor) は predictor を引数で受け取る純粋関数 → そのまま呼べる。
- classify(image) は内部で get_predictor() を呼ぶ入口 → モジュールの名前を monkeypatch して差し替える。
"""

from PIL import Image

import ui


def test_classify_none_image(fake_predictor):
    """画像が無い場合は予測せず、空の確率辞書とメッセージを返す。"""
    headline, probs = ui._classify(None, fake_predictor)
    assert "アップロードされていません" in headline
    assert probs == {}


def test_classify_returns_headline_and_probs(fake_predictor):
    """任意サイズの画像でも内部で 28x28 にリサイズし、決定的に class 7 を返す。"""
    img = Image.new("L", (100, 100), color=0)  # わざと 28x28 以外
    headline, probs = ui._classify(img, fake_predictor)
    assert headline.startswith("予測: 7")
    # gr.Label 用に "0".."9" をキーとした確率辞書になる
    assert set(probs) == {str(i) for i in range(10)}
    assert all(isinstance(v, float) for v in probs.values())


def test_classify_accepts_rgb(fake_predictor):
    """RGB 画像も convert('L') でグレースケール化されて通る。"""
    img = Image.new("RGB", (40, 40), color=(10, 20, 30))
    headline, _ = ui._classify(img, fake_predictor)
    assert headline.startswith("予測: 7")


def test_classify_entry_uses_shared_predictor(monkeypatch, fake_predictor):
    """入口 classify() は get_predictor() を呼ぶ。その名前を差し替えて検証する。

    ※ ui モジュール内に取り込まれた名前(ui.get_predictor)を上書きするのがポイント。
    """
    monkeypatch.setattr(ui, "get_predictor", lambda: fake_predictor)
    img = Image.new("L", (28, 28), color=0)
    headline, probs = ui.classify(img)
    assert headline.startswith("予測: 7")
    assert set(probs) == {str(i) for i in range(10)}
