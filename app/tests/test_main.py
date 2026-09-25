"""main.py の API テスト（TestClient 経由）。

実 HTTP に近い形でエンドポイントを叩く。
Predictor は client フィクスチャ側で偽物に差し替え済みなので、重みは不要。

注意:
- /predict, /predict_json は Depends(get_predictor_dep) を使うので
  dependency_overrides（client フィクスチャ）で差し替わる。
- /health は get_predictor() を「直接」呼ぶため override が効かない。
  そのため health のテストだけは main.get_predictor を monkeypatch する。
"""

import io

from PIL import Image

import main


def _png_bytes(size=(28, 28), color=0) -> io.BytesIO:
    """テスト用の PNG バイト列を作る。"""
    buf = io.BytesIO()
    Image.new("L", size, color=color).save(buf, format="PNG")
    buf.seek(0)
    return buf


# ---- /health（DI を通らないので monkeypatch で差し替える）----

def test_health_ok(monkeypatch, client, fake_predictor):
    monkeypatch.setattr(main, "get_predictor", lambda: fake_predictor)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_health_degraded(monkeypatch, client):
    def boom():
        raise RuntimeError("重みが読めない想定")

    monkeypatch.setattr(main, "get_predictor", boom)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


# ---- /predict（画像ファイル）----

def test_predict_with_image(client):
    files = {"file": ("digit.png", _png_bytes(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["class_id"] == 7
    assert len(body["probabilities"]) == 10
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_accepts_any_size_image(client):
    """28x28 以外でもサーバ側でリサイズされるので 200。"""
    files = {"file": ("big.png", _png_bytes(size=(200, 120)), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200


def test_predict_rejects_non_image_content_type(client):
    files = {"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400


def test_predict_rejects_corrupt_image(client):
    """content_type は image だが中身が壊れている → デコード失敗で 400。"""
    files = {"file": ("broken.png", io.BytesIO(b"not-a-real-png"), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400


# ---- /predict_json（画素配列）----

def test_predict_json_ok(client):
    pixels = [[0.0] * 28 for _ in range(28)]
    r = client.post("/predict_json", json={"pixels": pixels})
    assert r.status_code == 200
    assert r.json()["class_id"] == 7


def test_predict_json_rejects_bad_shape(client):
    """28x28 以外は Pydantic 検証で 422（500 ではない）。"""
    pixels = [[0.0] * 10 for _ in range(10)]
    r = client.post("/predict_json", json={"pixels": pixels})
    assert r.status_code == 422


# ---- ルーティング ----

def test_root_redirects_to_ui(client):
    """/ は /ui へリダイレクト（follow_redirects=False で 307 を確認）。"""
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui"
