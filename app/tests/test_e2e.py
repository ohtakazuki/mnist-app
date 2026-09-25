"""tests/test_e2e.py — エンドツーエンド（E2E）テスト。

これまでの単体 / 結合テストと違い、ここでは「本物」を一気通貫で通す:
- 実際の学習済み重み (model.pth) を使う（FakeModel は使わない）
- uvicorn で本物のサーバプロセスを起動し、実 HTTP で叩く（TestClient ではない）
- 実画像 / 実 MNIST サンプルで end-to-end の挙動と精度を確認する

そのぶん環境依存が強い:
- model.pth が無ければモジュールごと skip する（E2E はそもそも成立しない）
- すべて @pytest.mark.e2e 付き。通常 CI から切り離し `pytest -m e2e` で実行する想定
"""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image, ImageDraw

import inference
from inference import WEIGHTS_PATH

# このモジュールのテストはすべて E2E マーカー付き（pytest.ini に登録すること）
pytestmark = pytest.mark.e2e

# inference.py（= アプリ本体）が置かれたディレクトリ。サーバ起動時の作業ディレクトリにする
PROJECT_ROOT = Path(inference.__file__).resolve().parent

# 本物の重みが無ければ E2E は成立しない → モジュールごと skip
if not WEIGHTS_PATH.exists():
    pytest.skip(
        f"E2E には学習済み重みが必要です: {WEIGHTS_PATH} が見つかりません",
        allow_module_level=True,
    )


def _free_port() -> int:
    """OS に空きポートを 1 つ割り当ててもらう（ポート衝突を避ける）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_healthy(base_url: str, timeout: float = 30.0) -> None:
    """/health が 200 かつ model_loaded=True を返すまでポーリングする。"""
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("model_loaded") is True:
                return
        except Exception as e:  # 起動直後の接続拒否などは握って再試行
            last_err = e
        time.sleep(0.3)
    raise RuntimeError(f"サーバが時間内に起動しませんでした: last_err={last_err}")


@pytest.fixture(scope="session")
def server() -> str:
    """本物の uvicorn サーバをサブプロセスで起動し、base_url を返す。

    - TestClient ではなく実プロセス・実 HTTP で叩く点が結合テストとの最大の違い。
    - scope="session" なので全 E2E テストで 1 プロセスを共有する（起動コスト削減）。
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    # ログはパイプではなく一時ファイルへ（パイプ満杯でサーバが固まるのを防ぐ）
    log = tempfile.TemporaryFile(mode="w+b")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_until_healthy(base_url)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.seek(0)
        server_log = log.read().decode(errors="replace") or "(ログなし)"
        log.close()
        raise RuntimeError("サーバ起動に失敗。サーバログ:\n" + server_log)

    yield base_url

    # 後始末: プロセスを確実に停止する（テスト間にサーバを残さない）
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


def _digit_png(size: tuple[int, int] = (28, 28)) -> bytes:
    """黒地に白で数字を描いた PNG を作る（MNIST の「黒地に白」前提に合わせる）。

    描画フォント依存なので精度の assert には使わない。契約（レスポンス形）検証用。
    """
    img = Image.new("L", size, color=0)   # 黒地
    draw = ImageDraw.Draw(img)
    draw.text((8, 4), "7", fill=255)      # 白で 7 を描く
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _assert_prediction_contract(body: dict) -> None:
    """レスポンスがスキーマ通りかを検証する共通ヘルパ。"""
    assert set(body) == {"class_id", "class_name", "confidence", "probabilities"}
    assert 0 <= body["class_id"] <= 9
    assert body["class_name"] == str(body["class_id"])
    assert len(body["probabilities"]) == 10
    assert 0.0 <= body["confidence"] <= 1.0
    assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-4)


# ---- 起動・ルーティング ----

def test_health_reports_real_model_loaded(server):
    """本物の重みがロードされ、/health が ok を返す。"""
    r = httpx.get(f"{server}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_root_redirects_to_ui(server):
    """/ は /ui へリダイレクトする。"""
    r = httpx.get(f"{server}/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui"


def test_ui_page_is_served(server):
    """Gradio UI が /ui で配信されている（HTML が返る）。

    Gradio は /ui を /ui/ へ 307 リダイレクトする。
    httpx は既定でリダイレクトを追わないので follow_redirects=True を付ける。
    """
    r = httpx.get(f"{server}/ui", follow_redirects=True)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


# ---- 推論エンドポイント（本物のモデルを通す）----

def test_predict_image_contract(server):
    """画像アップロード → 本物のモデルで推論し、レスポンス契約を満たす。"""
    files = {"file": ("digit.png", _digit_png(), "image/png")}
    r = httpx.post(f"{server}/predict", files=files)
    assert r.status_code == 200
    _assert_prediction_contract(r.json())


def test_predict_json_contract(server):
    """画素配列 → 本物のモデルで推論し、レスポンス契約を満たす。"""
    pixels = [[0.0] * 28 for _ in range(28)]
    r = httpx.post(f"{server}/predict_json", json={"pixels": pixels})
    assert r.status_code == 200
    _assert_prediction_contract(r.json())


def test_predict_rejects_non_image(server):
    """画像以外の content-type は 400。"""
    files = {"file": ("note.txt", b"hello", "text/plain")}
    r = httpx.post(f"{server}/predict", files=files)
    assert r.status_code == 400


def test_predict_json_rejects_bad_shape(server):
    """28x28 以外は Pydantic 検証で 422。"""
    pixels = [[0.0] * 10 for _ in range(10)]
    r = httpx.post(f"{server}/predict_json", json={"pixels": pixels})
    assert r.status_code == 422


# ---- 精度（実 MNIST。torchvision とデータが無ければ skip）----

def test_accuracy_on_real_mnist(server):
    """実 MNIST テスト画像で end-to-end の精度を確認する。

    学習済みモデル + 実前処理 + 実 HTTP をすべて通した「本物の精度」を測る。
    モデルが壊れた / 取り違えた場合にここで落ちるのが E2E の価値。
    torchvision が無い / ダウンロードできない環境では skip。
    """
    pytest.importorskip("torchvision")
    from torchvision import datasets

    try:
        ds = datasets.MNIST(root="/tmp/mnist-e2e", train=False, download=True)
    except Exception as e:
        pytest.skip(f"MNIST データを取得できませんでした: {e}")

    n = 50
    correct = 0
    for i in range(n):
        img, label = ds[i]  # img: PIL Image(L, 28x28), label: int
        # MNIST は黒地に白＝学習時と同じ向き。/predict_json なら PNG 符号化の揺れを避けられる
        arr = np.asarray(img, dtype=np.float32) / 255.0
        r = httpx.post(f"{server}/predict_json", json={"pixels": arr.tolist()})
        assert r.status_code == 200
        if r.json()["class_id"] == label:
            correct += 1

    accuracy = correct / n
    # しきい値はモデルに応じて調整可。学習済み CNN なら通常 0.9 以上は出る
    assert accuracy >= 0.9, f"E2E 精度が低すぎます: {accuracy:.2%}（{correct}/{n}）"