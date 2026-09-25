"""テスト全体で共有するフィクスチャ群。

設計の肝は「本物のモデル重み（model.pth）に依存しないこと」。
- Predictor はモデルオブジェクトを引数で受け取れる → 偽モデル(FakeModel)を注入する。
- FastAPI は Depends(get_predictor_dep) なので dependency_overrides で差し替える。
これにより、学習済み重みが無い CI 上でも全テストが決定的に通る。
"""

import pytest
import torch
from fastapi.testclient import TestClient

import main
from inference import Predictor, get_predictor


class FakeModel:
    """Predictor に注入する偽の CNN。

    nn.Module の代わりに使う最小の二重（テストダブル）。
    - `eval()` を持つ（Predictor.__init__ が呼ぶため必須）。
    - `__call__` は常に target_class を最尤にする固定 logits を返す。
      → 推論結果が決定的になり、class_id を assert できる。
    """

    def __init__(self, target_class: int = 7):
        self.target_class = target_class
        self.eval_called = False  # eval() が呼ばれたかを検証できるよう記録

    def eval(self):
        # nn.Module.eval() は self を返す。その挙動に合わせておく。
        self.eval_called = True
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 入力 (B, 1, 28, 28) を想定。バッチ数ぶんの logits (B, 10) を返す。
        batch = x.shape[0]
        logits = torch.full((batch, 10), -10.0)
        logits[:, self.target_class] = 10.0  # softmax 後に target がほぼ 1.0 になる
        return logits


@pytest.fixture
def fake_model() -> FakeModel:
    """新しい FakeModel を1つ返す（eval 呼び出しの検証などに使う）。"""
    return FakeModel(target_class=7)


@pytest.fixture
def fake_predictor(fake_model) -> Predictor:
    """偽モデルを注入した Predictor。重みロード不要で predict をテストできる。"""
    return Predictor(fake_model)


@pytest.fixture(autouse=True)
def clear_predictor_cache():
    """get_predictor の lru_cache をテストごとにクリアし、状態の漏れを防ぐ。

    autouse=True なので全テストの前後で自動実行される。
    """
    get_predictor.cache_clear()
    yield
    get_predictor.cache_clear()


@pytest.fixture
def client(fake_predictor) -> TestClient:
    """偽 Predictor を注入した TestClient。

    エンドポイントは Depends(get_predictor_dep) を使うので、
    その関数自体を上書きする（get_predictor ではなく get_predictor_dep が差し替え対象）。
    """
    main.app.dependency_overrides[main.get_predictor_dep] = lambda: fake_predictor
    with TestClient(main.app) as c:
        yield c
    # 後始末。他テストに override が漏れないよう必ずクリアする。
    main.app.dependency_overrides.clear()
