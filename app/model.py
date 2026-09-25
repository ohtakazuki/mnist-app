"""MNIST分類用の軽量CNN。

学習スクリプト・推論サーバの双方から `from model import MnistCNN` で読み込まれる。
state_dict だけを保存・読み込みするため、このクラス定義が両側で一致していることが必須。
"""

import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 畳み込み層: 入力1ch(グレースケール) → 16ch → 32ch。padding=1 で縦横サイズを維持
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)          # 縦横を半分に縮小（28→14→7）
        self.fc1 = nn.Linear(32 * 7 * 7, 64)    # 32ch × 7 × 7 を1列に並べて全結合
        self.dropout = nn.Dropout(0.3)          # 学習時のみ30%を0にする（eval() で無効）
        self.fc2 = nn.Linear(64, 10)            # 出力は数字 0〜9 の10クラス分のスコア

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 32, 7, 7)
        x = x.view(x.size(0), -1)              # (B, 32*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x                               # (B, 10) の logits（softmax 前）
