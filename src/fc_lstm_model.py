"""FC-LSTM 基线模型，整理自原始 `LSTM对比实验.py`。"""

from __future__ import annotations

import torch


class FCLSTMPointModel(torch.nn.Module):
    """将 4 个协变量视作长度为 4 的序列，输出单步目标值。"""

    def __init__(self, hidden_size: int = 128, num_layers: int = 3):
        super().__init__()
        self.feature_embed = torch.nn.Linear(1, hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.feature_embed(features)
        output, _ = self.lstm(embedded, None)
        return self.output(output[:, -1, :])
