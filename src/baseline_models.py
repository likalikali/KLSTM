"""论文复现实验使用的窗口化基线模型。"""

from __future__ import annotations

import torch


class FeedForwardMultiStepModel(torch.nn.Module):
    """将窗口展开后直接回归未来多个时间步。"""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class Seq2SeqRNNBaseline(torch.nn.Module):
    """不带 FM 的窗口化 FC-LSTM / FC-GRU 基线。"""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        num_layers: int,
        forecast_step: int,
        cell_type: str = "gru",
        dropout: float = 0.2,
        teacher_prob: float = 0.5,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_step = forecast_step
        self.teacher_prob = teacher_prob
        self.cell_type = cell_type.lower()

        if self.cell_type not in {"gru", "lstm"}:
            raise ValueError(f"不支持的 cell_type: {cell_type}")

        input_dim = feature_size + 1
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.encoder_input = torch.nn.Linear(input_dim, hidden_size)
        self.decoder_input = torch.nn.Linear(input_dim, hidden_size)

        rnn_cls = torch.nn.GRU if self.cell_type == "gru" else torch.nn.LSTM
        self.encoder = rnn_cls(hidden_size, hidden_size, num_layers=num_layers, dropout=rnn_dropout)
        self.decoder = rnn_cls(hidden_size, hidden_size, num_layers=num_layers, dropout=rnn_dropout)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(
        self,
        histx: torch.Tensor,
        histz: torch.Tensor,
        futx: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_inputs = torch.cat((histx, histz), dim=-1)
        encoder_inputs = self.encoder_input(encoder_inputs).permute(1, 0, 2)
        _, hidden = self.encoder(encoder_inputs)

        previous = histz[:, -1, :]
        future_targets = None
        if target is not None:
            future_targets = target[:, -self.forecast_step :, :]

        outputs = []
        decoder_hidden = hidden
        for step in range(self.forecast_step):
            if self.training and future_targets is not None and torch.rand(1).item() < self.teacher_prob:
                previous = future_targets[:, step, :]

            decoder_step = torch.cat((futx[:, step, :], previous), dim=-1)
            decoder_step = self.decoder_input(decoder_step).unsqueeze(0)
            decoder_output, decoder_hidden = self.decoder(decoder_step, decoder_hidden)
            previous = self.output(decoder_output.squeeze(0))
            outputs.append(previous.unsqueeze(1))

        return torch.cat(outputs, dim=1)
