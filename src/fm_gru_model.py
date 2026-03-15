"""更接近论文 checkpoint 实现的 FM-GRU 模型。"""

from __future__ import annotations

import numpy as np
import torch


class Distribution(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.mu_layer = torch.nn.Linear(input_size, 1)
        self.sigma_layer = torch.nn.Linear(input_size, 1)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu_layer(hidden_state), self.sigma_layer(hidden_state)


class Attention(torch.nn.Module):
    def __init__(self, query_dim: int, key_dim: int, dim: int):
        super().__init__()
        self.linear_q = torch.nn.Linear(query_dim, dim)
        self.linear_k = torch.nn.Linear(key_dim, dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(query.squeeze(0))
        k = self.linear_k(keys.permute(1, 0, 2))
        attn = torch.bmm(k, q.unsqueeze(-1))
        attn = torch.softmax(attn, dim=1)
        return torch.bmm(attn.transpose(1, 2), values.permute(1, 0, 2)).squeeze(1)


class TPAMech(torch.nn.Module):
    def __init__(self, query_dim: int, key_dim: int, dim: int):
        super().__init__()
        self.linear_q = torch.nn.Linear(query_dim, dim)
        self.linear_k = torch.nn.Linear(key_dim, dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(query.squeeze(0))
        k = self.linear_k(keys.permute(0, 2, 1))
        attn = torch.bmm(k, q.unsqueeze(-1))
        attn = torch.softmax(attn, dim=1)
        return torch.bmm(values, attn).squeeze(-1)


class FMLayer(torch.nn.Module):
    def __init__(self, input_size: int, factor_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, 1)
        self.v = torch.nn.Parameter(torch.randn(input_size, factor_size))
        torch.nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(inputs)
        interaction_part_1 = torch.mm(inputs, self.v)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(inputs, 2), torch.pow(self.v, 2))
        interaction = 0.5 * torch.sum(interaction_part_2 - interaction_part_1, dim=1, keepdim=True)
        return linear_part + interaction


class FMGRUModel(torch.nn.Module):
    def __init__(
        self,
        target_size: int,
        feature_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.5,
        forecast_step: int = 12,
        encode_step: int = 24,
        teacher_prob: float = 0.5,
        fm_k: int = 84,
        use_fm: bool = True,
    ):
        super().__init__()
        self.forecast_step = forecast_step
        self.encode_step = encode_step
        self.teacher_prob = teacher_prob
        self.use_fm = use_fm

        self.fm = FMLayer(hidden_size, fm_k) if use_fm else None
        self.input_embed = torch.nn.Linear(target_size, hidden_size)
        self.feat_embed = torch.nn.Linear(feature_size, hidden_size)

        encoder_input_size = 2 * hidden_size + (1 if use_fm else 0)
        decoder_input_size = 2 * hidden_size + forecast_step + (1 if use_fm else 0)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn_encoder = torch.nn.GRU(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            dropout=rnn_dropout,
        )
        self.rnn_decoder = torch.nn.GRU(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            dropout=rnn_dropout,
        )

        self.attention = Attention(hidden_size, hidden_size, hidden_size)
        self.tpa_mech = TPAMech(hidden_size, forecast_step, hidden_size)
        self.output = torch.nn.Linear(hidden_size, 1)
        self.distribution = Distribution(hidden_size)

    def forward(
        self,
        histx: torch.Tensor,
        histz: torch.Tensor,
        futx: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        future_z = target[:, -self.forecast_step - 1 : -1]

        hidden = None
        encoder_states = []
        for step in range(self.encode_step):
            z_input = self.input_embed(histz[:, step, :])
            x_input = self.feat_embed(histx[:, step, :])
            encoder_parts = [z_input, x_input]
            if self.use_fm and self.fm is not None:
                encoder_parts.append(self.fm(x_input))
            cell_input = torch.cat(encoder_parts, dim=1).unsqueeze(0)
            output, hidden = self.rnn_encoder(cell_input, hidden)
            encoder_states.append(output)

        encoder_states = torch.cat(encoder_states, dim=0)

        decoder_states = []
        state = encoder_states[-1, :, :].unsqueeze(0)
        for step in range(self.forecast_step):
            feat_input = self.feat_embed(futx[:, step, :]).unsqueeze(0)

            z_input = state
            if self.training and np.random.rand() < self.teacher_prob:
                z_input = self.input_embed(future_z[:, step, :]).unsqueeze(0)

            attn_input = self.attention(state, encoder_states, encoder_states).unsqueeze(0)
            tpa_input = self.tpa_mech(state, futx, futx).unsqueeze(0)

            decoder_parts = [feat_input, attn_input, tpa_input]
            if self.use_fm and self.fm is not None:
                decoder_parts.append(self.fm(self.feat_embed(futx[:, step, :])).unsqueeze(0))
            decoder_input = torch.cat(decoder_parts, dim=-1)
            state, hidden = self.rnn_decoder(decoder_input, hidden)
            decoder_states.append(state)

        decoder_states = torch.cat(decoder_states, dim=0)
        all_states = torch.cat((encoder_states, decoder_states), dim=0)

        predictions = self.output(all_states.permute(1, 0, 2).relu())
        mu, sigma = self.distribution(all_states.permute(1, 0, 2))
        return predictions, predictions[:, -self.forecast_step :, :], (mu, sigma)
