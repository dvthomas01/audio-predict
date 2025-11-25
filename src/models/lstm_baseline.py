import torch
import torch.nn as nn
from typing import Tuple


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.transpose
        x = self.input_proj(x)
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.output_proj(lstm_out)
        return output, hidden

    def predict_autoregressive(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        ctx_out, hidden = self.forward(x)
        current_input = ctx_out[:, -1:, :]
        predictions = [current_input]
        for _ in range(horizon-1):
            output, hidden = self.forward(current_input, hidden)
            predictions.append(output)
            current_input = output

        return torch.cat(predictions, dim=2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
