"""
LSTM baseline model for audio frame prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMBaseline(nn.Module):
    """
    LSTM-based model for predicting future audio frames.
    
    Architecture:
    - Input projection to hidden dimension
    - Multi-layer LSTM
    - Output projection to prediction horizon
    """
    
    def __init__(
        self,
        input_dim: int,  # Frequency bins in spectrogram
        hidden_dim: int = 256,
        num_layers: int = 2,
        prediction_horizon: int = 5,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Args:
            input_dim: Number of frequency bins in input spectrogram
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            prediction_horizon: Number of future frames to predict (k)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim * prediction_horizon)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram frames of shape (batch_size, freq_bins, context_size)
            
        Returns:
            Predicted future frames of shape (batch_size, freq_bins, prediction_horizon)
        """
        batch_size, freq_bins, context_size = x.shape
        
        # Transpose to (batch_size, context_size, freq_bins)
        x = x.transpose(1, 2)
        
        # Project input
        x = self.input_proj(x)  # (batch_size, context_size, hidden_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, context_size, lstm_output_dim)
        
        # Use last output
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_output_dim)
        
        # Project to output
        output = self.output_proj(last_output)  # (batch_size, freq_bins * prediction_horizon)
        
        # Reshape to (batch_size, freq_bins, prediction_horizon)
        output = output.view(batch_size, freq_bins, self.prediction_horizon)
        
        return output


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    freq_bins = 128
    context_size = 10
    prediction_horizon = 5
    
    model = LSTMBaseline(
        input_dim=freq_bins,
        hidden_dim=256,
        num_layers=2,
        prediction_horizon=prediction_horizon
    )
    
    x = torch.randn(batch_size, freq_bins, context_size)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {count_parameters(model):,}")

