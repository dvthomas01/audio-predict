"""
Medium Transformer model for audio frame prediction.
"""

from .transformer_small import (
    TransformerSmall,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock
)
import torch
import torch.nn as nn


class TransformerMedium(nn.Module):
    """
    Medium Transformer model for audio frame prediction.
    
    Architecture:
    - Input embedding
    - Positional encoding
    - 4 transformer encoder blocks
    - Output projection to prediction horizon
    """
    
    def __init__(
        self,
        input_dim: int,  # Frequency bins
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        prediction_horizon: int = 5,
        max_seq_len: int = 1000,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Number of frequency bins in input spectrogram
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            prediction_horizon: Number of future frames to predict (k)
            max_seq_len: Maximum sequence length for positional encoding
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim * prediction_horizon)
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
        x = x.transpose(1, 2)  # (batch_size, context_size, freq_bins)
        
        # Embed input
        x = self.input_embedding(x)  # (batch_size, context_size, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Use mean pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Project to output
        output = self.output_proj(x)  # (batch_size, freq_bins * prediction_horizon)
        
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
    
    model = TransformerMedium(
        input_dim=freq_bins,
        d_model=256,
        num_heads=8,
        num_layers=4,
        prediction_horizon=prediction_horizon
    )
    
    x = torch.randn(batch_size, freq_bins, context_size)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {count_parameters(model):,}")

