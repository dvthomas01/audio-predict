"""
Training script for Transformer models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import TransformerSmall, TransformerMedium, TransformerLarge
from data.dataset import create_dataloaders


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for input_frames, target_frames in tqdm(train_loader, desc="Training"):
        input_frames = input_frames.to(device)
        target_frames = target_frames.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(input_frames)
        
        # Compute loss
        loss = criterion(predictions, target_frames)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_frames, target_frames in tqdm(val_loader, desc="Validating"):
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)
            
            predictions = model(input_frames)
            loss = criterion(predictions, target_frames)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_transformer(
    model_name: str,
    spectrogram_dir: str,
    context_size: int = 10,
    prediction_horizon: int = 5,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    save_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train a Transformer model.
    
    Args:
        model_name: "small", "medium", or "large"
        spectrogram_dir: Directory containing preprocessed spectrograms
        context_size: Number of input frames
        prediction_horizon: Number of future frames to predict
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        save_dir: Directory to save checkpoints
        device: Device to train on
    """
    device = torch.device(device)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        spectrogram_dir=spectrogram_dir,
        context_size=context_size,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size
    )
    
    # Get input dimension from first batch
    sample_input, _ = next(iter(train_loader))
    input_dim = sample_input.shape[1]  # freq_bins
    
    # Create model
    print(f"Creating {model_name} Transformer model...")
    if model_name == "small":
        model = TransformerSmall(
            input_dim=input_dim,
            prediction_horizon=prediction_horizon
        ).to(device)
    elif model_name == "medium":
        model = TransformerMedium(
            input_dim=input_dim,
            prediction_horizon=prediction_horizon
        ).to(device)
    elif model_name == "large":
        model = TransformerLarge(
            input_dim=input_dim,
            prediction_horizon=prediction_horizon
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': {
                    'model_name': model_name,
                    'input_dim': input_dim,
                    'prediction_horizon': prediction_horizon,
                    'context_size': context_size
                }
            }
            torch.save(
                checkpoint,
                save_path / f"transformer_{model_name}_best.pt"
            )
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                save_path / f"transformer_{model_name}_epoch_{epoch+1}.pt"
            )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    with open(save_path / f"transformer_{model_name}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument("--model", type=str, required=True, choices=["small", "medium", "large"])
    parser.add_argument("--spectrogram_dir", type=str, required=True)
    parser.add_argument("--context_size", type=int, default=10)
    parser.add_argument("--prediction_horizon", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    train_transformer(
        model_name=args.model,
        spectrogram_dir=args.spectrogram_dir,
        context_size=args.context_size,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        device=args.device
    )

