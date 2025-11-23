"""
Evaluation script to compare different architectures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import LSTMBaseline, TransformerSmall, TransformerMedium, TransformerLarge
from data.dataset import create_dataloaders


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted frames (batch_size, freq_bins, prediction_horizon)
        targets: Target frames (batch_size, freq_bins, prediction_horizon)
        
    Returns:
        Dictionary of metrics
    """
    # MSE
    mse = nn.MSELoss()(predictions, targets).item()
    
    # MAE
    mae = nn.L1Loss()(predictions, targets).item()
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # Spectrogram similarity (cosine similarity)
    pred_flat = predictions.view(predictions.shape[0], -1)
    target_flat = targets.view(targets.shape[0], -1)
    cosine_sim = torch.nn.functional.cosine_similarity(
        pred_flat, target_flat, dim=1
    ).mean().item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'cosine_similarity': cosine_sim
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate a model on test set."""
    model.eval()
    
    all_metrics = []
    
    with torch.no_grad():
        for input_frames, target_frames in tqdm(test_loader, desc="Evaluating"):
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)
            
            predictions = model(input_frames)
            metrics = compute_metrics(predictions, target_frames)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return avg_metrics


def load_model(checkpoint_path: str, model_type: str, device: torch.device) -> nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['model_config']
    
    if model_type == "lstm":
        model = LSTMBaseline(
            input_dim=config['input_dim'],
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            prediction_horizon=config['prediction_horizon']
        )
    elif model_type == "transformer_small":
        model = TransformerSmall(
            input_dim=config['input_dim'],
            prediction_horizon=config['prediction_horizon']
        )
    elif model_type == "transformer_medium":
        model = TransformerMedium(
            input_dim=config['input_dim'],
            prediction_horizon=config['prediction_horizon']
        )
    elif model_type == "transformer_large":
        model = TransformerLarge(
            input_dim=config['input_dim'],
            prediction_horizon=config['prediction_horizon']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model


def compare_architectures(
    checkpoint_dir: str,
    spectrogram_dir: str,
    context_size: int = 10,
    prediction_horizon: int = 5,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Compare all architectures on test set.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        spectrogram_dir: Directory containing spectrograms
        context_size: Context size used during training
        prediction_horizon: Prediction horizon used during training
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
    """
    device = torch.device(device)
    checkpoint_path = Path(checkpoint_dir)
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        spectrogram_dir=spectrogram_dir,
        context_size=context_size,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size
    )
    
    # Models to evaluate
    models_to_eval = [
        ("lstm", "lstm_best.pt"),
        ("transformer_small", "transformer_small_best.pt"),
        ("transformer_medium", "transformer_medium_best.pt"),
        ("transformer_large", "transformer_large_best.pt")
    ]
    
    results = {}
    
    for model_type, checkpoint_file in models_to_eval:
        checkpoint_file_path = checkpoint_path / checkpoint_file
        
        if not checkpoint_file_path.exists():
            print(f"Warning: {checkpoint_file} not found, skipping...")
            continue
        
        print(f"\nEvaluating {model_type}...")
        
        # Load model
        model = load_model(str(checkpoint_file_path), model_type, device)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        results[model_type] = metrics
        
        print(f"Results for {model_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    # Save results
    results_file = checkpoint_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Cosine Sim':<12}")
    print("-"*60)
    for model_type, metrics in results.items():
        print(f"{model_type:<20} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f} "
              f"{metrics['rmse']:<12.6f} {metrics['cosine_similarity']:<12.6f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare architectures")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--spectrogram_dir", type=str, required=True)
    parser.add_argument("--context_size", type=int, default=10)
    parser.add_argument("--prediction_horizon", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    compare_architectures(
        checkpoint_dir=args.checkpoint_dir,
        spectrogram_dir=args.spectrogram_dir,
        context_size=args.context_size,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        device=args.device
    )

