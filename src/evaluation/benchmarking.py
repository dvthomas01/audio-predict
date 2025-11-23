"""
Comprehensive benchmarking system for the four experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from models import TransformerSmall, TransformerMedium, TransformerLarge
from data.dataset import AudioFrameDataset, create_dataloaders
from evaluation.eval_architectures import compute_metrics, evaluate_model, load_model
from training.train_transformer import train_transformer
from training.train_lstm import train_lstm


def duration_to_frames(duration_seconds: float, sample_rate: int = 22050, 
                       hop_length: int = 512) -> int:
    """
    Convert audio duration in seconds to number of spectrogram frames.
    
    Args:
        duration_seconds: Duration in seconds
        sample_rate: Audio sample rate
        hop_length: Hop length used in spectrogram computation
        
    Returns:
        Number of spectrogram frames
    """
    # Number of audio samples
    num_samples = int(duration_seconds * sample_rate)
    # Number of spectrogram frames (each frame represents hop_length samples)
    num_frames = num_samples // hop_length
    return num_frames


def frames_to_duration(num_frames: int, sample_rate: int = 22050, 
                       hop_length: int = 512) -> float:
    """Convert number of spectrogram frames to duration in seconds."""
    num_samples = num_frames * hop_length
    duration = num_samples / sample_rate
    return duration


class BenchmarkRunner:
    """Runs all four benchmarking experiments."""
    
    def __init__(
        self,
        spectrogram_dir: str,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "benchmark_results",
        sample_rate: int = 22050,
        hop_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            spectrogram_dir: Directory containing preprocessed spectrograms
            checkpoint_dir: Directory to save/load model checkpoints
            results_dir: Directory to save benchmark results
            sample_rate: Audio sample rate (for duration conversion)
            hop_length: Hop length used in preprocessing (for duration conversion)
            device: Device to run experiments on
        """
        self.spectrogram_dir = spectrogram_dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.device = torch.device(device)
        
        # Get input dimension from a sample spectrogram
        spec_files = list(Path(spectrogram_dir).glob("*.pt"))
        if spec_files:
            sample_spec = torch.load(spec_files[0])
            self.input_dim = sample_spec.shape[0]  # freq_bins
        else:
            raise ValueError(f"No spectrogram files found in {spectrogram_dir}")
    
    def benchmark_1_vary_prediction_horizon(
        self,
        fixed_context_duration: float = 60.0,  # seconds (1 minute)
        prediction_horizons: List[int] = [1, 3, 5, 10, 20],
        model_types: List[str] = ["small", "medium", "large"],
        batch_size: int = 32,
        num_epochs: int = 20,
        train: bool = True
    ) -> Dict:
        """
        Benchmark 1: Fixed context size → vary prediction horizon (k)
        
        Args:
            fixed_context_duration: Fixed duration of input context in seconds
            prediction_horizons: List of k values to test
            model_types: Which transformer models to test
            batch_size: Batch size for training/evaluation
            num_epochs: Number of training epochs
            train: Whether to train models or just evaluate existing ones
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*70)
        print("BENCHMARK 1: Fixed Context Duration → Vary Prediction Horizon (k)")
        print("="*70)
        
        # Convert duration to frames
        context_size = duration_to_frames(fixed_context_duration, 
                                         self.sample_rate, self.hop_length)
        print(f"Fixed context duration: {fixed_context_duration}s = {context_size} frames")
        
        results = {}
        
        for k in prediction_horizons:
            print(f"\n--- Testing k={k} (prediction horizon: {k} frames) ---")
            k_results = {}
            
            for model_type in model_types:
                print(f"\n  Model: Transformer {model_type}")
                
                checkpoint_name = f"benchmark1_{model_type}_k{k}.pt"
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                
                # Train model if needed
                if train or not checkpoint_path.exists():
                    print(f"    Training model...")
                    train_transformer(
                        model_name=model_type,
                        spectrogram_dir=self.spectrogram_dir,
                        context_size=context_size,
                        prediction_horizon=k,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        save_dir=str(self.checkpoint_dir),
                        device=str(self.device)
                    )
                    # Rename checkpoint
                    best_checkpoint = self.checkpoint_dir / f"transformer_{model_type}_best.pt"
                    if best_checkpoint.exists():
                        best_checkpoint.rename(checkpoint_path)
                
                # Evaluate model
                print(f"    Evaluating model...")
                _, _, test_loader = create_dataloaders(
                    spectrogram_dir=self.spectrogram_dir,
                    context_size=context_size,
                    prediction_horizon=k,
                    batch_size=batch_size
                )
                
                model = load_model(str(checkpoint_path), f"transformer_{model_type}", self.device)
                metrics = evaluate_model(model, test_loader, self.device)
                k_results[model_type] = metrics
                
                print(f"    Results: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
            
            results[f"k={k}"] = k_results
        
        # Save results
        results_file = self.results_dir / "benchmark1_vary_k.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'Fixed context duration → vary prediction horizon',
                'fixed_context_duration_seconds': fixed_context_duration,
                'fixed_context_size_frames': context_size,
                'prediction_horizons': prediction_horizons,
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return results
    
    def benchmark_2_vary_context_size(
        self,
        fixed_k: int = 10,
        context_durations: List[float] = [5.0, 10.0, 20.0, 30.0, 60.0],  # seconds
        model_types: List[str] = ["small", "medium", "large"],
        batch_size: int = 32,
        num_epochs: int = 20,
        train: bool = True
    ) -> Dict:
        """
        Benchmark 2: Vary context size → fixed prediction horizon (k)
        
        Args:
            fixed_k: Fixed prediction horizon
            context_durations: List of context durations in seconds
            model_types: Which transformer models to test
            batch_size: Batch size for training/evaluation
            num_epochs: Number of training epochs
            train: Whether to train models or just evaluate existing ones
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*70)
        print("BENCHMARK 2: Vary Context Duration → Fixed Prediction Horizon (k)")
        print("="*70)
        print(f"Fixed k: {fixed_k} frames")
        
        results = {}
        
        for duration in context_durations:
            context_size = duration_to_frames(duration, self.sample_rate, self.hop_length)
            print(f"\n--- Testing context duration: {duration}s = {context_size} frames ---")
            duration_results = {}
            
            for model_type in model_types:
                print(f"\n  Model: Transformer {model_type}")
                
                checkpoint_name = f"benchmark2_{model_type}_ctx{int(duration)}s.pt"
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                
                # Train model if needed
                if train or not checkpoint_path.exists():
                    print(f"    Training model...")
                    train_transformer(
                        model_name=model_type,
                        spectrogram_dir=self.spectrogram_dir,
                        context_size=context_size,
                        prediction_horizon=fixed_k,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        save_dir=str(self.checkpoint_dir),
                        device=str(self.device)
                    )
                    # Rename checkpoint
                    best_checkpoint = self.checkpoint_dir / f"transformer_{model_type}_best.pt"
                    if best_checkpoint.exists():
                        best_checkpoint.rename(checkpoint_path)
                
                # Evaluate model
                print(f"    Evaluating model...")
                _, _, test_loader = create_dataloaders(
                    spectrogram_dir=self.spectrogram_dir,
                    context_size=context_size,
                    prediction_horizon=fixed_k,
                    batch_size=batch_size
                )
                
                model = load_model(str(checkpoint_path), f"transformer_{model_type}", self.device)
                metrics = evaluate_model(model, test_loader, self.device)
                duration_results[model_type] = metrics
                
                print(f"    Results: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
            
            results[f"{duration}s"] = duration_results
        
        # Save results
        results_file = self.results_dir / "benchmark2_vary_context.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'Vary context duration → fixed prediction horizon',
                'fixed_k': fixed_k,
                'context_durations_seconds': context_durations,
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return results
    
    def benchmark_3_vary_model_size(
        self,
        context_size: int = 10,
        prediction_horizon: int = 5,
        model_types: List[str] = ["small", "medium", "large"],
        batch_size: int = 32,
        num_epochs: int = 20,
        train: bool = True
    ) -> Dict:
        """
        Benchmark 3: Vary model size (small, medium, large)
        
        Args:
            context_size: Context size in frames
            prediction_horizon: Prediction horizon (k)
            model_types: Which transformer models to test
            batch_size: Batch size for training/evaluation
            num_epochs: Number of training epochs
            train: Whether to train models or just evaluate existing ones
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*70)
        print("BENCHMARK 3: Vary Model Size")
        print("="*70)
        print(f"Context size: {context_size} frames, k: {prediction_horizon}")
        
        results = {}
        
        for model_type in model_types:
            print(f"\n--- Testing Transformer {model_type} ---")
            
            checkpoint_name = f"benchmark3_{model_type}.pt"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # Train model if needed
            if train or not checkpoint_path.exists():
                print(f"  Training model...")
                train_transformer(
                    model_name=model_type,
                    spectrogram_dir=self.spectrogram_dir,
                    context_size=context_size,
                    prediction_horizon=prediction_horizon,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    save_dir=str(self.checkpoint_dir),
                    device=str(self.device)
                )
                # Rename checkpoint
                best_checkpoint = self.checkpoint_dir / f"transformer_{model_type}_best.pt"
                if best_checkpoint.exists():
                    best_checkpoint.rename(checkpoint_path)
            
            # Evaluate model
            print(f"  Evaluating model...")
            _, _, test_loader = create_dataloaders(
                spectrogram_dir=self.spectrogram_dir,
                context_size=context_size,
                prediction_horizon=prediction_horizon,
                batch_size=batch_size
            )
            
            model = load_model(str(checkpoint_path), f"transformer_{model_type}", self.device)
            metrics = evaluate_model(model, test_loader, self.device)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            metrics['num_parameters'] = num_params
            
            results[model_type] = metrics
            
            print(f"  Results: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, Params={num_params:,}")
        
        # Save results
        results_file = self.results_dir / "benchmark3_vary_model_size.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'Vary model size',
                'context_size': context_size,
                'prediction_horizon': prediction_horizon,
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return results
    
    def benchmark_4_vary_dataset_size(
        self,
        context_size: int = 10,
        prediction_horizon: int = 5,
        dataset_sizes: Dict[str, float] = {
            'small': 0.6,   # 60% of full dataset
            'medium': 0.8,  # 80% of full dataset
            'large': 1.0    # 100% of full dataset
        },
        model_types: List[str] = ["small", "medium", "large"],
        batch_size: int = 32,
        num_epochs: int = 20,
        train: bool = True
    ) -> Dict:
        """
        Benchmark 4: Vary dataset size
        
        Args:
            context_size: Context size in frames
            prediction_horizon: Prediction horizon (k)
            dataset_sizes: Dictionary mapping size names to fractions (0.0-1.0)
            model_types: Which transformer models to test
            batch_size: Batch size for training/evaluation
            num_epochs: Number of training epochs
            train: Whether to train models or just evaluate existing ones
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*70)
        print("BENCHMARK 4: Vary Dataset Size")
        print("="*70)
        print(f"Context size: {context_size} frames, k: {prediction_horizon}")
        
        # Get total number of files
        spec_files = list(Path(self.spectrogram_dir).glob("*.pt"))
        total_files = len(spec_files)
        print(f"Total files in dataset: {total_files}")
        
        results = {}
        
        for size_name, fraction in dataset_sizes.items():
            num_files = int(total_files * fraction)
            print(f"\n--- Testing {size_name} dataset: {num_files} files ({fraction*100:.0f}%) ---")
            size_results = {}
            
            for model_type in model_types:
                print(f"\n  Model: Transformer {model_type}")
                
                checkpoint_name = f"benchmark4_{model_type}_{size_name}.pt"
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                
                # Train model if needed
                if train or not checkpoint_path.exists():
                    print(f"    Training model on {num_files} files ({fraction*100:.0f}% of dataset)...")
                    
                    # Train using the training function with max_files parameter
                    # We'll need to modify create_dataloaders to support this
                    # For now, create a temporary dataset with limited files
                    train_loader, val_loader, _ = create_dataloaders(
                        spectrogram_dir=self.spectrogram_dir,
                        context_size=context_size,
                        prediction_horizon=prediction_horizon,
                        batch_size=batch_size,
                        max_files=num_files
                    )
                    
                    # Create model
                    if model_type == "small":
                        model = TransformerSmall(
                            input_dim=self.input_dim,
                            prediction_horizon=prediction_horizon
                        ).to(self.device)
                    elif model_type == "medium":
                        model = TransformerMedium(
                            input_dim=self.input_dim,
                            prediction_horizon=prediction_horizon
                        ).to(self.device)
                    elif model_type == "large":
                        model = TransformerLarge(
                            input_dim=self.input_dim,
                            prediction_horizon=prediction_horizon
                        ).to(self.device)
                    
                    # Train model
                    from training.train_transformer import train_epoch, validate
                    import torch.optim as optim
                    
                    criterion = nn.MSELoss()
                    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5, verbose=False
                    )
                    
                    best_val_loss = float('inf')
                    for epoch in range(num_epochs):
                        train_loss = train_epoch(model, train_loader, criterion, optimizer, self.device)
                        val_loss = validate(model, val_loader, criterion, self.device)
                        scheduler.step(val_loss)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss': val_loss,
                                'model_config': {
                                    'model_name': model_type,
                                    'input_dim': self.input_dim,
                                    'prediction_horizon': prediction_horizon,
                                    'context_size': context_size
                                }
                            }, checkpoint_path)
                    
                    print(f"    Training complete. Best val loss: {best_val_loss:.6f}")
                
                # Evaluate model
                print(f"    Evaluating model...")
                _, _, test_loader = create_dataloaders(
                    spectrogram_dir=self.spectrogram_dir,
                    context_size=context_size,
                    prediction_horizon=prediction_horizon,
                    batch_size=batch_size,
                    max_files=None  # Use full dataset for evaluation
                )
                
                model = load_model(str(checkpoint_path), f"transformer_{model_type}", self.device)
                metrics = evaluate_model(model, test_loader, self.device)
                metrics['num_training_files'] = num_files
                size_results[model_type] = metrics
                
                print(f"    Results: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
            
            results[size_name] = size_results
        
        # Save results
        results_file = self.results_dir / "benchmark4_vary_dataset_size.json"
        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'Vary dataset size',
                'context_size': context_size,
                'prediction_horizon': prediction_horizon,
                'total_files': total_files,
                'dataset_sizes': dataset_sizes,
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return results
    
    def run_all_benchmarks(
        self,
        benchmark_1_config: Optional[Dict] = None,
        benchmark_2_config: Optional[Dict] = None,
        benchmark_3_config: Optional[Dict] = None,
        benchmark_4_config: Optional[Dict] = None,
        train: bool = True
    ):
        """Run all four benchmarks with given configurations."""
        all_results = {}
        
        if benchmark_1_config:
            all_results['benchmark_1'] = self.benchmark_1_vary_prediction_horizon(
                train=train, **benchmark_1_config
            )
        
        if benchmark_2_config:
            all_results['benchmark_2'] = self.benchmark_2_vary_context_size(
                train=train, **benchmark_2_config
            )
        
        if benchmark_3_config:
            all_results['benchmark_3'] = self.benchmark_3_vary_model_size(
                train=train, **benchmark_3_config
            )
        
        if benchmark_4_config:
            all_results['benchmark_4'] = self.benchmark_4_vary_dataset_size(
                train=train, **benchmark_4_config
            )
        
        # Save summary
        summary_file = self.results_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"All benchmarks complete! Summary saved to {summary_file}")
        print(f"{'='*70}")
        
        return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmarking experiments")
    parser.add_argument("--spectrogram_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="benchmark_results")
    parser.add_argument("--benchmark", type=int, choices=[1, 2, 3, 4], help="Which benchmark to run (1-4), or omit to run all")
    parser.add_argument("--no_train", action="store_true", help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        spectrogram_dir=args.spectrogram_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        device=args.device
    )
    
    train = not args.no_train
    
    if args.benchmark == 1:
        runner.benchmark_1_vary_prediction_horizon(train=train)
    elif args.benchmark == 2:
        runner.benchmark_2_vary_context_size(train=train)
    elif args.benchmark == 3:
        runner.benchmark_3_vary_model_size(train=train)
    elif args.benchmark == 4:
        runner.benchmark_4_vary_dataset_size(train=train)
    else:
        # Run all benchmarks
        runner.run_all_benchmarks(
            benchmark_1_config={
                'fixed_context_duration': 60.0,
                'prediction_horizons': [1, 3, 5, 10, 20]
            },
            benchmark_2_config={
                'fixed_k': 10,
                'context_durations': [5.0, 10.0, 20.0, 30.0, 60.0]
            },
            benchmark_3_config={
                'context_size': 10,
                'prediction_horizon': 5
            },
            benchmark_4_config={
                'context_size': 10,
                'prediction_horizon': 5,
                'dataset_sizes': {'small': 0.6, 'medium': 0.8, 'large': 1.0}
            },
            train=train
        )

