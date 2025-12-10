"""
Benchmark Evaluation Script for Autoregressive Transformers
Copy each section into separate Colab cells.

This script contains three comprehensive benchmarks:
1. Same Context Input, Varying K Values: Test models at K=[20, 60, 100] with context=150
2. Varying Context Input Size: Test context=[50, 100, 150] with K=60
3. Model Size Comparison: Compare small, medium, and large models with context=150, K=60
"""

# ============================================================================
# CELL 1: Setup and Imports
# ============================================================================
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import tempfile
import sys
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Install gdown if needed
!pip install -q gdown
"""

# ============================================================================
# CELL 1.5: Audio Decoding Functions
# ============================================================================
"""
import torchaudio
import torchaudio.transforms as T
from IPython.display import Audio, display

def mel_spectrogram_to_audio(
    log_mel_spec: torch.Tensor,
    sample_rate: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    f_max: int = 22050
) -> torch.Tensor:
    \"\"\"Convert log mel spectrogram back to audio waveform.\"\"\"
    # Ensure correct shape: (freq_bins, time_frames)
    if log_mel_spec.shape[0] != n_mels:
        log_mel_spec = log_mel_spec.transpose(0, 1)

    # Convert from log scale to linear
    mel_spec = torch.exp(log_mel_spec)

    # Convert mel to linear spectrogram
    inverse_mel = T.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_max=f_max
    )

    linear_spec = inverse_mel(mel_spec.unsqueeze(0))

    # Griffin-Lim to reconstruct waveform
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=32
    )

    waveform = griffin_lim(linear_spec)
    return waveform.squeeze(0)
"""

# ============================================================================
# CELL 2: Load Data
# ============================================================================
"""
# Load test tensor from Google Drive
save_dir = '/content/drive/MyDrive/saved_tensors'
test_tensor = torch.load(os.path.join(save_dir, 'test_tensor.pt'))

print(f"Test tensor shape: {test_tensor.shape}")
print(f"Total samples: {test_tensor.shape[0]}")
print(f"Frames per sample: {test_tensor.shape[1]}")
print(f"Frequency bins: {test_tensor.shape[2]}")
"""

# ============================================================================
# CELL 3: Import Model Architecture
# ============================================================================
"""
# You need to copy the model architecture from your training script
# This includes: PositionalEncoding, MultiHeadAttention, TransformerBlock, AutoregressiveTransformer
# Copy those classes from Tranformer_Train_Script.ipynb here
"""

# ============================================================================
# CELL 4: Helper Functions (Google Drive)
# ============================================================================
"""
def extract_google_drive_id(url):
    \"\"\"Extract file ID from Google Drive URL.\"\"\"
    if '/file/d/' in url:
        return url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    else:
        return url  # Assume it's already an ID

def download_from_google_drive(file_id, destination):
    \"\"\"Download file from Google Drive using gdown.\"\"\"
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
    except ImportError:
        print("ERROR: gdown not installed. Install with: pip install gdown")
        sys.exit(1)
"""

# ============================================================================
# CELL 5: Load Model from Checkpoint
# ============================================================================
"""
def load_model_from_checkpoint(checkpoint_path_or_url: str, device: str = "cuda"):
    \"\"\"
    Load model from checkpoint file or Google Drive URL.
    
    Args:
        checkpoint_path_or_url: Path to checkpoint or Google Drive URL
        device: Device to load model on
    
    Returns:
        model, config
    \"\"\"
    # Handle Google Drive links
    if "drive.google.com" in checkpoint_path_or_url or len(checkpoint_path_or_url) == 33:
        file_id = extract_google_drive_id(checkpoint_path_or_url)
        temp_dir = Path(tempfile.gettempdir()) / "audio_predict_models"
        temp_dir.mkdir(exist_ok=True)
        checkpoint_path = temp_dir / f"{file_id}.pt"
        
        if not checkpoint_path.exists():
            print(f"Downloading model from Google Drive...")
            download_from_google_drive(file_id, str(checkpoint_path))
        else:
            print(f"Using cached model: {checkpoint_path}")
    else:
        checkpoint_path = checkpoint_path_or_url
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Infer max_seq_len from checkpoint if available
    if 'pos_encoding.pe' in checkpoint.get('model_state_dict', {}):
        pe_shape = checkpoint['model_state_dict']['pos_encoding.pe'].shape
        max_seq_len = pe_shape[1]  # (1, max_seq_len, d_model)
    else:
        max_seq_len = config.get('max_seq_len', config.get('context_size', 150) + config.get('target_frames', 102) + 10)
    
    # Reconstruct model (adjust based on your model architecture)
    model = AutoregressiveTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config.get('dropout', 0.1),
        max_seq_len=max_seq_len
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config
"""

# ============================================================================
# CELL 5.5: CORRECTION - Fixed compute_all_metrics Function
# ============================================================================
"""
⚠️ IMPORTANT: Replace your existing compute_all_metrics function with this corrected version!

The bug was in the Correlation calculation - it was using torch.sum() instead of torch.mean(),
which caused correlation values to be in the hundreds of thousands instead of between -1 and 1.

Copy this entire function and replace Cell 6.
"""

# ============================================================================
# CELL 6: Compute All Metrics (Reused from Training Script) - CORRECTED VERSION
# ============================================================================
"""
def compute_all_metrics(predictions, ground_truth, return_per_frame=False):
    \"\"\"
    Compute comprehensive metrics for spectrogram prediction.
    
    Args:
        predictions: Predicted spectrograms (batch, num_frames, freq_bins) or (num_frames, freq_bins)
        ground_truth: Ground truth spectrograms (batch, num_frames, freq_bins) or (num_frames, freq_bins)
        return_per_frame: If True, also return per-frame metrics
    
    Returns:
        Dictionary of metrics
    \"\"\"
    # Ensure same shape
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if ground_truth.dim() == 2:
        ground_truth = ground_truth.unsqueeze(0)
    
    # Handle different lengths
    min_frames = min(predictions.shape[1], ground_truth.shape[1])
    predictions = predictions[:, :min_frames, :]
    ground_truth = ground_truth[:, :min_frames, :]
    
    # Flatten for some metrics
    pred_flat = predictions.flatten()
    target_flat = ground_truth.flatten()
    
    # 1. MSE (Mean Squared Error)
    mse_tensor = torch.mean((predictions - ground_truth) ** 2)
    mse = mse_tensor.item()
    
    # 2. RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(mse_tensor).item()
    
    # 3. MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(predictions - ground_truth)).item()
    
    # 4. R² Score (Coefficient of Determination)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    r2 = r2.item()
    
    # 5. Correlation (Pearson correlation coefficient)
    # Formula: r = E[(X - E[X])(Y - E[Y])] / (std(X) * std(Y))
    pred_mean = torch.mean(pred_flat)
    target_mean = torch.mean(target_flat)
    # Use mean instead of sum to get proper correlation coefficient
    covariance = torch.mean((pred_flat - pred_mean) * (target_flat - target_mean))
    pred_std = torch.std(pred_flat)
    target_std = torch.std(target_flat)
    denominator = pred_std * target_std + 1e-8
    correlation = (covariance / denominator).item()
    # Clamp to [-1, 1] range to handle numerical errors
    correlation = max(-1.0, min(1.0, correlation))
    
    # 6. Frame Accuracy (within 10% of ground truth)
    # For each element, check if |pred - truth| < 0.1 * |truth|
    frame_errors = torch.abs(predictions - ground_truth)
    frame_accuracy = (frame_errors < 0.1 * torch.abs(ground_truth) + 1e-8).float().mean().item()
    
    # 7. Log Spectral Distance (LSD)
    # Convert to power spectrum (square of magnitude)
    pred_power = torch.clamp(predictions ** 2, min=1e-10)
    target_power = torch.clamp(ground_truth ** 2, min=1e-10)
    
    # Compute per-frame LSD
    log_pred = torch.log10(pred_power + 1e-10)
    log_target = torch.log10(target_power + 1e-10)
    lsd_per_frame = torch.sqrt(torch.mean((log_pred - log_target) ** 2, dim=2))  # (batch, frames)
    lsd = torch.mean(lsd_per_frame).item()
    
    # 8. Cosine Similarity
    pred_flat_norm = pred_flat / (torch.norm(pred_flat) + 1e-8)
    target_flat_norm = target_flat / (torch.norm(target_flat) + 1e-8)
    cosine_sim = torch.sum(pred_flat_norm * target_flat_norm).item()
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Correlation': correlation,
        'Frame Accuracy (10%)': frame_accuracy,
        'Log Spectral Distance': lsd,
        'Cosine Similarity': cosine_sim
    }
    
    if return_per_frame:
        # Per-frame MSE
        per_frame_mse = torch.mean((predictions - ground_truth) ** 2, dim=2).squeeze(0).cpu().numpy()
        metrics['Per-Frame MSE'] = per_frame_mse
    
    return metrics
"""

# ============================================================================
# CELL 7: Visualize Spectrograms (Reused from Training Script)
# ============================================================================
"""
def visualize_spectrograms(
    ground_truth: torch.Tensor,
    predicted: torch.Tensor,
    context: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Spectrogram Comparison"
):
    \"\"\"
    Visualize ground truth vs predicted spectrograms.
    
    Args:
        ground_truth: Ground truth spectrogram (num_frames, freq_bins) or (1, num_frames, freq_bins)
        predicted: Predicted spectrogram (num_frames, freq_bins) or (1, num_frames, freq_bins)
        context: Context spectrogram (context_len, freq_bins) or (1, context_len, freq_bins)
        save_path: Optional path to save figure
        title: Title for the plot
    \"\"\"
    # Remove batch dimension if present
    if ground_truth.dim() == 3:
        ground_truth = ground_truth.squeeze(0)
    if predicted.dim() == 3:
        predicted = predicted.squeeze(0)
    if context.dim() == 3:
        context = context.squeeze(0)
    
    # Convert to numpy
    context_np = context.cpu().numpy().T  # (freq_bins, context_len)
    ground_truth_np = ground_truth.cpu().numpy().T  # (freq_bins, num_frames)
    predicted_np = predicted.cpu().numpy().T  # (freq_bins, num_frames)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Context
    im1 = axes[0].imshow(context_np, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Context (Input)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (frames)')
    axes[0].set_ylabel('Frequency (bins)')
    plt.colorbar(im1, ax=axes[0])
    
    # Ground Truth
    im2 = axes[1].imshow(ground_truth_np, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (frames)')
    axes[1].set_ylabel('Frequency (bins)')
    plt.colorbar(im2, ax=axes[1])
    
    # Predicted
    im3 = axes[2].imshow(predicted_np, aspect='auto', origin='lower', cmap='viridis')
    axes[2].set_title('Predicted', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (frames)')
    axes[2].set_ylabel('Frequency (bins)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram comparison to {save_path}")
    
    plt.show()
"""

# ============================================================================
# CELL 8: Benchmark 1 - Varying K Values
# ============================================================================
"""
def benchmark_varying_k(
    model_checkpoints: Dict[str, str],  # {'model_name': path_or_url}
    test_tensor: torch.Tensor,
    context_size: int = 150,
    k_values: List[int] = [20, 60, 100],
    num_samples: Optional[int] = None,  # None = use all
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    visualize: bool = True,
    save_dir: str = "benchmark_results"
):
    \"\"\"
    Benchmark 1: Same context input, varying K values.
    
    Args:
        model_checkpoints: Dict mapping model names to checkpoint paths/URLs
        test_tensor: Test tensor of shape (N, T, F)
        context_size: Context size (150 frames)
        k_values: List of K values to test [20, 60, 100]
        num_samples: Number of samples to evaluate (None = all)
        device: Device to run on
        visualize: Whether to create visualizations
        save_dir: Directory to save results
    
    Returns:
        Dictionary of results: {model_name: {k: metrics}}
    \"\"\"
    print("="*80)
    print("BENCHMARK 1: SAME CONTEXT INPUT, VARYING K VALUES")
    print("="*80)
    print(f"Context size: {context_size} frames")
    print(f"K values to test: {k_values}")
    print(f"Number of models: {len(model_checkpoints)}")
    print("="*80)
    
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    # Select samples
    total_samples = test_tensor.shape[0]
    if num_samples is None or num_samples > total_samples:
        num_samples = total_samples
    
    # Randomly sample indices
    sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    all_results = {}
    
    for model_name, checkpoint_path in model_checkpoints.items():
        print(f"\\n{'='*80}")
        print(f"Evaluating {model_name.upper()} model...")
        print(f"{'='*80}")
        
        # Load model
        model, config = load_model_from_checkpoint(checkpoint_path, device)
        
        model_results = {}
        
        for k in k_values:
            print(f"\\nTesting K={k} frames...")
            
            # Check if we have enough frames
            if context_size + k > test_tensor.shape[1]:
                print(f"  ⚠️  Skipping K={k}: Need {context_size + k} frames but only {test_tensor.shape[1]} available")
                continue
            
            all_predictions = []
            all_ground_truths = []
            per_sample_mse = []  # Track per-sample MSE for K=60
            
            with torch.no_grad():
                for idx in tqdm(sample_indices, desc=f"  K={k}", leave=False):
                    # Get context and ground truth
                    context = test_tensor[idx:idx+1, :context_size, :].to(device)
                    ground_truth = test_tensor[idx, context_size:context_size+k, :]
                    
                    # Normalize context if needed
                    if 'normalization_stats' in config:
                        stats = config['normalization_stats']
                        if stats['method'] == 'standardize':
                            context_norm = (context - stats['mean']) / stats['std']
                        elif stats['method'] == 'minmax':
                            context_norm = (context - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                        else:
                            context_norm = context
                    else:
                        context_norm = context
                    
                    # Generate
                    generated_norm = model.generate(context_norm, num_frames=k)
                    
                    # Denormalize
                    if 'normalization_stats' in config:
                        stats = config['normalization_stats']
                        if stats['method'] == 'standardize':
                            generated = generated_norm * stats['std'] + stats['mean']
                        elif stats['method'] == 'minmax':
                            generated = generated_norm * (stats['max'] - stats['min']) + stats['min']
                        else:
                            generated = generated_norm
                    else:
                        generated = generated_norm
                    
                    generated_cpu = generated.cpu()
                    all_predictions.append(generated_cpu)
                    all_ground_truths.append(ground_truth.unsqueeze(0))
                    
                    # Track per-sample MSE for K=60 at context_size=150
                    if k == 60 and context_size == 150:
                        # Compute MSE for this single sample
                        sample_mse = torch.mean((generated_cpu.squeeze(0) - ground_truth) ** 2).item()
                        per_sample_mse.append((idx, sample_mse))
            
            # Concatenate and compute metrics
            predictions = torch.cat(all_predictions, dim=0)  # (num_samples, k, F)
            ground_truths = torch.cat(all_ground_truths, dim=0)  # (num_samples, k, F)
            
            metrics = compute_all_metrics(predictions, ground_truths, return_per_frame=True)
            model_results[k] = metrics
            
            # Find best sample for K=60 at context_size=150
            if k == 60 and context_size == 150 and per_sample_mse:
                best_sample_idx, best_mse = min(per_sample_mse, key=lambda x: x[1])
                print(f"\\n    📊 Best sample at K=60: index {best_sample_idx} (MSE: {best_mse:.6f})")
                # Store best sample index for this model
                if 'best_samples' not in all_results:
                    all_results['best_samples'] = {}
                all_results['best_samples'][model_name] = best_sample_idx
            
            # Print summary
            print(f"    MSE: {metrics['MSE']:.6f}")
            print(f"    Frame Accuracy: {metrics['Frame Accuracy (10%)']:.2%}")
            print(f"    R²: {metrics['R²']:.4f}")
            print(f"    Correlation: {metrics['Correlation']:.4f}")
            print(f"    LSD: {metrics['Log Spectral Distance']:.4f} dB")
        
        all_results[model_name] = model_results
    
    # Print summary of best samples
    if 'best_samples' in all_results:
        print(f"\\n{'='*80}")
        print("BEST SAMPLES AT CONTEXT_SIZE=150, K=60:")
        print(f"{'='*80}")
        for model_name, best_idx in all_results['best_samples'].items():
            print(f"  {model_name.upper()}: Sample index {best_idx}")
        print(f"{'='*80}\\n")
    
    # Visualize results (only MSE and Frame Accuracy)
    if visualize:
        metrics_to_plot = ['MSE', 'Frame Accuracy (10%)']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Exclude helper entries like 'best_samples' from plotting
        model_names = [name for name in all_results.keys() if name != 'best_samples']
        # Use consistent colors: Blue, Orange, Green for small, medium, large
        color_map = {'small': '#1f77b4', 'medium': '#ff7f0e', 'large': '#2ca02c'}
        colors = [color_map.get(name, plt.cm.Set3(i)) for i, name in enumerate(model_names)]
        
        # Create legend handles (will be used once at the end)
        legend_handles = []
        
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            for i, model_name in enumerate(model_names):
                values = []
                for k in k_values:
                    if k in all_results[model_name]:
                        values.append(all_results[model_name][k][metric_name])
                    else:
                        values.append(np.nan)
                
                line = ax.plot(k_values, values, marker='o', color=colors[i], linewidth=2, markersize=8, label=model_name)
                # Store handle for legend (only once per model)
                if idx == 0:  # Only add to legend from first plot
                    legend_handles.append(line[0])
            
            ax.set_xlabel('K Value (Frames)', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} vs K', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
        
        # Add single legend outside the plot area
        fig.legend(legend_handles, model_names, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle('Benchmark 1: Same Context, Varying K Values', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 0.96, 0.96])  # Leave space for legend and title
        
        save_path = Path(save_dir) / "benchmark1_varying_k.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        print(f"\\nVisualization saved to: {save_path}")
        plt.show()
    
    return all_results
"""

# ============================================================================
# CELL 9: Benchmark 2 - Varying Context Size
# ============================================================================
"""
def benchmark_varying_context(
    model_checkpoints: Dict[str, str],  # {'model_name': path_or_url}
    test_tensor: torch.Tensor,
    context_sizes: List[int] = [50, 100, 150],
    k_value: int = 60,
    num_samples: int = 100,  # Fewer samples for faster evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    visualize: bool = True,
    save_dir: str = "benchmark_results",
    sample_idx: Optional[int] = None  # Specific sample index to use for visualization/audio (from Benchmark 1)
):
    \"\"\"
    Benchmark 2: Varying context input size, constant K.
    
    Args:
        model_checkpoints: Dict mapping model names to checkpoint paths/URLs
        test_tensor: Test tensor of shape (N, T, F)
        context_sizes: List of context sizes to test [50, 100, 150]
        k_value: Constant K value (60 frames)
        num_samples: Number of samples to evaluate
        device: Device to run on
        visualize: Whether to create visualizations
        save_dir: Directory to save results
        sample_idx: Specific sample index to use for visualization/audio (from Benchmark 1)
    
    Returns:
        Dictionary of results: {model_name: {context_size: metrics}}
    \"\"\"
    print("="*80)
    print("BENCHMARK 2: VARYING CONTEXT INPUT SIZE, CONSTANT K")
    print("="*80)
    print(f"Context sizes to test: {context_sizes}")
    print(f"Constant K value: {k_value} frames")
    print(f"Number of models: {len(model_checkpoints)}")
    print("="*80)
    
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    # Select samples
    total_samples = test_tensor.shape[0]
    num_samples = min(num_samples, total_samples)
    
    # If specific sample_idx provided, use it; otherwise random sample
    if sample_idx is not None:
        if sample_idx >= total_samples:
            print(f"⚠️  Warning: sample_idx {sample_idx} >= total_samples {total_samples}. Using random sample.")
            sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        else:
            # Use the specified sample and fill rest randomly
            remaining_indices = [i for i in range(total_samples) if i != sample_idx]
            if len(remaining_indices) >= num_samples - 1:
                additional_indices = np.random.choice(remaining_indices, size=num_samples - 1, replace=False)
                sample_indices = np.concatenate([[sample_idx], additional_indices])
            else:
                sample_indices = np.concatenate([[sample_idx], remaining_indices])
            print(f"Using specified sample index {sample_idx} for visualization/audio")
    else:
        sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    all_results = {}
    
    # Store visualization data (use first model for visualization/audio)
    vis_context = None
    vis_ground_truth = None
    vis_predictions = {}  # Store predictions for each model
    vis_sample_idx = sample_idx if sample_idx is not None else sample_indices[0]
    
    for model_name, checkpoint_path in model_checkpoints.items():
        print(f"\\n{'='*80}")
        print(f"Evaluating {model_name.upper()} model...")
        print(f"{'='*80}")
        
        # Load model
        model, config = load_model_from_checkpoint(checkpoint_path, device)
        
        model_results = {}
        
        for context_size in context_sizes:
            print(f"\\n  Testing context_size={context_size} frames...")
            
            # Check if we have enough frames
            if context_size + k_value > test_tensor.shape[1]:
                print(f"  ⚠️  Skipping context_size={context_size}: Need {context_size + k_value} frames but only {test_tensor.shape[1]} available")
                continue
            
            all_predictions = []
            all_ground_truths = []
            
            # Store visualization data only for first model and context_size=150, K=60
            is_first_model = (list(model_checkpoints.keys()).index(model_name) == 0)
            
            with torch.no_grad():
                for i, idx in enumerate(tqdm(sample_indices, desc=f"    Context={context_size}", leave=False)):
                    # Get context and ground truth
                    context = test_tensor[idx:idx+1, :context_size, :].to(device)
                    ground_truth = test_tensor[idx, context_size:context_size+k_value, :]
                    
                    # Store specified sample for visualization (only for first model)
                    if idx == vis_sample_idx and is_first_model:
                        vis_context = context.cpu()
                        vis_ground_truth = ground_truth
                    
                    # Normalize context if needed
                    if 'normalization_stats' in config:
                        stats = config['normalization_stats']
                        if stats['method'] == 'standardize':
                            context_norm = (context - stats['mean']) / stats['std']
                        elif stats['method'] == 'minmax':
                            context_norm = (context - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                        else:
                            context_norm = context
                    else:
                        context_norm = context
                    
                    # Generate
                    generated_norm = model.generate(context_norm, num_frames=k_value)
                    
                    # Denormalize
                    if 'normalization_stats' in config:
                        stats = config['normalization_stats']
                        if stats['method'] == 'standardize':
                            generated = generated_norm * stats['std'] + stats['mean']
                        elif stats['method'] == 'minmax':
                            generated = generated_norm * (stats['max'] - stats['min']) + stats['min']
                        else:
                            generated = generated_norm
                    else:
                        generated = generated_norm
                    
                    # Store prediction for visualization (only for first model at context_size=150, K=60)
                    if idx == vis_sample_idx and is_first_model and context_size == 150 and k_value == 60:
                        vis_predictions[model_name] = generated.cpu().squeeze(0)
                    
                    all_predictions.append(generated.cpu())
                    all_ground_truths.append(ground_truth.unsqueeze(0))
            
            # Concatenate and compute metrics
            predictions = torch.cat(all_predictions, dim=0)  # (num_samples, k, F)
            ground_truths = torch.cat(all_ground_truths, dim=0)  # (num_samples, k, F)
            
            metrics = compute_all_metrics(predictions, ground_truths, return_per_frame=True)
            model_results[context_size] = metrics
            
            # Print summary
            print(f"    MSE: {metrics['MSE']:.6f}")
            print(f"    Frame Accuracy: {metrics['Frame Accuracy (10%)']:.2%}")
            print(f"    R²: {metrics['R²']:.4f}")
            print(f"    Correlation: {metrics['Correlation']:.4f}")
            print(f"    LSD: {metrics['Log Spectral Distance']:.4f} dB")
        
        all_results[model_name] = model_results
    
    # Visualize spectrograms and audio (only for first model at context_size=150, K=60)
    if visualize and vis_context is not None and vis_ground_truth is not None:
        # Use first model's prediction for visualization
        first_model_name = list(model_checkpoints.keys())[0]
        if first_model_name in vis_predictions:
            visualize_spectrograms(
                ground_truth=vis_ground_truth,
                predicted=vis_predictions[first_model_name],
                context=vis_context.squeeze(0),
                title=f"Context Size=150, K=60",
                save_path=Path(save_dir) / f"benchmark2_context150_spectrogram.png"
            )
            
            # Audio decoding for this sample (first model only)
            print(f"\\n🎵 Generating audio at context_size=150, K=60...")
            
            # Prepare spectrograms for audio conversion (transpose to freq_bins x time_frames)
            context_spec = vis_context.squeeze(0).transpose(0, 1)  # (128, context_size)
            generated_spec = vis_predictions[first_model_name].transpose(0, 1)  # (128, k_value)
            ground_truth_spec = vis_ground_truth.transpose(0, 1)  # (128, k_value)
            
            # Convert to audio
            try:
                context_audio = mel_spectrogram_to_audio(
                    context_spec,
                    sample_rate=44100,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128,
                    f_max=22050
                )
                
                generated_audio = mel_spectrogram_to_audio(
                    generated_spec,
                    sample_rate=44100,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128,
                    f_max=22050
                )
                
                ground_truth_audio = mel_spectrogram_to_audio(
                    ground_truth_spec,
                    sample_rate=44100,
                    n_fft=2048,
                    hop_length=512,
                    n_mels=128,
                    f_max=22050
                )
                
                # Save audio files
                audio_dir = Path(save_dir) / "audio_samples"
                audio_dir.mkdir(exist_ok=True, parents=True)
                
                context_audio_path = audio_dir / "context.wav"
                generated_audio_path = audio_dir / "generated.wav"
                ground_truth_audio_path = audio_dir / "ground_truth.wav"
                
                torchaudio.save(str(context_audio_path), context_audio.unsqueeze(0), 44100)
                torchaudio.save(str(generated_audio_path), generated_audio.unsqueeze(0), 44100)
                torchaudio.save(str(ground_truth_audio_path), ground_truth_audio.unsqueeze(0), 44100)
                
                print(f"✅ Audio saved:")
                print(f"   Context: {context_audio_path}")
                print(f"   Generated: {generated_audio_path}")
                print(f"   Ground Truth: {ground_truth_audio_path}")
                
                # Display audio players in Colab
                print(f"\\n📻 Audio Players:")
                print(f"Context Audio:")
                display(Audio(str(context_audio_path), rate=44100))
                print(f"Generated Audio:")
                display(Audio(str(generated_audio_path), rate=44100))
                print(f"Ground Truth Audio:")
                display(Audio(str(ground_truth_audio_path), rate=44100))
                
            except Exception as e:
                print(f"⚠️  Error generating audio: {e}")
                print("   Make sure torchaudio and torchcodec are installed: !pip install -q torchcodec")
    
    # Visualize metrics comparison (only MSE and Frame Accuracy)
    if visualize:
        metrics_to_plot = ['MSE', 'Frame Accuracy (10%)']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Exclude helper entries from plotting
        model_names = list(all_results.keys())
        # Use consistent colors: Blue, Orange, Green for small, medium, large
        color_map = {'small': '#1f77b4', 'medium': '#ff7f0e', 'large': '#2ca02c'}
        colors = [color_map.get(name, plt.cm.Set3(i)) for i, name in enumerate(model_names)]
        
        # Create legend handles (will be used once at the end)
        legend_handles = []
        
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            for i, model_name in enumerate(model_names):
                values = []
                valid_context_sizes = []
                for ctx_size in context_sizes:
                    if ctx_size in all_results[model_name]:
                        values.append(all_results[model_name][ctx_size][metric_name])
                        valid_context_sizes.append(ctx_size)
                    else:
                        values.append(np.nan)
                
                # Use line plot with consistent colors
                line = ax.plot(valid_context_sizes, [v for v in values if not np.isnan(v)], 
                       marker='o', color=colors[i], linewidth=2, markersize=8, label=model_name)
                # Store handle for legend (only once per model)
                if idx == 0:  # Only add to legend from first plot
                    legend_handles.append(line[0])
            
            ax.set_xlabel('Context Size (frames)', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} vs Context Size', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3, linestyle='--')
        
        # Add single legend outside the plot area
        fig.legend(legend_handles, model_names, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle('Benchmark 2: Varying Context Size, Constant K=60', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 0.96, 0.96])  # Leave space for legend and title
        
        save_path = Path(save_dir) / "benchmark2_varying_context.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        print(f"\\nVisualization saved to: {save_path}")
        plt.show()
    
    return all_results
"""

# ============================================================================
# CELL 10: Benchmark 3 - Model Size Comparison
# ============================================================================
"""
def benchmark_model_sizes(
    model_checkpoints: Dict[str, str],  # {'small': path, 'medium': path, 'large': path}
    test_tensor: torch.Tensor,
    context_size: int = 150,
    k_value: int = 60,
    num_samples: Optional[int] = None,  # None = use all
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    visualize: bool = True,
    save_dir: str = "benchmark_results"
):
    \"\"\"
    Benchmark 3: Compare model sizes (small, medium, large).
    
    Args:
        model_checkpoints: Dict mapping model sizes to checkpoint paths/URLs
        test_tensor: Test tensor of shape (N, T, F)
        context_size: Context size (150 frames)
        k_value: Number of frames to predict (60)
        num_samples: Number of samples to evaluate (None = all)
        device: Device to run on
        visualize: Whether to create visualizations
        save_dir: Directory to save results
    
    Returns:
        Dictionary of results: {model_size: metrics}
    \"\"\"
    print("="*80)
    print("BENCHMARK 3: MODEL SIZE COMPARISON")
    print("="*80)
    print(f"Context size: {context_size} frames")
    print(f"K value: {k_value} frames")
    print(f"Models to compare: {list(model_checkpoints.keys())}")
    print("="*80)
    
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    # Select samples
    total_samples = test_tensor.shape[0]
    if num_samples is None or num_samples > total_samples:
        num_samples = total_samples
    
    sample_indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    all_results = {}
    
    for model_size, checkpoint_path in model_checkpoints.items():
        print(f"\\n{'='*80}")
        print(f"Evaluating {model_size.upper()} model...")
        print(f"{'='*80}")
        
        # Load model
        model, config = load_model_from_checkpoint(checkpoint_path, device)
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for idx in tqdm(sample_indices, desc=f"  {model_size}"):
                # Get context and ground truth
                context = test_tensor[idx:idx+1, :context_size, :].to(device)
                ground_truth = test_tensor[idx, context_size:context_size+k_value, :]
                
                # Normalize context if needed
                if 'normalization_stats' in config:
                    stats = config['normalization_stats']
                    if stats['method'] == 'standardize':
                        context_norm = (context - stats['mean']) / stats['std']
                    elif stats['method'] == 'minmax':
                        context_norm = (context - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
                    else:
                        context_norm = context
                else:
                    context_norm = context
                
                # Generate
                generated_norm = model.generate(context_norm, num_frames=k_value)
                
                # Denormalize
                if 'normalization_stats' in config:
                    stats = config['normalization_stats']
                    if stats['method'] == 'standardize':
                        generated = generated_norm * stats['std'] + stats['mean']
                    elif stats['method'] == 'minmax':
                        generated = generated_norm * (stats['max'] - stats['min']) + stats['min']
                    else:
                        generated = generated_norm
                else:
                    generated = generated_norm
                
                all_predictions.append(generated.cpu())
                all_ground_truths.append(ground_truth.unsqueeze(0))
        
        # Concatenate and compute metrics
        predictions = torch.cat(all_predictions, dim=0)  # (num_samples, k, F)
        ground_truths = torch.cat(all_ground_truths, dim=0)  # (num_samples, k, F)
        
        metrics = compute_all_metrics(predictions, ground_truths, return_per_frame=True)
        all_results[model_size] = metrics
        
        # Print summary
        print(f"\\n{model_size.upper()} Results:")
        print(f"    MSE: {metrics['MSE']:.6f}")
        print(f"    Frame Accuracy: {metrics['Frame Accuracy (10%)']:.2%}")
        print(f"    R²: {metrics['R²']:.4f}")
        print(f"    Correlation: {metrics['Correlation']:.4f}")
        print(f"    LSD: {metrics['Log Spectral Distance']:.4f} dB")
    
    # Visualize comparison (only MSE and Frame Accuracy)
    if visualize:
        metrics_to_plot = ['MSE', 'Frame Accuracy (10%)']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        model_sizes = list(all_results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for idx, metric_name in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [all_results[model][metric_name] for model in model_sizes]
            
            bars = ax.bar(model_sizes, values, color=colors[:len(model_sizes)], alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                # Format based on metric type
                if 'Accuracy' in metric_name:
                    label_text = f'{height:.2%}'
                else:
                    label_text = f'{height:.4f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label_text,
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add single legend outside the plot area
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], edgecolor='black', alpha=0.7, label=model_sizes[i]) for i in range(len(model_sizes))]
        fig.legend(legend_elements, model_sizes, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11, frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle('Benchmark 3: Model Size Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 0.96, 0.96])  # Leave space for legend and title
        
        save_path = Path(save_dir) / "benchmark3_model_sizes.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        print(f"\\nVisualization saved to: {save_path}")
        plt.show()
    
    return all_results
"""

# ============================================================================
# CELL 11: Configure and Run Benchmarks
# ============================================================================
"""
# Configure model checkpoints (Google Drive URLs or local paths)
# You can specify any number of models for Benchmark 1 and 3
model_checkpoints = {
    'small': 'YOUR_SMALL_MODEL_URL_OR_PATH',
    'medium': 'YOUR_MEDIUM_MODEL_URL_OR_PATH',
    'large': 'YOUR_LARGE_MODEL_URL_OR_PATH'
}

# For Benchmark 2, specify a single model (ideally your best model)
best_model_checkpoint = 'YOUR_BEST_MODEL_URL_OR_PATH'
"""

# ============================================================================
# CELL 12: Run Benchmark 1
# ============================================================================
"""
results_b1 = benchmark_varying_k(
    model_checkpoints=model_checkpoints,
    test_tensor=test_tensor,
    context_size=150,
    k_values=[20, 60, 100],
    num_samples=2000,  # Use all or specify a number
    visualize=True,
    save_dir="benchmark_results"
)
"""

# ============================================================================
# CELL 13: Run Benchmark 2
# ============================================================================
"""
results_b2 = benchmark_varying_context(
    model_checkpoint=best_model_checkpoint,
    test_tensor=test_tensor,
    context_sizes=[50, 100, 150],
    k_value=60,
    num_samples=100,  # Fewer samples for faster evaluation
    visualize=True,
    save_dir="benchmark_results"
)
"""

# ============================================================================
# CELL 14: Run Benchmark 3
# ============================================================================
"""
results_b3 = benchmark_model_sizes(
    model_checkpoints=model_checkpoints,
    test_tensor=test_tensor,
    context_size=150,
    k_value=60,
    num_samples=2000,  # Use all or specify a number
    visualize=True,
    save_dir="benchmark_results"
)
"""

