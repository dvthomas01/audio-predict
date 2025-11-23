"""
Audio reconstruction from predicted spectrograms.
"""

import torch
import torchaudio
from pathlib import Path
import numpy as np
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.preprocess import AudioPreprocessor


def reconstruct_audio_from_predictions(
    predictions: torch.Tensor,
    preprocessor: AudioPreprocessor,
    output_path: str,
    sample_rate: int = 22050
):
    """
    Reconstruct audio waveform from predicted spectrogram frames.
    
    Args:
        predictions: Predicted spectrogram frames (freq_bins, prediction_horizon)
        preprocessor: AudioPreprocessor instance used for preprocessing
        output_path: Path to save reconstructed audio
        sample_rate: Sample rate for output audio
    """
    # Convert spectrogram back to audio
    waveform = preprocessor.spectrogram_to_audio(predictions)
    
    # Ensure correct shape: (1, T) for torchaudio
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Save audio
    torchaudio.save(output_path, waveform, sample_rate)
    print(f"Reconstructed audio saved to {output_path}")


def reconstruct_batch_predictions(
    predictions: torch.Tensor,
    preprocessor: AudioPreprocessor,
    output_dir: str,
    sample_rate: int = 22050,
    prefix: str = "reconstructed"
):
    """
    Reconstruct audio from a batch of predictions.
    
    Args:
        predictions: Predicted spectrograms (batch_size, freq_bins, prediction_horizon)
        preprocessor: AudioPreprocessor instance
        output_dir: Directory to save reconstructed audio files
        sample_rate: Sample rate for output audio
        prefix: Prefix for output filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    batch_size = predictions.shape[0]
    
    for i in range(batch_size):
        pred_spec = predictions[i]  # (freq_bins, prediction_horizon)
        output_file = output_path / f"{prefix}_{i:04d}.wav"
        reconstruct_audio_from_predictions(
            pred_spec,
            preprocessor,
            str(output_file),
            sample_rate
        )


def concatenate_predictions(
    predictions: torch.Tensor,
    overlap: int = 0
) -> torch.Tensor:
    """
    Concatenate multiple predicted frames into a continuous spectrogram.
    
    Args:
        predictions: Predicted frames (num_frames, freq_bins, prediction_horizon)
        overlap: Number of overlapping frames between predictions
        
    Returns:
        Concatenated spectrogram (freq_bins, total_time)
    """
    num_frames, freq_bins, prediction_horizon = predictions.shape
    
    if overlap == 0:
        # Simple concatenation
        total_time = num_frames * prediction_horizon
        concatenated = predictions.transpose(0, 1).reshape(freq_bins, total_time)
    else:
        # Overlapping concatenation with averaging
        stride = prediction_horizon - overlap
        total_time = (num_frames - 1) * stride + prediction_horizon
        concatenated = torch.zeros(freq_bins, total_time)
        counts = torch.zeros(total_time)
        
        for i in range(num_frames):
            start = i * stride
            end = start + prediction_horizon
            concatenated[:, start:end] += predictions[i]
            counts[start:end] += 1
        
        # Average overlapping regions
        concatenated = concatenated / counts.clamp(min=1)
    
    return concatenated


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconstruct audio from predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions .pt file")
    parser.add_argument("--output", type=str, required=True, help="Output audio path")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--spectrogram_type", type=str, default="mel", choices=["mel", "stft"])
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=args.sample_rate,
        spectrogram_type=args.spectrogram_type
    )
    
    # Load predictions
    predictions = torch.load(args.predictions)
    
    # Reconstruct
    reconstruct_audio_from_predictions(
        predictions,
        preprocessor,
        args.output,
        args.sample_rate
    )

