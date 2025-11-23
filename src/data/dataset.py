"""
Dataset classes for audio frame prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

# Check if MPS is available (Apple Silicon)
try:
    MPS_AVAILABLE = torch.backends.mps.is_available()
except AttributeError:
    MPS_AVAILABLE = False


class AudioFrameDataset(Dataset):
    """
    Dataset for audio frame prediction.
    Each sample consists of:
    - Input: context_size frames of spectrogram
    - Target: next k frames of spectrogram
    """
    
    def __init__(
        self,
        spectrogram_dir: str,
        context_size: int = 10,
        prediction_horizon: int = 5,
        frame_size: int = 1,
        stride: int = 1,
        max_files: Optional[int] = None
    ):
        """
        Args:
            spectrogram_dir: Directory containing preprocessed spectrogram .pt files
            context_size: Number of input frames to use as context
            prediction_horizon: Number of future frames to predict (k)
            frame_size: Number of time steps per frame (default 1)
            stride: Stride for sliding window (default 1)
            max_files: Maximum number of files to load (None = all)
        """
        self.spectrogram_dir = Path(spectrogram_dir)
        self.context_size = context_size
        self.prediction_horizon = prediction_horizon
        self.frame_size = frame_size
        self.stride = stride
        
        # Load all spectrogram files
        spec_files = list(self.spectrogram_dir.glob("*.pt"))
        if max_files is not None:
            spec_files = spec_files[:max_files]
        
        self.spectrograms = []
        self.file_indices = []  # Track which file each sample comes from
        
        print(f"Loading {len(spec_files)} spectrogram files...")
        for file_idx, spec_file in enumerate(spec_files):
            try:
                spec = torch.load(spec_file)
                # spec shape: (freq_bins, time_frames)
                if spec.dim() == 2:
                    self.spectrograms.append(spec)
                    # Calculate number of samples from this file
                    time_frames = spec.shape[1]
                    num_samples = max(0, (time_frames - context_size - prediction_horizon) // stride + 1)
                    self.file_indices.extend([file_idx] * num_samples)
            except Exception as e:
                print(f"Error loading {spec_file}: {e}")
        
        print(f"Loaded {len(self.spectrograms)} spectrograms")
        print(f"Total samples: {len(self.file_indices)}")
    
    def __len__(self) -> int:
        return len(self.file_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_frames: (freq_bins, context_size) - input context
            target_frames: (freq_bins, prediction_horizon) - target future frames
        """
        file_idx = self.file_indices[idx]
        spec = self.spectrograms[file_idx]
        
        # Calculate position in this file
        # Count how many samples come from files before this one
        samples_before = sum(1 for i in range(len(self.file_indices)) 
                           if i < idx and self.file_indices[i] == file_idx)
        
        # Calculate start position
        start_pos = samples_before * self.stride
        
        # Extract input and target frames
        input_end = start_pos + self.context_size
        target_start = input_end
        target_end = target_start + self.prediction_horizon
        
        # Handle edge cases
        if target_end > spec.shape[1]:
            # Pad if necessary
            padding_size = target_end - spec.shape[1]
            spec = torch.nn.functional.pad(spec, (0, padding_size), mode='constant', value=0)
        
        input_frames = spec[:, start_pos:input_end]  # (freq_bins, context_size)
        target_frames = spec[:, target_start:target_end]  # (freq_bins, prediction_horizon)
        
        return input_frames, target_frames


def create_dataloaders(
    spectrogram_dir: str,
    context_size: int = 10,
    prediction_horizon: int = 5,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    frame_size: int = 1,
    stride: int = 1,
    max_files: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        spectrogram_dir: Directory containing spectrograms
        context_size: Number of input frames
        prediction_horizon: Number of future frames to predict
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        frame_size: Frames per sample
        stride: Stride for sliding window
        max_files: Maximum files to load
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = AudioFrameDataset(
        spectrogram_dir=spectrogram_dir,
        context_size=context_size,
        prediction_horizon=prediction_horizon,
        frame_size=frame_size,
        stride=stride,
        max_files=max_files
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    # pin_memory is not supported on MPS (Apple Silicon), so disable it
    use_pin_memory = torch.cuda.is_available() and not MPS_AVAILABLE
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

