"""
Audio preprocessing utilities for converting audio to spectrograms.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import librosa


class AudioPreprocessor:
    """
    Preprocesses audio files into spectrogram representations.
    Supports both STFT and Mel spectrograms.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        spectrogram_type: str = "mel",  # "mel" or "stft"
        normalize: bool = True,
        device: str = "cpu"
    ):
        """
        Args:
            sample_rate: Target sample rate for audio
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel filter banks (only for mel spectrogram)
            spectrogram_type: "mel" or "stft"
            normalize: Whether to normalize spectrograms
            device: Device to run computations on
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.spectrogram_type = spectrogram_type
        self.normalize = normalize
        self.device = device
        
        # Initialize transforms
        if spectrogram_type == "mel":
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            ).to(device)
        else:
            self.stft_transform = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length
            ).to(device)
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform tensor of shape (1, T)
        """
        try:
            # Try torchaudio first (preferred)
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.to(self.device)
        except Exception as e:
            # Fallback to librosa if torchaudio fails (e.g., missing torchcodec for MP3)
            try:
                import librosa
                waveform, sr = librosa.load(audio_path, sr=None, mono=False)
                # Convert to torch tensor
                waveform = torch.from_numpy(waveform).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # Add channel dimension
                waveform = waveform.to(self.device)
            except Exception as librosa_error:
                raise RuntimeError(
                    f"Failed to load audio with both torchaudio and librosa. "
                    f"torchaudio error: {e}, librosa error: {librosa_error}"
                )
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def audio_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to spectrogram.
        
        Args:
            waveform: Audio tensor of shape (1, T) or (T,)
            
        Returns:
            Spectrogram tensor of shape (freq_bins, time_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        if self.spectrogram_type == "mel":
            spec = self.mel_transform(waveform)
        else:
            spec = self.stft_transform(waveform)
        
        # Convert to log scale (dB)
        spec = torch.log10(spec + 1e-10)
        
        # Normalize if requested
        if self.normalize:
            spec = self._normalize_spectrogram(spec)
        
        # Remove batch dimension
        return spec.squeeze(0)
    
    def _normalize_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram to [0, 1] range."""
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max > spec_min:
            spec = (spec - spec_min) / (spec_max - spec_min)
        return spec
    
    def spectrogram_to_audio(
        self, 
        spectrogram: torch.Tensor,
        method: str = "griffin_lim"
    ) -> torch.Tensor:
        """
        Convert spectrogram back to audio waveform.
        
        Args:
            spectrogram: Spectrogram tensor of shape (freq_bins, time_frames)
            method: Reconstruction method ("griffin_lim" or "istft")
            
        Returns:
            Audio waveform tensor
        """
        # Denormalize if normalized
        if self.normalize:
            # This is approximate - in practice, you'd want to store normalization params
            spec = spectrogram * 10.0 - 5.0  # Rough inverse
        
        # Convert from log scale
        spec = torch.pow(10.0, spec) - 1e-10
        
        if method == "griffin_lim":
            if self.spectrogram_type == "mel":
                # Need to convert mel to linear first
                # For now, use inverse mel
                inverse_mel = torchaudio.transforms.InverseMelScale(
                    n_stft=self.n_fft // 2 + 1,
                    n_mels=self.n_mels,
                    sample_rate=self.sample_rate
                ).to(self.device)
                # Approximate: convert mel to linear magnitude
                linear_spec = inverse_mel(spec.unsqueeze(0))
            else:
                linear_spec = spec.unsqueeze(0)
            
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ).to(self.device)
            waveform = griffin_lim(linear_spec)
        else:
            # Direct ISTFT
            waveform = torch.istft(
                spec.unsqueeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
        
        return waveform.squeeze(0)
    
    def split_into_frames(
        self, 
        spectrogram: torch.Tensor, 
        frame_size: int,
        overlap: int = 0
    ) -> torch.Tensor:
        """
        Split spectrogram into fixed-size frames.
        
        Args:
            spectrogram: Spectrogram of shape (freq_bins, time_frames)
            frame_size: Number of time frames per chunk
            overlap: Number of overlapping frames between chunks
            
        Returns:
            Tensor of shape (num_chunks, freq_bins, frame_size)
        """
        freq_bins, time_frames = spectrogram.shape
        stride = frame_size - overlap
        
        frames = []
        for i in range(0, time_frames - frame_size + 1, stride):
            frame = spectrogram[:, i:i + frame_size]
            frames.append(frame)
        
        if len(frames) == 0:
            # Pad if too short
            padding = frame_size - time_frames
            padded = torch.nn.functional.pad(spectrogram, (0, padding))
            frames = [padded]
        
        return torch.stack(frames)


def preprocess_audio_directory(
    input_dir: str,
    output_dir: str,
    preprocessor: AudioPreprocessor,
    file_extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a")
):
    """
    Preprocess all audio files in a directory and save spectrograms.
    
    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory to save processed spectrograms
        preprocessor: AudioPreprocessor instance
        file_extensions: Audio file extensions to process
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in audio_files:
        try:
            waveform = preprocessor.load_audio(str(audio_file))
            spectrogram = preprocessor.audio_to_spectrogram(waveform)
            
            # Save as .pt file
            output_file = output_path / f"{audio_file.stem}.pt"
            torch.save(spectrogram, output_file)
            print(f"Processed: {audio_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
    
    print(f"Preprocessing complete. Saved {len(audio_files)} spectrograms to {output_dir}")

