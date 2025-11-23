# Audio Frame Prediction: Transformers vs LSTMs

A PyTorch implementation comparing Transformer and LSTM architectures for predicting future audio frames from spectrogram representations.

## Project Structure

```
audio-predict/
├── data/
│   ├── raw_audio/              # Place raw audio files here
│   └── processed_spectrograms/  # Preprocessed spectrograms (auto-generated)
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset classes for audio frame prediction
│   │   └── preprocess.py       # Audio preprocessing utilities
│   ├── models/
│   │   ├── lstm_baseline.py    # LSTM baseline model
│   │   ├── transformer_small.py
│   │   ├── transformer_medium.py
│   │   └── transformer_large.py
│   ├── training/
│   │   ├── train_lstm.py       # Training script for LSTM
│   │   └── train_transformer.py  # Training script for Transformers
│   └── evaluation/
│       ├── eval_architectures.py  # Compare all architectures
│       ├── audio_reconstruction.py  # Convert predictions to audio
│       ├── benchmarking.py     # Comprehensive benchmarking system
│       └── generate_report.py  # Generate benchmark reports
├── config.yaml                 # Configuration file
├── requirements.txt
└── README.md
```

## Installation

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### 1. Preprocess Audio Data

Place your audio files in `data/raw_audio/`, then preprocess them:

```python
from src.data.preprocess import AudioPreprocessor, preprocess_audio_directory

# Create preprocessor
preprocessor = AudioPreprocessor(
    sample_rate=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    spectrogram_type="mel"
)

# Preprocess all audio files
preprocess_audio_directory(
    input_dir="data/raw_audio",
    output_dir="data/processed_spectrograms",
    preprocessor=preprocessor
)
```

### 2. Train Models

#### Train LSTM Baseline

```bash
python src/training/train_lstm.py \
    --spectrogram_dir data/processed_spectrograms \
    --context_size 10 \
    --prediction_horizon 5 \
    --batch_size 32 \
    --num_epochs 50 \
    --save_dir checkpoints
```

#### Train Transformer (Small/Medium/Large)

```bash
python src/training/train_transformer.py \
    --model small \
    --spectrogram_dir data/processed_spectrograms \
    --context_size 10 \
    --prediction_horizon 5 \
    --batch_size 32 \
    --num_epochs 50 \
    --save_dir checkpoints
```

### 3. Evaluate Models

Compare all trained models:

```bash
python src/evaluation/eval_architectures.py \
    --checkpoint_dir checkpoints \
    --spectrogram_dir data/processed_spectrograms \
    --context_size 10 \
    --prediction_horizon 5
```

### 4. Run Benchmarks

Run comprehensive benchmarking experiments:

```bash
# Run all benchmarks
python src/evaluation/benchmarking.py \
    --spectrogram_dir data/processed_spectrograms \
    --results_dir benchmark_results

# Run specific benchmark
python src/evaluation/benchmarking.py \
    --spectrogram_dir data/processed_spectrograms \
    --benchmark 1

# Generate report
python src/evaluation/generate_report.py \
    --results_dir benchmark_results
```

## Model Architectures

### LSTM Baseline
- Multi-layer LSTM with configurable hidden dimension and depth
- Input projection → LSTM → Output projection
- Default: 2 layers, 256 hidden dim

### Transformer Small
- 2 transformer encoder blocks
- 128 model dimension, 4 attention heads
- ~500K parameters

### Transformer Medium
- 4 transformer encoder blocks
- 256 model dimension, 8 attention heads
- ~2M parameters

### Transformer Large
- 8 transformer encoder blocks
- 512 model dimension, 16 attention heads
- ~10M parameters

## Evaluation Metrics

- MSE: Mean Squared Error
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- Cosine Similarity: Spectrogram similarity metric

## Benchmarking Experiments

The project includes a comprehensive benchmarking system for four experiments:

1. Fixed context duration → Vary prediction horizon (k)
   - Fixed input duration (e.g., 1 minute)
   - Test k values: [1, 3, 5, 10, 20] frames
   - Measure how prediction accuracy changes with prediction distance

2. Vary context duration → Fixed k
   - Fixed k = 10 frames
   - Test context durations: [5s, 10s, 20s, 30s, 60s]
   - Measure how context size affects prediction quality

3. Vary model size
   - Compare Transformer small, medium, and large
   - Measure parameter count vs performance trade-offs

4. Vary dataset size
   - Train on 60%, 80%, and 100% of full dataset
   - Measure how dataset size affects learning

## Supported Audio Formats

- WAV (.wav) - Recommended, lossless
- MP3 (.mp3) - Most common format
- FLAC (.flac) - Lossless compression
- M4A (.m4a) - Apple format

For MP4 video files, extract audio first:
```bash
ffmpeg -i video.mp4 -vn -acodec copy data/raw_audio/audio.m4a
```

## Configuration

Edit `config.yaml` to adjust:
- Audio preprocessing parameters
- Model architectures
- Training hyperparameters
- Experiment settings

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Notes

- The project uses Mel spectrograms by default (can be changed to STFT)
- All models predict spectrogram frames, which can be converted back to audio
- Training checkpoints are saved automatically
- Evaluation results are saved as JSON files



## License

This project is for educational/research purposes.

