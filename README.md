# Attention vs. Recurrence: Benchmarking Transformers and LSTMs for Music Spectral Forecasting
**Dami Thomas · Liam Sheldon · Suraj Reddy**  
_Final Project for 6.7960 (MIT)_

---

## Abstract
With the rise of large-scale multimodal generation, coherent music forecasting has become an increasingly relevant sequence modeling problem. While **LSTMs** historically dominated music time-series modeling due to strong inductive bias for temporal recurrence, **Transformers** have recently shown competitive performance in many sequence domains via global self-attention. However, their effectiveness in **continuous audio forecasting**—where evaluation must reflect both numerical accuracy and perceptual coherence—remains contested.  

We benchmark **standard autoregressive Transformers** against **LSTM baselines** on **mel-spectrogram next-frame forecasting**, using Medley-Solos-DB. We evaluate across (i) **future prediction horizon** (varying *K*), (ii) **context sensitivity** (varying input history), and (iii) **qualitative audio reconstruction** via Griffin–Lim. Across settings, the Transformer exhibits stronger retention of global musical structure and improved perceptual quality, while LSTMs display stronger smoothing and instability in long-horizon generation. Interestingly, we observe non-monotonic scaling for Transformers (medium outperforming large), suggesting dataset size and optimization sensitivity at larger parameter counts.

---

## 1. Introduction
Generative music forecasting requires capturing both local texture (transients, harmonics) and long-range structure (phrasing, melodic contour). Recurrent Neural Networks (RNNs), particularly **Long Short-Term Memory (LSTM)** networks, historically excelled at this task due to explicit temporal state retention [1].  

More recently, **self-attention** has enabled Transformers to dominate many sequence modeling problems, raising the question of whether attention-driven models outperform recurrence for audio prediction [2]. Unlike NLP, audio forecasting must preserve fine-grained texture while remaining **audibly coherent**, not merely numerically consistent.

**Research question:** *Does the computational overhead of Transformers translate into better musical “listening” than efficient recurrence?*

---

## 2. Related Work
Early work established LSTMs as a core method for algorithmic composition (e.g., blues improvisation modeling) [3]. In other continuous forecasting domains, results are mixed: LSTMs can outperform Transformers in stock forecasting [4], while Transformers capture long-range trends better in video modeling [5].  

In music/audio, Transformer success often depends on **symbolic representations** (e.g., MIDI events) rather than raw audio [6]. Audio-specific Transformer variants such as **SpecTNT** address limitations of naïve frequency-vector representations via specialized time-frequency modeling [8]. A gap remains for **standard Transformers** in **generative forecasting** tasks on continuous spectral representations.

---

## 3. Goals
We introduce two primary shifts relative to prior audio Transformer work:
1. Focus on **generative autoregressive forecasting** (next-*K* prediction), not classification.
2. Perform an **architectural ablation**: compare **standard Transformer** baselines against LSTMs rather than specialized designs (e.g., TNT).

We train **Small / Medium / Large** versions of both families to isolate architecture vs. capacity effects.

---

## 4. Methods

### 4.1 Data
We use **Medley-Solos-DB** [9], containing single-instrument solos across 7+ instrument classes (e.g., clarinet, electric guitar, flute, piano, trumpet, violin). This setup produces relatively clean timbral structure while remaining challenging for autoregressive forecasting.

We preprocess audio into **128-bin Mel Spectrograms**, reducing ~65k waveform samples into ~253 time frames while retaining pitch/timbre cues.

**Figure 1. Project overview graphic**  
![Figure 1: Introduction of methods in research project.](./images/intrographic.png)

**Figure 2. Example Mel Spectrogram**  
![Figure 2: Example of Mel Spectrogram Data](./images/examplespectrogram.png)

---

### 4.2 Audio Reconstruction (Mel → Waveform)
To evaluate perceptual quality, we reconstruct audio from predicted Mel Spectrograms:

1. Convert log magnitudes to linear via exponentiation  
2. Map Mel magnitudes to linear-frequency spectrogram using inverse Mel operator  
3. Estimate missing phase using **Griffin–Lim** iterations  
4. Output 1D waveform signal for listening-based comparison  

---

### 4.3 Architectures

#### 4.3.1 Autoregressive Transformer
Causal Transformer with sinusoidal positional encodings. Variants:

- **Small:** 2 layers, 4 heads, d=128  
- **Medium:** 4 layers, 8 heads, d=256  
- **Large:** 8 layers, 16 heads, d=512  

#### 4.3.2 LSTM Baseline
Standard LSTM stacked with projection layers:

- **Small:** 1 layer, hidden=256  
- **Medium:** 2 layers, hidden=512  
- **Large:** 4 layers, hidden=1024  

| Variant | Transformer Params | LSTM Params |
|---|---:|---:|
| Small | 544,768 | ~600,000 |
| Medium | 4,111,232 | ~4,300,000 |
| Large | 26,598,016 | ~33,850,496 |

---

### 4.4 Training Details
- Optimizer: **AdamW**
- Learning rate: `1e-4`
- Batch size: `64`
- Epochs: `35`

---

### 4.5 Experimental Design

#### Benchmark 1: Future Horizon Forecasting (varying *K*)
Given fixed context of 150 frames, predict next:
- *K* ∈ {20, 60, 100}

#### Benchmark 2: Context Sensitivity (varying context)
Fix horizon to 60 frames; vary context:
- context ∈ {50, 100, 150}

#### Benchmark 3: Audio Comparison
Qualitative audio evaluation comparing:
- Transformer (Medium)
- LSTM (Large)
- Ground truth

---

### 4.6 Evaluation Metrics
- **Mean Squared Error (MSE)** over spectrogram pixels
- **Frame Accuracy**: % frames within 10% of ground truth
- **Visual inspection**: harmonic definition, transient clarity
- **Audio inspection**: listening-based evaluation

---

## 5. Results

### 5.1 Benchmark 1: Forecast Horizon (varying *K*)

#### LSTM
Small/Medium LSTMs degrade sharply at longer horizons, while the Large LSTM remains more stable (lowest MSE at *K*=60 and *K*=100). This suggests a more “classic” scaling trend where additional capacity improves long-horizon stability.

**Figure 3. LSTM performance vs. forecast horizon**  
![Figure 3: LSTM performance with Varying K-frame horizon](./images/LSTMs/benchmark1_lstm.png)

#### Transformer
Transformer performance shows non-monotonic scaling: **Medium outperforms Large** across horizons, suggesting possible overfitting or optimization instability in the 26M+ parameter regime.

**Figure 4. Transformer performance vs. forecast horizon**  
![Figure 4: Transformer MSE and Frame Accuracy vs. K](./images/Transformers/benchmark1_varying_k%20(1).png)

---

### 5.2 Benchmark 2: Context Sensitivity (varying input)

#### LSTM
Large LSTM benefits from increased context (MSE decreases from context=50 to 150). However, frame accuracy remains low overall, consistent with visually smoothed predictions.

**Figure 5. LSTM context sensitivity**  
![Figure 5: LSTM MSE vs. Context Size](./images/LSTMs/benchmark2_lstm.png)

**Figure 6. LSTM (Large) spectrogram prediction, context=50**  
![Figure 6: Large LSTM spectrogram output for context=50](./images/LSTMs/spec50.png)

**Figure 7. LSTM (Large) spectrogram prediction, context=100**  
![Figure 7: Large LSTM spectrogram output for context=100](./images/LSTMs/spec100.png)

**Figure 8. LSTM (Large) spectrogram prediction, context=150**  
![Figure 8: Large LSTM spectrogram output for context=150](./images/LSTMs/spec150.png)

#### Transformer
Transformer benefits most strongly from the longest context (150), but exhibits a surprising dip at context=100, suggesting mid-range context may introduce ambiguity without providing enough structure.

**Figure 9. Transformer context sensitivity**  
![Figure 9: Transformer Context Sensitivity](./images/Transformers/benchmark2_varying_context.png)

**Figure 10. Transformer (Medium) spectrogram prediction, context=50**  
![Figure 10: Transformer spectrogram output for context=50](./images/Transformers/benchmark2_context50_spectrogram.png)

**Figure 11. Transformer (Medium) spectrogram prediction, context=100**  
![Figure 11: Transformer spectrogram output for context=100](./images/Transformers/benchmark2_context100_spectrogram.png)

**Figure 12. Transformer (Medium) spectrogram prediction, context=150**  
![Figure 12: Transformer spectrogram output for context=150](./images/Transformers/benchmark2_context150_spectrogram.png)

---

### 5.3 Benchmark 3: Audio Comparison

> **Audio files:** place these in `./audio/` (or update paths accordingly)

- Context: `audio/transformercontext.m4a`  
- Transformer prediction: `audio/transformersgenerative.m4a`  
- LSTM prediction: `audio/predicted_LSTM.m4a`  
- Ground truth: `audio/transformerground.m4a`

GitHub does not render inline `<audio>` players in Markdown reliably across clients, so we provide direct links:

- **Context:** [transformercontext.m4a](./audio/transformercontext.m4a)  
- **Transformer prediction:** [transformersgenerative.m4a](./audio/transformersgenerative.m4a)  
- **LSTM prediction:** [predicted_LSTM.m4a](./audio/predicted_LSTM.m4a)  
- **Ground truth:** [transformerground.m4a](./audio/transformerground.m4a)

**Figure 13. Transformer spectrogram corresponding to generated audio**  
![Figure 13: Transformer audio spectrogram](./images/Transformers/spectrogram_comparison.png)

**Figure 14. LSTM spectrogram corresponding to generated audio**  
![Figure 14: LSTM audio spectrogram](./images/LSTMs/audiolstmspec.png)

Qualitatively, Transformer audio better preserves pitch contour and loudness variation, while the LSTM output diverges rapidly and often becomes noisy/static. Visually, both struggle with sharp vertical transients, though Transformers preserve harmonic band structure more consistently.

---

## 6. Discussion
Our initial expectation was that Transformers would require substantially more data to demonstrate benefits over recurrence, especially given the compression inherent in mel-spectrogram representations. The empirical results contradict this: Transformers yield lower error and stronger perceptual fidelity, suggesting the dataset is sufficient for attention to learn meaningful global musical dependencies.

A notable result is **non-monotonic Transformer scaling**, where the Medium model outperforms the Large model. This likely reflects sensitivity to optimization and overfitting at high parameter count without sufficient training stabilization (e.g., learning-rate scheduling, warmup, or regularization).

Both architectures exhibit **spectral smoothing**, and neither reliably captures transient vertical structures. This suggests that while attention improves global structure, both models lack inductive biases for local time-frequency detail.

---

## 7. Future Work
- Stabilize large Transformer training with improved schedules (warmup, decay, lower LR)
- Explore hybrid models: attention + convolution for local spectral detail
- Data augmentation: pitch shift, time stretch, frequency masking
- Larger and more diverse datasets for stronger generalization
- Alternative audio reconstruction and perceptual metrics beyond Griffin–Lim

---

## References
[1] Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.*  
[2] Vaswani, A., et al. (2017). *Attention Is All You Need.*  
[3] Eck, D., & Schmidhuber, J. (2002). *A First Look at Music Composition Using LSTM RNNs.*  
[4] Hittawe, M., Sidahmed, H., & Elshiekh, S. (2024). *Comparison of LSTM and Transformer for Time Series Data Forecasting.*  
[5] Weissenborn, D., Täckström, O., & Uszkoreit, J. (2020). *Scaling Autoregressive Video Models.*  
[6] Huang, C. Z. A., et al. (2018). *Music Transformer.*  
[7] Vasquez, S., & Lewis, M. (2019). *MelNet.*  
[8] Lu, W. T., et al. (2021). *SpecTNT: A Time-Frequency Transformer for Music Audio.*  
[9] Lostanlen, V., et al. (2019). *Medley-solos-DB.*  

---
