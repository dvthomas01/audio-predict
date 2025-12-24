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

---

## 2. Related Work
Early work established LSTMs as a core method for algorithmic composition [3]. In other continuous forecasting domains, results are mixed [4,5]. In music/audio, Transformer success often depends on symbolic representations [6], while audio-specific architectures such as SpecTNT introduce specialized inductive biases [8]. A gap remains for **standard Transformers** in **continuous generative forecasting**.

---

## 3. Goals
We focus on **generative autoregressive forecasting** and perform an **architectural ablation** comparing standard Transformers against LSTMs. We train **Small / Medium / Large** versions of both architectures to isolate architectural effects from model capacity.

---

## 4. Methods

### 4.1 Data
We use **Medley-Solos-DB** [9], containing single-instrument solos across multiple instrument classes.

**Figure 1. Introduction of methods in research project**  
![Figure 1](blog/images/intrographic.png)

**Figure 2. Example of Mel Spectrogram Data**  
![Figure 2](blog/images/examplespectrogram.png)

---

### 4.2 Audio Reconstruction
Predicted Mel Spectrograms are converted back to waveform audio by reversing the preprocessing pipeline: exponentiating log magnitudes, applying the inverse Mel transform, and estimating phase using the **Griffin–Lim** algorithm.

---

### 4.3 Architectures

#### Autoregressive Transformer
Causal Transformer with sinusoidal positional encoding.

#### LSTM Baseline
Stacked LSTM with linear projection back to spectral space.

---

### 4.4 Training
- Optimizer: AdamW  
- Learning Rate: 1e-4  
- Batch Size: 64  
- Epochs: 35  

---

## 5. Results

### 5.1 Future Horizon Forecasting

**Figure 3. LSTM performance vs. K-frame horizon**  
![Figure 3](blog/images/LSTMs/benchmark1_lstm.png)

**Figure 4. Transformer MSE and Frame Accuracy vs. K**  
![Figure 4](blog/images/Transformers/benchmark1_varying_k%20(1).png)

---

### 5.2 Context Sensitivity

**Figure 5. LSTM MSE vs. Context Size**  
![Figure 5](blog/images/LSTMs/benchmark2_lstm.png)

**Figure 6. LSTM Large — context = 50**  
![Figure 6](blog/images/LSTMs/spec50.png)

**Figure 7. LSTM Large — context = 100**  
![Figure 7](blog/images/LSTMs/spec100.png)

**Figure 8. LSTM Large — context = 150**  
![Figure 8](blog/images/LSTMs/spec150.png)

**Figure 9. Transformer Context Sensitivity**  
![Figure 9](blog/images/Transformers/benchmark2_varying_context.png)

**Figure 10. Transformer Medium — context = 50**  
![Figure 10](blog/images/Transformers/benchmark2_context50_spectrogram.png)

**Figure 11. Transformer Medium — context = 100**  
![Figure 11](blog/images/Transformers/benchmark2_context100_spectrogram.png)

**Figure 12. Transformer Medium — context = 150**  
![Figure 12](blog/images/Transformers/benchmark2_context150_spectrogram.png)

---

### 5.3 Audio Comparison

Audio files are linked directly (GitHub Markdown does not reliably embed players):

- Context: `audio/transformercontext.m4a`
- Transformer prediction: `audio/transformersgenerative.m4a`
- LSTM prediction: `audio/predicted_LSTM.m4a`
- Ground truth: `audio/transformerground.m4a`

**Figure 13. Transformer spectrogram corresponding to generated audio**  
![Figure 13](blog/images/Transformers/spectrogram_comparison.png)

**Figure 14. LSTM spectrogram corresponding to generated audio**  
![Figure 14](blog/images/LSTMs/audiolstmspec.png)

---

## 6. Discussion
The Transformer consistently produced lower error and higher perceptual quality than the LSTM, despite expectations that attention would require larger datasets. Model scaling was non-monotonic for Transformers, with the Medium model outperforming the Large, likely due to optimization instability.

---

## 7. Future Work
Future work includes improved Transformer optimization, hybrid attention–convolution architectures, dataset expansion, and richer perceptual evaluation metrics.

---

## References
[1] Hochreiter & Schmidhuber (1997)  
[2] Vaswani et al. (2017)  
[3] Eck & Schmidhuber (2002)  
[4] Hittawe et al. (2024)  
[5] Weissenborn et al. (2020)  
[6] Huang et al. (2018)  
[7] Vasquez & Lewis (2019)  
[8] Lu et al. (2021)  
[9] Lostanlen et al. (2019)

---
