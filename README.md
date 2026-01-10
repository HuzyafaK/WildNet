# WildNet: Wildlife Audio Classification Using CNN-Attention-Transformer Architecture

<p align="center">
  <img alt="TensorFlow" src="https://img.shields.io/badge/Made%20with-TensorFlow-orange?style=for-the-badge&logo=tensorflow" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---
## 🏆 Competition Performance

**BirdCLEF 2025 Challenge Results:**
- **Score:** 0.705 ROC-AUC (macro-averaged, ignoring zero-positive classes)
- **Rank:** Top 15% globally (76% of winning solution: 0.93)
- **Key Innovation:** Architectural generalization > data engineering

### Performance Breakdown

| Configuration | ROC-AUC | Delta | Notes |
|--------------|---------|-------|-------|
| Baseline (train only) | 0.623 | - | No augmentation, no external data |
| + 20 augmented soundscapes | **0.705** | **+13%** | Minimal preprocessing |
| Competition winner | 0.93 | +32% | Heavy ensemble + extensive augmentation |

### Why This Matters

**Data Efficiency Proof:**
- 0.623 → 0.705 with just 20 samples proves architecture learns robust features
- Critical for domains with expensive labeled data (medical, neural, rare species)

**Production Readiness:**
- ⚡ **14 samples/sec** CPU-only inference (8,400 examples in 10 min)
- 🎯 Suitable for edge deployment (no GPU required)
- 📦 Single model, no ensemble complexity

---

## 🏗️ Architecture Innovations

### 1. Hierarchical Channel Attention
Unlike standard approaches that apply attention once, WildNet uses channel attention 
**after each of the 3 CNN stages**:
```python
# Stage 1: 64 filters → Channel Attention → Enhances low-level features
# Stage 2: 128 filters → Channel Attention → Enhances mid-level patterns  
# Stage 3: 256 filters → Channel Attention → Enhances high-level abstractions
```
This captures discriminative patterns at multiple frequency scales.

### 2. Learned Positional Embeddings
Standard transformers use fixed sinusoidal encodings. WildNet learns:
```python
position_embeddings = Embedding(max_len, d_model)
scaled_embeddings = Dense(d_model)(position_embeddings)  # Adaptive scaling
```
This adapts to audio-specific temporal structure.

### 3. Focal Loss for Extreme Imbalance
With classes ranging from 2 to 990 samples (1:500 ratio):
```python
focal_loss = -α * (1 - p_t)^γ * log(p_t)
# α=0.25, γ=2.0 → dynamically focuses on hard examples
```

### 4. Bilinear Spectrogram Interpolation
Preserves acoustic content vs zero-padding:
```python
spec_resized = tf.image.resize(spec, (128, 206), method='bilinear')
# No truncation or padding → retains full frequency-time information
```

---

## 🔬 Technical Specifications

**Model Architecture:**
- **CNN Blocks:** 3 stages (64 → 128 → 256 filters)
- **Attention:** Channel attention after each block
- **Transformer:** 2 encoders, 8 heads, key_dim=64, ff_dim=1024
- **Pooling:** Global average (vs token-wise decoding)
- **Parameters:** ~8M (efficient for edge deployment)

**Training:**
- **Loss:** Focal loss (α=0.25, γ=2.0)
- **Optimizer:** Adam (lr=1e-4)
- **Metrics:** AUC, PR-AUC, Precision@Recall, Recall@Precision
- **Hardware:** Tesla P100 GPU
- **Epochs:** 20 (early stopping on validation AUC)

**Preprocessing:**
- **Audio:** 32kHz resampling, mono channel
- **Spectrograms:** 128 mel bands, bilinear resize to 206 width
- **Normalization:** dB scale + min-max [0,1]
- **Labels:** Multi-label one-hot encoding (primary + secondary species)

---

## 🎯 Applications Beyond Birdsong

This architecture is **domain-agnostic** and applicable to:

✅ **Medical Audio:** Heart/lung sound classification (imbalanced disease classes)  
✅ **Neural Decoding:** EEG/MEG signal classification (subject-specific imbalance)  
✅ **Urban Sound:** Environmental noise monitoring (rare event detection)  
✅ **Marine Bioacoustics:** Whale/dolphin call classification (overlapping vocalizations)  
✅ **Speech Pathology:** Voice disorder detection (limited patient samples)

**Key principle:** When labeled data is expensive and imbalanced, 
architectural design > brute-force data collection.
