# WildNet: Wildlife Audio Classification Using CNN-Attention-Transformer Architecture

<p align="center">
  <img alt="TensorFlow" src="https://img.shields.io/badge/Made%20with-TensorFlow-orange?style=for-the-badge&logo=tensorflow" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img alt="DataSet" src="https://img.shields.io/badge/DataSet-BirdCLEF%202025-blueviolet?style=for-the-badge" />
</p>

---

## 📚 Overview

**WildNet** is a deep learning model designed for wildlife audio classification, focusing on birds, mammals, and amphibians.  
It combines **CNNs**, **Channel Attention**, and **Transformer Encoders** for effective multi-label classification of species-specific vocalizations.

This work is part of our upcoming research:  
**_"WildNet: A Hybrid CNN-Attention-Transformer Architecture for Wildlife Audio Classification"_**

The full draft of the research paper is available `  draft.pdf  `.

The model is currently optimized for the **BirdCLEF 2025** dataset but is **generalizable** to other spectrogram-based audio classification tasks.

---

## 🛠️ Features

- Hybrid **CNN-Attention-Transformer** architecture
- Input: **Mel-Spectrograms** (128x206) from audio recordings
- Handles **multi-label classification** (multiple species per clip)
- **Focal Loss** to manage extreme class imbalance
- **Stratified Multi-Label K-Fold Cross-Validation**
- **TensorFlow 2.x** implementation
- **Generalizable** to other spectrogram classification domains
- **Future extensions**: event localization, real-time edge deployment

---

## 🐦 Dataset

- **Source**: [BirdCLEF 2025 - LifeCLEF Challenge](https://www.imageclef.org/LifeCLEF2025/BirdCLEF)
- **Size**: 25,000+ audio recordings (.ogg)
- **Labels**: Primary and secondary species
- **Metadata**: Geolocation, timestamps, recording quality

---

## 🔥 Architecture Overview

- **Input**: 128 x 206 x 1 Mel-Spectrograms (from 32kHz resampled audio)
- **CNN Backbone**: 3-stage feature extractor
- **Channel Attention Modules** after each CNN block
- **Transformer Encoder** for modeling long-range temporal dependencies
- **Classifier Head**: Global pooling → Dense layers → Sigmoid activation

---

⚙️ Installation
---------------

`   git clone https://github.com/yourusername/WildNet.git  cd WildNet  pip install -r requirements.txt   `

📆 Requirements
---------------

*   TensorFlow >= 2.10
    
*   numpy

*   ast
    
*   librosa
    
*   joblib
    
*   scikit-multilearn
    
*   pandas

*   tqdm
    

🚀 Quickstart
-------------

### Prepare Dataset

*   Place your .ogg inside data/audio/
    
*   Prepare a labels.csv inside data/ with filename → labels mapping.
    

### Preprocess Audio

Generate Mel-Spectrograms from audio files:

`   python preprocess.py   `

### Train the Model

`   python train.py   `

### Evaluate

`   python evaluate.py   `

### Run Inference
To get predictions on a new audio file, simply run:

`   Inference.py   `


📈 Evaluation Metrics
---------------------

*   Macro-averaged ROC-AUC
    
*   Precision-Recall AUC (PR-AUC)
    
*  Recall@Precision
    
*   Precision@Recall
    

🔮 Future Work
--------------

*   Integrate audio event localization for fine-grained classification
    
*   Apply wavelet or harmonic-percussive transforms for richer feature extraction
    
*   Develop a lightweight, real-time edge model for deployment on mobile and embedded devices
    
*   Generalize WildNet architecture across broader spectrogram classification tasks, including:
    
    *   Urban sound classification
        
    *   Marine mammal detection
        
    *   Environmental noise analysis
        

🧐 Project Structure
--------------------


`WildNet/  │  
├── data/                     # Raw audio, labels, metadata               
├── models/                    # Model components  
|  ├── cnn_backbone.py    
|  ├── attention_module.py    
|  ├── transformer_encoder.py     
|  └── classifier_head.py 
├── train.py                   # Training pipeline 
├── evaluate.py                # Evaluation and metrics 
├── preprocess.py              # Audio preprocessing (audio → spectrograms) 
├── utils.py                   # Utilities (stratified split, plotting, etc.) 
├── model.keras
├── inference.py
├── requirements.txt           # Python dependencies  
└── README.md                  # This documentation   `

📝 License
----------

This project is licensed under the MIT License.Feel free to use, modify, and distribute it for research and educational purposes!

👌 Acknowledgements
-------------------

*   TensorFlow team and contributors
    
*   BirdCLEF & LifeCLEF challenge organizers
    
*   Open-source libraries: librosa, scikit-learn, matplotlib
    
*   Researchers advancing wildlife monitoring and bioacoustics
