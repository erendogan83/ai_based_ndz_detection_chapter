# 🌞 AI-Based Islanding Detection in Photovoltaic Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE 1547](https://img.shields.io/badge/Standard-IEEE%201547-orange.svg)](https://standards.ieee.org/ieee/1547/6733/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![DOI](https://img.shields.io/badge/DOI-Pending-yellow.svg)]()

> **Advanced machine learning solutions for detecting islanding in grid-connected solar power plants**

<div align="center">
  <img src="https://img.shields.io/badge/Accuracy-93.18%25-brightgreen?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/NDZ%20Detection-85%25-blue?style=for-the-badge" alt="NDZ Detection">
  <img src="https://img.shields.io/badge/Test%20Cases-361-orange?style=for-the-badge" alt="Test Cases">
</div>

---

## 👨‍🔬 Author Information

**Eren DOĞAN**  
*Electrical-Electronics Engineer*  
📧 Email: erendogan@gibtu.edu.tr  
🎓 Institution: Gaziantep İslam Bilim ve Teknoloji Üniversitesi  
🏢 Department: Electrical-Electronics Engineering

**Supervisor:**  
**Prof. Dr. Mehmet Ali ÖZÇELİK**  
📧 Email: maozcelik@gibtu.edu.tr  
🎓 Institution: Gaziantep İslam Bilim ve Teknoloji Üniversitesi

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance Metrics](#-performance-metrics)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)
- [Figures](#-figures)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

This repository contains the **complete implementation** of AI-based islanding detection methods for photovoltaic (PV) systems connected to electrical grids. The research addresses the critical **Non-Detection Zone (NDZ)** problem where traditional passive methods fail to detect islanding conditions.

### 🔬 Research Highlights

- ✅ **IEEE 1547 Compliant** - All methods follow IEEE standards
- ✅ **4,900 Synthetic Samples** - Comprehensive training dataset
- ✅ **361 NDZ Test Cases** - Challenging boundary conditions
- ✅ **Load-Independent** - No load type information required
- ✅ **Three ML Models** - Random Forest, SVM, ANN comparison
- ✅ **Open Source** - Full code and data availability

---

## ⭐ Key Features

### 🚀 **Superior Performance**
- **Random Forest:** 93.18% ± 0.95% cross-validation accuracy
- **SVM:** 91.90% ± 0.80% cross-validation accuracy  
- **ANN:** 91.76% ± 0.69% cross-validation accuracy

### 🎯 **NDZ Detection Rates** (361 challenging test cases)
| Method | Detection Rate | Cases Detected |
|--------|---------------|----------------|
| 🌲 **Random Forest** | **85.0%** | **307/361** |
| 🔷 **SVM** | **82.0%** | **296/361** |
| 🧠 **ANN** | **79.0%** | **285/361** |
| 📊 ROCOF (Traditional) | 60.4% | 218/361 |
| 📈 Vector Surge | 27.7% | 100/361 |
| ⚠️ OUF/OUV | 0.0% | 0/361 |

### 💡 **Technical Innovation**
- 9 measurable features (no exotic sensors required)
- Dynamic parameters > Static parameters (ROCOF: 37.76% importance)
- Fast inference: <1ms per sample
- Compatible with standard inverter microcontrollers

---

## 📊 Performance Metrics

### Cross-Validation Results (5-Fold)

<div align="center">

| Model | Accuracy | Precision | Recall | F1-Score | Std Dev |
|-------|----------|-----------|--------|----------|---------|
| 🌲 Random Forest | **93.18%** | 95.81% | 90.41% | 93.03% | ±0.95% |
| 🔷 SVM | **91.90%** | 96.66% | 86.87% | 91.51% | ±0.80% |
| 🧠 ANN | **91.76%** | 95.55% | 87.68% | 91.44% | ±0.69% |

</div>

### NDZ Performance Comparison

```
████████████████████████████████████████████ Random Forest: 85.0% (307/361)
█████████████████████████████████████████    SVM: 82.0% (296/361)
████████████████████████████████████         ANN: 79.0% (285/361)
████████████████████████                     ROCOF: 60.4% (218/361)
████████                                     Vector Surge: 27.7% (100/361)
                                             OUF/OUV: 0.0% (0/361)
```

---

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
NumPy >= 1.19.0
Pandas >= 1.1.0
Scikit-learn >= 0.24.0
Matplotlib >= 3.3.0
Seaborn >= 0.11.0
```

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/erendogan83/ai_based_ndz_detection_chapter.git
cd ai_based_ndz_detection_chapter

# Install required packages
pip install -r requirements.txt
```

### Create `requirements.txt`

```txt
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

---

## 🚀 Quick Start

### Run Complete Analysis

```python
python islanding_detection_english.py
```

This will:
1. ✅ Generate IEEE 1547-compliant synthetic dataset (4,900 samples)
2. ✅ Create 361 NDZ test cases
3. ✅ Train Random Forest, SVM, and ANN models
4. ✅ Evaluate traditional passive methods (OUF/OUV, ROCOF, Vector Surge)
5. ✅ Generate all figures (5 professional plots)
6. ✅ Save results to CSV files

### Expected Output

```
======================================================================
ISLANDING DETECTION ANALYSIS
======================================================================

Generating figures...
✓ Figure 1: NDZ Region
✓ Figure 2: System Diagram
✓ Figure 3: Confusion Matrix
✓ Figure 4: Feature Importance
✓ Figure 5: Performance Comparison

✓ Analysis completed!
======================================================================
```

---

## 📦 Dataset

### Synthetic Dataset Generation

The dataset follows **IEEE 1547-2018** standards with balanced load type distribution:

| Load Type | Samples | Quality Factor (Q) | Characteristics |
|-----------|---------|-------------------|-----------------|
| **R** (Resistive) | 1,225 | 0.5 - 1.0 | Low Q, easy detection |
| **RL** (Inductive) | 1,225 | 1.0 - 2.5 | Medium Q |
| **RC** (Capacitive) | 1,225 | 1.0 - 2.5 | Medium Q |
| **RLC** (Resonant) | 1,225 | 2.5 - 5.0 | **High Q, challenging NDZ** |

### Feature Vector (9 Parameters)

```python
Features = [
    'Frequency',        # Grid frequency (Hz)
    'Voltage',          # Voltage at PCC (p.u.)
    'Power_Factor',     # Cosine of phase angle
    'THD',              # Total Harmonic Distortion (%)
    'Delta_Freq',       # Frequency deviation (Hz)
    'Delta_Volt',       # Voltage deviation (p.u.)
    'ROCOF',            # Rate of Change of Frequency (Hz/s) ⭐
    'Phase_Jump',       # Phase angle change (degrees)
    'Quality_Factor'    # Load resonance characteristic (Q)
]
```

---

## 🤖 Models

### 1. 🌲 Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
```

### 2. 🔷 Support Vector Machine (SVM)

```python
SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    random_state=42
)
```

### 3. 🧠 Artificial Neural Network (ANN)

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    random_state=42
)
```

---

## 📈 Results

### Feature Importance (Random Forest)

| Feature | Importance | Category |
|---------|-----------|----------|
| **ROCOF** | **37.76%** | 🔴 Dynamic |
| **Phase Jump** | **25.40%** | 🔴 Dynamic |
| **THD** | **9.21%** | 🟡 Semi-Dynamic |
| **ΔV** | **8.15%** | 🟢 Static |
| **ΔF** | **7.92%** | 🟢 Static |

---

## 🖼️ Figures

All figures are generated automatically and saved to the `outputs/` directory.

---

## 📚 Citation

```bibtex
@article{dogan2024islanding,
  title={AI-Based Solutions for Islanding Detection in Solar Power Plants},
  author={Doğan, Eren and Özçelik, Mehmet Ali},
  year={2024},
  institution={Gaziantep İslam Bilim ve Teknoloji Üniversitesi}
}
```

---

## 📄 License

MIT License - Copyright (c) 2024 Eren Doğan

---

## 📞 Contact

**Eren DOĞAN**

- 📧 Email: erendogan@gibtu.edu.tr
- 🐙 GitHub: [@erendogan83](https://github.com/erendogan83)
- 🎓 Institution: Gaziantep İslam Bilim ve Teknoloji Üniversitesi

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for the renewable energy community**

</div>
