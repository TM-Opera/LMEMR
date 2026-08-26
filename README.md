# Large Model Enhanced Multimodal Representations (LMEMR)

**LMEMR** is a novel framework for accurately predicting fine-grained urban mobility patterns using only static geospatial data. By leveraging large vision-language models and advanced multimodal learning, LMEMR offers a scalable, privacy-friendly solution for smart city applications.

## 🔍 Overview

Accurately predicting fine-grained urban mobility is essential for optimizing transportation, accessibility, and urban management. However, existing approaches often depend on high-cost, privacy-sensitive dynamic data such as trajectories or signaling records, limiting their scalability and cross-city applicability.

This study proposes the **Large Model Enhanced Multimodal Representations (LMEMR)** framework to learn hourly parcel-level mobility dynamics solely from static geospatial data, including:
- Remote sensing imagery
- 3D building footprints
- Street view imagery 
- Points of interest (POI)

## 🤖 FrameWork

![FrameWork](picture/framework.png)

## ⚙️ Key Features

### 🌐 **Static Data Driven**
Predicts dynamic mobility without requiring real-time tracking or personal data, ensuring user privacy and enabling broader deployment.

### 🧠 **Semantic Enhancement with Large Models**
Employs large vision-language models (VLMs/LLMs) to generate natural-language descriptions for each modality, enriching static inputs with human-understandable semantics.

### 🔗 **Dual-Level Contrastive Learning**
Aligns features both within and across modalities through a dual-level contrastive strategy, reducing semantic gaps and improving multimodal consistency.

### 📊 **Spatial-Temporal Modeling**
- **Spatial Dependencies**: Modeled via a Graph Attention Network (GAT).
- **Temporal Dynamics**: Captured using a Transformer encoder to generate 24-hour mobility sequences.


## 🔧 Applications

LMEMR enables effective urban planning and management in scenarios where dynamic data is unavailable or restricted, offering an interpretable and transferable solution for:
- Traffic forecasting
- Public transit optimization
- Urban accessibility analysis
- Smart infrastructure planning


## 📂 Repository Structure

```
project-root/
├── data/region_ID/
│   ├── ID_Building.png
│   ├── ID_RSI.png
│   ├── ID_POI.csv
│   └── streetviews/
└── model/
    ├── DataSet.py
    ├── GATNetwork.py
    ├── Multimodal_Semantic_Enhancer.py
    ├── TrainWithGraph.py
    └── ...
```

## 🚀 Running

### Requirements

- Python >= 3.7
- PyTorch >= 1.10
- CUDA

### Quick Start

```bash
python main.py --root_dir /path/to/your/data
```

### View All Parameters

```bash
python main.py --help
```

### Common Parameter Examples

```bash
# Custom training parameters
python main.py --root_dir ./data --epochs 500 --lr 0.001

# Use only specific modalities (disable text and SVIS)
python main.py --root_dir ./data --no_text --no_svis

# CPU training
python main.py --root_dir ./data --device cpu

# Custom output paths
python main.py --root_dir ./data \
    --save_path ./output/my_model.pth \
    --metrics_save_path ./output/my_metrics.npz \
    --confusion_matrix_path ./output/my_matrix.npz
```

### Output Files

- `model.pth` - Trained model weights
- `metrics.npz` - Training metrics (accuracy, F1, etc.)
- `matrix.npz` - Confusion matrix


## 🔖 Citation

A paper about the work was published in IEEE Transactions on Intelligent Transportation Systems.

If you like this work and would like to use it in a scientific context, please cite this article.

```bibtex
@ARTICLE{11540094,
  author={Zhao, Tianhong and Li, Jianbin and Cao, Jinzhou and Tu, Wei and Biljecki, Filip and Yi, Shengao and Yuan, Zhilu},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Learning Fine-Grained Urban Mobility Dynamics Through Large Model-Enhanced Multimodal Representations}, 
  year={2026},
  volume={},
  number={},
  pages={1-15},
  keywords={Modeling;Urban areas;Learning (artificial intelligence);Fluid flow;Dynamics;Modules (abstract algebra);Semantics;Buildings;Educational institutions;Transformers;Urban mobility prediction;multimodal learning;vision–language models;contrastive learning;graph attention networks},
  doi={10.1109/TITS.2026.3696956}}

```
