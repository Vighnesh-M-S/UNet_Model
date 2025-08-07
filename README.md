# 🎨 Color Polygon UNet

This project trains a Conditional UNet model to generate filled polygon shapes conditioned on both **shape masks** and **color hints**.

## 🏗 Project Structure

```
UNet_Model/
├── my_datasets/
│   └── color_polygon_dataset.py
├── models/
│   └── DoubleUnet.py
├── scripts/
│   ├── train.py
│   └── predict.py
├── dataset/
│   └── training/
│       ├── inputs/
│       ├── outputs/
│       └── data.json
├── checkpoints/
│   └── color_polygon_unet.pth
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd UNet_Model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Training

```bash
PYTHONPATH=. python3 scripts/train.py
```

### 4. Run Prediction

```bash
PYTHONPATH=. python3 scripts/predict.py
```

---

## 🛠 Tech Stack

- Python 🐍
- PyTorch 🔥
- WandB 📊
- Matplotlib & PIL for visualization

---
