# 🛰️ Satellite Imagery-Based Property Valuation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A multimodal machine learning system that predicts property prices by combining tabular property data with satellite imagery features.

## 📊 Results

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| XGBoost Baseline | $129,486 | $74,709 | 0.8664 |
| **EfficientNet + LightGBM + KNN** | **$111,857** | **$67,230** | **0.9003** |

**🎯 13.6% RMSE improvement over baseline**

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Satellite      │     │   Tabular        │     │   Geographic    │
│  Images         │     │   Features       │     │   Coordinates   │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       │                        ▼
┌─────────────────┐              │              ┌─────────────────┐
│  EfficientNet   │              │              │   KNN Features  │
│  B0 Encoder     │              │              │   (15 neighbors)│
│  (256-dim)      │              │              │   (7 features)  │
└────────┬────────┘              │              └────────┬────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Feature Fusion       │
                    │   (294 total features) │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   LightGBM Regressor   │
                    │   (2000 estimators)    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Price Prediction     │
                    └────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM recommended
- GPU optional (for faster image feature extraction)

### Installation

```bash
# Clone the repository
git clone https://github.com/a-man-kr/satellite-property-valuation.git
cd satellite-property-valuation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

Place your data files in the `data/` directory:
```
data/
├── raw/
│   ├── train.csv    # Training data with prices
│   └── test.csv     # Test data for predictions
└── images/          # Satellite images (optional)
    └── {property_id}.png
```

### Run the Pipeline

```bash
# Run the improved model (recommended)
python run_improved_pipeline.py

# Or run the baseline model
python run_pipeline.py
```

### Output

Predictions are saved to `outputs/predictions.csv`:
```csv
id,predicted_price
2591820310,369735.74
7974200820,771959.46
...
```

## 📁 Project Structure

```
satellite_property_valuation/
├── 📂 data/
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Cached processed data
│   └── images/                 # Satellite images
├── 📂 src/
│   ├── data_fetcher.py         # Satellite image download
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   ├── explainability.py       # Grad-CAM visualization
│   └── models/
│       ├── cnn_encoder.py      # ResNet18/EfficientNet encoders
│       ├── multimodal_model.py # Baseline fusion models
│       └── improved_model.py   # EfficientNet + LightGBM + KNN
├── 📂 notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data preprocessing
│   └── 03_model_training.ipynb # Model training experiments
├── 📂 outputs/
│   ├── predictions.csv         # Final predictions
│   └── figures/                # Visualizations
├── 📂 docs/
│   └── architecture.md         # Technical documentation
├── run_improved_pipeline.py    # Main script (recommended)
├── run_pipeline.py             # Baseline pipeline
├── main.py                     # CLI interface
├── requirements.txt
├── PROJECT_REPORT.md           # Detailed project report
└── README.md
```

## 🔧 Configuration

### Model Parameters

```python
# LightGBM Configuration
n_estimators = 2000
learning_rate = 0.03
max_depth = 10
num_leaves = 64
early_stopping_rounds = 50

# EfficientNet-B0
embedding_dim = 256
pretrained = True  # ImageNet weights

# KNN Features
n_neighbors = 15
metric = "haversine"
```

### Environment Variables

```bash
# Optional: For fetching satellite images
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

## 📈 Features

### Tabular Features (30)
- **Property**: bedrooms, bathrooms, sqft_living, sqft_lot, floors, grade, condition
- **Location**: lat, long, zipcode, waterfront, view
- **Derived**: age, years_since_renovation, living_lot_ratio, quality_score

### Image Features (256)
- EfficientNet-B0 embeddings from satellite imagery
- Captures visual characteristics: roof type, lot size, neighborhood density

### KNN Features (7)
- Neighborhood price statistics based on geographic proximity
- mean, median, std, min, max, count, density

## 📊 Data

| Dataset | Samples | Description |
|---------|---------|-------------|
| Training | 16,209 | Properties with price labels |
| Test | 5,404 | Properties for prediction |
| Images | 2,524 | Satellite images (256×256 px) |

## 🧪 Experiments

See `notebooks/` for detailed experiments:
- `01_eda.ipynb` - Data exploration and visualization
- `02_preprocessing.ipynb` - Feature engineering analysis
- `03_model_training.ipynb` - Model comparison and tuning

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{satellite_property_valuation,
  title = {Satellite Imagery-Based Property Valuation},
  year = {2026},
  url = {https://github.com/a-man-kr/satellite-property-valuation}
}
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [EfficientNet](https://arxiv.org/abs/1905.11946) for image feature extraction
- [LightGBM](https://lightgbm.readthedocs.io/) for gradient boosting
- King County housing dataset
