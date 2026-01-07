# Satellite Imagery-Based Property Valuation

Predicting property prices using satellite imagery and tabular features with multimodal machine learning.

## Project Structure

```
├── 22322004_final.csv       # Final predictions (id, predicted_price)
├── 22322004_report.pdf      # Project report with EDA and visualizations
├── data_fetcher.py          # Downloads satellite images using Mapbox API
├── preprocessing.ipynb      # Data cleaning, EDA, feature engineering
├── model_training.ipynb     # Model training and evaluation
├── requirements.txt         # Python dependencies
└── README.md
```

## Results

- **Model**: EfficientNet-B0 + LightGBM + KNN
- **R² Score**: 0.9003
- **RMSE**: $111,857

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Run `data_fetcher.py` to download satellite images
2. Run `preprocessing.ipynb` for EDA and feature engineering
3. Run `model_training.ipynb` for model training and predictions

## Enrollment Number
22322004
