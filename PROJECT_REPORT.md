# Satellite Imagery-Based Property Valuation - Project Report

## Executive Summary

This project develops a multimodal machine learning system for predicting property prices using both tabular features and satellite imagery. The final model achieves **R² = 0.9003** on the validation set, representing a **13.6% improvement** in RMSE over the baseline.

## Problem Statement

Predict property market values using:
- Tabular data: Property characteristics (bedrooms, bathrooms, sqft, location, etc.)
- Satellite imagery: Aerial views of property locations

## Methodology

### 1. Data Preprocessing
- Cleaned missing values using median imputation
- Engineered 30 features including:
  - Property age and renovation status
  - Size ratios (living/lot, above/basement)
  - Neighborhood comparisons (sqft vs neighbors)
  - Quality scores (grade × condition)
  - Location binning

### 2. Image Feature Extraction
- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Output**: 256-dimensional embeddings per image
- **Coverage**: 2,524 satellite images (256×256 pixels)

### 3. KNN Neighborhood Features
- 15 nearest neighbors using haversine distance on lat/long
- Features extracted:
  - Mean, median, std, min, max neighbor prices
  - Mean distance to neighbors
  - Price density (sum prices / sum distances)

### 4. Final Model
- **Algorithm**: LightGBM Gradient Boosting
- **Input**: 294 combined features (30 tabular + 256 image + 1 has_image + 7 KNN)
- **Training**: Early stopping with 50-round patience

## Results

| Metric | Baseline (XGBoost) | Improved Model | Improvement |
|--------|-------------------|----------------|-------------|
| RMSE   | $129,486          | $111,857       | -13.6%      |
| MAE    | $74,709           | $67,230        | -10.0%      |
| R²     | 0.8664            | 0.9003         | +0.034      |

## Key Innovations

1. **EfficientNet-B0** instead of ResNet18 for more efficient feature extraction
2. **KNN neighborhood features** to capture local market context
3. **Two-stage architecture**: CNN for images → gradient boosting for final prediction
4. **LightGBM** with tuned hyperparameters for optimal performance

## Model Configuration

```python
# LightGBM
n_estimators=2000, learning_rate=0.03, max_depth=10
num_leaves=64, min_child_samples=20
reg_lambda=1.0, reg_alpha=0.1
colsample_bytree=0.8, subsample=0.8

# EfficientNet-B0
embedding_dim=256, pretrained=True

# KNN
n_neighbors=15, metric='haversine'
```

## Files

| File | Description |
|------|-------------|
| `outputs/predictions.csv` | Final predictions (5,404 samples) |
| `run_improved_pipeline.py` | Main training script |
| `src/models/improved_model.py` | Model implementation |
| `src/preprocessing.py` | Data preprocessing |

## Reproducibility

```bash
cd satellite_property_valuation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_improved_pipeline.py
```

## Conclusion

The two-stage multimodal approach successfully combines tabular property data with satellite imagery features, achieving R² = 0.9003. The KNN neighborhood features proved particularly valuable for capturing local market dynamics.
