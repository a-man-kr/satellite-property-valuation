# Multimodal Property Valuation Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTIMODAL PROPERTY VALUATION                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌─────────────────────┐
│   TABULAR DATA      │     │   SATELLITE IMAGE   │
│   (CSV Features)    │     │   (lat/long → API)  │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Feature Engineering│     │  CNN Encoder        │
│  - Age calculation  │     │  (ResNet18)         │
│  - Size ratios      │     │  - Pretrained       │
│  - Location bins    │     │  - 256-dim output   │
│  - Quality scores   │     │                     │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Tabular Encoder    │     │  Image Embedding    │
│  (MLP: 128→64)      │     │  (256 dimensions)   │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          └───────────┬───────────────┘
                      │
                      ▼
          ┌─────────────────────┐
          │   FUSION LAYER      │
          │   (Concatenation)   │
          │   128 dimensions    │
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │   PREDICTION HEAD   │
          │   (MLP → Price)     │
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │   PREDICTED PRICE   │
          │   ($ value)         │
          └─────────────────────┘
```

## Fusion Strategies

### 1. Early Fusion
```
Tabular Features ─┐
                  ├─→ Concatenate ─→ Shared MLP ─→ Price
Image Features ───┘
```
- Combines raw features before any processing
- Simple but may miss modality-specific patterns

### 2. Late Fusion (Recommended)
```
Tabular Features ─→ Tabular MLP ─┐
                                 ├─→ Concatenate ─→ Fusion MLP ─→ Price
Image Features ───→ Image MLP ───┘
```
- Processes each modality separately first
- Combines learned representations
- Best performance in experiments

### 3. Attention Fusion
```
Tabular Features ─→ Tabular MLP ─┐
                                 ├─→ Multi-Head Attention ─→ Fusion MLP ─→ Price
Image Features ───→ Image MLP ───┘
```
- Uses attention to weight modality importance
- More complex but can learn dynamic weighting

## Data Flow

```
1. DATA ACQUISITION
   ├── Load CSV data (train.csv, test.csv)
   └── Fetch satellite images via API (Mapbox/Google/OSM)

2. PREPROCESSING
   ├── Clean missing values
   ├── Engineer tabular features (30+ features)
   ├── Extract image embeddings (256-dim per image)
   └── Scale/normalize features

3. MODEL TRAINING
   ├── Train XGBoost baseline (tabular only)
   ├── Train multimodal neural network
   └── Compare performance (RMSE, R², MAE)

4. INFERENCE
   ├── Process test data
   ├── Generate predictions
   └── Output predictions.csv

5. EXPLAINABILITY
   ├── Feature importance analysis
   ├── Grad-CAM visualization
   └── Visual feature analysis
```

## Key Components

### Tabular Features (30+)
- **Original**: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, sqft_living15, sqft_lot15
- **Engineered**: age, years_since_renovation, living_lot_ratio, above_living_ratio, basement_ratio, living_vs_neighbors, lot_vs_neighbors, total_rooms, sqft_per_room, quality_score, has_basement, was_renovated, year_sold, month_sold

### Image Features
- **Source**: Satellite imagery from property coordinates
- **Encoder**: ResNet18 pretrained on ImageNet
- **Output**: 256-dimensional embedding
- **Captures**: Green cover, water proximity, road density, building patterns

### Loss Function
- Mean Squared Error (MSE) for regression
- AdamW optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau

## Performance Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error - primary metric |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |

## Explainability

### Grad-CAM
- Visualizes which image regions influence predictions
- Highlights features like water, vegetation, roads
- Helps understand model decision-making

### Feature Importance
- XGBoost built-in importance scores
- Permutation importance for neural networks
- Comparison between tabular and multimodal models
