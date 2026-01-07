"""
Generate PDF Report for Satellite Property Valuation Project
Enrollment: 22322004
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

def create_report():
    """Generate comprehensive PDF report with EDA and results."""
    
    # Load data
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    predictions = pd.read_csv('22322004_final.csv')
    
    with PdfPages('22322004_report.pdf') as pdf:
        
        # ============ PAGE 1: Title Page ============
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Satellite Imagery-Based Property Valuation', 
                 ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.55, 'Multimodal Machine Learning for Price Prediction', 
                 ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.4, 'Enrollment Number: 22322004', 
                 ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.3, 'Final Model: EfficientNet-B0 + LightGBM + KNN', 
                 ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.2, 'R² Score: 0.9003 | RMSE: $111,857', 
                 ha='center', va='center', fontsize=14, fontweight='bold', color='green')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 2: Executive Summary ============
        fig = plt.figure(figsize=(11, 8.5))
        summary_text = """
EXECUTIVE SUMMARY

OBJECTIVE
Predict property prices using tabular features and satellite imagery.

APPROACH
• Two-stage multimodal architecture combining:
  - EfficientNet-B0 for satellite image feature extraction (256-dim embeddings)
  - KNN-based neighborhood features using geographic coordinates
  - LightGBM gradient boosting for final prediction

DATA
• Training samples: 16,209 properties with price labels
• Test samples: 5,404 properties for prediction
• Satellite images: 2,524 images (256×256 pixels)
• Features: 294 total (30 tabular + 256 image + 7 KNN + 1 has_image)

RESULTS
┌─────────────────────────────┬───────────┬──────────┬──────────┐
│ Model                       │ RMSE      │ MAE      │ R²       │
├─────────────────────────────┼───────────┼──────────┼──────────┤
│ XGBoost Baseline            │ $129,486  │ $74,709  │ 0.8664   │
│ EfficientNet+LightGBM+KNN   │ $111,857  │ $67,230  │ 0.9003   │
└─────────────────────────────┴───────────┴──────────┴──────────┘

IMPROVEMENT: 13.6% RMSE reduction over baseline
        """
        fig.text(0.1, 0.9, summary_text, ha='left', va='top', fontsize=10, 
                 family='monospace', transform=fig.transFigure)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 3: Price Distribution (EDA) ============
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Exploratory Data Analysis: Price Distribution', fontsize=14, fontweight='bold')
        
        # Price histogram
        axes[0, 0].hist(train_df['price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(train_df['price'].median(), color='red', linestyle='--', 
                          label=f'Median: ${train_df["price"].median():,.0f}')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].legend()
        
        # Log price
        axes[0, 1].hist(np.log1p(train_df['price']), bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Log(Price)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Log-Transformed Price')
        
        # Box plot
        axes[1, 0].boxplot(train_df['price'])
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].set_title('Price Box Plot')
        
        # Price statistics
        stats_text = f"""
Price Statistics:
─────────────────
Mean:   ${train_df['price'].mean():,.0f}
Median: ${train_df['price'].median():,.0f}
Std:    ${train_df['price'].std():,.0f}
Min:    ${train_df['price'].min():,.0f}
Max:    ${train_df['price'].max():,.0f}

Training: {len(train_df):,} samples
Test:     {len(test_df):,} samples
        """
        axes[1, 1].text(0.1, 0.9, stats_text, ha='left', va='top', fontsize=11, 
                       family='monospace', transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 4: Feature Correlations (EDA) ============
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.suptitle('Exploratory Data Analysis: Feature Correlations', fontsize=14, fontweight='bold')
        
        # Correlation with price
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        correlations = train_df[numeric_cols].corr()['price'].sort_values(ascending=True)
        correlations = correlations.drop('price')
        
        colors = ['green' if x > 0 else 'red' for x in correlations.values]
        axes[0].barh(correlations.index, correlations.values, color=colors, alpha=0.7)
        axes[0].set_xlabel('Correlation with Price')
        axes[0].set_title('Feature Correlations')
        axes[0].axvline(0, color='black', linewidth=0.5)
        
        # Top features heatmap
        top_features = ['price', 'sqft_living', 'grade', 'sqft_above', 'bathrooms', 'view', 'bedrooms']
        corr_matrix = train_df[top_features].corr()
        im = axes[1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_xticks(range(len(top_features)))
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_xticklabels(top_features, rotation=45, ha='right')
        axes[1].set_yticklabels(top_features)
        axes[1].set_title('Correlation Heatmap (Top Features)')
        
        # Add correlation values
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                axes[1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 5: Key Features Analysis (EDA) ============
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        fig.suptitle('Exploratory Data Analysis: Key Features', fontsize=14, fontweight='bold')
        
        # Sqft Living vs Price
        axes[0, 0].scatter(train_df['sqft_living'], train_df['price'], alpha=0.3, s=3)
        axes[0, 0].set_xlabel('Sqft Living')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Living Space vs Price')
        
        # Grade vs Price
        grade_prices = train_df.groupby('grade')['price'].mean()
        axes[0, 1].bar(grade_prices.index, grade_prices.values, color='steelblue')
        axes[0, 1].set_xlabel('Grade')
        axes[0, 1].set_ylabel('Avg Price ($)')
        axes[0, 1].set_title('Grade vs Price')
        
        # Waterfront vs Price
        wf_prices = train_df.groupby('waterfront')['price'].mean()
        axes[0, 2].bar(['No', 'Yes'], wf_prices.values, color=['gray', 'blue'])
        axes[0, 2].set_xlabel('Waterfront')
        axes[0, 2].set_ylabel('Avg Price ($)')
        axes[0, 2].set_title(f'Waterfront Premium: +${wf_prices[1]-wf_prices[0]:,.0f}')
        
        # View vs Price
        view_prices = train_df.groupby('view')['price'].mean()
        axes[1, 0].bar(view_prices.index, view_prices.values, color='green')
        axes[1, 0].set_xlabel('View Rating')
        axes[1, 0].set_ylabel('Avg Price ($)')
        axes[1, 0].set_title('View vs Price')
        
        # Bedrooms vs Price
        bed_prices = train_df[train_df['bedrooms'] <= 8].groupby('bedrooms')['price'].mean()
        axes[1, 1].bar(bed_prices.index, bed_prices.values, color='orange')
        axes[1, 1].set_xlabel('Bedrooms')
        axes[1, 1].set_ylabel('Avg Price ($)')
        axes[1, 1].set_title('Bedrooms vs Price')
        
        # Bathrooms vs Price
        bath_prices = train_df[train_df['bathrooms'] <= 6].groupby('bathrooms')['price'].mean()
        axes[1, 2].bar(bath_prices.index.astype(str), bath_prices.values, color='purple')
        axes[1, 2].set_xlabel('Bathrooms')
        axes[1, 2].set_ylabel('Avg Price ($)')
        axes[1, 2].set_title('Bathrooms vs Price')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 6: Geographic Analysis (EDA) ============
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.suptitle('Exploratory Data Analysis: Geographic Patterns', fontsize=14, fontweight='bold')
        
        # Price by location
        scatter = axes[0].scatter(train_df['long'], train_df['lat'], 
                                  c=train_df['price'], cmap='RdYlGn_r', 
                                  alpha=0.5, s=3)
        plt.colorbar(scatter, ax=axes[0], label='Price ($)')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title('Property Prices by Location')
        
        # Waterfront locations
        non_wf = train_df[train_df['waterfront'] == 0]
        wf = train_df[train_df['waterfront'] == 1]
        axes[1].scatter(non_wf['long'], non_wf['lat'], alpha=0.3, s=2, c='gray', label='Non-Waterfront')
        axes[1].scatter(wf['long'], wf['lat'], alpha=0.8, s=15, c='blue', marker='*', label='Waterfront')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title(f'Waterfront Properties ({len(wf)} total)')
        axes[1].legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 7: Model Architecture ============
        fig = plt.figure(figsize=(11, 8.5))
        arch_text = """
MODEL ARCHITECTURE

┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE MULTIMODAL ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: FEATURE EXTRACTION
────────────────────────────

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Satellite      │     │   Tabular        │     │   Geographic    │
│  Images         │     │   Features       │     │   Coordinates   │
│  (256×256 px)   │     │   (30 features)  │     │   (lat, long)   │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       │                        ▼
┌─────────────────┐              │              ┌─────────────────┐
│  EfficientNet   │              │              │   KNN Features  │
│  B0 (ImageNet)  │              │              │  (15 neighbors) │
│  → 256-dim      │              │              │  → 7 features   │
└────────┬────────┘              │              └────────┬────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Feature Concatenation │
                    │   (294 total features)  │
                    └────────────┬───────────┘

STAGE 2: PREDICTION
───────────────────
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   LightGBM Regressor   │
                    │   • 2000 estimators    │
                    │   • learning_rate=0.03 │
                    │   • max_depth=10       │
                    │   • early_stopping=50  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Price Prediction ($) │
                    └────────────────────────┘
        """
        fig.text(0.05, 0.95, arch_text, ha='left', va='top', fontsize=9, 
                 family='monospace', transform=fig.transFigure)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 8: Feature Engineering ============
        fig = plt.figure(figsize=(11, 8.5))
        features_text = """
FEATURE ENGINEERING

TABULAR FEATURES (30)
─────────────────────
Original Features:
  • bedrooms, bathrooms, sqft_living, sqft_lot, floors
  • waterfront, view, condition, grade
  • sqft_above, sqft_basement, yr_built, yr_renovated
  • zipcode, lat, long, sqft_living15, sqft_lot15

Engineered Features:
  • age = 2015 - yr_built
  • years_since_renovation = 2015 - yr_renovated (if renovated)
  • living_lot_ratio = sqft_living / sqft_lot
  • above_living_ratio = sqft_above / sqft_living
  • basement_ratio = sqft_basement / sqft_living
  • living_vs_neighbors = sqft_living / sqft_living15
  • lot_vs_neighbors = sqft_lot / sqft_lot15
  • total_rooms = bedrooms + bathrooms
  • sqft_per_room = sqft_living / total_rooms
  • quality_score = grade × condition
  • has_basement = 1 if sqft_basement > 0
  • was_renovated = 1 if yr_renovated > 0

IMAGE FEATURES (256)
────────────────────
  • EfficientNet-B0 pretrained on ImageNet
  • Input: 256×256 RGB satellite images
  • Output: 256-dimensional embedding vector
  • Captures: roof type, lot size, vegetation, neighborhood density

KNN NEIGHBORHOOD FEATURES (7)
─────────────────────────────
  • Based on 15 nearest neighbors (haversine distance)
  • knn_price_mean: Average price of neighbors
  • knn_price_median: Median price of neighbors
  • knn_price_std: Price standard deviation
  • knn_price_min: Minimum neighbor price
  • knn_price_max: Maximum neighbor price
  • knn_count: Number of neighbors found
  • knn_density: Neighbor density metric

TOTAL: 294 FEATURES
        """
        fig.text(0.05, 0.95, features_text, ha='left', va='top', fontsize=10, 
                 family='monospace', transform=fig.transFigure)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 9: Results & Predictions ============
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Model Results & Predictions', fontsize=14, fontweight='bold')
        
        # Prediction distribution
        axes[0, 0].hist(predictions['predicted_price'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(predictions['predicted_price'].median(), color='red', linestyle='--',
                          label=f'Median: ${predictions["predicted_price"].median():,.0f}')
        axes[0, 0].set_xlabel('Predicted Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Test Predictions Distribution')
        axes[0, 0].legend()
        
        # Train vs Prediction comparison
        axes[0, 1].hist(train_df['price'], bins=50, alpha=0.5, label='Training (Actual)', color='blue')
        axes[0, 1].hist(predictions['predicted_price'], bins=50, alpha=0.5, label='Test (Predicted)', color='orange')
        axes[0, 1].set_xlabel('Price ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Training vs Test Distribution')
        axes[0, 1].legend()
        
        # Model comparison
        models = ['XGBoost\nBaseline', 'EfficientNet+\nLightGBM+KNN']
        rmse_vals = [129486, 111857]
        colors = ['gray', 'green']
        bars = axes[1, 0].bar(models, rmse_vals, color=colors)
        axes[1, 0].set_ylabel('RMSE ($)')
        axes[1, 0].set_title('Model Comparison - RMSE')
        for bar, val in zip(bars, rmse_vals):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000, 
                           f'${val:,}', ha='center', va='bottom', fontsize=10)
        
        # R² comparison
        r2_vals = [0.8664, 0.9003]
        bars = axes[1, 1].bar(models, r2_vals, color=colors)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Model Comparison - R²')
        axes[1, 1].set_ylim(0.8, 0.95)
        for bar, val in zip(bars, r2_vals):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                           f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # ============ PAGE 10: Conclusion ============
        fig = plt.figure(figsize=(11, 8.5))
        conclusion_text = """
CONCLUSION

KEY ACHIEVEMENTS
────────────────
✓ Achieved R² = 0.9003 (target was > 0.90)
✓ RMSE reduced from $129,486 to $111,857 (13.6% improvement)
✓ Successfully integrated satellite imagery with tabular data
✓ KNN neighborhood features significantly improved predictions

METHODOLOGY HIGHLIGHTS
──────────────────────
1. EfficientNet-B0 effectively extracts visual features from satellite images
2. KNN-based neighborhood features capture local market dynamics
3. LightGBM with early stopping prevents overfitting
4. Two-stage architecture allows flexible feature engineering

TECHNICAL SPECIFICATIONS
────────────────────────
• Training samples: 16,209
• Test samples: 5,404
• Total features: 294
• Image encoder: EfficientNet-B0 (pretrained)
• Final model: LightGBM (2000 estimators)
• Validation strategy: 80/20 train/val split

PREDICTION STATISTICS
─────────────────────
• Predictions generated: 5,404
• Mean predicted price: ${:,.0f}
• Median predicted price: ${:,.0f}
• Min predicted price: ${:,.0f}
• Max predicted price: ${:,.0f}

FILES SUBMITTED
───────────────
• 22322004_final.csv - Predictions (id, predicted_price)
• 22322004_report.pdf - This report
• data_fetcher.py - Satellite image fetching
• preprocessing.ipynb - Data preprocessing
• model_training.ipynb - Model training
• README.md - Project documentation
        """.format(
            predictions['predicted_price'].mean(),
            predictions['predicted_price'].median(),
            predictions['predicted_price'].min(),
            predictions['predicted_price'].max()
        )
        fig.text(0.05, 0.95, conclusion_text, ha='left', va='top', fontsize=10, 
                 family='monospace', transform=fig.transFigure)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    print("✅ Report generated: 22322004_report.pdf")

if __name__ == "__main__":
    create_report()
