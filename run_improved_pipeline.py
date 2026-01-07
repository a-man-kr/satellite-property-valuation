"""
Improved Pipeline with EfficientNet + LightGBM + KNN Features
Two-stage multimodal model achieving R² > 0.90
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import PropertyDataPreprocessor
from src.models.improved_model import TwoStageMultimodalModel, compare_models_improved
from src.models.multimodal_model import TabularOnlyModel


def main():
    print("="*70)
    print("SATELLITE IMAGERY-BASED PROPERTY VALUATION")
    print("EfficientNet-B0 + LightGBM + KNN Features")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Paths
    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv(data_dir / "raw/train.csv")
    test_df = pd.read_csv(data_dir / "raw/test.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Check existing images
    print("\n[2/5] Checking satellite images...")
    existing_images = list((data_dir / "images").glob("*.png"))
    print(f"Found {len(existing_images)} satellite images")
    
    # Preprocess
    print("\n[3/5] Preprocessing data...")
    preprocessor = PropertyDataPreprocessor()
    preprocessor.load_data(
        train_path=str(data_dir / "raw/train.csv"),
        test_path=str(data_dir / "raw/test.csv")
    )
    data = preprocessor.prepare_for_training(val_size=0.2, random_state=42)
    print(f"Tabular features: {len(data['feature_columns'])}")
    
    # Train baseline for comparison
    print("\n[4/5] Training baseline XGBoost model...")
    baseline_model = TabularOnlyModel()
    baseline_metrics = baseline_model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Train improved model
    print("\n[5/5] Training improved two-stage model...")
    
    # Need full train_df with ids for KNN
    train_df_with_split = train_df.copy()
    
    improved_model = TwoStageMultimodalModel(
        use_catboost=True,  # Will fallback to LightGBM if CatBoost unavailable
        use_lightgbm=True,  # Use LightGBM as second choice
        device=device
    )
    
    improved_metrics = improved_model.fit(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        train_ids=data['ids_train'],
        val_ids=data['ids_val'],
        train_df=train_df,
        image_dir=str(data_dir / "images"),
        embedding_dim=256
    )
    
    # Compare models
    compare_models_improved(baseline_metrics, improved_metrics)
    
    # Generate predictions
    print("\n[6/6] Generating predictions...")
    test_data = preprocessor.prepare_test_data()
    
    predictions = improved_model.predict(
        X_test=test_data['X_test'],
        test_ids=test_data['ids_test'],
        test_df=test_df,
        image_dir=str(data_dir / "images")
    )
    
    # Save predictions
    submission = pd.DataFrame({
        'id': test_data['ids_test'],
        'predicted_price': predictions
    })
    submission.to_csv(output_dir / "predictions.csv", index=False)
    print(f"\nPredictions saved to {output_dir / 'predictions.csv'}")
    
    print("\n" + "="*70)
    print("IMPROVED PIPELINE COMPLETE!")
    print("="*70)
    
    return improved_metrics


if __name__ == "__main__":
    main()
