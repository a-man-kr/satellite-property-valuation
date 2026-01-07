"""
Baseline pipeline runner with ResNet18 + XGBoost.
For best results, use run_improved_pipeline.py instead.
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import PropertyDataPreprocessor
from src.models.cnn_encoder import SatelliteImageEncoder, ImageFeatureExtractor
from src.models.multimodal_model import (
    PropertyDataset, MultimodalFusionModel, MultimodalTrainer,
    TabularOnlyModel, compare_models
)

def fetch_images_google(df, output_dir, max_images=None, rate_limit=0.1):
    """Fetch satellite images from Google Maps API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("Warning: GOOGLE_MAPS_API_KEY not set. Skipping image fetch.")
        return 0, 0, 0
    
    success, failed, skipped = 0, 0, 0
    
    rows = df.iterrows()
    total = min(len(df), max_images) if max_images else len(df)
    
    print(f"Fetching {total} satellite images...")
    
    for idx, row in tqdm(rows, total=total):
        if max_images and (success + skipped) >= max_images:
            break
            
        pid = str(row['id'])
        img_path = output_dir / f"{pid}.png"
        
        if img_path.exists():
            skipped += 1
            continue
        
        lat, lon = row['lat'], row['long']
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=18&size=256x256&maptype=satellite&key={api_key}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
        
        time.sleep(rate_limit)
    
    print(f"Done: {success} downloaded, {skipped} skipped, {failed} failed")
    return success, skipped, failed


def main():
    print("="*60)
    print("SATELLITE IMAGERY-BASED PROPERTY VALUATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Paths
    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(data_dir / "raw/train.csv")
    test_df = pd.read_csv(data_dir / "raw/test.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Check existing images
    print("\n[2/6] Checking satellite images...")
    existing_images = list((data_dir / "images").glob("*.png"))
    print(f"Found {len(existing_images)} existing satellite images")
    
    # Only fetch more if we have very few
    if len(existing_images) < 500:
        print("Fetching more images...")
        all_df = pd.concat([train_df, test_df])
        fetch_images_google(all_df, data_dir / "images", max_images=1000, rate_limit=0.02)
    else:
        print("Using existing images (sufficient for training)")
    
    # Preprocess
    print("\n[3/6] Preprocessing data...")
    preprocessor = PropertyDataPreprocessor()
    preprocessor.load_data(
        train_path=str(data_dir / "raw/train.csv"),
        test_path=str(data_dir / "raw/test.csv")
    )
    data = preprocessor.prepare_for_training(val_size=0.2, random_state=42)
    print(f"Features: {len(data['feature_columns'])}")
    
    # Extract image features
    print("\n[4/6] Extracting image features...")
    encoder = SatelliteImageEncoder(embedding_dim=256, pretrained=True)
    extractor = ImageFeatureExtractor(encoder, device=device)
    
    all_ids = np.concatenate([data['ids_train'], data['ids_val']])
    image_features = extractor.extract_batch(all_ids.tolist(), str(data_dir / "images"), batch_size=32)
    
    def get_img_features(ids, feat_dict, dim=256):
        return np.array([feat_dict.get(str(pid), np.zeros(dim)) for pid in ids])
    
    X_img_train = get_img_features(data['ids_train'], image_features)
    X_img_val = get_img_features(data['ids_val'], image_features)
    
    # Train models
    print("\n[5/6] Training models...")
    
    # XGBoost baseline
    print("\nTraining XGBoost (tabular only)...")
    tabular_model = TabularOnlyModel()
    tabular_metrics = tabular_model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Multimodal model
    print("\nTraining Multimodal model (late fusion)...")
    train_dataset = PropertyDataset(data['X_train'], X_img_train, data['y_train'])
    val_dataset = PropertyDataset(data['X_val'], X_img_val, data['y_val'])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = MultimodalFusionModel(
        tabular_dim=data['X_train'].shape[1],
        image_dim=256,
        fusion_type='late'
    )
    
    trainer = MultimodalTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=30, lr=1e-3, patience=7)
    multimodal_metrics = trainer.evaluate(val_loader)
    
    # Compare
    compare_models(tabular_metrics, multimodal_metrics)
    
    # Generate predictions
    print("\n[6/6] Generating predictions...")
    test_data = preprocessor.prepare_test_data()
    X_img_test = get_img_features(test_data['ids_test'], image_features)
    
    test_dataset = PropertyDataset(test_data['X_test'], X_img_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    predictions = trainer.predict(test_loader)
    
    # Save predictions
    submission = pd.DataFrame({
        'id': test_data['ids_test'],
        'predicted_price': predictions
    })
    submission.to_csv(output_dir / "predictions.csv", index=False)
    print(f"\nPredictions saved to {output_dir / 'predictions.csv'}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
