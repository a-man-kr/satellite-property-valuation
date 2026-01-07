"""
Main script for Satellite Imagery-Based Property Valuation
Run the complete pipeline: data fetching, preprocessing, training, and prediction.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data_fetcher import SatelliteImageFetcher, create_placeholder_images
from src.preprocessing import PropertyDataPreprocessor
from src.models.cnn_encoder import SatelliteImageEncoder, ImageFeatureExtractor
from src.models.multimodal_model import (
    PropertyDataset, MultimodalFusionModel, MultimodalTrainer,
    TabularOnlyModel, compare_models
)


def main(args):
    """Run the complete pipeline."""
    
    # Setup paths
    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Fetch satellite images
    if args.fetch_images:
        print("\n" + "="*50)
        print("STEP 1: Fetching Satellite Images")
        print("="*50)
        
        train_df = pd.read_csv(data_dir / "raw/train.csv")
        test_df = pd.read_csv(data_dir / "raw/test.csv")
        all_df = pd.concat([train_df, test_df])
        
        if args.use_placeholder:
            create_placeholder_images(all_df, str(data_dir / "images"))
        else:
            fetcher = SatelliteImageFetcher(
                api_provider=args.api_provider,
                output_dir=str(data_dir / "images")
            )
            fetcher.fetch_all_images(all_df)
    
    # Step 2: Preprocess data
    print("\n" + "="*50)
    print("STEP 2: Preprocessing Data")
    print("="*50)
    
    preprocessor = PropertyDataPreprocessor()
    preprocessor.load_data(
        train_path=str(data_dir / "raw/train.csv"),
        test_path=str(data_dir / "raw/test.csv")
    )
    
    data = preprocessor.prepare_for_training(val_size=0.2, random_state=42)
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Features: {len(data['feature_columns'])}")
    
    # Step 3: Extract image features
    print("\n" + "="*50)
    print("STEP 3: Extracting Image Features")
    print("="*50)
    
    image_encoder = SatelliteImageEncoder(embedding_dim=256, pretrained=True)
    feature_extractor = ImageFeatureExtractor(image_encoder, device=device)
    
    all_ids = np.concatenate([data['ids_train'], data['ids_val']])
    
    features_path = data_dir / "processed/image_features.npz"
    if features_path.exists() and not args.reextract:
        print("Loading cached image features...")
        image_features = feature_extractor.load_features(str(features_path))
    else:
        print("Extracting image features...")
        image_features = feature_extractor.extract_batch(
            all_ids.tolist(), 
            str(data_dir / "images")
        )
        feature_extractor.save_features(image_features, str(features_path))
    
    # Prepare image feature arrays
    def get_image_features(ids, features_dict, embedding_dim=256):
        features = []
        for pid in ids:
            if str(pid) in features_dict:
                features.append(features_dict[str(pid)])
            else:
                features.append(np.zeros(embedding_dim))
        return np.array(features)
    
    X_img_train = get_image_features(data['ids_train'], image_features)
    X_img_val = get_image_features(data['ids_val'], image_features)
    
    # Step 4: Train models
    print("\n" + "="*50)
    print("STEP 4: Training Models")
    print("="*50)
    
    # Train XGBoost baseline
    print("\nTraining XGBoost baseline...")
    tabular_model = TabularOnlyModel()
    tabular_metrics = tabular_model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Train multimodal model
    print(f"\nTraining Multimodal model ({args.fusion_type} fusion)...")
    
    train_dataset = PropertyDataset(
        data['X_train'], X_img_train, data['y_train'], data['ids_train']
    )
    val_dataset = PropertyDataset(
        data['X_val'], X_img_val, data['y_val'], data['ids_val']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = MultimodalFusionModel(
        tabular_dim=data['X_train'].shape[1],
        image_dim=256,
        fusion_type=args.fusion_type
    )
    
    trainer = MultimodalTrainer(model, device=device)
    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience
    )
    
    multimodal_metrics = trainer.evaluate(val_loader)
    
    # Compare models
    compare_models(tabular_metrics, multimodal_metrics)
    
    # Step 5: Generate predictions
    print("\n" + "="*50)
    print("STEP 5: Generating Predictions")
    print("="*50)
    
    test_data = preprocessor.prepare_test_data()
    X_img_test = get_image_features(test_data['ids_test'], image_features)
    
    test_dataset = PropertyDataset(
        test_data['X_test'], X_img_test, property_ids=test_data['ids_test']
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    predictions = trainer.predict(test_loader)
    
    # Save predictions
    submission = pd.DataFrame({
        'id': test_data['ids_test'],
        'predicted_price': predictions
    })
    submission.to_csv(output_dir / "predictions.csv", index=False)
    print(f"Saved predictions to {output_dir / 'predictions.csv'}")
    
    # Save models
    torch.save(model.state_dict(), output_dir / "multimodal_model.pth")
    preprocessor.save_preprocessor(str(output_dir / "preprocessor.pkl"))
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    print(f"\nFinal Results:")
    print(f"  Tabular-only RMSE: ${tabular_metrics['rmse']:,.0f}")
    print(f"  Multimodal RMSE: ${multimodal_metrics['rmse']:,.0f}")
    print(f"  Improvement: {(tabular_metrics['rmse'] - multimodal_metrics['rmse']) / tabular_metrics['rmse'] * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Property Valuation Pipeline")
    
    # Data arguments
    parser.add_argument("--fetch-images", action="store_true",
                        help="Fetch satellite images")
    parser.add_argument("--use-placeholder", action="store_true",
                        help="Use placeholder images instead of real satellite images")
    parser.add_argument("--api-provider", type=str, default="mapbox",
                        choices=["mapbox", "google", "osm"],
                        help="API provider for satellite images")
    parser.add_argument("--reextract", action="store_true",
                        help="Re-extract image features even if cached")
    
    # Model arguments
    parser.add_argument("--fusion-type", type=str, default="late",
                        choices=["early", "late", "attention"],
                        help="Fusion strategy for multimodal model")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    args = parser.parse_args()
    main(args)
