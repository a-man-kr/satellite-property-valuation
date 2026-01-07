"""
Multimodal Property Valuation Model
Combines tabular features with satellite image embeddings for price prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class PropertyDataset(Dataset):
    """Dataset for multimodal property data."""
    
    def __init__(self, tabular_features: np.ndarray, image_features: np.ndarray,
                 targets: np.ndarray = None, property_ids: np.ndarray = None):
        self.tabular = torch.FloatTensor(tabular_features)
        self.images = torch.FloatTensor(image_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.ids = property_ids
        
    def __len__(self):
        return len(self.tabular)
    
    def __getitem__(self, idx):
        item = {
            'tabular': self.tabular[idx],
            'image': self.images[idx]
        }
        if self.targets is not None:
            item['target'] = self.targets[idx]
        if self.ids is not None:
            item['id'] = self.ids[idx]
        return item


class TabularEncoder(nn.Module):
    """Neural network encoder for tabular features."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], 
                 output_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal model combining tabular and image features.
    
    Fusion strategies:
    - 'early': Concatenate features before processing
    - 'late': Process separately then combine
    - 'attention': Use attention mechanism for fusion
    """
    
    def __init__(self, tabular_dim: int, image_dim: int = 256,
                 fusion_type: str = 'late', hidden_dim: int = 128):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'early':
            # Early fusion: concatenate then process
            combined_dim = tabular_dim + image_dim
            self.fusion = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )
            
        elif fusion_type == 'late':
            # Late fusion: process separately then combine
            self.tabular_encoder = TabularEncoder(tabular_dim, [128, 64], 64)
            self.image_encoder = nn.Sequential(
                nn.Linear(image_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64)
            )
            self.fusion = nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1)
            )
            
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.tabular_encoder = TabularEncoder(tabular_dim, [128, 64], 64)
            self.image_encoder = nn.Sequential(
                nn.Linear(image_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, tabular: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == 'early':
            combined = torch.cat([tabular, image], dim=1)
            return self.fusion(combined)
        
        elif self.fusion_type == 'late':
            tab_emb = self.tabular_encoder(tabular)
            img_emb = self.image_encoder(image)
            combined = torch.cat([tab_emb, img_emb], dim=1)
            return self.fusion(combined)
        
        elif self.fusion_type == 'attention':
            tab_emb = self.tabular_encoder(tabular).unsqueeze(1)
            img_emb = self.image_encoder(image).unsqueeze(1)
            
            # Stack as sequence
            seq = torch.cat([tab_emb, img_emb], dim=1)
            attn_out, _ = self.attention(seq, seq, seq)
            
            # Pool and predict
            pooled = attn_out.mean(dim=1)
            combined = torch.cat([pooled, tab_emb.squeeze(1)], dim=1)
            return self.fusion(combined)


class MultimodalTrainer:
    """Training and evaluation for multimodal model."""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, lr: float = 1e-3, patience: int = 10):
        """Train the model with early stopping."""
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_r2': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                tabular = batch['tabular'].to(self.device)
                image = batch['image'].to(self.device)
                target = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(tabular, image).squeeze()
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Update scheduler
            scheduler.step(val_metrics['mse'])
            
            # Logging
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['mse'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_r2'].append(val_metrics['r2'])
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val RMSE: {val_metrics['rmse']:.2f} - "
                  f"Val R²: {val_metrics['r2']:.4f}")
            
            # Early stopping
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        self.model.load_state_dict(self.best_state)
        return history
    
    def evaluate(self, data_loader: DataLoader) -> dict:
        """Evaluate model on a dataset."""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                tabular = batch['tabular'].to(self.device)
                image = batch['image'].to(self.device)
                
                pred = self.model(tabular, image).squeeze()
                predictions.extend(pred.cpu().numpy())
                
                if 'target' in batch:
                    targets.extend(batch['target'].numpy())
        
        predictions = np.array(predictions)
        
        if targets:
            targets = np.array(targets)
            return {
                'mse': mean_squared_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions)
            }
        
        return {'predictions': predictions}
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Generate predictions."""
        return self.evaluate(data_loader)['predictions']


class TabularOnlyModel:
    """XGBoost baseline using only tabular features."""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        """Train XGBoost model."""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        # Evaluate
        val_pred = self.model.predict(X_val)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'r2': r2_score(y_val, val_pred)
        }
        print(f"Validation - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X)
    
    def feature_importance(self) -> dict:
        """Get feature importance scores."""
        return dict(zip(
            range(len(self.model.feature_importances_)),
            self.model.feature_importances_
        ))


def compare_models(tabular_metrics: dict, multimodal_metrics: dict):
    """Compare tabular-only vs multimodal model performance."""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"{'Metric':<15} {'Tabular Only':<15} {'Multimodal':<15} {'Improvement':<15}")
    print("-"*50)
    
    for metric in ['rmse', 'mae', 'r2']:
        tab_val = tabular_metrics.get(metric, 0)
        mm_val = multimodal_metrics.get(metric, 0)
        
        if metric == 'r2':
            improvement = (mm_val - tab_val) * 100
            print(f"{metric.upper():<15} {tab_val:<15.4f} {mm_val:<15.4f} {improvement:+.2f}%")
        else:
            improvement = (tab_val - mm_val) / tab_val * 100
            print(f"{metric.upper():<15} {tab_val:<15.2f} {mm_val:<15.2f} {improvement:+.2f}%")
    
    print("="*50)


if __name__ == "__main__":
    # Test model architectures
    tabular_dim = 30
    image_dim = 256
    batch_size = 4
    
    # Test data
    tabular = torch.randn(batch_size, tabular_dim)
    image = torch.randn(batch_size, image_dim)
    
    # Test each fusion type
    for fusion in ['early', 'late', 'attention']:
        model = MultimodalFusionModel(tabular_dim, image_dim, fusion_type=fusion)
        out = model(tabular, image)
        print(f"{fusion.capitalize()} fusion output shape: {out.shape}")
