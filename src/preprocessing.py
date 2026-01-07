"""
Data Preprocessing for Property Valuation
Handles data cleaning, feature engineering, and train/val splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class PropertyDataPreprocessor:
    """Preprocess property data for model training."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'price'
        
    def load_data(self, train_path: str, test_path: str = None):
        """Load train and test datasets."""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path) if test_path else None
        
        print(f"Train shape: {self.train_df.shape}")
        if self.test_df is not None:
            print(f"Test shape: {self.test_df.shape}")
        
        return self
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and handle missing values."""
        df = df.copy()
        
        # Convert date to datetime features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
            df['year_sold'] = df['date'].dt.year
            df['month_sold'] = df['date'].dt.month
            df.drop('date', axis=1, inplace=True)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        df = df.copy()
        
        # Age of property
        current_year = 2015  # Based on dataset timeframe
        df['age'] = current_year - df['yr_built']
        df['years_since_renovation'] = np.where(
            df['yr_renovated'] > 0,
            current_year - df['yr_renovated'],
            df['age']
        )
        
        # Size ratios
        df['living_lot_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
        df['above_living_ratio'] = df['sqft_above'] / (df['sqft_living'] + 1)
        df['basement_ratio'] = df['sqft_basement'] / (df['sqft_living'] + 1)
        
        # Neighborhood comparison
        df['living_vs_neighbors'] = df['sqft_living'] / (df['sqft_living15'] + 1)
        df['lot_vs_neighbors'] = df['sqft_lot'] / (df['sqft_lot15'] + 1)
        
        # Room features
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        df['sqft_per_room'] = df['sqft_living'] / (df['total_rooms'] + 1)
        
        # Location features (binned)
        df['lat_bin'] = pd.cut(df['lat'], bins=20, labels=False)
        df['long_bin'] = pd.cut(df['long'], bins=20, labels=False)
        
        # Quality score
        df['quality_score'] = df['grade'] * df['condition']
        
        # Has basement
        df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
        
        # Was renovated
        df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
        
        return df
    
    def get_feature_columns(self) -> list:
        """Define feature columns for modeling."""
        return [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
            'sqft_living15', 'sqft_lot15',
            # Engineered features
            'age', 'years_since_renovation', 'living_lot_ratio',
            'above_living_ratio', 'basement_ratio', 'living_vs_neighbors',
            'lot_vs_neighbors', 'total_rooms', 'sqft_per_room',
            'quality_score', 'has_basement', 'was_renovated',
            'year_sold', 'month_sold'
        ]
    
    def prepare_for_training(self, val_size: float = 0.2, random_state: int = 42):
        """Prepare data for model training."""
        # Clean data
        train_clean = self.clean_data(self.train_df)
        
        # Engineer features
        train_features = self.engineer_features(train_clean)
        
        # Get feature columns
        self.feature_columns = self.get_feature_columns()
        
        # Ensure all columns exist
        available_cols = [c for c in self.feature_columns if c in train_features.columns]
        
        # Prepare X and y
        X = train_features[available_cols].values
        y = train_features[self.target_column].values
        ids = train_features['id'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/val split
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_scaled, y, ids, test_size=val_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'ids_train': ids_train,
            'ids_val': ids_val,
            'feature_columns': available_cols
        }
    
    def prepare_test_data(self):
        """Prepare test data for prediction."""
        if self.test_df is None:
            raise ValueError("Test data not loaded")
        
        # Clean and engineer features
        test_clean = self.clean_data(self.test_df)
        test_features = self.engineer_features(test_clean)
        
        # Get available columns
        available_cols = [c for c in self.feature_columns if c in test_features.columns]
        
        X_test = test_features[available_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_test': X_test_scaled,
            'ids_test': test_features['id'].values
        }
    
    def save_preprocessor(self, path: str):
        """Save preprocessor state."""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
    
    def load_preprocessor(self, path: str):
        """Load preprocessor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.scaler = state['scaler']
            self.feature_columns = state['feature_columns']


def get_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract geospatial features using coordinates."""
    df = df.copy()
    
    # Distance to Seattle downtown (approximate center)
    seattle_lat, seattle_lon = 47.6062, -122.3321
    
    df['dist_to_downtown'] = np.sqrt(
        (df['lat'] - seattle_lat)**2 + (df['long'] - seattle_lon)**2
    )
    
    # Distance to water (approximate Puget Sound)
    water_lon = -122.4
    df['dist_to_water'] = np.abs(df['long'] - water_lon)
    
    return df


if __name__ == "__main__":
    # Example usage
    preprocessor = PropertyDataPreprocessor()
    preprocessor.load_data(
        train_path="data/raw/train.csv",
        test_path="data/raw/test.csv"
    )
    
    data = preprocessor.prepare_for_training()
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Features: {len(data['feature_columns'])}")
    
    # Save preprocessor
    preprocessor.save_preprocessor("data/processed/preprocessor.pkl")
