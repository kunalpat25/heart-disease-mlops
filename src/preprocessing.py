"""
Preprocessing Pipeline for Heart Disease Prediction
Provides reusable preprocessing transformers for reproducibility.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer for data cleaning.
    Handles missing values and data type conversions.
    """
    
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = None
        self.feature_names = None
    
    def fit(self, X, y=None):
        """Fit the imputer on training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        """Transform the data by imputing missing values."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.imputer.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names."""
        if input_features is not None:
            return input_features
        return self.feature_names


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature scaling.
    Uses StandardScaler for normalization.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """Fit the scaler on training data."""
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        """Transform the data by scaling features."""
        return self.scaler.transform(X)


def create_preprocessing_pipeline():
    """
    Create a complete preprocessing pipeline.
    
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline
    """
    pipeline = Pipeline([
        ('cleaner', DataCleaner(strategy='median')),
        ('scaler', FeatureScaler())
    ])
    
    return pipeline


def save_pipeline(pipeline, path: str):
    """Save the preprocessing pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to {path}")


def load_pipeline(path: str):
    """Load a preprocessing pipeline from disk."""
    return joblib.load(path)


# Feature names for the heart disease dataset
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male, 0 = female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes, 0 = no)',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment (0-2)',
    'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
    'thal': 'Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)'
}


def validate_input(data: dict) -> dict:
    """
    Validate input data for prediction.
    
    Args:
        data: Dictionary with feature names as keys
    
    Returns:
        dict: Validated and cleaned data
        
    Raises:
        ValueError: If required features are missing or invalid
    """
    validated = {}
    
    for feature in FEATURE_NAMES:
        if feature not in data:
            raise ValueError(f"Missing required feature: {feature}")
        
        value = data[feature]
        
        # Convert to appropriate type
        try:
            validated[feature] = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {feature}: {value}")
    
    # Validate ranges
    if not (0 <= validated['age'] <= 120):
        raise ValueError("Age must be between 0 and 120")
    
    if validated['sex'] not in [0, 1]:
        raise ValueError("Sex must be 0 or 1")
    
    if validated['cp'] not in [0, 1, 2, 3]:
        raise ValueError("Chest pain type must be 0, 1, 2, or 3")
    
    if validated['fbs'] not in [0, 1]:
        raise ValueError("Fasting blood sugar must be 0 or 1")
    
    if validated['exang'] not in [0, 1]:
        raise ValueError("Exercise induced angina must be 0 or 1")
    
    return validated


if __name__ == "__main__":
    # Test the pipeline
    print("Testing preprocessing pipeline...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'age': [63, 37, 41],
        'sex': [1, 1, 0],
        'cp': [3, 2, 1],
        'trestbps': [145, 130, 130],
        'chol': [233, 250, 204],
        'fbs': [1, 0, 0],
        'restecg': [0, 1, 0],
        'thalach': [150, 187, 172],
        'exang': [0, 0, 0],
        'oldpeak': [2.3, 3.5, 1.4],
        'slope': [0, 0, 2],
        'ca': [0, 0, 0],
        'thal': [1, 2, 2]
    })
    
    # Create and fit pipeline
    pipeline = create_preprocessing_pipeline()
    transformed = pipeline.fit_transform(sample_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Transformed shape: {transformed.shape}")
    print("✓ Pipeline test successful!")
