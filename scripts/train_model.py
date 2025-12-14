"""
Model Training Script
Command-line script for training the heart disease prediction model.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import train_model, evaluate_model, cross_validate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Heart Disease Prediction Model')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['logistic_regression', 'random_forest', 'gradient_boosting'],
                        help='Type of model to train')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    
    # Try to load processed data first
    processed_path = os.path.join(data_dir, 'processed', 'heart_cleaned.csv')
    raw_path = os.path.join(data_dir, 'raw', 'heart.csv')
    
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        logger.info(f"Loaded processed data from {processed_path}")
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        logger.info(f"Loaded raw data from {raw_path}")
    else:
        # Download data
        logger.info("Data not found. Downloading...")
        sys.path.insert(0, data_dir)
        from download_data import download_dataset
        df = download_dataset()
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        logger.info("Handling missing values...")
        df = df.fillna(df.median())
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Train model
    logger.info(f"Training {args.model_type} model...")
    model, params = train_model(
        X_train_scaled, y_train,
        model_type=args.model_type,
        tune_hyperparameters=args.tune
    )
    logger.info(f"Model parameters: {params}")
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_results = cross_validate_model(model, X_train_scaled, y_train, cv=5)
    logger.info(f"CV Accuracy: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    logger.info("=" * 50)
    logger.info("TEST SET METRICS")
    logger.info("=" * 50)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("=" * 50)
    
    # Save model
    model_path = os.path.join(models_dir, f'best_model_{args.model_type}.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names
    feature_names_path = os.path.join(models_dir, 'feature_names.joblib')
    joblib.dump(list(X.columns), feature_names_path)
    logger.info(f"Feature names saved to {feature_names_path}")
    
    # Save processed data
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
    
    logger.info("\n✓ Training completed successfully!")
    
    return metrics


if __name__ == "__main__":
    main()
