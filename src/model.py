"""
Heart Disease Prediction Model Module
Provides functions for model training, evaluation, and inference.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
from typing import Dict, Tuple, Any, Optional


# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'class': LogisticRegression,
        'default_params': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        },
        'param_grid': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'default_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'default_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        },
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'random_forest',
    params: Optional[Dict] = None,
    tune_hyperparameters: bool = False
) -> Tuple[Any, Dict]:
    """
    Train a classification model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('logistic_regression', 'random_forest', 'gradient_boosting')
        params: Model parameters (if None, uses defaults)
        tune_hyperparameters: Whether to perform hyperparameter tuning
    
    Returns:
        Tuple of (trained model, parameters used)
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    model_class = config['class']
    
    if params is None:
        params = config['default_params'].copy()
    
    if tune_hyperparameters:
        print(f"Tuning hyperparameters for {model_type}...")
        base_model = model_class(**config['default_params'])
        grid_search = GridSearchCV(
            base_model,
            config['param_grid'],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        params = grid_search.best_params_
        print(f"Best parameters: {params}")
    else:
        model = model_class(**params)
        model.fit(X_train, y_train)
    
    return model, params


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Dict[str, float]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to validate
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric
    
    Returns:
        Dictionary with mean and std of scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'cv_mean': scores.mean(),
        'cv_std': scores.std(),
        'cv_scores': scores.tolist()
    }


def predict(
    model: Any,
    scaler: Any,
    features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        features: Dictionary of input features
    
    Returns:
        Dictionary with prediction and probabilities
    """
    from .preprocessing import FEATURE_NAMES, validate_input
    
    # Validate and order features
    validated = validate_input(features)
    feature_array = np.array([[validated[f] for f in FEATURE_NAMES]])
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    return {
        'prediction': int(prediction),
        'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'probability_no_disease': float(probabilities[0]),
        'probability_disease': float(probabilities[1]),
        'confidence': float(max(probabilities))
    }


def save_model(model: Any, path: str):
    """Save a trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> Any:
    """Load a trained model from disk."""
    return joblib.load(path)


def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importance attributes")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    # Test the module
    print("Testing model module...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 13)
    y = (np.random.rand(100) > 0.5).astype(int)
    
    # Train a model
    model, params = train_model(X, y, model_type='random_forest')
    print(f"Model trained with parameters: {params}")
    
    # Evaluate
    metrics = evaluate_model(model, X, y)
    print(f"Metrics: {metrics}")
    
    print("✓ Model module test successful!")
