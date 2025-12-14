"""
Unit Tests for Model Module
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import (
    train_model,
    evaluate_model,
    cross_validate_model,
    get_feature_importance,
    MODEL_CONFIGS
)


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 13)
    y = (np.random.rand(100) > 0.5).astype(int)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Train a sample model."""
    X, y = sample_data
    model, params = train_model(X, y, model_type='random_forest')
    return model


class TestModelConfigs:
    """Test model configurations."""
    
    def test_configs_exist(self):
        """Test that model configs are defined."""
        assert 'logistic_regression' in MODEL_CONFIGS
        assert 'random_forest' in MODEL_CONFIGS
        assert 'gradient_boosting' in MODEL_CONFIGS
    
    def test_config_has_required_keys(self):
        """Test that each config has required keys."""
        for model_type, config in MODEL_CONFIGS.items():
            assert 'class' in config
            assert 'default_params' in config
            assert 'param_grid' in config


class TestTrainModel:
    """Test cases for train_model function."""
    
    def test_train_logistic_regression(self, sample_data):
        """Test training logistic regression."""
        X, y = sample_data
        model, params = train_model(X, y, model_type='logistic_regression')
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_random_forest(self, sample_data):
        """Test training random forest."""
        X, y = sample_data
        model, params = train_model(X, y, model_type='random_forest')
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'feature_importances_')
    
    def test_train_gradient_boosting(self, sample_data):
        """Test training gradient boosting."""
        X, y = sample_data
        model, params = train_model(X, y, model_type='gradient_boosting')
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_train_invalid_model_type(self, sample_data):
        """Test that invalid model type raises error."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Unknown model type"):
            train_model(X, y, model_type='invalid_model')
    
    def test_custom_params(self, sample_data):
        """Test training with custom parameters."""
        X, y = sample_data
        custom_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model, params = train_model(X, y, model_type='random_forest', params=custom_params)
        
        assert model.n_estimators == 50
        assert model.max_depth == 5


class TestEvaluateModel:
    """Test cases for evaluate_model function."""
    
    def test_evaluate_returns_metrics(self, sample_data, trained_model):
        """Test that evaluation returns all expected metrics."""
        X, y = sample_data
        
        metrics = evaluate_model(trained_model, X, y)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
    
    def test_metrics_in_valid_range(self, sample_data, trained_model):
        """Test that metrics are in valid range [0, 1]."""
        X, y = sample_data
        
        metrics = evaluate_model(trained_model, X, y)
        
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"


class TestCrossValidateModel:
    """Test cases for cross_validate_model function."""
    
    def test_cross_validation(self, sample_data, trained_model):
        """Test cross-validation."""
        X, y = sample_data
        
        cv_results = cross_validate_model(trained_model, X, y, cv=3)
        
        assert 'cv_mean' in cv_results
        assert 'cv_std' in cv_results
        assert 'cv_scores' in cv_results
        assert len(cv_results['cv_scores']) == 3
    
    def test_cv_mean_in_range(self, sample_data, trained_model):
        """Test that CV mean is in valid range."""
        X, y = sample_data
        
        cv_results = cross_validate_model(trained_model, X, y, cv=3)
        
        assert 0 <= cv_results['cv_mean'] <= 1


class TestGetFeatureImportance:
    """Test cases for get_feature_importance function."""
    
    def test_feature_importance_random_forest(self, sample_data):
        """Test feature importance for random forest."""
        X, y = sample_data
        model, _ = train_model(X, y, model_type='random_forest')
        
        feature_names = [f'feature_{i}' for i in range(13)]
        importance_df = get_feature_importance(model, feature_names)
        
        assert len(importance_df) == 13
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_feature_importance_logistic_regression(self, sample_data):
        """Test feature importance for logistic regression."""
        X, y = sample_data
        model, _ = train_model(X, y, model_type='logistic_regression')
        
        feature_names = [f'feature_{i}' for i in range(13)]
        importance_df = get_feature_importance(model, feature_names)
        
        assert len(importance_df) == 13


class TestModelPrediction:
    """Test model prediction functionality."""
    
    def test_predict_single_sample(self, sample_data, trained_model):
        """Test prediction on a single sample."""
        X, _ = sample_data
        
        prediction = trained_model.predict(X[0:1])
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    def test_predict_proba(self, sample_data, trained_model):
        """Test probability prediction."""
        X, _ = sample_data
        
        proba = trained_model.predict_proba(X[0:1])
        
        assert proba.shape == (1, 2)
        assert np.abs(proba.sum() - 1.0) < 1e-10  # Probabilities sum to 1
    
    def test_predict_batch(self, sample_data, trained_model):
        """Test batch prediction."""
        X, _ = sample_data
        
        predictions = trained_model.predict(X[:10])
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
