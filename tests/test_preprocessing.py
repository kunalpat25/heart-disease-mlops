"""
Unit Tests for Data Processing Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import (
    create_preprocessing_pipeline,
    validate_input,
    DataCleaner,
    FeatureScaler,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS
)


class TestDataCleaner:
    """Test cases for DataCleaner transformer."""
    
    def test_fit_transform_no_missing(self):
        """Test fitting and transforming data without missing values."""
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        cleaner = DataCleaner(strategy='median')
        result = cleaner.fit_transform(data)
        
        assert result.shape == (3, 2)
        assert not np.isnan(result).any()
    
    def test_fit_transform_with_missing(self):
        """Test that missing values are imputed."""
        data = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [4, 5, np.nan]
        })
        
        cleaner = DataCleaner(strategy='median')
        result = cleaner.fit_transform(data)
        
        assert result.shape == (3, 2)
        assert not np.isnan(result).any()
    
    def test_median_imputation(self):
        """Test that median imputation works correctly."""
        data = pd.DataFrame({
            'a': [1, np.nan, 5]  # median is 3
        })
        
        cleaner = DataCleaner(strategy='median')
        result = cleaner.fit_transform(data)
        
        # The imputed value should be (1 + 5) / 2 = 3
        assert result[1, 0] == 3.0


class TestFeatureScaler:
    """Test cases for FeatureScaler transformer."""
    
    def test_scaling(self):
        """Test that features are scaled correctly."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = FeatureScaler()
        result = scaler.fit_transform(data)
        
        # Check that mean is approximately 0 and std is approximately 1
        assert np.abs(result.mean()) < 1e-10
        assert np.abs(result.std() - 1) < 0.1
    
    def test_shape_preserved(self):
        """Test that shape is preserved after scaling."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        
        scaler = FeatureScaler()
        result = scaler.fit_transform(data)
        
        assert result.shape == data.shape


class TestPreprocessingPipeline:
    """Test cases for the complete preprocessing pipeline."""
    
    def test_pipeline_creation(self):
        """Test that pipeline is created correctly."""
        pipeline = create_preprocessing_pipeline()
        
        assert pipeline is not None
        assert len(pipeline.steps) == 2
    
    def test_pipeline_fit_transform(self):
        """Test pipeline fit_transform."""
        pipeline = create_preprocessing_pipeline()
        
        data = pd.DataFrame({
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
        
        result = pipeline.fit_transform(data)
        
        assert result.shape == (3, 13)
        assert not np.isnan(result).any()


class TestValidateInput:
    """Test cases for input validation."""
    
    def test_valid_input(self):
        """Test validation with valid input."""
        valid_data = {
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        result = validate_input(valid_data)
        
        assert len(result) == 13
        assert result['age'] == 63.0
    
    def test_missing_feature(self):
        """Test that missing feature raises ValueError."""
        invalid_data = {
            'age': 63,
            'sex': 1
            # Missing other features
        }
        
        with pytest.raises(ValueError, match="Missing required feature"):
            validate_input(invalid_data)
    
    def test_invalid_age(self):
        """Test that invalid age raises ValueError."""
        invalid_data = {
            'age': 150,  # Invalid: > 120
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        with pytest.raises(ValueError, match="Age must be between"):
            validate_input(invalid_data)
    
    def test_invalid_sex(self):
        """Test that invalid sex value raises ValueError."""
        invalid_data = {
            'age': 63,
            'sex': 2,  # Invalid: not 0 or 1
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        with pytest.raises(ValueError, match="Sex must be 0 or 1"):
            validate_input(invalid_data)


class TestFeatureConstants:
    """Test feature name constants."""
    
    def test_feature_names_count(self):
        """Test that we have 13 features."""
        assert len(FEATURE_NAMES) == 13
    
    def test_feature_descriptions_complete(self):
        """Test that all features have descriptions."""
        for feature in FEATURE_NAMES:
            assert feature in FEATURE_DESCRIPTIONS
            assert len(FEATURE_DESCRIPTIONS[feature]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
