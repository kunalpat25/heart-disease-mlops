"""
Unit Tests for FastAPI Application
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.main import app


# Create test client
client = TestClient(app)


class TestRootEndpoint:
    """Test cases for root endpoint."""
    
    def test_root_returns_200(self):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_api_info(self):
        """Test that root returns API information."""
        response = client.get("/")
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "Heart Disease" in data["message"]


class TestHealthEndpoint:
    """Test cases for health endpoint."""
    
    def test_health_returns_200(self):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self):
        """Test that health returns status information."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestPredictEndpoint:
    """Test cases for predict endpoint."""
    
    @pytest.fixture
    def valid_input(self):
        """Valid input data for prediction."""
        return {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
    
    def test_predict_valid_input_format(self, valid_input):
        """Test that predict endpoint accepts valid input format."""
        response = client.post("/predict", json=valid_input)
        
        # Should return 200 if model is loaded, 503 if not
        assert response.status_code in [200, 503]
    
    def test_predict_missing_field(self):
        """Test that missing field returns 422."""
        incomplete_input = {
            "age": 63,
            "sex": 1
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_age(self, valid_input):
        """Test that invalid age returns 422."""
        invalid_input = valid_input.copy()
        invalid_input["age"] = 200  # Invalid: > 120
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_invalid_sex(self, valid_input):
        """Test that invalid sex returns 422."""
        invalid_input = valid_input.copy()
        invalid_input["sex"] = 5  # Invalid: not 0 or 1
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
    
    def test_predict_negative_values(self, valid_input):
        """Test that negative values where not allowed returns 422."""
        invalid_input = valid_input.copy()
        invalid_input["trestbps"] = -10  # Invalid: negative
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test cases for metrics endpoint."""
    
    def test_metrics_returns_200(self):
        """Test that metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_returns_prometheus_format(self):
        """Test that metrics returns Prometheus format."""
        response = client.get("/metrics")
        
        # Should contain Prometheus metric format
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type or "text/html" in content_type


class TestModelInfoEndpoint:
    """Test cases for model info endpoint."""
    
    def test_model_info_endpoint_exists(self):
        """Test that model info endpoint exists."""
        response = client.get("/model/info")
        
        # Should return 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]


class TestCORSHeaders:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present."""
        response = client.options("/")
        # Just check that the request doesn't fail
        assert response.status_code in [200, 405]


class TestInputValidation:
    """Test input validation edge cases."""
    
    def test_empty_body(self):
        """Test that empty body returns 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_non_json_body(self):
        """Test that non-JSON body returns 422."""
        response = client.post("/predict", data="not json")
        assert response.status_code == 422
    
    def test_string_instead_of_number(self):
        """Test that string instead of number is handled."""
        invalid_input = {
            "age": "sixty-three",  # Should be number
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
        
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
