# Heart Disease Prediction - MLOps Project

![CI/CD Pipeline](https://github.com/yourusername/heart-disease-mlops/workflows/ML%20Pipeline%20CI/CD/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Project Overview

This project implements an end-to-end MLOps pipeline for predicting heart disease risk based on patient health data. It demonstrates modern MLOps best practices including:

- **Data Acquisition & EDA**: Automated data download and exploratory analysis
- **Model Development**: Multiple ML models with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for tracking experiments
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Containerization**: Docker-based deployment
- **Orchestration**: Kubernetes manifests for cloud deployment
- **Monitoring**: Prometheus + Grafana for observability

## 🏗️ Project Structure

```
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml     # GitHub Actions CI/CD
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI application
├── data/
│   ├── download_data.py        # Data acquisition script
│   ├── raw/                    # Raw dataset
│   └── processed/              # Cleaned dataset
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   └── configmap.yaml
├── models/                     # Trained models and artifacts
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb # Model training with MLflow
├── screenshots/               # Screenshots for documentation
├── scripts/
│   └── train_model.py         # Training script
├── src/
│   ├── __init__.py
│   ├── model.py               # Model training utilities
│   └── preprocessing.py       # Data preprocessing pipeline
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git
- (Optional) kubectl for Kubernetes deployment
- (Optional) Minikube or Docker Desktop for local K8s

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/heart-disease-mlops.git
cd heart-disease-mlops
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
python data/download_data.py
```

### 4. Train Model

```bash
python scripts/train_model.py --model-type random_forest
```

### 5. Run API Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## 📊 Exploratory Data Analysis

Run the EDA notebook to explore the dataset:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Key Findings:
- Dataset contains 303 samples with 13 features
- Relatively balanced classes (~54% no disease, ~46% disease)
- Top correlated features: cp (chest pain), thalach (max heart rate), exang (exercise angina)

## 🤖 Model Training

### Available Models:
1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble method (default)
3. **Gradient Boosting** - Advanced ensemble

### Training with MLflow:

```bash
# Basic training
python scripts/train_model.py --model-type random_forest

# With hyperparameter tuning
python scripts/train_model.py --model-type random_forest --tune

# View MLflow UI
mlflow ui --backend-store-uri file:mlruns
```

Open MLflow UI at: http://localhost:5000

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov=app

# Run specific test file
pytest tests/test_api.py -v
```

## 🐳 Docker Deployment

### Build and Run Container

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest

# Test the API
curl http://localhost:8000/health
```

### Docker Compose (with Monitoring)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

Services:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

## ☸️ Kubernetes Deployment

### Local Deployment (Docker Desktop)

```bash
# Enable Kubernetes in Docker Desktop:
# Settings → Kubernetes → Enable Kubernetes → Apply & Restart

# Build Docker image
docker build -t heart-disease-api:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl get pods -n heart-disease-ml
kubectl get services -n heart-disease-ml

# Access the service (if using NodePort or LoadBalancer)
# http://localhost:80
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | Prometheus metrics |
| `/model/info` | GET | Model information |
| `/docs` | GET | Swagger documentation |

### Sample Prediction Request

```bash
curl -X POST "http://localhost:80/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Sample Response

```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease",
  "probability_no_disease": 0.23,
  "probability_disease": 0.77,
  "confidence": 0.77,
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

## 📈 Monitoring

### Prometheus Metrics

Available metrics:
- `predictions_total` - Total predictions by result
- `prediction_latency_seconds` - Prediction latency histogram
- `api_requests_total` - Total API requests by endpoint

### Grafana Dashboard

1. Access Grafana: http://localhost:3000
2. Login: admin / admin123
3. Navigate to Dashboards → Heart Disease Prediction API

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline includes:

1. **Linting** - flake8, black
2. **Unit Tests** - pytest with coverage
3. **Model Training** - Automated training
4. **Docker Build** - Build and test container
5. **Security Scan** - Dependency vulnerability check
6. **Deploy** - Production deployment (manual trigger)

## 📁 Dataset Information

**Heart Disease UCI Dataset**

- **Source**: UCI Machine Learning Repository
- **Samples**: 303
- **Features**: 13 attributes
- **Target**: Binary (0 = No disease, 1 = Disease)

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1=male, 0=female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST |
| ca | Major vessels colored by fluoroscopy |
| thal | Thalassemia type |

## 📝 License

This project is licensed under the MIT License.
