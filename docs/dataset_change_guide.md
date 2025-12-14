# Guide: Using a Different UCI Dataset

This guide explains how to adapt the MLOps pipeline for a new dataset from UCI Machine Learning Repository.

## Table of Contents

1. [Quick Overview](#quick-overview)
2. [Step-by-Step Process](#step-by-step-process)
3. [Model Versioning Strategy](#model-versioning-strategy)
4. [Example: Diabetes Dataset](#example-diabetes-dataset)

---

## Quick Overview

### Files to Modify

| File | Changes Needed |
|------|----------------|
| `data/download_data.py` | Update dataset ID, URL, column names |
| `src/preprocessing.py` | Update feature validation, data types |
| `notebooks/01_EDA.ipynb` | Run new EDA for the dataset |
| `notebooks/02_Model_Training.ipynb` | Retrain models |
| `app/main.py` | Update `PredictionRequest` schema |
| `tests/*.py` | Update test cases |

---

## Step-by-Step Process

### Step 1: Find Your Dataset

1. Go to [UCI ML Repository](https://archive.ics.uci.edu/)
2. Search for your dataset
3. Note down:
   - **Dataset ID** (e.g., 45 for Heart Disease, 17 for Diabetes)
   - **Feature names and types**
   - **Target variable name**

### Step 2: Update Data Download Script

Edit `data/download_data.py`:

```python
# Change these constants
UCI_DATASET_ID = 17  # New dataset ID
UCI_DATASET_NAME = "diabetes"  # Dataset folder name
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.data"

# Update FEATURE_NAMES based on new dataset
FEATURE_NAMES = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age'
]

TARGET_NAME = 'outcome'  # or 'class', 'label', etc.
```

### Step 3: Update Preprocessing

Edit `src/preprocessing.py`:

```python
# Update expected columns
EXPECTED_COLUMNS = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age'
]

def validate_input(data: dict) -> bool:
    """Validate input data for new dataset."""
    required_fields = EXPECTED_COLUMNS
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")
    
    # Add dataset-specific validations
    if data['glucose'] < 0 or data['glucose'] > 500:
        raise ValueError("Glucose must be between 0 and 500")
    
    return True
```

### Step 4: Update API Schema

Edit `app/main.py`:

```python
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Input schema for new dataset."""
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: float = Field(..., ge=0, le=500)
    blood_pressure: float = Field(..., ge=0, le=200)
    skin_thickness: float = Field(..., ge=0, le=100)
    insulin: float = Field(..., ge=0, le=900)
    bmi: float = Field(..., ge=0, le=70)
    diabetes_pedigree: float = Field(..., ge=0, le=3)
    age: int = Field(..., ge=0, le=120)

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree": 0.627,
                "age": 50
            }
        }

# Update FEATURE_NAMES in predict endpoint
FEATURE_NAMES = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age'
]
```

### Step 5: Run New EDA

1. Open `notebooks/01_EDA.ipynb`
2. Clear all outputs
3. Run all cells to generate new visualizations
4. Save screenshots to `screenshots/` folder

### Step 6: Retrain Models

1. Open `notebooks/02_Model_Training.ipynb`
2. Clear all outputs
3. Run training cells
4. Models will be saved with new version

### Step 7: Update Tests

Edit `tests/test_api.py`:

```python
def test_predict_valid_input():
    """Test prediction with valid input for new dataset."""
    response = client.post(
        "/predict",
        json={
            "pregnancies": 6,
            "glucose": 148,
            "blood_pressure": 72,
            "skin_thickness": 35,
            "insulin": 0,
            "bmi": 33.6,
            "diabetes_pedigree": 0.627,
            "age": 50
        }
    )
    assert response.status_code == 200
```

### Step 8: Rebuild and Deploy

```powershell
# Rebuild Docker image
docker build -t heart-disease-api:v2.0.0 .

# Update Kubernetes deployment
kubectl set image deployment/heart-disease-api api=heart-disease-api:v2.0.0 -n ml-production
```

---

## Model Versioning Strategy

### Current Versioning Approach

The project uses **semantic versioning** for models:

```
MAJOR.MINOR.PATCH
  │     │     └── Bug fixes, minor improvements
  │     └──────── New features, hyperparameter changes
  └────────────── New dataset, architecture changes
```

### Version Tracking Methods

#### Method 1: File-Based Versioning

```python
# In app/main.py
MODEL_VERSION = "1.0.0"  # Update this when retraining

# Save model with version
joblib.dump(model, f'models/model_v{MODEL_VERSION}.joblib')
```

#### Method 2: MLflow Model Registry (Recommended)

```python
import mlflow

# Register model with version
mlflow.sklearn.log_model(
    model,
    "heart-disease-model",
    registered_model_name="HeartDiseaseClassifier"
)

# Load specific version
model_uri = "models:/HeartDiseaseClassifier/1"
model = mlflow.sklearn.load_model(model_uri)
```

#### Method 3: Git Tags

```bash
# Tag model release
git tag -a model-v1.0.0 -m "Heart disease model v1.0.0"
git push origin model-v1.0.0

# For new dataset
git tag -a model-v2.0.0 -m "Diabetes model v2.0.0"
git push origin model-v2.0.0
```

### Implementing MLflow Model Registry

#### Step 1: Update Training Notebook

```python
import mlflow
from mlflow.tracking import MlflowClient

# Set up MLflow
mlflow.set_tracking_uri("file:mlruns")

# Train and register model
with mlflow.start_run():
    model.fit(X_train, y_train)
    
    # Log model to registry
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="HeartDiseaseClassifier"
    )
    
# Transition to production
client = MlflowClient()
client.transition_model_version_stage(
    name="HeartDiseaseClassifier",
    version=1,
    stage="Production"
)
```

#### Step 2: Update API to Load from Registry

```python
# app/main.py
import mlflow

# Load production model
MODEL_NAME = "HeartDiseaseClassifier"
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
```

#### Step 3: Version Comparison Dashboard

Create a comparison script:

```python
# scripts/compare_versions.py
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

print("Model Versions:")
print("-" * 60)
for v in versions:
    print(f"Version: {v.version}")
    print(f"  Stage: {v.current_stage}")
    print(f"  Created: {v.creation_timestamp}")
    print(f"  Run ID: {v.run_id}")
    print()
```

---

## Example: Diabetes Dataset

Here's a complete example for switching to the Pima Indians Diabetes dataset:

### 1. Dataset Information

- **UCI ID:** 17
- **URL:** https://archive.ics.uci.edu/dataset/17/diabetes
- **Features:** 8 numerical
- **Target:** Binary (diabetes/no diabetes)
- **Samples:** 768

### 2. Updated download_data.py

```python
"""Download Pima Indians Diabetes dataset."""
import os
import pandas as pd

UCI_DATASET_ID = 17
DATASET_NAME = "diabetes"

FEATURE_NAMES = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age'
]
TARGET_NAME = 'outcome'

def load_diabetes_data():
    """Load diabetes dataset."""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        return df
    except Exception as e:
        print(f"UCI API failed: {e}")
        # Fallback to direct URL
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        df = pd.read_csv(url, header=None)
        df.columns = FEATURE_NAMES + [TARGET_NAME]
        return df

if __name__ == "__main__":
    df = load_diabetes_data()
    df.to_csv("data/diabetes.csv", index=False)
    print(f"Saved {len(df)} records")
```

### 3. Version Bump Checklist

When switching datasets:

- [ ] Update `download_data.py` with new dataset info
- [ ] Update `src/preprocessing.py` with new validations
- [ ] Update `app/main.py` with new schema
- [ ] Run EDA notebook and save new visualizations
- [ ] Retrain models and log to MLflow
- [ ] Update tests with new sample data
- [ ] Increment MODEL_VERSION to 2.0.0
- [ ] Tag release: `git tag model-v2.0.0`
- [ ] Rebuild Docker image with new tag
- [ ] Update Kubernetes deployment

---

## Quick Reference Commands

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Download new dataset
python data/download_data.py

# 3. Run EDA notebook
jupyter notebook notebooks/01_EDA.ipynb

# 4. Train models
jupyter notebook notebooks/02_Model_Training.ipynb

# 5. Run tests
pytest tests/ -v

# 6. Build new Docker image
docker build -t my-ml-api:v2.0.0 .

# 7. Tag and push
git add .
git commit -m "Switch to diabetes dataset v2.0.0"
git tag model-v2.0.0
git push origin master --tags

# 8. Update Kubernetes
kubectl set image deployment/heart-disease-api api=my-ml-api:v2.0.0 -n ml-production
```

---

## Summary

| Aspect | Current State | After Dataset Change |
|--------|---------------|----------------------|
| Dataset | Heart Disease (ID: 45) | New UCI Dataset |
| Features | 13 | Varies by dataset |
| Model Version | 1.0.0 | 2.0.0 |
| Docker Tag | latest | v2.0.0 |

The pipeline is designed to be **dataset-agnostic** - you only need to update the configuration files and retrain the models. The CI/CD, containerization, and deployment infrastructure remains the same.
