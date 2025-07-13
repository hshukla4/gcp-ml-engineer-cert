#!/bin/bash
# Script to create __init__.py files for all Python modules in the project

echo "📦 Creating __init__.py files for all modules..."

# Get the project root directory
PROJECT_ROOT="$(pwd)"

# Create __init__.py for main project
cat > __init__.py << 'EOF'
"""
GCP ML Engineer Certification Study Project

A comprehensive study repository for Google Cloud Professional Machine Learning Engineer certification.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Make key modules easily accessible
from . import utils
from . import config

__all__ = ["utils", "config"]
EOF

# Create utils module
mkdir -p utils
cat > utils/__init__.py << 'EOF'
"""
Utility functions for GCP ML Engineer certification project.
"""

from .gcp_utils import *
from .ml_utils import *
from .data_utils import *

__all__ = ["gcp_utils", "ml_utils", "data_utils"]
EOF

# Create utility submodules
cat > utils/gcp_utils.py << 'EOF'
"""
Google Cloud Platform utility functions.
"""

from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage
import os


def init_vertex_ai(project_id: str = None, location: str = "us-central1"):
    """Initialize Vertex AI with project and location."""
    if not project_id:
        project_id = os.environ.get("GCP_PROJECT_ID", "ml-engineer-study-462417")
    
    aiplatform.init(project=project_id, location=location)
    print(f"✅ Vertex AI initialized for project: {project_id}, location: {location}")
    return project_id, location


def get_gcs_bucket(bucket_name: str = None, project_id: str = None):
    """Get or create a GCS bucket."""
    if not project_id:
        project_id = os.environ.get("GCP_PROJECT_ID", "ml-engineer-study-462417")
    if not bucket_name:
        bucket_name = f"{project_id}-ml-artifacts"
    
    client = storage.Client(project=project_id)
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"✅ Using existing bucket: {bucket_name}")
    except:
        bucket = client.create_bucket(bucket_name, location="us-central1")
        print(f"✅ Created new bucket: {bucket_name}")
    
    return bucket


def get_bigquery_client(project_id: str = None):
    """Get BigQuery client."""
    if not project_id:
        project_id = os.environ.get("GCP_PROJECT_ID", "ml-engineer-study-462417")
    
    client = bigquery.Client(project=project_id)
    print(f"✅ BigQuery client initialized for project: {project_id}")
    return client
EOF

cat > utils/ml_utils.py << 'EOF'
"""
Machine Learning utility functions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Prepare data for ML training."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)
    
    print(f"\n{model_name} Evaluation:")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return y_pred


def plot_feature_importance(model, feature_names, top_n: int = 10):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
EOF

cat > utils/data_utils.py << 'EOF'
"""
Data processing utility functions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_data_from_gcs(bucket_name: str, file_path: str) -> pd.DataFrame:
    """Load data from Google Cloud Storage."""
    gcs_path = f"gs://{bucket_name}/{file_path}"
    df = pd.read_csv(gcs_path)
    print(f"✅ Loaded {len(df)} rows from {gcs_path}")
    return df


def generate_synthetic_data(n_samples: int = 1000, scenario: str = "classification") -> pd.DataFrame:
    """Generate synthetic data for testing."""
    np.random.seed(42)
    
    if scenario == "classification":
        # Customer churn scenario
        data = {
            'customer_age': np.random.randint(18, 70, n_samples),
            'account_length_days': np.random.randint(30, 3650, n_samples),
            'monthly_charges': np.random.normal(50, 20, n_samples),
            'total_charges': np.random.normal(1000, 500, n_samples),
            'number_of_products': np.random.randint(1, 5, n_samples),
            'tech_support_calls': np.random.poisson(2, n_samples),
            'churned': np.random.binomial(1, 0.3, n_samples)
        }
    elif scenario == "regression":
        # Sales prediction scenario
        data = {
            'store_size': np.random.randint(1000, 10000, n_samples),
            'staff_count': np.random.randint(5, 50, n_samples),
            'marketing_spend': np.random.normal(5000, 2000, n_samples),
            'competition_distance': np.random.normal(5, 2, n_samples),
            'sales': np.random.normal(50000, 15000, n_samples)
        }
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {n_samples} samples for {scenario}")
    return df


def create_time_series_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Create time series features from a date column."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract time features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    
    print(f"✅ Created time series features from {date_column}")
    return df
EOF

# Create config module
mkdir -p config
cat > config/__init__.py << 'EOF'
"""
Configuration management for GCP ML Engineer certification project.
"""

from .settings import *

__all__ = ["Settings", "get_settings"]
EOF

cat > config/settings.py << 'EOF'
"""
Project settings and configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Project settings."""
    # GCP Settings
    project_id: str = "ml-engineer-study-462417"
    region: str = "us-central1"
    zone: str = "us-central1-a"
    
    # Storage Settings
    bucket_name: Optional[str] = None
    
    # BigQuery Settings
    bq_dataset: str = "ml_datasets"
    
    # Model Settings
    model_artifact_path: str = "models"
    pipeline_root: Optional[str] = None
    
    def __post_init__(self):
        """Set derived settings."""
        if not self.bucket_name:
            self.bucket_name = f"{self.project_id}-ml-artifacts"
        if not self.pipeline_root:
            self.pipeline_root = f"gs://{self.bucket_name}/pipelines"
        
        # Override from environment variables
        self.project_id = os.getenv("GCP_PROJECT_ID", self.project_id)
        self.region = os.getenv("GCP_REGION", self.region)


_settings = None


def get_settings() -> Settings:
    """Get singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
EOF

# Create models module
mkdir -p models
cat > models/__init__.py << 'EOF'
"""
Machine Learning models for various scenarios.
"""

from .classification import *
from .regression import *
from .clustering import *

__all__ = ["classification", "regression", "clustering"]
EOF

cat > models/classification.py << 'EOF'
"""
Classification models and pipelines.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def get_classifier_pipeline(model_type: str = "random_forest"):
    """Get a classification pipeline with preprocessing."""
    
    # Preprocessor
    preprocessor = StandardScaler()
    
    # Model selection
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(random_state=42),
        "xgboost": xgb.XGBClassifier(n_estimators=100, random_state=42)
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(models.keys())}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', models[model_type])
    ])
    
    return pipeline
EOF

cat > models/regression.py << 'EOF'
"""
Regression models and pipelines.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def get_regressor_pipeline(model_type: str = "random_forest"):
    """Get a regression pipeline with preprocessing."""
    
    # Preprocessor
    preprocessor = StandardScaler()
    
    # Model selection
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "xgboost": xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(models.keys())}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', models[model_type])
    ])
    
    return pipeline
EOF

cat > models/clustering.py << 'EOF'
"""
Clustering models and analysis.
"""

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_clustering_pipeline(model_type: str = "kmeans", n_clusters: int = 3):
    """Get a clustering pipeline with preprocessing."""
    
    # Preprocessor
    preprocessor = StandardScaler()
    
    # Model selection
    models = {
        "kmeans": KMeans(n_clusters=n_clusters, random_state=42),
        "dbscan": DBSCAN(eps=0.5, min_samples=5)
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(models.keys())}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clustering', models[model_type])
    ])
    
    return pipeline
EOF

# Create tests module
mkdir -p tests
cat > tests/__init__.py << 'EOF'
"""
Tests for GCP ML Engineer certification project.
"""
EOF

cat > tests/test_utils.py << 'EOF'
"""
Test utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from utils.data_utils import generate_synthetic_data
from utils.ml_utils import prepare_data


def test_generate_synthetic_data():
    """Test synthetic data generation."""
    # Test classification data
    df_class = generate_synthetic_data(n_samples=100, scenario="classification")
    assert len(df_class) == 100
    assert "churned" in df_class.columns
    
    # Test regression data
    df_reg = generate_synthetic_data(n_samples=100, scenario="regression")
    assert len(df_reg) == 100
    assert "sales" in df_reg.columns


def test_prepare_data():
    """Test data preparation."""
    df = generate_synthetic_data(n_samples=100, scenario="classification")
    X_train, X_test, y_train, y_test = prepare_data(df, "churned", test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
EOF

# Create hands-on-labs __init__.py files
for lab_dir in hands-on-labs/*; do
    if [ -d "$lab_dir" ]; then
        echo "Creating __init__.py for $lab_dir"
        touch "$lab_dir/__init__.py"
    fi
done

# Create projects __init__.py files
for project_dir in projects/*; do
    if [ -d "$project_dir" ]; then
        echo "Creating __init__.py for $project_dir"
        cat > "$project_dir/__init__.py" << 'EOF'
"""
ML project module.
"""
EOF
    fi
done

# Create .env.example file
cat > .env.example << 'EOF'
# GCP Configuration
GCP_PROJECT_ID=ml-engineer-study-462417
GCP_REGION=us-central1
GCP_ZONE=us-central1-a

# Storage Configuration
GCS_BUCKET_NAME=${GCP_PROJECT_ID}-ml-artifacts

# BigQuery Configuration
BQ_DATASET=ml_datasets

# API Keys (if needed)
# OPENAI_API_KEY=your_key_here
EOF

# Create setup.py for the project
cat > setup.py << 'EOF'
"""
Setup configuration for GCP ML Engineer Certification project.
"""

from setuptools import setup, find_packages

setup(
    name="gcp-ml-engineer-cert",
    version="0.1.0",
    description="GCP ML Engineer Certification Study Project",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform>=1.38.0",
        "google-cloud-bigquery>=3.13.0",
        "google-cloud-storage>=2.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.11",
)
EOF

echo "✅ All __init__.py files and module structure created!"
echo ""
echo "📁 Project structure is now properly organized as Python packages"
echo ""
echo "You can now use imports like:"
echo "  from utils.gcp_utils import init_vertex_ai"
echo "  from utils.ml_utils import evaluate_model"
echo "  from models.classification import get_classifier_pipeline"
echo "  from config import get_settings"
