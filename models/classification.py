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
