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
