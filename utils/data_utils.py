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
