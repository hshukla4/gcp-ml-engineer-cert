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
