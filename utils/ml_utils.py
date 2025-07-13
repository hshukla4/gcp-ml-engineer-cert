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
