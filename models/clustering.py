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
