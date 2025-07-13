"""
Alternative pipeline example using Vertex AI SDK directly
(Works with Python 3.12+)
"""

from google.cloud import aiplatform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

PROJECT_ID = "ml-engineer-study-462417"
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-ml-artifacts"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

def create_and_train_model():
    """Simple model training without KFP"""
    
    # Create sample data
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.3f}")
    
    # Save model locally
    joblib.dump(model, "model.pkl")
    
    # Upload to Vertex AI
    vertex_model = aiplatform.Model.upload(
        display_name="iris-classifier",
        artifact_uri=f"gs://{BUCKET_NAME}/models/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    )
    
    return vertex_model

if __name__ == "__main__":
    print("Training model with Vertex AI SDK...")
    model = create_and_train_model()
    print(f"Model uploaded: {model.display_name}")
