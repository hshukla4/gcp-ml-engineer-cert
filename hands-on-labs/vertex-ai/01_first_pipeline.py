"""
Vertex AI Pipeline Lab: Build Your First ML Pipeline
This lab demonstrates how to create a simple ML pipeline using Vertex AI
"""

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import kfp
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

# Initialize Vertex AI
PROJECT_ID = "ml-engineer-study-462417"
REGION = "us-central1"
PIPELINE_ROOT = f"gs://{PROJECT_ID}-ml-artifacts/pipeline-root"

aiplatform.init(project=PROJECT_ID, location=REGION)

@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.9"
)
def load_data(dataset: Output[Dataset]):
    """Component to load and prepare data"""
    import pandas as pd
    from sklearn.datasets import load_iris
    
    # Load iris dataset as example
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Save to output
    df.to_csv(dataset.path, index=False)

@component(
    packages_to_install=["pandas", "scikit-learn"],
    base_image="python:3.9"
)
def train_model(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics]
):
    """Component to train a model"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    
    # Load data
    df = pd.read_csv(dataset.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    metrics.log_metric("accuracy", accuracy)
    
    # Save model
    joblib.dump(clf, model.path)

@dsl.pipeline(
    name="iris-classification-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def ml_pipeline():
    """Define the pipeline"""
    # Load data
    load_data_op = load_data()
    
    # Train model
    train_model_op = train_model(
        dataset=load_data_op.outputs["dataset"]
    )

# Compile and run pipeline
if __name__ == "__main__":
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="iris_pipeline.json"
    )
    
    # Create pipeline job
    job = pipeline_jobs.PipelineJob(
        display_name="iris-classification-pipeline",
        template_path="iris_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=True,
    )
    
    # Submit pipeline job
    job.submit()
