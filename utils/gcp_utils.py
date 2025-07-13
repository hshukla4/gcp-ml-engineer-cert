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
