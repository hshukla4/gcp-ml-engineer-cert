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
