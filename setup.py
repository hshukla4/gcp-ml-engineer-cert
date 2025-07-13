"""
Setup configuration for GCP ML Engineer Certification project.
"""

from setuptools import setup, find_packages

setup(
    name="gcp-ml-engineer-cert",
    version="0.1.0",
    description="GCP ML Engineer Certification Study Project",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform>=1.38.0",
        "google-cloud-bigquery>=3.13.0",
        "google-cloud-storage>=2.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.11",
)
