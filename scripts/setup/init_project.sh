#!/bin/bash

# Project initialization script
PROJECT_ID="ml-engineer-study-462417"
REGION="us-central1"
BUCKET_NAME="${PROJECT_ID}-ml-artifacts"

echo "�� Initializing GCP ML Engineer Study Project..."

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📡 Enabling required APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    compute.googleapis.com \
    dataflow.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    notebooks.googleapis.com \
    storage-component.googleapis.com

# Create storage bucket
echo "🪣 Creating storage bucket..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME/

# Create BigQuery dataset
echo "📊 Creating BigQuery datasets..."
bq mk -d --location=$REGION --project_id=$PROJECT_ID ml_datasets
bq mk -d --location=$REGION --project_id=$PROJECT_ID ml_features

echo "✅ Project initialization complete!"
