#!/bin/bash
echo "🚀 GCP ML Engineer Cert - Quick Start"
echo "====================================="

# Check Python version
python --version

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements_essential.txt
fi

# Set up GCP project
echo ""
echo "📋 Next steps:"
echo "1. Set your GCP project: gcloud config set project YOUR_PROJECT_ID"
echo "2. Authenticate: gcloud auth application-default login"
echo "3. Run first lab: python hands-on-labs/vertex-ai/lab1_ml_problem_framing.py"
echo "4. Create diagram: python scripts/create_diagram.py --lab lab1"
