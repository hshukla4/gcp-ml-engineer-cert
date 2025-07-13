#!/bin/bash
# Save as install_packages.sh

echo "📦 Installing GCP ML packages step by step..."

# Core packages first
echo "1️⃣ Installing core GCP packages..."
pip install google-cloud-aiplatform==1.38.1
pip install google-cloud-bigquery==3.13.0
pip install google-cloud-storage==2.10.0

# Basic ML packages
echo "2️⃣ Installing ML libraries..."
pip install scikit-learn==1.3.2 pandas==2.0.3 numpy==1.24.3
pip install matplotlib==3.7.2 seaborn==0.12.2 xgboost==1.7.6

# TensorFlow
echo "3️⃣ Installing TensorFlow..."
pip install tensorflow==2.15.0

# Try pipeline components
echo "4️⃣ Attempting pipeline components..."
pip install kfp==2.3.0 --no-deps
pip install kfp-server-api==2.0.1
pip install google-cloud-pipeline-components==2.5.0

# Development tools
echo "5️⃣ Installing development tools..."
pip install jupyter notebook pytest black flake8

# Test installation
echo "✅ Testing installations..."
python << 'EOF'
import sys
print(f"Python: {sys.version}")

packages = {
    "Vertex AI": "from google.cloud import aiplatform",
    "BigQuery": "from google.cloud import bigquery",
    "Storage": "from google.cloud import storage",
    "TensorFlow": "import tensorflow as tf",
    "Scikit-learn": "import sklearn",
    "Pandas": "import pandas",
    "KFP": "import kfp"
}

for name, import_stmt in packages.items():
    try:
        exec(import_stmt)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {str(e)[:50]}")
EOF
