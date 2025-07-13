#!/bin/bash

# Enhanced setup script with Python version handling

echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Function to compare versions
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Create virtual environment with appropriate Python version
if version_ge $PYTHON_VERSION "3.12"; then
    echo "⚠️  Python 3.12+ detected. Some pipeline components may not be compatible."
    echo "Creating environment with compatible packages only..."
    
    # Create requirements file for Python 3.12+
    cat > requirements_py312.txt << 'EOF'
# GCP Libraries - Core essentials
google-cloud-aiplatform>=1.38.0
google-cloud-bigquery>=3.13.0
google-cloud-storage>=2.10.0

# ML Libraries
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=1.7.0

# For neural networks, use JAX or PyTorch instead of TensorFlow if needed
torch>=2.0.0
jax[cpu]>=0.4.0

# Development Tools
jupyter>=1.0.0
notebook>=7.0.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
pyyaml>=6.0.0
tqdm>=4.65.0
requests>=2.31.0
EOF
    
    REQUIREMENTS_FILE="requirements_py312.txt"
else
    echo "✅ Python version compatible with all packages"
    REQUIREMENTS_FILE="requirements.txt"
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📦 Installing packages..."
pip install -r $REQUIREMENTS_FILE

# Install kfp separately with specific version for pipeline work
if version_ge $PYTHON_VERSION "3.12"; then
    echo "📝 Note: For pipeline components, consider using:"
    echo "   - Vertex AI SDK's built-in pipeline features"
    echo "   - Cloud Build for CI/CD"
    echo "   - Cloud Composer (Airflow) for orchestration"
else
    # Try to install pipeline components for older Python versions
    pip install kfp==2.3.0 --no-deps
    pip install kfp-server-api==2.0.1
fi

# Create a simple test script
cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
"""Test that all essential imports work"""

print("Testing GCP imports...")
try:
    from google.cloud import aiplatform
    print("✅ google.cloud.aiplatform")
except ImportError as e:
    print(f"❌ google.cloud.aiplatform: {e}")

try:
    from google.cloud import bigquery
    print("✅ google.cloud.bigquery")
except ImportError as e:
    print(f"❌ google.cloud.bigquery: {e}")

try:
    from google.cloud import storage
    print("✅ google.cloud.storage")
except ImportError as e:
    print(f"❌ google.cloud.storage: {e}")

print("\nTesting ML imports...")
try:
    import sklearn
    print("✅ scikit-learn")
except ImportError as e:
    print(f"❌ scikit-learn: {e}")

try:
    import pandas as pd
    print("✅ pandas")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import numpy as np
    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy: {e}")

print("\nEnvironment ready for GCP ML Engineer certification!")
EOF

# Run the test
echo ""
echo "🧪 Testing imports..."
python test_imports.py

# Create alternative pipeline script using Vertex AI SDK
cat > vertex_pipeline_example.py << 'EOF'
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
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Notes for Python 3.12+ users:"
echo "   - Some pipeline components (KFP, TFX) are not yet compatible"
echo "   - Use Vertex AI SDK directly for most tasks"
echo "   - Consider Cloud Build or Cloud Composer for complex pipelines"
echo ""
echo "🚀 Next steps:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Set up GCP auth: gcloud auth application-default login"
echo "   3. Try the example: python vertex_pipeline_example.py"
