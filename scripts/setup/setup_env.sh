#!/bin/bash

# Environment setup script
echo "🔧 Setting up Python environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

echo "✅ Environment setup complete!"
echo "Activate with: source venv/bin/activate"
