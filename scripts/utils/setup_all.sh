#!/bin/bash
# Master setup script that calls other scripts in order

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 GCP ML Engineer Certification - Complete Setup"
echo "Project root: $PROJECT_ROOT"
echo "Scripts directory: $SCRIPT_DIR"

cd "$PROJECT_ROOT"

# Step 1: Initialize GCP project
if [ -f "$SCRIPT_DIR/setup/init_project.sh" ]; then
    echo "1️⃣ Initializing GCP project..."
    bash "$SCRIPT_DIR/setup/init_project.sh"
else
    echo "⚠️  init_project.sh not found in scripts/setup/"
fi

# Step 2: Setup Python environment
if [ -f "$SCRIPT_DIR/setup/setup_env.sh" ]; then
    echo "2️⃣ Setting up Python environment..."
    bash "$SCRIPT_DIR/setup/setup_env.sh"
else
    echo "⚠️  setup_env.sh not found in scripts/setup/"
fi

# Step 3: Install packages step by step
if [ -f "$SCRIPT_DIR/install_packages.sh" ]; then
    echo "3️⃣ Installing packages..."
    bash "$SCRIPT_DIR/install_packages.sh"
else
    echo "⚠️  install_packages.sh not found"
fi

echo "✅ Setup complete!"
