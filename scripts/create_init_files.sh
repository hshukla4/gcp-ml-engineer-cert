#!/bin/bash
# Script to create __init__.py files for all Python modules in the project

echo "📦 Creating __init__.py files for all modules..."

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "Working in project root: $PROJECT_ROOT"

# [Rest of the script content from the artifact above]
# ... (all the content)
