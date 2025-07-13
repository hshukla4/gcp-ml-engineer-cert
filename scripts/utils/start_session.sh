#!/bin/bash
# Quick start script for daily study sessions

echo "🎓 Starting GCP ML Engineer Study Session"
echo "========================================"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    echo "   Run: python3.11 -m venv venv"
    exit 1
fi

# Set GCP project
PROJECT_ID="ml-engineer-study-462417"
gcloud config set project $PROJECT_ID
echo "✅ GCP project set to: $PROJECT_ID"

# Quick verification
echo ""
echo "Quick check:"
python -c "
from google.cloud import aiplatform
print('✅ Vertex AI SDK loaded')
print(f'   Project: {aiplatform.init.project}')
" 2>/dev/null || echo "❌ Vertex AI SDK not working"

# Show today's study suggestions
echo ""
echo "📚 Today's study suggestions:"
echo "1. Run: jupyter notebook"
echo "2. Open: hands-on-labs/vertex-ai/01_first_pipeline.py"
echo "3. Review: study-notes/01-ml-problem-framing/"
echo ""
echo "Happy studying! 🚀"
