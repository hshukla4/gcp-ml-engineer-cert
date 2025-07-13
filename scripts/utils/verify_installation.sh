#!/bin/bash
# Verify all installations are working

echo "🔍 Verifying GCP ML Engineer Setup"
echo "=================================="

# Check Python version
echo -n "Python version: "
python --version

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "❌ No virtual environment active"
    echo "   Run: source venv/bin/activate"
fi

# Test imports
echo ""
echo "Testing package imports..."
python << 'EOF'
import importlib
import sys

packages = [
    ("google.cloud.aiplatform", "Vertex AI SDK"),
    ("google.cloud.bigquery", "BigQuery"),
    ("google.cloud.storage", "Cloud Storage"),
    ("sklearn", "Scikit-learn"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("tensorflow", "TensorFlow"),
    ("matplotlib", "Matplotlib"),
    ("jupyter", "Jupyter"),
]

print("\nPackage Status:")
print("-" * 40)

working = []
failed = []

for module_name, display_name in packages:
    try:
        if '.' in module_name:
            parts = module_name.split('.')
            module = importlib.import_module(parts[0])
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = importlib.import_module(module_name)
        
        version = getattr(module, '__version__', 'unknown')
        if module_name == 'google.cloud.aiplatform':
            from google.cloud import aiplatform
            version = aiplatform.__version__
        
        print(f"✅ {display_name:<20} {version}")
        working.append(display_name)
    except ImportError as e:
        print(f"❌ {display_name:<20} Not installed")
        failed.append(display_name)
    except Exception as e:
        print(f"⚠️  {display_name:<20} Error: {str(e)[:30]}")
        failed.append(display_name)

print("-" * 40)
print(f"\nSummary: {len(working)} working, {len(failed)} failed")

if failed:
    print(f"\nFailed packages: {', '.join(failed)}")
    print("\nTo fix, try:")
    print("pip install google-cloud-aiplatform google-cloud-bigquery google-cloud-storage")
