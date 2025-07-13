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
