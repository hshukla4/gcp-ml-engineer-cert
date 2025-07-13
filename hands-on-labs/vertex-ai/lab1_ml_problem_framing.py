#!/usr/bin/env python3
"""
Lab 1: ML Problem Framing with Vertex AI
Key concepts for GCP ML Engineer Certification
"""

from utils.gcp_utils import init_vertex_ai
from utils.data_utils import generate_synthetic_data
from utils.ml_utils import prepare_data, evaluate_model
from models.classification import get_classifier_pipeline
from config import get_settings

# Initialize settings
settings = get_settings()

# ===== PART 1: ML PROBLEM FRAMING =====
print("🎯 ML Problem Framing")
print("=" * 50)

# Scenario: Customer Churn Prediction
problem_definition = """
Business Problem: Reduce customer churn by 20%
ML Problem: Binary classification to predict churn probability
Success Metrics: 
  - Business: Reduce churn rate from 30% to 24%
  - ML: Achieve 85% recall for churned customers
  - ML: Maintain precision above 70%
"""
print(problem_definition)

# ===== PART 2: DATA PREPARATION =====
print("\n📊 Data Preparation")
print("=" * 50)

# Generate synthetic customer data
df = generate_synthetic_data(n_samples=5000, scenario="classification")
print(f"Dataset shape: {df.shape}")
print(f"\nFeatures: {list(df.drop('churned', axis=1).columns)}")
print(f"Target: churned")
print(f"\nClass distribution:")
print(df['churned'].value_counts(normalize=True))

# ===== PART 3: VERTEX AI INITIALIZATION =====
print("\n☁️ Vertex AI Setup")
print("=" * 50)

try:
    project_id, location = init_vertex_ai(settings.project_id, settings.region)
    print(f"✅ Connected to Vertex AI")
except Exception as e:
    print(f"⚠️  Vertex AI initialization failed: {e}")
    print("Continue with local training...")

# ===== PART 4: MODEL TRAINING =====
print("\n🤖 Model Training")
print("=" * 50)

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(df, "churned")

# Train multiple models for comparison
models_to_test = ["logistic_regression", "random_forest", "xgboost"]

results = {}
for model_type in models_to_test:
    print(f"\nTraining {model_type}...")
    pipeline = get_classifier_pipeline(model_type)
    pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import recall_score, precision_score, f1_score
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[model_type] = {
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
    
    print(f"  Recall: {recall:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  F1-Score: {f1:.3f}")

# ===== PART 5: MODEL SELECTION =====
print("\n🏆 Model Selection")
print("=" * 50)

# Check which model meets our business requirements
print("Business requirement: 85% recall for churned customers")
print("\nModel evaluation:")

best_model = None
best_recall = 0

for model_name, metrics in results.items():
    meets_recall = "✅" if metrics['recall'] >= 0.85 else "❌"
    meets_precision = "✅" if metrics['precision'] >= 0.70 else "❌"
    
    print(f"\n{model_name}:")
    print(f"  Recall: {metrics['recall']:.3f} {meets_recall}")
    print(f"  Precision: {metrics['precision']:.3f} {meets_precision}")
    
    if metrics['recall'] >= 0.85 and metrics['recall'] > best_recall:
        best_model = model_name
        best_recall = metrics['recall']

if best_model:
    print(f"\n✅ Recommended model: {best_model}")
else:
    print("\n⚠️  No model meets the business requirements. Consider:")
    print("  - Adjusting model hyperparameters")
    print("  - Using different algorithms")
    print("  - Engineering better features")
    print("  - Collecting more data")

# ===== EXAM TIPS =====
print("\n📚 Exam Tips")
print("=" * 50)
print("1. Always start with business metrics, then map to ML metrics")
print("2. Consider the cost of false positives vs false negatives")
print("3. For churn: false negatives (missing churners) are usually more costly")
print("4. Vertex AI services to consider:")
print("   - AutoML: When you have limited ML expertise")
print("   - Custom Training: When you need full control")
print("   - Model Registry: For model versioning and deployment")
