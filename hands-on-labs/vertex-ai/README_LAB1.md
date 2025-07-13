# Lab 1: ML Problem Framing - Detailed Code Analysis

## Overview
This lab demonstrates the fundamental concept that **default ML models rarely meet business requirements**. Through practical implementation, we discover why hyperparameter tuning and advanced techniques are essential for production ML systems.

## Business Scenario
- **Problem**: Customer churn prediction for a subscription service
- **Business Goal**: Reduce churn rate by 20% (from 30% to 24%)
- **ML Translation**: Binary classification with focus on identifying potential churners
- **Critical Requirement**: Catch 85% of churners (85% recall)

## Code Structure

### 1. Problem Framing Section
```python
problem_definition = """
Business Problem: Reduce customer churn by 20%
ML Problem: Binary classification to predict churn probability
Success Metrics: 
  - Business: Reduce churn rate from 30% to 24%
  - ML: Achieve 85% recall for churned customers
  - ML: Maintain precision above 70%
"""
```

**Key Learning**: Always start with business metrics and translate them to ML metrics. The 85% recall requirement means we need to catch 85 out of every 100 actual churners.

### 2. Data Generation
```python
df = generate_synthetic_data(n_samples=5000, scenario="classification")
```

**Generated Features**:
- `customer_age`: Customer demographics (18-70 years)
- `account_length_days`: Customer tenure (30-3650 days)
- `monthly_charges`: Subscription cost (~$50 ± $20)
- `total_charges`: Lifetime value (~$1000 ± $500)
- `number_of_products`: Products subscribed (1-5)
- `tech_support_calls`: Support interactions (Poisson distributed)
- `churned`: Target variable (30% positive class)

**Data Characteristics**:
- 5000 samples total
- 70/30 class split (imbalanced, realistic for churn)
- No missing values (synthetic data advantage)
- Features have different scales (important for model selection)

### 3. Data Preparation
```python
X_train, X_test, y_train, y_test = prepare_data(df, "churned", test_size=0.2)
```

**Split Details**:
- Training: 4000 samples
- Testing: 1000 samples
- Stratified split maintains 30% churn rate in both sets

### 4. Model Training & Evaluation

#### Logistic Regression
```python
pipeline = get_classifier_pipeline("logistic_regression")
```
**Results**:
- Recall: 0.000 (Complete failure!)
- Precision: 0.000
- F1-Score: 0.000

**Why it failed**: The default LogisticRegression with unscaled, imbalanced data predicts all samples as non-churn (majority class).

#### Random Forest
```python
pipeline = get_classifier_pipeline("random_forest")
```
**Results**:
- Recall: 0.050 (Catches only 5% of churners)
- Precision: 0.375
- F1-Score: 0.089

**Why low performance**: Default parameters don't handle class imbalance. The model is conservative, predicting very few positive cases.

#### XGBoost
```python
pipeline = get_classifier_pipeline("xgboost")
```
**Results**:
- Recall: 0.134 (Catches 13.4% of churners)
- Precision: 0.317
- F1-Score: 0.189

**Best of three, but still fails**: Even the most sophisticated algorithm with default settings achieves only 13.4% recall vs. 85% requirement.

### 5. Business Impact Analysis

```python
if best_recall:
    print(f"\n✅ Recommended model: {best_model}")
else:
    print("\n⚠️  No model meets the business requirements. Consider:")
    print("  - Adjusting model hyperparameters")
    print("  - Using different algorithms")
    print("  - Engineering better features")
    print("  - Collecting more data")
```

**Critical Insight**: With 13.4% recall, we would miss 86.6% of churners. In business terms:
- If 300 customers would churn (out of 1000)
- We'd only identify 40 of them
- We'd lose 260 customers we could have saved
- Massive revenue impact!

## Key Learnings

### 1. The Recall-Precision Trade-off
- **High Recall**: Catch most churners but many false alarms
- **High Precision**: When we predict churn, we're usually right
- **Business Decision**: False negatives (missing churners) cost more than false positives (unnecessary retention offers)

### 2. Why Models Failed
1. **Class Imbalance**: 70/30 split biases models toward majority class
2. **Default Thresholds**: 0.5 threshold inappropriate for imbalanced data
3. **No Feature Scaling**: Logistic regression sensitive to scale
4. **Equal Class Weights**: Models don't know churners are more important

### 3. Solutions for Lab 2
- **Adjust Class Weights**: Make minority class more important
- **Threshold Tuning**: Lower threshold to catch more positives
- **SMOTE/Resampling**: Balance the training data
- **Ensemble Methods**: Combine models for better coverage
- **Custom Loss Functions**: Optimize directly for recall

## Code Flow Diagram

```
1. IMPORT & SETUP
   ├── Load utilities (gcp_utils, ml_utils, data_utils)
   ├── Initialize settings
   └── Set up Vertex AI connection

2. PROBLEM DEFINITION
   ├── Define business problem
   ├── Map to ML problem
   └── Set success metrics (85% recall)

3. DATA PREPARATION
   ├── Generate synthetic data (5000 samples)
   ├── Explore class distribution (30% churn)
   └── Split data (80/20)

4. MODEL TRAINING
   ├── Logistic Regression → 0% recall ❌
   ├── Random Forest → 5% recall ❌
   └── XGBoost → 13.4% recall ❌

5. EVALUATION
   ├── Compare against 85% requirement
   ├── All models fail
   └── Identify need for tuning

6. INSIGHTS
   └── Document solutions for Lab 2
```

## Visualization Evolution

The 10 PNG diagrams in `learning/diagrams/lab1/` show the iterative process:
1. **Initial attempts**: Messy arrows, overlapping boxes
2. **Improvements**: Better layout, clearer flow
3. **Final version**: Clean pipeline with proper visual hierarchy

This mirrors the learning process - understanding improves through iteration!

## Key Takeaways for Certification

1. **Always Start with Business Requirements**
   - Don't jump to model selection
   - Understand the cost of errors
   - Define success before coding

2. **Baseline Models Are Just That - Baselines**
   - Never deploy default parameters
   - Always check class distribution
   - Consider business impact of predictions

3. **Metrics Matter**
   - Accuracy is misleading for imbalanced data
   - Choose metrics that align with business goals
   - Monitor multiple metrics, optimize for one

4. **Vertex AI Considerations**
   - AutoML might handle imbalance better
   - Custom training gives more control
   - Model monitoring would catch this issue

## Next Steps

Lab 2 will implement solutions:
- Class weight adjustment
- Threshold optimization  
- Resampling techniques
- Ensemble methods
- Custom scoring functions

The goal: Transform the 13.4% recall baseline into a production-ready 85%+ recall model!