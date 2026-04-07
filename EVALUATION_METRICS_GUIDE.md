# 📊 Evaluation Metrics Guide

## Overview

Comprehensive evaluation metrics for both Classification and Regression models, including visualizations and interpretations.

---

## 🎯 Classification Metrics

### Available Metrics:

#### Basic Metrics:
1. **Accuracy** - Overall correctness
2. **Balanced Accuracy** - Accuracy adjusted for class imbalance
3. **Precision** - Positive prediction accuracy
4. **Recall** - True positive detection rate
5. **F1-Score** - Harmonic mean of precision and recall

#### Advanced Metrics:
6. **ROC-AUC** - Area under ROC curve
7. **Log Loss** - Probabilistic prediction quality
8. **Matthews Correlation Coefficient** - Balanced measure for imbalanced data
9. **Cohen's Kappa** - Agreement beyond chance

#### Multi-Class Specific:
- **Macro Average** - Unweighted mean across classes
- **Weighted Average** - Weighted by class support

---

### 📈 Classification Visualizations

#### 1. Confusion Matrix
- **What**: Shows actual vs predicted classifications
- **Interpretation**: 
  - Diagonal = Correct predictions
  - Off-diagonal = Errors
- **Interactive**: Hover for exact counts

#### 2. ROC Curve (Binary Classification)
- **What**: True Positive Rate vs False Positive Rate
- **Interpretation**:
  - AUC = 1.0: Perfect classifier
  - AUC = 0.5: Random classifier
  - AUC > 0.8: Good classifier
- **Use**: Compare model discrimination ability

#### 3. Precision-Recall Curve (Binary Classification)
- **What**: Precision vs Recall trade-off
- **Interpretation**:
  - High area: Good performance
  - Useful for imbalanced datasets
- **Use**: When false positives/negatives have different costs

---

### 📊 Metric Interpretations

#### Accuracy
```
Accuracy = (TP + TN) / Total
```
- **Range**: 0 to 1 (higher is better)
- **Good**: > 0.85
- **Use**: Balanced datasets
- **Caution**: Misleading with imbalanced data

#### Precision
```
Precision = TP / (TP + FP)
```
- **Range**: 0 to 1 (higher is better)
- **Meaning**: "Of all positive predictions, how many were correct?"
- **High Precision**: Few false positives
- **Use**: When false positives are costly (e.g., spam detection)

#### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- **Range**: 0 to 1 (higher is better)
- **Meaning**: "Of all actual positives, how many did we find?"
- **High Recall**: Few false negatives
- **Use**: When false negatives are costly (e.g., disease detection)

#### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Range**: 0 to 1 (higher is better)
- **Meaning**: Balanced measure of precision and recall
- **Good**: > 0.80
- **Use**: When you need balance between precision and recall

#### ROC-AUC
```
AUC = Area under ROC curve
```
- **Range**: 0 to 1 (higher is better)
- **Interpretation**:
  - 1.0: Perfect classifier
  - 0.9-1.0: Excellent
  - 0.8-0.9: Good
  - 0.7-0.8: Fair
  - 0.5-0.7: Poor
  - 0.5: Random
- **Use**: Overall model discrimination ability

#### Matthews Correlation Coefficient
```
MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
- **Range**: -1 to 1
- **Interpretation**:
  - 1: Perfect prediction
  - 0: Random prediction
  - -1: Total disagreement
- **Use**: Imbalanced datasets, more reliable than accuracy

#### Cohen's Kappa
```
κ = (Po - Pe) / (1 - Pe)
```
- **Range**: -1 to 1
- **Interpretation**:
  - < 0: No agreement
  - 0-0.20: Slight
  - 0.21-0.40: Fair
  - 0.41-0.60: Moderate
  - 0.61-0.80: Substantial
  - 0.81-1.00: Almost perfect
- **Use**: Measures agreement beyond chance

---

## 📊 Regression Metrics

### Available Metrics:

#### Error Metrics:
1. **MSE** - Mean Squared Error
2. **RMSE** - Root Mean Squared Error
3. **MAE** - Mean Absolute Error
4. **Median Absolute Error** - Robust to outliers
5. **MAPE** - Mean Absolute Percentage Error
6. **Max Error** - Worst case error

#### Performance Metrics:
7. **R² Score** - Proportion of variance explained
8. **Adjusted R²** - R² adjusted for number of features
9. **Explained Variance** - Variance explained by model

---

### 📈 Regression Visualizations

#### 1. Residual Plot
- **What**: Predicted values vs residuals (errors)
- **Interpretation**:
  - Random scatter around zero: Good model
  - Pattern: Model missing something
  - Funnel shape: Heteroscedasticity
- **Use**: Check model assumptions

#### 2. Actual vs Predicted
- **What**: Actual values vs predicted values
- **Interpretation**:
  - Points on diagonal: Perfect predictions
  - Scatter around diagonal: Prediction error
  - Systematic deviation: Bias
- **Use**: Visual assessment of prediction quality

#### 3. Error Distribution
- **What**: Histogram of residuals
- **Interpretation**:
  - Normal distribution: Good
  - Skewed: Systematic bias
  - Heavy tails: Outliers
- **Use**: Check normality assumption

---

### 📊 Metric Interpretations

#### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```
- **Range**: -∞ to 1 (higher is better)
- **Interpretation**:
  - 1.0: Perfect fit
  - 0.9-1.0: Excellent
  - 0.7-0.9: Good
  - 0.5-0.7: Moderate
  - < 0.5: Poor
  - < 0: Worse than mean
- **Meaning**: Proportion of variance explained

#### Adjusted R²
```
Adj R² = 1 - (1-R²)(n-1)/(n-p-1)
```
- **Range**: -∞ to 1
- **Meaning**: R² adjusted for number of features
- **Use**: Compare models with different numbers of features
- **Better than R²**: Penalizes unnecessary features

#### Mean Squared Error (MSE)
```
MSE = (1/n) Σ(y_true - y_pred)²
```
- **Range**: 0 to ∞ (lower is better)
- **Meaning**: Average squared error
- **Sensitive to**: Outliers (squared errors)
- **Units**: Squared units of target variable

#### Root Mean Squared Error (RMSE)
```
RMSE = sqrt(MSE)
```
- **Range**: 0 to ∞ (lower is better)
- **Meaning**: Average error magnitude
- **Advantage**: Same units as target variable
- **Use**: Most common regression metric

#### Mean Absolute Error (MAE)
```
MAE = (1/n) Σ|y_true - y_pred|
```
- **Range**: 0 to ∞ (lower is better)
- **Meaning**: Average absolute error
- **Robust to**: Outliers (no squaring)
- **Interpretation**: Average prediction error

#### Mean Absolute Percentage Error (MAPE)
```
MAPE = (100/n) Σ|y_true - y_pred|/|y_true|
```
- **Range**: 0 to ∞ (lower is better)
- **Units**: Percentage
- **Interpretation**:
  - < 10%: Excellent
  - 10-20%: Good
  - 20-50%: Acceptable
  - > 50%: Poor
- **Caution**: Undefined when y_true = 0

#### Median Absolute Error
```
MedAE = median(|y_true - y_pred|)
```
- **Range**: 0 to ∞ (lower is better)
- **Meaning**: Median of absolute errors
- **Robust to**: Outliers
- **Use**: When outliers are present

---

## 🎯 Choosing the Right Metric

### For Classification:

| Scenario | Recommended Metric |
|----------|-------------------|
| **Balanced classes** | Accuracy, F1-Score |
| **Imbalanced classes** | Balanced Accuracy, MCC, Cohen's Kappa |
| **False positives costly** | Precision, Specificity |
| **False negatives costly** | Recall, Sensitivity |
| **Need balance** | F1-Score |
| **Probability predictions** | ROC-AUC, Log Loss |
| **Multi-class** | Macro/Weighted F1, ROC-AUC (OvR) |

### For Regression:

| Scenario | Recommended Metric |
|----------|-------------------|
| **General use** | RMSE, R² |
| **Outliers present** | MAE, Median Absolute Error |
| **Percentage errors** | MAPE |
| **Compare models** | Adjusted R² |
| **Interpretability** | MAE (same units as target) |
| **Penalize large errors** | MSE, RMSE |

---

## 📊 Example Interpretations

### Classification Example:

```
Model: Random Forest Classifier
Dataset: Email Spam Detection (1000 samples)

Metrics:
- Accuracy: 0.95 (95% correct)
- Precision: 0.93 (93% of spam predictions are correct)
- Recall: 0.97 (97% of actual spam is detected)
- F1-Score: 0.95 (balanced performance)
- ROC-AUC: 0.98 (excellent discrimination)

Confusion Matrix:
              Predicted
              Not Spam  Spam
Actual Not Spam   580     20
       Spam        30    370

Interpretation:
✅ Excellent overall performance (95% accuracy)
✅ High recall (97%) - catches most spam
✅ Good precision (93%) - few false positives
⚠️ 20 legitimate emails marked as spam (FP)
⚠️ 30 spam emails missed (FN)

Recommendation: Production-ready, but monitor false positives
```

### Regression Example:

```
Model: Random Forest Regressor
Dataset: House Price Prediction (500 samples)

Metrics:
- R²: 0.87 (87% variance explained)
- Adjusted R²: 0.85
- RMSE: $25,000 (average error)
- MAE: $18,000 (typical error)
- MAPE: 8.5% (percentage error)

Residual Plot: Random scatter around zero ✅
Actual vs Predicted: Points close to diagonal ✅
Error Distribution: Approximately normal ✅

Interpretation:
✅ Strong predictive power (R² = 0.87)
✅ Low percentage error (MAPE = 8.5%)
✅ Typical prediction off by $18,000
✅ Model assumptions satisfied

Recommendation: Excellent model, ready for deployment
```

---

## 🎓 Best Practices

### 1. Use Multiple Metrics
- Don't rely on a single metric
- Different metrics reveal different aspects
- Consider domain-specific requirements

### 2. Understand Trade-offs
- Precision vs Recall
- Bias vs Variance
- Complexity vs Interpretability

### 3. Consider Context
- Cost of errors
- Class imbalance
- Business requirements
- Regulatory constraints

### 4. Validate Properly
- Use cross-validation
- Test on holdout set
- Check for overfitting
- Validate assumptions

### 5. Visualize Results
- Confusion matrix
- ROC/PR curves
- Residual plots
- Error distributions

---

## 🚀 Using the Evaluation Metrics

### In the Application:

1. **Train Models** in Model Training section
2. **View Results** in Model Performance table
3. **Select Model** for detailed evaluation
4. **Review Metrics** in Comprehensive Evaluation section
5. **Analyze Visualizations** (ROC, PR, Residuals)
6. **Compare Models** using different metrics
7. **Download Best Model** for deployment

### Workflow:

```
1. Train multiple models
   ↓
2. Compare basic metrics (Accuracy/R²)
   ↓
3. Select top 2-3 models
   ↓
4. Review comprehensive metrics
   ↓
5. Analyze visualizations
   ↓
6. Consider business requirements
   ↓
7. Choose final model
   ↓
8. Download and deploy
```

---

## 📚 Metric Formulas Reference

### Classification:

| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) |
| Specificity | TN / (TN + FP) |
| FPR | FP / (FP + TN) |
| FNR | FN / (FN + TP) |

### Regression:

| Metric | Formula |
|--------|---------|
| MSE | (1/n) Σ(y - ŷ)² |
| RMSE | √MSE |
| MAE | (1/n) Σ\|y - ŷ\| |
| MAPE | (100/n) Σ\|y - ŷ\|/\|y\| |
| R² | 1 - SS_res/SS_tot |
| Adj R² | 1 - (1-R²)(n-1)/(n-p-1) |

---

## ✅ Summary

### Classification Metrics:
✅ **10+ metrics** for comprehensive evaluation  
✅ **3 visualizations** (Confusion Matrix, ROC, PR)  
✅ **Binary & Multi-class** support  
✅ **Imbalanced data** metrics included  

### Regression Metrics:
✅ **9+ metrics** covering all aspects  
✅ **3 visualizations** (Residuals, Actual vs Predicted, Distribution)  
✅ **Robust metrics** for outliers  
✅ **Percentage errors** for interpretability  

### Features:
✅ **Interactive visualizations** with Plotly  
✅ **Model selection** for detailed analysis  
✅ **Comprehensive interpretation** guide  
✅ **Production-ready** evaluation  

---

**Use comprehensive evaluation metrics to make informed model selection decisions!** 📊✨
