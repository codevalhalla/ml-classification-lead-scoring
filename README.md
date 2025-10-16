# Lead Scoring Classification

## Project Overview

This project implements a lead scoring classification system to predict whether a lead will convert into a customer. It uses logistic regression to analyze lead characteristics and determine conversion probability based on features like lead source, industry, interaction count, and more.

## Dataset

The project uses `course_lead_scoring.csv` with the following features:

- **lead_source**: Channel through which the lead was acquired
- **industry**: Industry classification (retail, finance, technology, healthcare, education, manufacturing, other)
- **number_of_courses_viewed**: Count of courses viewed by the lead
- **annual_income**: Annual income of the lead
- **employment_status**: Employment status classification
- **location**: Geographic location of the lead
- **interaction_count**: Number of interactions with the company
- **lead_score**: Pre-existing lead score
- **converted**: Target variable (1 = converted, 0 = not converted)

### Data Statistics

- Total records: 1,462
- Missing values handled: Categorical columns filled with 'NA', numerical columns filled with 0

## Key Findings

### Data Quality
- **lead_source**: 128 missing values
- **industry**: 134 missing values  
- **annual_income**: 181 missing values
- **employment_status**: 100 missing values
- **location**: 63 missing values

### Feature Importance (Mutual Information Score)
- **lead_source**: 0.04 (highest relevance)
- **employment_status**: 0.01
- **industry**: 0.01
- **location**: 0.00 (lowest relevance)

### Feature Correlation with Lead Score
- **interaction_count**: 0.01 (very weak correlation)
- **number_of_courses_viewed**: -0.00 (negligible correlation)

## Methodology

### 1. Data Preprocessing
- Handled missing values using appropriate strategies for data types
- No outlier removal or scaling applied

### 2. Train-Validation-Test Split
- 80% training data, 20% test data (random_state=42)
- Training data further split: 75% train, 25% validation

### 3. Feature Engineering
- Converted dataframes to dictionary format
- Used `DictVectorizer` for one-hot encoding of categorical features

### 4. Model: Logistic Regression
- **Solver**: liblinear
- **Max iterations**: 1000
- **Regularization strengths tested**: 0.01, 0.1, 1, 10, 100

## Results

### Model Performance
- **Validation Accuracy**: 69.97%
- **Optimal Regularization Strength**: All values (0.01 to 100) performed equally with 69.97% accuracy

### Feature Elimination Analysis
- Removing **industry**: No accuracy change (0% drop)
- Removing **employment_status**: 0.34% accuracy drop
- Removing **lead_score**: 0.68% improvement in accuracy

**Insight**: Lead_score feature had a negative impact on model performance when included, suggesting possible multicollinearity or redundancy.

## Usage

```python
# Import and load data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# Load dataset
df = pd.read_csv("./course_lead_scoring.csv")

# Preprocess data
# ... (refer to code for preprocessing steps)

# Train model
model = LogisticRegression(solver='liblinear', C=1, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict_proba(X_val)[:, 1]
```

## Dependencies

- pandas
- numpy
- scikit-learn

Install via:
```bash
pip install pandas numpy scikit-learn
```

## Files

- `course_lead_scoring.csv`: Input dataset
- `lead_scoring_classification.py`: Main analysis script (or Jupyter notebook)

## Future Improvements

- Experiment with alternative classification algorithms (Random Forest, XGBoost, SVM)
- Implement cross-validation for more robust accuracy estimates
- Address class imbalance if present
- Explore feature engineering and interaction terms
- Tune hyperparameters more systematically
- Implement ensemble methods
- Add feature scaling if using distance-based models

## Notes

- The current model achieves ~70% accuracy with minimal hyperparameter tuning
- Regularization strength showed no significant impact on performance
- The feature elimination technique revealed that lead_score may be introducing noise
- Consider domain expertise to validate feature importance rankings
