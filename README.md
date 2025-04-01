# Loan Default Prediction Model

This project aims to develop a predictive model for assessing the likelihood of loan defaults. Using a dataset that includes various borrower characteristics, the model leverages machine learning techniques to provide insights that can help financial institutions minimize risk, optimize lending strategies, and enhance decision-making.

[See Main Report Here!](loan_default_model_and_EDA.ipynb)

---

## Table of Contents
1. [Objective](#objective)
2. [Data Exploration](#data-exploration)
   - Snapshot of Loan Data
   - Target Variable Distribution
   - Age, Income, and Loan Amount Distributions
   - Home Ownership and Loan Percent Income Distributions
   - Loan Grade and Previous Defaults
3. [Modeling Process](#modeling-process)
   - Dataset Preparation
   - Feature Engineering
   - Model Selection
   - Model Finalization and Stacking
4. [Visualizations](#visualizations)
   - Confusion Matrix
   - Feature Importance
   - SHAP Analysis
5. [Results](#results)
6. [Files and Directory Structure](#files-and-directory-structure)

---

### 1. Objective <a name="objective"></a>

The primary goal is to build a robust loan default prediction model based on borrower demographics, loan features, and credit history. By identifying high-risk loans, the model can assist financial institutions in minimizing financial exposure to defaults.

---

### 2. Data Exploration <a name="data-exploration"></a>

This section examines the loan dataset, with each row representing an individual loan and associated borrower characteristics.

- **Snapshot of Loan Data**: Provides a quick overview of the variables, including the target `loan_status`, which represents whether a loan was defaulted or fully paid.
- **Target Variable Distribution**: The target is imbalanced, with approximately 78% of loans being fully paid and 22% defaulting.
- **Distribution Analyses**:
  - **Age**: Default rates are higher among younger borrowers, particularly those aged 22.
  - **Home Ownership**: Default rates are highest among renters.
  - **Income**: Income alone does not seem to predict loan default likelihood.
  - **Loan Amounts**: Higher loan amounts show a higher propensity for default.
  - **Interest Rates**: Loans with high-interest rates have a higher risk of default.
  - **Loan-to-Income Ratios**: High loan-to-income ratios are associated with increased default risk.
  - **Loan Grade and Previous Defaults**: Loans with lower grades and those with prior defaults show a clear pattern of higher risk.

---

### 3. Modeling Process <a name="modeling-process"></a>

#### 3.1 Dataset Preparation

Data preprocessing includes encoding categorical variables and handling imbalances using SMOTE to enhance model performance on the minority class (defaults).

#### 3.2 Feature Engineering

Features such as debt-to-income (DTI) ratios were calculated, along with standardized income and loan amount fields. These features help in capturing key financial characteristics of borrowers.

#### 3.3 Model Selection

We tested various classification models, including:
- **Traditional Models**: Logistic Regression, Decision Tree, and Random Forest.
- **Ensemble Models**: Gradient Boosting, Extra Trees, XGBoost, CatBoost, and LightGBM.
- **Stacking**: Combining predictions from multiple models for optimal accuracy.

#### 3.4 Model Finalization and Stacking

The best-performing model was a stacking classifier, combining top models with a logistic regression as the final estimator. This approach achieved a weighted average F1-score of 94%.

---

### 4. Visualizations <a name="visualizations"></a>

#### 4.1 Confusion Matrix

The confusion matrix visualization provides insight into the model's predictive performance, showing that while it performs well overall, there is a higher risk of false negatives.

#### 4.2 Feature Importance

Feature importance analysis highlights the most influential variables in predicting loan defaults. This analysis provides interpretable insights into which borrower characteristics are most impactful.

#### 4.3 SHAP Analysis

SHAP (SHapley Additive exPlanations) values are used to interpret model predictions, illustrating how individual features impact the probability of loan default.

---

### 5. Results <a name="results"></a>

The final model demonstrated strong performance, with a weighted average F1-score of 94%. Key findings include:
- Higher interest rates and larger loan amounts significantly increase default risk.
- Borrowers with prior defaults or lower loan grades are more prone to default.
- The stacking model is effective in minimizing risk and improving prediction accuracy.

---

### 6. Files and Directory Structure <a name="files-and-directory-structure"></a>

The project files are organized as follows:

```
├── data/
│   ├── credit_risk_dataset.csv          # Original dataset
│   ├── non_scaled.csv                   # Non-scaled dataset post feature engineering
│   ├── train.csv                        # Training dataset after SMOTE
│   └── test.csv                         # Test dataset after SMOTE
├── models/
│   └── stacking_model.pkl               # Final stacking model for deployment
├── charts/
│   ├── Age_Distribution.png
│   ├── Loan_Percent_Income_Distribution.png
│   ├── Target_Distribution.png
│   ├── Loan_Grade_Distribution.png
│   ├── Previous_Defaults_Distribution.png
│   ├── Home_Ownership_Distribution.png
│   ├── feature_importance.html
│   └── confusion_matrix.html
```

---

This README provides an overview and guidance for reproducing the loan default prediction model, including steps for data preprocessing, feature engineering, model selection, and final evaluation.
