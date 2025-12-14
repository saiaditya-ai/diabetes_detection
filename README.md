# Diabetes Prediction Using Machine Learning

## Problem Statement

Predict whether a patient has diabetes based on diagnostic measurements from the Pima Indians Diabetes Database. This is a binary classification problem where the goal is to identify individuals at risk of diabetes using medical features.

## Dataset

The dataset contains 768 observations with 8 medical predictor variables and 1 target variable:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (0 = No Diabetes, 1 = Diabetes)

## Approach

### 1. Statistical Foundation

- Correlation analysis to identify relationships between features
- Bayesian probability calculations
- T-tests to compare diabetic vs non-diabetic groups
- Probability distribution analysis using KDE plots
- Outlier detection using IQR method

### 2. Data Preprocessing

- Handled missing values (zeros) using KNN Imputation
- Removed outliers using z-score method (threshold > 3)
- Created engineered features:
  - BMI category
  - Glucose-to-insulin ratio
  - Age groups
- Applied StandardScaler for feature normalization
- Balanced classes using SMOTE oversampling

### 3. Model Training and Selection

Trained and compared 8 machine learning algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- XGBoost

Each model was optimized using GridSearchCV with 5-fold cross-validation to find the best hyperparameters. Models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC metrics.

### 4. Dimensionality Reduction

- Applied Principal Component Analysis (PCA) to understand feature variance
- Visualized data in 2D principal component space
- Analyzed cumulative explained variance

### 5. Model Evaluation and Fairness

- Generated confusion matrix and classification report
- Plotted ROC curve with AUC score
- Conducted fairness analysis across different age groups (TPR comparison)
- Analyzed feature importance for interpretability

## Results

The final model was selected based on F1-score and ROC-AUC performance. After comparing all 8 algorithms, XGBoost emerged as the best performing model, demonstrating superior ability to identify diabetes cases while maintaining optimal balance between precision and recall.

## Requirements

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- scipy

## Usage

Run the Jupyter notebook `diabetes-prediction-using-machine-learning.ipynb` to reproduce the analysis and results.
