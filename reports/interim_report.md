# Interim Report: Insurance Risk Analytics & Predictive Modeling

## 1. Introduction
This interim report summarizes the exploratory data analysis (EDA) and preliminary modeling performed on the insurance dataset. The goal is to understand the data, visualize patterns, and prepare for predictive modeling.

## 2. Dataset Overview
- Dataset contains insurance information with columns:
  - `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`
- Loaded from: `data/raw/insurance.csv`
- Missing values: None

## 3. Data Cleaning & Encoding
- Binary columns (`sex`, `smoker`) were mapped to numeric:
  - `sex`: male → 0, female → 1
  - `smoker`: no → 0, yes → 1
- Multi-category column `region` was one-hot encoded.

## 4. Exploratory Data Analysis (EDA)
- Age distribution: histogram plotted
- Charges distribution: histogram plotted
- Pairplot: visualizing numeric column correlations
- Correlation heatmap created to check relationships
- Boxplots:
  - Charges vs Sex
  - Charges vs Smoker

## 5. Preliminary Modeling
- Split data into training (80%) and test (20%) sets
- Linear Regression model trained on all features
- Model performance on test set:
  - R² Score: `your_value_here`  
  - RMSE: `your_value_here`

## 6. Feature Importance
| Feature | Coefficient |
|---------|------------|
| `list features` | `list coefficients` |

## 7. Next Steps
- Explore advanced models: Random Forest, XGBoost
- Validate model performance using cross-validation
- Prepare the final report for submission
