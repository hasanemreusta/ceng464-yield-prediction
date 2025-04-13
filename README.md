# ceng464-yield-prediction
# 🌾 Grain Yield Prediction with Machine Learning

This project compares the performance of various classification algorithms for predicting grain yield based on soil and sowing-related data.

> 📍 Developed as part of the **CENG464** course project at Çankaya University.

---

## 📊 Overview

Grain yield prediction is essential for sustainable agriculture. In this project, we applied several machine learning classification algorithms to predict grain yield categories (A, B, C) based on processed agricultural data.

### 🛠️ Machine Learning Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- Artificial Neural Networks (ANN)
- Random Forest
- Gradient Boosting
- AdaBoost
- Support Vector Machines (SVM)
- XGBoost

---

## 🧪 Methodology

1. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical variables
   - Standardized features

2. **Feature Selection**
   - SelectKBest
   - Recursive Feature Elimination (RFE)

3. **Model Evaluation**
   - Accuracy, F1 Score, Precision, Recall, MCC, ROC-AUC
   - ROC Curves & Confusion Matrices

---

## 📁 Files

- `CENG464 Project - Python Files.py`:  
   This Python script includes both **Hasan's** and **Gökay's** contributions merged into a single file.  
   - **Hasan's part**: Uses Recursive Feature Elimination (RFE), performs 5-fold cross-validation, evaluates models using multiple metrics, and visualizes ROC curves and confusion matrices.  
   - **Gökay's part**: Uses SelectKBest for feature selection, applies train/test split, compares classifiers, and exports results to Excel for further analysis.

- `CENG464 Project - Report.pdf`:  
   Detailed report explaining the problem, methodology, data preprocessing steps, and final results.

- `CENG464 Project - Data.xlsx`:  
   Final merged output with performance metrics from both implementations (Hasan & Gökay) using both all features and selected features.

---

## 🏆 Best Performing Model

**Random Forest** achieved the highest accuracy of **78.45%** when using all features, due to its robustness and ensemble learning nature.

---



## 📌 Notes

- Developed collaboratively by **Gökay Çetinakdoğan** and **Hasan Emre Usta**
- Part of the course **CENG464 – Data Mining** project at Çankaya University.


