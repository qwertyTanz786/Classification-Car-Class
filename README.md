# 🚗 Car Evaluation Classification using Random Forest
---

## 📌 Overview

This project builds a Machine Learning classification model to predict the overall evaluation (`class`) of a car based on categorical attributes such as buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level.

The model achieves **97% accuracy** using a Random Forest Classifier.

---

## 📂 Dataset Information

The dataset contains the following features:

| Feature  | Description |
|-----------|------------|
| buying    | Buying price category |
| maint     | Maintenance cost category |
| doors     | Number of doors |
| persons   | Passenger capacity |
| lug_boot  | Luggage boot size |
| safety    | Safety level |
| class     | Overall car evaluation (Target Variable) |

---

## 🎯 Problem Statement

Given car attributes, predict the overall car acceptability category:

- unacc (unacceptable)  
- acc (acceptable)  
- good  
- vgood  

This is a **multi-class classification problem**.

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Random Forest Classifier  
- GridSearchCV  

---

## 🔄 Workflow

1. Data Loading  
2. Feature and Target Separation  
3. Label Encoding of Categorical Variables  
4. Stratified Train-Test Split (80/20)  
5. Model Training  
6. Hyperparameter Tuning using GridSearchCV  
7. Model Evaluation using Classification Report  

---

## ⚙️ Model Details

### Algorithm Used
**Random Forest Classifier**

Two experiments were conducted:

- Default parameters  
- Hyperparameter tuning with GridSearchCV (3-fold cross-validation)  

### Hyperparameters Tuned

- `n_estimators`  
- `max_depth`  
- `min_samples_split`  

---

## 📊 Results

| Metric | Score |
|--------|--------|
| Accuracy | 97% |
| Macro F1-score | 0.92 |
| Weighted F1-score | 0.97 |

The model performs consistently well both with and without hyperparameter tuning due to the structured and rule-based nature of the dataset.

---

## 🔍 Key Insights

- Stratified splitting significantly improved evaluation reliability.
- Hyperparameter tuning did not significantly change performance.
- Random Forest handles categorical encoded data effectively.
- The dataset is clean and highly separable.

---

## 📈 Future Improvements

- Compare with Logistic Regression and XGBoost  
- Visualize Confusion Matrix  
- Perform Feature Importance Analysis  
- Deploy as a Web App using Flask or Streamlit  

---
