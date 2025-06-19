#  Optimized Treatment Predictor

A machine learning-based system to predict the optimal treatment for a patient (Standard vs. Intensive) based on vital health data. Built using Python, Scikit-learn, and Pandas.


# Project Overview

This project is a healthcare predictive model that helps classify whether a patient should receive **Standard Treatment** or **Intensive Care**. It uses vital health indicators (e.g., age, blood pressure, heart rate, temperature) and patient-specific information (e.g., gender, blood type, allergies).

The model is trained using a synthetic dataset and includes preprocessing, hyperparameter tuning, model evaluation, and explainable prediction output.


# Features

- Label encoding for categorical data (e.g., gender, blood type, allergies)
- Standard scaling and imputation for numerical features
- RandomForestClassifier with hyperparameter tuning via RandomizedSearchCV
- ROC-AUC and Precision-Recall AUC scoring
- Confusion matrix visualization
- Predictive confidence score for new patient data
- Personalized treatment guidance


# How to Run

# 1. Clone the repository:
```bash
git clone https://github.com/yourusername/optimized-treatment-predictor.git
cd optimized-treatment-predictor
