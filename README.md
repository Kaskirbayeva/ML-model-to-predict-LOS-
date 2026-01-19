# Hospital Length of Stay Prediction

## Overview
This project develops machine learning models to predict hospital length of stay (LOS) using administrative hospital data. The aim is to compare predictive performance across algorithms and identify the most important predictors of LOS.

## Data
The analysis uses anonymized hospital administrative data covering inpatient admissions.
- Time period: 2018–2022
- Outcome variable: Length of stay (days)
- Predictors include demographic, clinical, and admission-related characteristics.

Due to data privacy restrictions, the raw data cannot be shared publicly.

## Methodology
- Models: XGBoost, Artificial Neural Network (ANN)
- Feature selection: Top 20 features identified by XGBoost
- Evaluation metrics: RMSE, MAE
- Train–test split with cross-validation

## Repository Structure
