# Hospital Length of Stay Prediction

## Overview
This project develops and evaluates machine learning models to predict hospital length of stay (LOS) using routinely collected administrative hospital data. The main objectives are to compare predictive performance across different algorithms, quantify uncertainty in class-specific performance, and identify the most important factors associated with hospitalisation.



## Data
The analysis is based on anonymised inpatient administrative records from hospital admissions.

- **Time period:** 2018–2022 (with extended processing for 2023 in supplementary pipeline)
- **Unit of analysis:** Hospital admission episode
- **Outcome variable:** Length of stay (LOS), modelled as a categorical class
- **Predictor domains:**
  - Demographic characteristics
  - Clinical diagnoses
  - Admission type and outcomes
  - Socioeconomic and insurance status
  - Hospital organisational characteristics

Due to data privacy restrictions, raw data are not publicly available. Only derived and anonymised analytical datasets are used in this repository.



## Feature Engineering

### 1. ICD-10 Classification
- ICD-10 diagnosis codes were truncated to the 3-character level.
- Codes were grouped into WHO ICD-10 chapters (I–XXII).
- Chapters were further aggregated into clinically meaningful disease categories (e.g., infectious diseases, circulatory system, neoplasms).
- A binary diagnosis-item matrix was constructed for modelling purposes.


### 2. Demographic Features
- Age was categorised into five groups: newborn, child, young adult, middle-aged adult, and senior.
- Sex was standardised into binary categories (male/female).
- Both variables were transformed into one-hot encoded representations.



### 3. Socioeconomic Status (Employment / Insurance Type)
- Employment status was used to infer Mandatory Health Insurance (MSHI) scheme contribution type (self-paid vs MSHI).
- Derived categorical indicators were encoded as binary features.



### 4. Admission Characteristics
- Admission type was classified as **planned vs emergency**.
- Admission outcomes were harmonised into four categories: discharged, referred, death, and self-discharge.
- Clinical complication status was binarised (presence vs absence of complication).



### 5. Clinical Specialty Mapping
Hospital ward profiles (`prof_koiki`) were mapped into broader clinical domains:

- Internal medicine  
- Surgery  
- Pediatrics  
- Oncology / Hematology  
- Neurology / Neurosurgery  
- Orthopedics / Trauma  
- Other specialties  

This aggregation reduces dimensionality while preserving clinical interpretability.



## Hospital-Level Characteristics
Hospital administrative data were merged using facility identifiers (`mo_name`) to enrich patient-level records.

Derived variables include:

- Hospital level: regional, city, rural, republican  
- Ownership type: public vs private  
- Geographical region: North, South, East, West, Central, National status  

All hospital-level variables were transformed into binary indicator matrices for modelling.


## Spatial and Geographic Features
Hospital regional identifiers were used to classify facilities into macro-geographical zones of Kazakhstan. This enables spatial analysis of service provision patterns and regional variation in hospital utilisation.


## Final Analytical Dataset
All engineered feature blocks were merged at the patient-episode level using a unique identifier. The final dataset includes:

- Diagnosis features  
- Demographic variables  
- Socioeconomic indicators  
- Admission characteristics  
- Clinical specialty indicators  
- Hospital-level attributes  
- Outcome variable (LOS)

The resulting dataset is a high-dimensional binary feature matrix designed for statistical modelling and machine learning applications.



## Input Files
- df2022.rds, df2023.rds – final feature-engineered dataset for modelling  




## Methodology
| Model | Script | Description |
|---|---|---|
| Random Forest | `02_RandomForest_tuning.R` | Bagged decision tree ensemble (`randomForest`/`ranger`); hyperparameters (mtry, ntree, node size) tuned via 5-fold CV grid search on a 10% stratified subsample, final model refit on the full training set and evaluated on the held-out test set |
| XGBoost | `03_XGBoost_tuning.R` | Gradient-boosted trees; hyperparameters (max depth, learning rate, subsample, colsample) tuned via 5-fold CV grid search on a 10% stratified subsample, final model refit on the full training set and evaluated on the held-out test set |
| LightGBM | `04_LightGBM_tuning.R` | Gradient-boosted trees with leaf-wise growth (`lightgbm`); hyperparameters (num_leaves, learning rate, feature_fraction, bagging_fraction) tuned via 5-fold CV grid search on a 10% stratified subsample, final model refit on the full training set and evaluated on the held-out test set |
| Artificial Neural Network (ANN) | `05_ANN_tuning.R` | Single-hidden-layer feedforward network (`nnet`); hyperparameters (hidden units, weight decay) tuned via 5-fold CV grid search on a 10% stratified subsample with centered/scaled inputs, final model refit on the full training set and evaluated on the held-out test set |
| Multinomial Logistic Regression | `06_MultinomialLR.R` | Baseline multinomial logit model (`nnet::multinom`), fit directly on the full training set and evaluated on the held-out test set, used for interpretable benchmarking |

## Hyperparameter Tuning

- All tunable models (XGBoost, ANN) were optimised using 5-fold cross-validation on a 10% stratified subsample of the training data, prior to fitting the final model on the full training set.
Macro-averaged F1 score was used as the selection criterion across CV folds to account for class imbalance in LOS categories

- All tunable models (XGBoost, ANN) were optimised using 5-fold cross-validation on a 10% stratified subsample of the training data, prior to fitting the final model on the full training set.
Macro-averaged F1 score was used as the selection criterion across CV folds to account for class imbalance in LOS categories.


## Evaluation Metrics

For each model and each LOS class, the following one-vs-rest metrics were computed on the held-out test set:

- Precision (Positive Predictive Value)
- Recall (Sensitivity)
- F1 score
- AUC (ROC)

** Uncertainty quantification: ** 95% confidence intervals for all metrics were estimated via non-parametric bootstrap resampling (B = 1,000 resamples) of the test set.

## Feature Importance

- Top predictive features were identified using XGBoost gain-based importance scores.
- For the multinomial logistic regression model, variable importance was approximated as the mean absolute coefficient across outcome classes.

## Repository Structure
project/
├── mapping.R                       # Data cleaning and feature engineering
├── 01_data_preparation.R           # Train/test split and preprocessing
├── 03_XGBoost_tuning.R             # XGBoost hyperparameter tuning and final model
├── 04_ANN_tuning.R                 # ANN hyperparameter tuning and final model
├── 06_MultinomialLR.R              # Multinomial logistic regression model
├── evaluation_bootstrap_CI.R       # Class-specific metrics with 95% bootstrap CIs
├── README.md                       # Project documentation
├── data/                           # Example or processed datasets (if shareable)
└── figures/                        # Output plots and visualisations
