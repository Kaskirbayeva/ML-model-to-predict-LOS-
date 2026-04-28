# Hospital Length of Stay Prediction

## Overview
This project develops and evaluates machine learning models to predict hospital length of stay (LOS) using routinely collected administrative hospital data. The main objectives are to compare predictive performance across different algorithms and to identify the most important factors associated with prolonged hospitalisation.

---

## Data
The analysis is based on anonymised inpatient administrative records from hospital admissions.

- **Time period:** 2018–2022 (with extended processing for 2023 in supplementary pipeline)
- **Unit of analysis:** Hospital admission episode
- **Outcome variable:** Length of stay (LOS), measured in days
- **Predictor domains:**
  - Demographic characteristics
  - Clinical diagnoses
  - Admission type and outcomes
  - Socioeconomic and insurance status
  - Hospital organisational characteristics

Due to data privacy restrictions, raw data are not publicly available. Only derived and anonymised analytical datasets are used in this repository.

---

## Feature Engineering

### 1. ICD-10 Classification
- ICD-10 diagnosis codes were truncated to the 3-character level.
- Codes were grouped into WHO ICD-10 chapters (I–XXII).
- Chapters were further aggregated into clinically meaningful disease categories (e.g., infectious diseases, circulatory system, neoplasms).
- A binary diagnosis-item matrix was constructed for modelling purposes.

---

### 2. Demographic Features
- Age was categorised into five groups: newborn, child, young adult, middle-aged adult, and senior.
- Sex was standardised into binary categories (male/female).
- Both variables were transformed into one-hot encoded representations.

---

### 3. Socioeconomic Status (Employment / Insurance Type)
- Employment status was used to infer OSMS contribution type (self-paid vs government-covered).
- Derived categorical indicators were encoded as binary features.

---

### 4. Admission Characteristics
- Admission type was classified as **planned vs emergency**.
- Admission outcomes were harmonised into four categories: discharged, referred, death, and self-discharge.
- Clinical complication status was binarised (presence vs absence of complication).

---

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

---

## Hospital-Level Characteristics
Hospital administrative data were merged using facility identifiers (`mo_name`) to enrich patient-level records.

Derived variables include:

- Hospital level: regional, city, rural, republican  
- Ownership type: public vs private  
- Geographical region: North, South, East, West, Central, National status  

All hospital-level variables were transformed into binary indicator matrices for modelling.

---

## Spatial and Geographic Features
Hospital regional identifiers were used to classify facilities into macro-geographical zones of Kazakhstan. This enables spatial analysis of service provision patterns and regional variation in hospital utilisation.

---

## Final Analytical Dataset
All engineered feature blocks were merged at the patient-episode level using a unique identifier.

The final dataset includes:

- Diagnosis features  
- Demographic variables  
- Socioeconomic indicators  
- Admission characteristics  
- Clinical specialty indicators  
- Hospital-level attributes  
- Outcome variable (LOS)

The resulting dataset is a high-dimensional binary feature matrix designed for statistical modelling and machine learning applications.

---

## Output Files
- df_new.RData – cleaned episode-level dataset  
- dff_new.RData – dataset merged with hospital-level features  
- df_all_new.RData – final feature-engineered dataset for modelling  

Equivalent datasets were also constructed for 2023 to enable temporal comparison.

---

## Methodology
- Models: XGBoost, Artificial Neural Network (ANN)  
- Feature selection: Top 20 features identified using XGBoost importance scores  
- Evaluation metrics: RMSE, MAE  
- Validation strategy: Train–test split with cross-validation  

---

## Repository Structure
project/
├── analysis.R        # Model training and evaluation
├── mapping.R         # Data cleaning and feature engineering
├── README.md         # Project documentation
├── data/             # Example or processed datasets (if shareable)
├── figures/          # Output plots and visualisations
