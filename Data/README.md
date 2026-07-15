# Data Directory

## Data Notice

This directory contains synthetic placeholder data only. The files included here were generated to approximate the structure, variable types, and distributional properties of the original administrative dataset used in this study, solely to demonstrate how the modelling pipeline runs end-to-end. They do **not** contain real patient records and must not be used for any clinical, epidemiological, or research inference.

The procedure used to generate these synthetic datasets is not included in this repository, as it falls outside the scope of the modelling pipeline presented here.

Access to the real administrative data is restricted due to data privacy and governance requirements, and must be formally requested from the Ministry of Health, subject to their data-sharing policies and any required institutional or ethical approvals. Researchers interested in accessing the underlying data should contact the data holder or the corresponding author for guidance on the request procedure.

## Files

| File | Description |
|---|---|
| `synthetic_train.rds` | Synthetic training dataset, approximating the structure of the 2022 training split (80%) used for model development and hyperparameter tuning |
| `synthetic_test.rds` | Synthetic test dataset, approximating the structure of the 2022 held-out test split (20%) used for internal model evaluation |
| `synthetic_external2023.rds` | Synthetic external validation dataset, approximating the structure of the independent 2023 dataset used for external temporal validation |
