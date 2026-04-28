############################################################
# 1. Load packages
############################################################
library(dplyr)
library(xgboost)
library(tidyr)

############################################################
# 2. Load external dataset
############################################################
load("df_all_23.RData")  # your external dataset
df_2023 <- df_all_23

# Create LOS classes
df_2023$los_class <- cut(
  df_2023$los,
  breaks = c(0,5,10,20,30,90),
  labels = 0:4,
  right = FALSE
)
df_2023$los_class <- as.factor(df_2023$los_class)

# Remove unnecessary columns
df_2023 <- df_2023 %>% select(-id, -los, -'NA') %>% drop_na()

############################################################
# 3. Prepare feature matrix
############################################################
X_ext <- model.matrix(los_class ~ . -1, df_2023)

############################################################
# 4. Load trained XGBoost model
############################################################
setwd("~/Jasgalym/4. R model")
xgb_model <- readRDS("~/xgb_model_final.rds")
# Get training feature names
train_features <- xgb_model$feature_names

# Align external features with training features
ext_features <- colnames(X_ext)

# Add missing columns with zeros
missing_cols <- setdiff(train_features, ext_features)
for (col in missing_cols) {
  X_ext <- cbind(X_ext, rep(0, nrow(X_ext)))
  colnames(X_ext)[ncol(X_ext)] <- col
}

# Remove extra columns
extra_cols <- setdiff(ext_features, train_features)
if(length(extra_cols) > 0){
  X_ext <- X_ext[, !(colnames(X_ext) %in% extra_cols)]
}

# Reorder columns to match training
X_ext <- X_ext[, train_features]

# Convert to DMatrix
dext <- xgb.DMatrix(data = X_ext)

############################################################
# 5. Predict LOS classes
############################################################
pred_prob_ext <- predict(xgb_model, dext)
pred_matrix_ext <- matrix(pred_prob_ext, ncol = 5, byrow = TRUE)
pred_xgb_ext <- max.col(pred_matrix_ext) - 1

# Add predictions to dataset
df_2023$LOS_Pred <- pred_xgb_ext

# Assign descriptive LOS categories
df_2023$LOS_Category <- cut(
  df_2023$LOS_Pred,
  breaks = c(-1,0,1,2,3,4),
  labels = c("0-5 days","5-10 days","10-20 days","20-30 days","30-90 days")
)

# Assign priority flags
df_2023 <- df_2023 %>%
  mutate(Priority_Flag = case_when(
    LOS_Pred >= 3 ~ "High",
    LOS_Pred == 2 ~ "Moderate",
    TRUE ~ "Low"
  ))

############################################################
# 6. Ready for Shiny EHR simulation
############################################################
patients <- df_2023  # use this in your Shiny app
head(patients)
# Save your prepared patients dataset for Shiny
setwd("~/Jasgalym/4. R model")
saveRDS(patients, "patients.rds")
