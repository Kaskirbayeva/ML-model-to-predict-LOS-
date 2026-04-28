############################################################
# XGBoost LOS Prediction & Hospital Impact Simulation
############################################################

# 1. Load required libraries
library(dplyr)
library(xgboost)

############################################################
# 2. Load external dataset
############################################################
setwd("~/Jasgalym/1.Data preparation")
load("df_all_23.RData")  # 
df_2023 <- df_all_23

# Create LOS classes
df_2023$los_class <- cut(
  df_2023$los,
  breaks = c(0,5,10,20,30,90),
  labels = 0:4,
  right = FALSE
)
df_2023$los_class <- as.factor(df_2023$los_class)

# Remove unnecessary columns and drop missing values
df_2023_clean <- df_2023 %>% select(-id, -los, -'NA') %>% drop_na()

############################################################
# 3. Prepare feature matrix
############################################################
X_ext <- model.matrix(los_class ~ . -1, df_2023_clean)

############################################################
# 4. Load trained XGBoost model and align features
############################################################
setwd("~/Jasgalym/4. R model")
xgb_model <- readRDS("xgb_model.rds")
train_features <- xgb_model$feature_names
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

# Reorder columns to match training features
X_ext <- X_ext[, train_features]

# Convert to XGBoost DMatrix
dext <- xgb.DMatrix(data = X_ext)

############################################################
# 5. Predict LOS classes with XGBoost
############################################################
pred_prob_ext <- predict(xgb_model, dext)
pred_matrix <- matrix(pred_prob_ext, ncol = 5, byrow = TRUE)
pred_xgb_class <- max.col(pred_matrix) - 1  # 0-based classes

# Map predicted classes to readable labels
patients <- df_2023_clean
patients$LOS_Pred <- factor(pred_xgb_class,
                            labels = c("0-5 days", "5-10 days", "10-20 days",
                                       "20-30 days", "30-90 days"))

# Optional: Assign priority flag based on predicted LOS
patients$Priority_Flag <- ifelse(
  patients$LOS_Pred %in% c("20-30 days", "30-90 days"), "High",
  ifelse(patients$LOS_Pred %in% c("10-20 days"), "Moderate", "Low")
)

############################################################
# 6. Map LOS classes to numeric midpoints
############################################################
patients$Predicted_LOS_days <- case_when(
  trimws(as.character(patients$LOS_Pred)) == "0-5 days"   ~ 3,
  trimws(as.character(patients$LOS_Pred)) == "5-10 days"  ~ 7,
  trimws(as.character(patients$LOS_Pred)) == "10-20 days" ~ 15,
  trimws(as.character(patients$LOS_Pred)) == "20-30 days" ~ 25,
  trimws(as.character(patients$LOS_Pred)) == "30-90 days" ~ 60,
  TRUE ~ NA_real_
)

patients$Baseline_LOS_days <- case_when(
  df_2023_clean$los_class == 0 ~ 3,
  df_2023_clean$los_class == 1 ~ 7,
  df_2023_clean$los_class == 2 ~ 15,
  df_2023_clean$los_class == 3 ~ 25,
  df_2023_clean$los_class == 4 ~ 60,
  TRUE ~ NA_real_
)

# Remove any rows with NA after mapping
patients <- patients %>% filter(!is.na(Predicted_LOS_days) & !is.na(Baseline_LOS_days))

############################################################
# 7. Calculate impact metrics
############################################################
# Total bed-days
total_beddays_baseline <- sum(patients$Baseline_LOS_days)
total_beddays_predicted <- sum(patients$Predicted_LOS_days)
total_beddays_improvement <- (total_beddays_baseline - total_beddays_predicted) / total_beddays_baseline * 100

# Delayed discharges (>10 days)
baseline_delays <- sum(patients$Baseline_LOS_days > 10)
predicted_delays <- sum(patients$Predicted_LOS_days > 10)
reduction_delays <- (baseline_delays - predicted_delays) / baseline_delays * 100

# Average bed turnover
avg_bed_turnover_baseline <- mean(patients$Baseline_LOS_days)
avg_bed_turnover_predicted <- mean(patients$Predicted_LOS_days)
bed_turnover_improvement <- (avg_bed_turnover_baseline - avg_bed_turnover_predicted) / avg_bed_turnover_baseline * 100

############################################################
# 8. Compile impact summary table
############################################################
impact_summary <- data.frame(
  Metric = c("Total Bed-Days", "Delayed Discharges (>10d)", "Average Bed Turnover (days)"),
  Baseline = c(total_beddays_baseline, baseline_delays, round(avg_bed_turnover_baseline,2)),
  Intervention = c(total_beddays_predicted, predicted_delays, round(avg_bed_turnover_predicted,2)),
  Improvement_Percent = c(
    round(total_beddays_improvement,1),
    round(reduction_delays,1),
    round(bed_turnover_improvement,1)
  )
)

print(impact_summary)

