############################################################
# External Validation Dataset (2023)
############################################################

library(dplyr)
library(xgboost)
library(nnet)
library(caret)

############################################################
# Load data
############################################################

load("~/Jasgalym/1.Data preparation/df_all_23.RData")
df_2023 <- df_all_23

############################################################
# Create outcome (LOS classes)
############################################################

df_2023$los_class <- cut(
  df_2023$los,
  breaks = c(0, 5, 10, 20, 30, 90),
  labels = 0:4,
  right = FALSE
)

df_2023$los_class <- as.factor(df_2023$los_class)

############################################################
# Clean dataset
############################################################

df_2023_clean <- df_2023 %>%
  select(-id, -los, -`NA`) %>%
  drop_na()

y_ext <- as.numeric(df_2023_clean$los_class) - 1

############################################################
# ==========================================================
# XGBoost External Validation
# ==========================================================
############################################################

X_ext <- model.matrix(los_class ~ . - 1, df_2023_clean)
X_ext_df <- as.data.frame(X_ext)

# Align columns with training set
missing_cols <- setdiff(colnames(X_train), colnames(X_ext_df))
for (col in missing_cols) {
  X_ext_df[[col]] <- 0
}

extra_cols <- setdiff(colnames(X_ext_df), colnames(X_train))
X_ext_df <- X_ext_df[, !(colnames(X_ext_df) %in% extra_cols)]

X_ext_df <- X_ext_df[, colnames(X_train)]
X_ext <- as.matrix(X_ext_df)

dext <- xgb.DMatrix(data = X_ext, label = y_ext)

# Prediction
pred_prob_xgb_ext <- predict(xgb_model, dext)
pred_matrix_ext <- matrix(pred_prob_xgb_ext, ncol = 5, byrow = TRUE)
pred_xgb_ext <- max.col(pred_matrix_ext) - 1

accuracy_xgb_ext <- mean(pred_xgb_ext == y_ext)

############################################################
# ==========================================================
# ANN External Validation
# ==========================================================
############################################################
############################################################
# 1. Create dummy variables
############################################################

train_dummies <- model.matrix(los_class ~ . - 1, train)
ext_dummies <- model.matrix(los_class ~ . - 1, df_2023_clean)

############################################################
# 2. Align feature space (SAFE VERSION)
############################################################

train_cols <- colnames(train_dummies)
ext_cols <- colnames(ext_dummies)

# Add missing columns in external set
missing_cols <- setdiff(train_cols, ext_cols)
if (length(missing_cols) > 0) {
  ext_dummies <- cbind(ext_dummies, matrix(0, nrow = nrow(ext_dummies), ncol = length(missing_cols)))
  colnames(ext_dummies)[(ncol(ext_dummies) - length(missing_cols) + 1):ncol(ext_dummies)] <- missing_cols
}

# Remove extra columns not in training
extra_cols <- setdiff(ext_cols, train_cols)
if (length(extra_cols) > 0) {
  ext_dummies <- ext_dummies[, !(colnames(ext_dummies) %in% extra_cols)]
}

# Ensure identical column order
ext_dummies <- ext_dummies[, train_cols]

############################################################
# 3. Scale features (IMPORTANT for ANN)
############################################################

preProc <- preProcess(train_dummies, method = c("center", "scale"))

train_scaled <- predict(preProc, train_dummies)
ext_scaled <- predict(preProc, ext_dummies)

############################################################
# 4. One-hot encode outcome
############################################################

y_train_ann <- class.ind(train$los_class)

############################################################
# 5. Train ANN model
############################################################

ann_model <- nnet(
  x = train_scaled,
  y = y_train_ann,
  size = 10,
  maxit = 200,
  decay = 0.01,
  trace = FALSE
)

############################################################
# 6. External prediction
############################################################

pred_prob_ann_ext <- predict(ann_model, ext_scaled, type = "raw")
pred_ann_ext <- max.col(pred_prob_ann_ext) - 1

############################################################
# ==========================================================
# Metrics Function
# ==========================================================
############################################################

compute_metrics <- function(pred, actual) {
  actual_factor <- factor(actual, levels = 0:4)
  pred_factor <- factor(pred, levels = 0:4)
  
  cm <- confusionMatrix(pred_factor, actual_factor)
  
  acc <- cm$overall["Accuracy"]
  
  per_class <- as.data.frame(cm$byClass[, c("Precision", "Recall", "F1")])
  per_class$Class <- rownames(per_class)
  
  per_class$F1[is.na(per_class$F1)] <- 0
  per_class$Precision[is.na(per_class$Precision)] <- 0
  per_class$Recall[is.na(per_class$Recall)] <- 0
  
  macro_F1 <- mean(per_class$F1)
  
  list(
    accuracy = acc,
    macro_F1 = macro_F1,
    per_class = per_class
  )
}

############################################################
# Log Loss
############################################################

log_loss <- function(y_true, y_prob) {
  eps <- 1e-15
  y_prob <- pmax(pmin(y_prob, 1 - eps), eps)
  
  y_mat <- class.ind(factor(y_true, levels = 0:4))
  
  -mean(rowSums(y_mat * log(y_prob)))
}

############################################################
# Metrics
############################################################

metrics_xgb_ext <- compute_metrics(pred_xgb_ext, y_ext)
metrics_ann_ext <- compute_metrics(pred_ann_ext, y_ext)

logloss_xgb_ext <- log_loss(y_ext, pred_matrix_ext)
logloss_ann_ext <- log_loss(y_ext, pred_prob_ann_ext)

############################################################
# Final results table
############################################################

results_ext <- data.frame(
  Model = c("XGBoost", "ANN"),
  Accuracy = c(metrics_xgb_ext$accuracy, metrics_ann_ext$accuracy),
  Macro_F1 = c(metrics_xgb_ext$macro_F1, metrics_ann_ext$macro_F1),
  LogLoss = c(logloss_xgb_ext, logloss_ann_ext)
)

print(results_ext)
