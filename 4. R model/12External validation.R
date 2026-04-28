
setwd("~/Jasgalym/4. R model")
ann_model <- readRDS("ann_model_2022.rds")
xgb_model <- readRDS("xgb_model_2022.rds")

load("~/Jasgalym/1.Data preparation/df_all_2023.RData")
load("~/Jasgalym/1.Data preparation/df_all.RData")
df_all_2023$los_class <- cut(
  df_all_2023$los,
  breaks = c(0, 5, 10, 20, 30, 90),
  labels = 0:4,
  right = FALSE  # interval includes the lower bound, excludes the upper
)

df_all_2023$los_class <- as.numeric(as.character(df_all_2023$los_class))

### Make aligned df_all_2023 with df_all 
# Remove ID and target columns
features_2022 <- df_all[, !(names(df_all) %in% c("id", "los", "los_class"))]
features_2023 <- df_all_2023[, !(names(df_all_2023) %in% c("id", "los", "los_class"))]

X_2022 <- model.matrix(~ . -1, data = features_2022)
X_2023 <- model.matrix(~ . -1, data = features_2023)

cols_2022 <- colnames(X_2022)
cols_2023 <- colnames(X_2023)

# Columns missing in 2023
missing_in_2023 <- setdiff(cols_2022, cols_2023)
# Extra columns in 2023
extra_in_2023 <- setdiff(cols_2023, cols_2022)

# Print
cat("Missing in 2023:\n")
print(missing_in_2023)

cat("\nExtra in 2023:\n")
print(extra_in_2023)

identical(cols_2022, cols_2023)
X_2023_aligned <- X_2023[, cols_2022, drop = FALSE]

######
# Target
target_2023 <- df_all_2023$los_class
dtest_2023 <- xgb.DMatrix(data = X_2023_aligned)

y_2023 <- df_all_2023$los_class
dtest_2023 <- xgb.DMatrix(data = X_2023_aligned, label = y_2023)

pred_probs_xgb_2023 <- predict(xgb_model, newdata = dtest_2023)

pred_matrix_xgb_2023 <- matrix(pred_probs_xgb_2023, ncol = 5, byrow = TRUE)
pred_class_xgb_2023 <- max.col(pred_matrix_xgb_2023) - 1

# Accuracy
mean(pred_class_xgb_2023 == y_2023)

# Confusion matrix
table(Predicted = pred_class_xgb_2023, Actual = y_2023)


# Convert to DMatrix
dtest_2023 <- xgb.DMatrix(data = X_2023_aligned, label = y_2023)

# Predict class probabilities
pred_probs_xgb_2023 <- predict(xgb_model, newdata = dtest_2023)

# Convert to matrix of probs (n x 5) and get predicted class
pred_matrix_xgb_2023 <- matrix(pred_probs_xgb_2023, ncol = 5, byrow = TRUE)
pred_class_xgb_2023 <- max.col(pred_matrix_xgb_2023) - 1

# Accuracy
acc_xgb_2023 <- mean(pred_class_xgb_2023 == y_2023)
cat("XGBoost Accuracy (2023):", round(acc_xgb_2023, 4), "\n")

# Confusion matrix
table(Predicted = pred_class_xgb_2023, Actual = y_2023)

# One-hot encode the actual labels
library(caret)
y_2023_onehot <- class2ind(as.factor(y_2023))

# Reshape predicted probabilities into matrix
num_class <- 5
pred_matrix_xgb_2023 <- matrix(pred_probs_xgb_2023, ncol = num_class, byrow = TRUE)

# Define log loss function
log_loss <- function(actual, predicted, eps = 1e-15) {
  predicted <- pmin(pmax(predicted, eps), 1 - eps)
  -sum(actual * log(predicted)) / nrow(actual)
}

# Compute log loss
ll_xgb_2023 <- log_loss(y_2023_onehot, pred_matrix_xgb_2023)
cat("Log Loss (XGBoost on 2023):", round(ll_xgb_2023, 4), "\n")


### External validation for ANN 
importance_matrix <- xgb.importance(model = xgb_model)
top20_features <- importance_matrix$Feature[1:20]

# Match structure with training data
X_2023 <- model.matrix(~ . -1, data = features_2023)

# Align columns with the top 20 used in ANN
X_2023_ann <- as.data.frame(X_2023[, top20_features])

# Ensure the column order matches training data
X_2023_ann <- X_2023_ann[, top20_features]

# -------------------------------
# 2. Predict using the trained ANN model
# -------------------------------

# pred_probs_ann_2023 will be a matrix of probabilities
pred_probs_ann_2023 <- predict(ann_model, X_2023_ann, type = "raw")

# -------------------------------
# 3. Evaluate accuracy and log loss
# -------------------------------

# Predicted class
pred_class_ann_2023 <- apply(pred_probs_ann_2023, 1, which.max) - 1  # match 0-based labels

# True class
y_2023 <- target_2023

# Accuracy
accuracy_ann_2023 <- mean(pred_class_ann_2023 == y_2023)
cat("Accuracy (ANN on 2023):", round(accuracy_ann_2023, 4), "\n")

# Log loss (if y_2023 is not one-hot encoded)
library(Metrics)
y_2023_onehot <- model.matrix(~ factor(y_2023) - 1)
logloss_ann_2023 <- logLoss(y_2023_onehot, pred_probs_ann_2023)
cat("Log Loss (ANN on 2023):", round(logloss_ann_2023, 4), "\n")


### RMSE 
# --- One-hot encode true labels ---
library(caret)
y_2023_onehot <- model.matrix(~ factor(y_2023) - 1)

# --- RMSE function ---
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# --- XGBoost RMSE ---
rmse_xgb_2023 <- rmse(y_2023_onehot, pred_probs_xgb_2023)
cat("RMSE (XGBoost on 2023):", round(rmse_xgb_2023, 4), "\n")

# --- ANN RMSE ---
rmse_ann_2023 <- rmse(y_2023_onehot, pred_probs_ann_2023)
cat("RMSE (ANN on 2023):", round(rmse_ann_2023, 4), "\n")






