############################################################
# 1. Load packages
############################################################

library(dplyr)
library(xgboost)
library(nnet)
library(caret)
library(ggplot2)
library(reshape2)

############################################################
# 2. Load data
############################################################

load("~/Jasgalym/1.Data preparation/df_all.RData")

############################################################
# 3. Create LOS classes
############################################################

df_all$los_class <- cut(
  df_all$los,
  breaks = c(0,5,10,20,30,90),
  labels = 0:4,
  right = FALSE
)

df_all$los_class <- as.factor(df_all$los_class)

# Remove unnecessary variables
df_all <- df_all %>% select(-id, -los)

############################################################
# 4. Train-test split
############################################################

set.seed(123)

index <- sample(1:nrow(df_all), 0.8*nrow(df_all))

train <- df_all[index,]
test  <- df_all[-index,]

predictors <- setdiff(names(df_all),"los_class")

############################################################
# 5. Baseline Model: Multinomial Logistic Regression
############################################################

library(nnet)

logit_model <- multinom(
  los_class ~ .,
  data = train,
  trace = FALSE
)

pred_logit <- predict(logit_model, test)

accuracy_logit <- mean(pred_logit == test$los_class)

############################################################
# 6. Prepare data for XGBoost
############################################################

X_train <- model.matrix(los_class ~ . -1, train)
X_test  <- model.matrix(los_class ~ . -1, test)

y_train <- as.numeric(train$los_class) - 1
y_test  <- as.numeric(test$los_class) - 1

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

############################################################
# 7. XGBoost with cross-validation
############################################################

params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  num_class = 5,
  eval_metric = "mlogloss",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  lambda = 1,
  alpha = 0.1
)

cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 0
)

best_nrounds <- cv_model$best_iteration

############################################################
# 8. Train final XGBoost model
############################################################

xgb_model <- xgboost(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

############################################################
# 9. XGBoost predictions
############################################################

pred_prob <- predict(xgb_model, dtest)

pred_matrix <- matrix(pred_prob, ncol = 5, byrow = TRUE)

pred_xgb <- max.col(pred_matrix) - 1

accuracy_xgb <- mean(pred_xgb == y_test)

############################################################
# 10. ANN model
############################################################

# Scale numeric predictors
scale_vars <- sapply(train[,predictors], is.numeric)

train_scaled <- train
test_scaled  <- test

train_matrix <- scale(train[, predictors[scale_vars]])
train_scaled <- train
train_scaled[, predictors[scale_vars]] <- train_matrix

test_scaled <- test

test_scaled[, predictors[scale_vars]] <- scale(
  test[, predictors[scale_vars]],
  center = attr(train_matrix, "scaled:center"),
  scale  = attr(train_matrix, "scaled:scale")
)

ann_model <- nnet(
  los_class ~ .,
  data = train_scaled,
  size = 5,
  maxit = 200,
  decay = 0.01,
  trace = FALSE
)

pred_ann <- predict(ann_model, test_scaled, type="class")
table(pred_ann)

accuracy_ann <- mean(pred_ann == test_scaled$los_class)
#accuracy_ann <- mean(actual_ann == test_scaled$los_class)
############################################################
# 11. Model comparison
############################################################


results <- data.frame(
  Model = c("Logistic Regression","XGBoost","ANN"),
  Accuracy = c(accuracy_logit, accuracy_xgb, accuracy_ann)
)

print(results)

############################################################
# 12. Confusion Matrix for best model (XGBoost example)
############################################################

confusionMatrix(
  factor(pred_xgb),
  factor(y_test)
)

############################################################
# 13. Feature importance (XGBoost)
############################################################

# 1. Get feature importance
importance <- xgb.importance(
  feature_names = colnames(X_train),
  model = xgb_model
)

# 2. Select top 20 features
top20_importance <- importance[1:50, ]  # importance is already sorted by Gain

# Plot top 20 features with larger font
xgb.plot.importance(
  top20_importance,
  rel_to_first = TRUE,
  top_n = 50,
  main = "Top 20 Feature Importance for XGBoost Model"
)

############################################################
# 14. Optional: SHAP interpretation
############################################################

# install.packages("SHAPforxgboost")

library(SHAPforxgboost)

shap_values <- shap.values(xgb_model, X_train)

shap.plot.summary(shap_values$shap_score)

############################################################
# 15. Save model
############################################################

saveRDS(xgb_model, "~/Jasgalym/4. R model/xgb_model_final.rds")


############################################################
# Model Evaluation: Accuracy, Precision, Recall, F1
############################################################

library(caret)
library(dplyr)
library(tidyr)

# 1. Ensure all predicted and true labels are factors with same levels
class_levels <- levels(df_all$los_class)  # 0:4 factor levels

# Ensure all classes are present
class_levels <- levels(test_scaled$los_class)

# XGBoost predictions
pred_xgb_factor <- factor(pred_xgb, levels = 0:4)
y_test_factor   <- factor(y_test, levels = 0:4)

# ANN predictions
pred_ann <- factor(pred_ann, levels = class_levels)
actual_ann <- factor(test_scaled$los_class, levels = class_levels)

# Logistic Regression predictions
pred_logit_factor <- factor(pred_logit, levels = class_levels)
y_logit_factor    <- factor(test$los_class, levels = class_levels)

# 2. Function to compute metrics

compute_metrics <- function(pred, actual) {
  # Ensure factors have the same levels
  all_levels <- sort(unique(c(levels(actual), levels(pred))))
  pred <- factor(pred, levels = all_levels)
  actual <- factor(actual, levels = all_levels)
  
  cm <- confusionMatrix(pred, actual)
  
  # Overall accuracy
  acc <- cm$overall["Accuracy"]
  
  # Per-class metrics
  metrics <- as.data.frame(cm$byClass[, c("Precision","Recall","F1")])
  metrics$Class <- rownames(metrics)
  
  # Replace NA F1 (from zero predictions) with 0
  metrics$F1[is.na(metrics$F1)] <- 0
  metrics$Precision[is.na(metrics$Precision)] <- 0
  metrics$Recall[is.na(metrics$Recall)] <- 0
  
  # Macro F1 (average across all classes including zero-predicted classes)
  macro_f1 <- mean(metrics$F1)
  
  list(
    accuracy = acc,
    macro_F1 = macro_f1,
    per_class = metrics
  )
}


# 3. Compute for all models
metrics_xgb   <- compute_metrics(pred_xgb_factor, y_test_factor)
metrics_ann <- compute_metrics(pred_ann, actual_ann)
metrics_logit <- compute_metrics(pred_logit_factor, y_logit_factor)

# 4. Create summary table
summary_table <- data.frame(
  Model = c("Logistic Regression", "XGBoost", "ANN"),
  Accuracy = c(metrics_logit$accuracy,
               metrics_xgb$accuracy,
               metrics_ann$accuracy),
  Macro_F1 = c(metrics_logit$macro_F1,
               metrics_xgb$macro_F1,
               metrics_ann$macro_F1)
)

print(summary_table)

############################################################
# 5. Optional: Visualize per-class F1 for all models
############################################################

# Combine per-class metrics
per_class_df <- bind_rows(
  metrics_logit$per_class %>% mutate(Model="Logistic Regression"),
  metrics_xgb$per_class %>% mutate(Model="XGBoost"),
  metrics_ann$per_class %>% mutate(Model="ANN")
)

# Pivot longer for plotting
plot_df <- per_class_df %>%
  pivot_longer(cols = c("Precision","Recall","F1"),
               names_to = "Metric",
               values_to = "Score")

ggplot(plot_df, aes(x = Class, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  facet_wrap(~ Model) +
  labs(title = "Per-class Precision, Recall, and F1-score",
       x = "LOS Class", y = "Score") +
  scale_fill_manual(values = c("#1b9e77","#d95f02","#7570b3")) +
  theme_minimal()

ggplot(plot_df, aes(x = Class, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  facet_wrap(~ Model) +
  labs(title = "Per-class Precision, Recall, and F1-score",
       x = "LOS Class", y = "Score") +
  # Color-blind friendly palette (Okabe & Ito)
  scale_fill_manual(values = c("#E69F00", "#56B4E9", "#009E73")) +
  theme_minimal() +
  theme(
    text = element_text(size = 12),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )

ggplot(plot_df, aes(x = Class, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  facet_wrap(~ Model) +
  labs(title = "Per-class Precision, Recall, and F1-score",
       x = "LOS Class", y = "Score") +
  # Grayscale fill for print-friendly bars
  scale_fill_grey(start = 0.3, end = 0.8) +
  theme_minimal() +
  theme(
    text = element_text(size = 12),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )


ggplot(plot_df, aes(x = factor(Class), y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  facet_wrap(~ Model) +
  labs(x = "LOS Class (days)", y = "Score") +
  # Grayscale fill for print-friendly bars
  scale_fill_grey(start = 0.3, end = 0.8) +
  # Custom x-axis labels
  scale_x_discrete(
    labels = c("0-4", "5-9", "10-19", "20-29", "30-90")  # one label per class
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 0, hjust = 0.5)
  )



#### LOG LOSS 
# XGBoost predicted probabilities
pred_prob_xgb <- predict(xgb_model, newdata = dtest)  # dtest is xgb.DMatrix
pred_prob_xgb <- matrix(pred_prob_xgb, ncol = 5, byrow = TRUE)

# ANN predicted probabilities
pred_prob_ann <- predict(ann_model, test_scaled, type = "raw")  # type = "raw" returns probs

library(nnet)
pred_prob_logit <- predict(logit_model, test, type = "probs")

log_loss <- function(y_true, y_prob){
  # y_true: factor of actual class
  # y_prob: matrix of predicted probabilities
  eps <- 1e-15
  y_prob <- pmax(pmin(y_prob, 1 - eps), eps)  # avoid log(0)
  y_mat <- model.matrix(~ y_true - 1)  # one-hot encoding
  -mean(rowSums(y_mat * log(y_prob)))
}

logloss_xgb <- log_loss(test$los_class, pred_prob_xgb)
logloss_ann <- log_loss(test_scaled$los_class, pred_prob_ann)
logloss_logit <- log_loss(test$los_class, pred_prob_logit)

### RMSE
# Assuming you have a multinom model from nnet
library(nnet)

# Predict probabilities first
pred_prob_logit <- predict(logit_model, test, type = "probs")

# Convert to predicted class (0–4)
pred_logit_class <- max.col(pred_prob_logit) - 1

# Predicted probabilities already in matrix
pred_prob_xgb <- matrix(predict(xgb_model, dtest), ncol = 5, byrow = TRUE)

# Predicted class (0–4)
pred_xgb_class <- max.col(pred_prob_xgb) - 1

# pred_prob_ann: predicted probabilities from ANN
pred_ann_class <- max.col(pred_prob_ann) - 1

rmse <- function(y_true, y_pred){
  sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2))
}

rmse_logit <- rmse(test$los_class, pred_logit_class)
rmse_xgb   <- rmse(test$los_class, pred_xgb_class)
rmse_ann   <- rmse(test_scaled$los_class, pred_ann_class)

### All in one 

results <- data.frame(
  Model = c("Logistic Regression","XGBoost","ANN"),
  Accuracy = c(accuracy_logit, accuracy_xgb, accuracy_ann),
  LogLoss = c(logloss_logit, logloss_xgb, logloss_ann),
  RMSE = c(rmse_logit, rmse_xgb, rmse_ann)
)

print(results)

#### ROC curve

install.packages("pROC")  # if not installed
library(pROC)

####### Colored ROC ######
plot_multiclass_roc <- function(actual, pred_probs, model_name){
  classes <- levels(factor(actual))
  
  roc_list <- list()
  auc_list <- c()
  
  for(i in seq_along(classes)){
    this_class <- classes[i]
    # one-vs-rest: current class vs all others
    binary_actual <- ifelse(actual == this_class, 1, 0)
    roc_obj <- roc(binary_actual, pred_probs[, i])
    roc_list[[this_class]] <- roc_obj
    auc_list[this_class] <- auc(roc_obj)
  }
  
  # Plot all curves
  plot(roc_list[[1]], col = 1, main = paste("ROC per LOS class -", model_name))
  for(i in 2:length(classes)){
    plot(roc_list[[i]], col = i, add = TRUE)
  }
  legend("bottomright", legend = paste(classes, "AUC:", round(auc_list, 3)),
         col = 1:length(classes), lwd = 2)
}

# Logistic regression
plot_multiclass_roc(test$los_class, pred_prob_logit, "Logistic Regression")

# XGBoost
plot_multiclass_roc(test$los_class, pred_prob_xgb, "XGBoost")

# ANN
plot_multiclass_roc(test_scaled$los_class, pred_prob_ann, "ANN")
#####


####### B/W ROC plot ######
library(pROC)

plot_multiclass_roc_bw <- function(actual, pred_probs, model_name){
  # Convert actual to factor and relabel starting from 1
  classes <- levels(factor(actual))
  new_labels <- seq_along(classes)  # 1, 2, 3, ...
  
  roc_list <- list()
  auc_list <- c()
  
  lty_values <- 1:length(classes)  # different line types for classes
  
  # Compute ROC for each class
  for(i in seq_along(classes)){
    this_class <- classes[i]
    binary_actual <- ifelse(actual == this_class, 1, 0)
    roc_obj <- roc(binary_actual, pred_probs[, i])
    roc_list[[this_class]] <- roc_obj
    auc_list[this_class] <- auc(roc_obj)
  }
  
  # Plot first ROC
  plot(roc_list[[1]], col = "black", lty = lty_values[1],
       lwd = 2, main = paste("", model_name), 
       cex.main = 1.8,      # title size
       cex.lab = 1.5,       # axis titles size
       cex.axis = 1.3)
  
  # Add remaining ROCs
  for(i in 2:length(classes)){
    plot(roc_list[[i]], col = "black", lty = lty_values[i], lwd = 2, add = TRUE)
  }
  
  # Add legend with class labels starting from 1
  legend("bottomright", 
         legend = paste(new_labels, "AUC:", round(auc_list, 3)),
         col = "black", lty = lty_values, lwd = 2, cex = 0.8)
}
##############

# Logistic regression
plot_multiclass_roc_bw(test$los_class, pred_prob_logit, "Logistic Regression")

# XGBoost
plot_multiclass_roc_bw(test$los_class, pred_prob_xgb, "XGBoost")

# ANN
plot_multiclass_roc_bw(test_scaled$los_class, pred_prob_ann, "ANN")



######  GET AUC values 
library(pROC)

# Function to compute AUCs and optionally plot ROC curves
get_multiclass_auc <- function(actual, pred_probs, model_name = NULL, plot = TRUE){
  classes <- levels(factor(actual))
  roc_list <- list()
  auc_list <- c()
  lty_values <- 1:length(classes)
  
  for(i in seq_along(classes)){
    this_class <- classes[i]
    binary_actual <- ifelse(actual == this_class, 1, 0)
    roc_obj <- roc(binary_actual, pred_probs[, i])
    roc_list[[this_class]] <- roc_obj
    auc_list[this_class] <- auc(roc_obj)
  }
  
  if(plot){
    plot(roc_list[[1]], col = "black", lty = lty_values[1],
         lwd = 2, main = ifelse(is.null(model_name), "ROC per class", paste("ROC per LOS class -", model_name)),
         cex.main = 1.8,
         cex.lab = 1.5,
         cex.axis = 1.3)
    
    for(i in 2:length(classes)){
      plot(roc_list[[i]], col = "black", lty = lty_values[i], lwd = 2, add = TRUE)
    }
    
    legend("bottomright", legend = paste(classes, "AUC:", round(auc_list, 3)),
           col = "black", lty = lty_values, lwd = 2, cex = 0.8)
  }
  
  return(round(auc_list, 3))
}

# Logistic Regression
auc_logit <- get_multiclass_auc(test$los_class, pred_prob_logit, "Logistic Regression", plot = FALSE)

# XGBoost
auc_xgb <- get_multiclass_auc(test$los_class, pred_prob_xgb, "XGBoost", plot = FALSE)

# ANN
auc_ann <- get_multiclass_auc(test_scaled$los_class, pred_prob_ann, "ANN", plot = FALSE)

# Combine into a data frame
auc_df <- data.frame(
  Model = c("Logistic Regression", "XGBoost", "ANN"),
  `Class 0-5` = c(auc_logit[1], auc_xgb[1], auc_ann[1]),
  `Class 5-10` = c(auc_logit[2], auc_xgb[2], auc_ann[2]),
  `Class 10-20` = c(auc_logit[3], auc_xgb[3], auc_ann[3]),
  `Class 20-30` = c(auc_logit[4], auc_xgb[4], auc_ann[4]),
  `Class 90+` = c(auc_logit[5], auc_xgb[5], auc_ann[5])
)

print(auc_df)

#### Save XGBoost model 
setwd("~/Jasgalym/4. R model")
saveRDS(xgb_model, "xgb_model.rds")
