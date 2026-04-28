############################################################
# External Validation Dataset (2023)
############################################################

load("~/Jasgalym/1.Data preparation/df_all_23.RData")
df_2023<-df_all_23

df_2023$los_class <- cut(
  df_2023$los,
  breaks = c(0,5,10,20,30,90),
  labels = 0:4,
  right = FALSE
)

df_2023$los_class <- as.factor(df_2023$los_class)

df_2023 <- df_2023 %>% select(-id, -los, -'NA')

df_2023_clean <- df_2023 %>% 
  drop_na()

X_ext <- model.matrix(los_class ~ . -1, df_2023_clean)

y_ext <- as.numeric(df_2023_clean$los_class) - 1

dext <- xgb.DMatrix(data = X_ext, label = y_ext)

############################################################
# External validation – XGBoost
############################################################
X_ext <- model.matrix(los_class ~ . -1, df_2023_clean)
X_ext_df <- as.data.frame(X_ext)
missing_cols <- setdiff(colnames(X_train), colnames(X_ext_df))

for(col in missing_cols){
  X_ext_df[, col] <- 0
}

extra_cols <- setdiff(colnames(X_ext_df), colnames(X_train))

X_ext_df <- X_ext_df[, !(colnames(X_ext_df) %in% extra_cols)]

X_ext_df <- X_ext_df[, colnames(X_train)]

X_ext <- as.matrix(X_ext_df)
dext <- xgb.DMatrix(data = X_ext, label = y_ext)


pred_prob_xgb_ext <- predict(xgb_model, dext)
pred_matrix_ext <- matrix(pred_prob_xgb_ext, ncol = 5, byrow = TRUE)
pred_xgb_ext <- max.col(pred_matrix_ext) - 1
accuracy_xgb_ext <- mean(pred_xgb_ext == y_ext)


############################################################
# External validation – ANN
############################################################
library(nnet)
library(caret)

# Convert training factors to dummy variables
train_dummies <- model.matrix(los_class ~ . - 1, train)

# Response
y_train_ann <- class.ind(train$los_class)  # one-hot encoding

# Fit ANN
ann_model_dummies <- nnet(
  x = train_dummies,
  y = y_train_ann,
  size = 5,
  maxit = 200,
  decay = 0.01,
  trace = FALSE
)

# Ensure columns match training dummies
ext_dummies_df <- as.data.frame(ext_dummies)

# Add missing columns (set to 0)
missing_cols <- setdiff(colnames(train_dummies), colnames(ext_dummies_df))

for(col in missing_cols){
  ext_dummies_df[[col]] <- 0
}

# Remove extra columns
extra_cols <- setdiff(colnames(ext_dummies_df), colnames(train_dummies))
ext_dummies_df <- ext_dummies_df[, !(colnames(ext_dummies_df) %in% extra_cols)]

# Reorder columns to match training
ext_dummies_df <- ext_dummies_df[, colnames(train_dummies)]
ext_dummies <- as.matrix(ext_dummies_df)


pred_prob_ann_ext <- predict(ann_model_dummies, ext_dummies, type = "raw")
pred_ann_ext <- max.col(pred_prob_ann_ext) - 1
y_ext <- as.numeric(df_2023_clean$los_class) - 1
accuracy_ann_ext <- mean(pred_ann_ext == y_ext)


### Compute metrics 

library(caret)

compute_metrics <- function(pred, actual){
  actual_factor <- factor(actual, levels = 0:4)
  pred_factor <- factor(pred, levels = 0:4)
  
  cm <- confusionMatrix(pred_factor, actual_factor)
  acc <- cm$overall["Accuracy"]
  per_class <- as.data.frame(cm$byClass[, c("Precision","Recall","F1")])
  per_class$Class <- rownames(per_class)
  per_class$F1[is.na(per_class$F1)] <- 0
  per_class$Precision[is.na(per_class$Precision)] <- 0
  per_class$Recall[is.na(per_class$Recall)] <- 0
  macro_F1 <- mean(per_class$F1)
  
  list(accuracy = acc, macro_F1 = macro_F1, per_class = per_class)
}

metrics_xgb_ext <- compute_metrics(pred_xgb_ext, y_ext)
metrics_ann_ext <- compute_metrics(pred_ann_ext, y_ext)

#Log loss 
log_loss <- function(y_true, y_prob){
  eps <- 1e-15
  y_prob <- pmax(pmin(y_prob, 1 - eps), eps)
  y_mat <- model.matrix(~ factor(y_true) - 1)
  -mean(rowSums(y_mat * log(y_prob)))
}

logloss_xgb_ext <- log_loss(y_ext, pred_matrix_ext)
logloss_ann_ext <- log_loss(y_ext, pred_prob_ann_ext)

results_ext <- data.frame(
  Model = c("XGBoost","ANN"),
  Accuracy = c(metrics_xgb_ext$accuracy, metrics_ann_ext$accuracy),
  Macro_F1 = c(metrics_xgb_ext$macro_F1, metrics_ann_ext$macro_F1),
  LogLoss = c(logloss_xgb_ext, logloss_ann_ext)
)

print(results_ext)
