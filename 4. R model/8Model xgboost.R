
#install.packages("xgboost")
library(caret)
library(gbm)
# Load required libraries
#load("~/Jasgalym/1.Data preparation/df_all.RData")
load("~/Jasgalym/1.Data preparation/df_all_new.RData")

# Set seed for reproducibility
set.seed(42)

library(dplyr)
#Remove id column as it is character variable 
df_all <- df_all[, !(names(df_all) %in% "id")]
features <- df_all[, !(names(df_all) %in% "los")]
LOS <- df_all$los  # returns a vector


# Split into 80% train, 20% test
train_index <- sample(seq_len(nrow(df_all)), size = 0.8 * nrow(df_all))#636453

X_train <- features[train_index, ]  #636453
X_test  <- features[-train_index, ] #159114
y_train <- LOS[train_index]  # vector
y_test  <- LOS[-train_index] # vector

library(xgboost) # for xgboost
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dtest  <- xgb.DMatrix(as.matrix(X_test), label = y_test)

#xgboost model
model <- xgboost(data = dtrain, nrounds = 100)

# generate predictions for our held-out testing data
pred <- predict(model, dtest)

# 2. Compute RMSE on test set
library(Metrics)

# Predict the mean LOS for all test observations
baseline_pred <- mean(y_train)

# Repeat that prediction to match test set length
baseline_preds <- rep(baseline_pred, length(y_test))

# Compute RMSE
baseline_rmse <- sqrt(mean((y_test - baseline_preds)^2))
cat("Baseline RMSE:", round(baseline_rmse, 3), "\n")

# 1. Make predictions on the test set
preds <- predict(model, dtest)

# 2. Calculate RMSE
model_rmse <- sqrt(mean((y_test - preds)^2))

# 3. Print the result
cat("Model RMSE:", round(model_rmse, 3), "\n")


#xgb.train()
library(xgboost)
library(ggplot2)

# 1. Prepare data
dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(as.matrix(X_test), label = y_test)

# 2. Watchlist for tracking train and test RMSE
watchlist <- list(train = dtrain, eval = dtest)

# 3. Train the model with evaluation logging
model <- xgb.train(
  data = dtrain,
  nrounds = 100,
  watchlist = watchlist,
  objective = "reg:squarederror",
  eval_metric = "rmse",
  verbose = 0
)

# 4. Extract evaluation log
eval_log1 <- model$evaluation_log

# 5. Plot using ggplot2
ggplot(eval_log1, aes(x = iter)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE")) +
  geom_line(aes(y = eval_rmse, color = "Test RMSE")) +
  labs(
    title = "XGBoost Learning Curve",
    x = "Boosting Iteration",
    y = "RMSE",
    color = "Dataset"
  ) +
  theme_minimal()

## Optimizing 
#Step 1: Early stopping to automatically pick the best number of rounds
#“Train up to 500 trees, but stop early if RMSE on test set doesn't improve for 10 rounds.”
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,  # start with a large number
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

#Stopping. Best iteration:
#[186]	train-rmse:5.089501	eval-rmse:5.254511

#Step 2: Hyperparameter Tuning (Grid Search)
# Let’s tune important hyperparameters using caret:

library(caret)
library(xgboost)

# Convert to training matrix
train_matrix <- xgb.DMatrix(as.matrix(X_train), label = y_train)

# Grid of hyperparameters
grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1),
  gamma = 0,
  colsample_bytree = c(0.7, 1),
  min_child_weight = 1,
  subsample = c(0.7, 1)
)

# Define training control
 
# Step 3: Review Best Model
# xgb_tuned$bestTune
# nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
# 48     200         9 0.1     0                1                1         1

#This means: the best hyperparameters have been found:
# nrounds           = 200
# max_depth         = 9
# eta               = 0.1
# gamma             = 0
# colsample_bytree  = 1
# min_child_weight  = 1
# subsample         = 1

# Next Step: Retrain the Final Model on Full Training Data
# Now that you’ve found the optimal parameters, retrain your final XGBoost model 
# using these on the full training data (X_train, y_train) — 
# with a test set watchlist and early stopping for safety:

# 1. Convert training and test sets to DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest  <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# 2. Define parameters based on bestTune
final_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 9,
  eta = 0.01,
  gamma = 0,
  colsample_bytree = 0.5,
  min_child_weight = 1,
  subsample = 1
)

# 3. Train the final model
final_model <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 500,  # from bestTune
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

#Evaluate on test set
preds <- predict(final_model, newdata = dtest)

library(Metrics)
rmse_test <- rmse(y_test, preds)
cat("Final Test RMSE:", round(rmse_test, 4), "\n")
eval_log2 <- final_model$evaluation_log

library(ggplot2)
ggplot(eval_log2, aes(x = iter)) +
  geom_line(aes(y = train_rmse, color = "Train RMSE")) +
  geom_line(aes(y = eval_rmse, color = "Test RMSE")) +
  labs(title = "Final XGBoost Learning Curve",
       x = "Boosting Iteration",
       y = "RMSE",
       color = "Dataset") +
  theme_minimal()
## Feature Importance in XGBoost
importance_matrix <- xgb.importance(model = final_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix, top_n = 20)


# Plot top 20 features
top_n <- 20
importance_df <- xgb.importance(model = final_model)
top_features <- importance_df[1:top_n, ]

ggplot(top_features, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#55a868", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Top Feature Importances from XGBoost Model",
    x = "Feature",
    y = "Gain (Importance Score)"
  ) +
  theme_minimal()

head(importance_df, 20)

#Now select the most important features, and drop least important
top_features <- importance_df$Feature[1:40]
X_train_top <- X_train[, top_features]
X_test_top  <- X_test[, top_features]

dtrain_top <- xgb.DMatrix(data = as.matrix(X_train_top), label = y_train)
dtest_top  <- xgb.DMatrix(data = as.matrix(X_test_top), label = y_test)

final_model_top <- xgb.train(
  params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 9,
    eta = 0.1,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  ),
  data = dtrain_top,
  watchlist = list(train = dtrain_top, eval = dtest_top),
  nrounds = 200,
  early_stopping_rounds = 10,
  verbose = 1
)

preds_top <- predict(final_model_top, dtest_top)

library(Metrics)
rmse_top <- rmse(y_test, preds_top)
cat("RMSE with top features only:", round(rmse_top, 4), "\n")
#with 40 top features: 5.253, whereas: 
#Model Variant	Test RMSE	Interpretation
# Full model (all features)	5.2455	Best performance — model fits well
# Baseline model          	5.273	Your reference point (slightly worse)
# Top 20 features only	    5.4023	Worse — dropping features reduced accuracy

# #Regularisation without dropping features 
params$lambda <- 1    # L2 regularization
params$alpha <- 0.1   # L1 regularization

#Get more robust RMSE estimates and avoid test/train split sensitivity:
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 200,
  nfold = 5,
  early_stopping_rounds = 10,
  verbose = 1,
  metrics = "rmse"
)


head(cv_model$evaluation_log)
cv_model$best_iteration

eval_log <- cv_model$evaluation_log

ggplot(eval_log, aes(x = iter)) +
  geom_line(aes(y = train_rmse_mean, color = "Train RMSE")) +
  geom_line(aes(y = test_rmse_mean, color = "Test RMSE")) +
  labs(title = "XGBoost Cross-Validation Learning Curve",
       x = "Boosting Iteration",
       y = "RMSE",
       color = "Dataset") +
  theme_minimal()
min(cv_model$evaluation_log$test_rmse_mean)


###install.packages("SHAPforxgboost")
###library(SHAPforxgboost)
# shap_values <- shap.values(xgb_model = final_model, X_train = as.matrix(X_train))
# shap.plot.summary(shap_values$shap_score)

xgb.save(final_model, "final_model_new.xgb")
###
ggplot(data.frame(True = y_test, Predicted = preds), aes(x = True, y = Predicted)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Predicted vs Actual LOS", x = "Actual LOS", y = "Predicted LOS") +
  theme_minimal()


###
residuals <- y_test - preds

ggplot(data.frame(True = y_test, Residuals = residuals), aes(x = True, y = Residuals)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residuals vs Actual LOS", x = "Actual LOS", y = "Residuals") +
  theme_minimal()


#save.image(file = "~/Jasgalym/1.Data preparation/8XGboost.RData")

#### Regression 

# 1. Prepare data (reuse train/test split from XGBoost)
# Features and target
X_train_reg <- X_train
X_test_reg  <- X_test
y_train_reg <- y_train
y_test_reg  <- y_test



# 2. Fit multiple linear regression model
lm_model <- lm(y_train_reg ~ ., data = X_train_reg)

# 3. Predict on test set
preds_lm <- predict(lm_model, newdata = X_test_reg)

# 4. Compute RMSE
rmse_lm <- rmse(y_test_reg, preds_lm)
cat("Linear Regression Test RMSE:", round(rmse_lm, 4), "\n")

# 5. Predicted vs Actual plot
ggplot(data.frame(True = y_test_reg, Predicted = preds_lm), aes(x = True, y = Predicted)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Linear Regression: Predicted vs Actual LOS", x = "Actual LOS", y = "Predicted LOS") +
  theme_minimal()

# 6. Residual plot
residuals_lm <- y_test_reg - preds_lm
ggplot(data.frame(True = y_test_reg, Residuals = residuals_lm), aes(x = True, y = Residuals)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Linear Regression Residuals", x = "Actual LOS", y = "Residuals") +
  theme_minimal()

# =========================================
# Optional: Regularized Regression (LASSO/Ridge)
# =========================================
# LASSO example using glmnet
library(glmnet)

# Convert data to matrix
X_train_mat <- as.matrix(X_train_reg)
X_test_mat  <- as.matrix(X_test_reg)

# Fit LASSO with cross-validation
cv_lasso <- cv.glmnet(X_train_mat, y_train_reg, alpha = 1, nfolds = 5)
lasso_best_lambda <- cv_lasso$lambda.min

# Predict using best lambda
preds_lasso <- predict(cv_lasso, s = lasso_best_lambda, newx = X_test_mat)
rmse_lasso <- rmse(y_test_reg, preds_lasso)
cat("LASSO Regression Test RMSE:", round(rmse_lasso, 4), "\n")



