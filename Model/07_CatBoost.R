##############################################################################
# 07_CatBoost.R
# CATBOOST MULTICLASS MODEL 
# Hyperparameter tuning + Final model
# For demonstration purposes, synthetic data approximating the distributional 
# properties of the original administrative dataset is used in place of the real data.
##############################################################################
# LOAD DATA
##############################################################################

train_df <- readRDS("synthetic_train.rds")
test_df  <- readRDS("synthetic_test.rds.rds")
##############################################################################
# 1. DEFINE HYPERPARAMETER GRID FOR MODEL TUNING
##############################################################################

# Grid of CatBoost hyperparameters to evaluate.
#
# depth:
#   Controls tree depth.
#   Higher values allow more complex models but increase risk of overfitting.
#
# learning_rate:
#   Step size during boosting.
#   Lower values generally require more iterations but can improve generalisation.
#
# l2_leaf_reg:
#   L2 regularisation coefficient controlling model complexity.
#
cat_grid <- expand.grid(
  
  depth = c(4,6,8),
  
  learning_rate = c(
    0.03,
    0.05
  ),
  
  l2_leaf_reg = c(
    3,
    5
  )
  
)


##############################################################################
# 2. PREPARE TRAINING DATA FOR CATBOOST POOL
##############################################################################

# Remove outcome variable from predictors
predictors <- setdiff(
  names(train_tune),
  "los_class"
)


# Create predictor matrix
train_x <- as.data.frame(
  train_tune[, predictors]
)


# Create CatBoost Pool object.
#
# Pool is CatBoost's internal data structure containing:
# - predictor variables
# - target labels
#
# CatBoost multiclass labels must start from zero,
# therefore subtract 1 from factor encoding.
#
train_pool <- catboost.load_pool(
  
  data = train_x,
  
  label =
    as.numeric(train_tune$los_class)-1
  
)

##############################################################################
# 3. HYPERPARAMETER TUNING USING 5-FOLD CROSS-VALIDATION
##############################################################################

# Store tuning results
results_cat <- data.frame()


# Loop through every hyperparameter combination
for(i in 1:nrow(cat_grid)){
  
  cat(
    "\nModel",
    i,
    "of",
    nrow(cat_grid),
    "\n"
  )
    
  # Define CatBoost training parameters
  params <- list(
    
    # Multiclass classification objective
    loss_function = "MultiClass",
    
    # Metric used during CV optimisation
    eval_metric = "MultiClass",
    
    
    # Maximum number of boosting iterations
    iterations = 1000,
    
    
    # Hyperparameters from grid
    depth =
      cat_grid$depth[i],
    
    learning_rate =
      cat_grid$learning_rate[i],
    
    l2_leaf_reg =
      cat_grid$l2_leaf_reg[i],
    
    
    # Reproducibility
    random_seed = 123,
    
    
    # Early stopping settings
    # Training stops if validation performance does not improve
    od_type = "Iter",
    od_wait = 50,
    
    verbose = 0
    
  )
  
  
  # Perform 5-fold cross-validation
  cv <- catboost.cv(
    
    pool = train_pool,
    
    params = params,
    
    fold_count = 5,
    
    shuffle = TRUE,
    
    partition_random_seed = 123
    
  )
  
  
  # Identify optimal number of iterations
  best_iter <-
    which.min(
      cv$test.MultiClass.mean
    )
  
  
  # Save CV results
  results_cat <-
    rbind(
      
      results_cat,
      
      data.frame(
        
        depth =
          cat_grid$depth[i],
        
        learning_rate =
          cat_grid$learning_rate[i],
        
        l2_leaf_reg =
          cat_grid$l2_leaf_reg[i],
        
        iterations =
          best_iter,
        
        logloss =
          min(
            cv$test.MultiClass.mean
          )
        
      )
      
    )
  
  
  print(results_cat)
  
}

##############################################################################
# 4. SELECT BEST MODEL PARAMETERS
##############################################################################

# Select parameter combination with lowest validation log loss

best_row <- dplyr::slice_min(
  results_cat,
  logloss,
  n = 1
)

best_row

##############################################################################
# 5. TRAIN FINAL CATBOOST MODEL
##############################################################################

# Prepare final training and testing datasets

train_x <- train_df[, predictors]

test_x <- test_df[, predictors]


# Convert datasets into CatBoost Pool objects

train_pool <- catboost.load_pool(
  
  train_x,
  
  label =
    as.numeric(train_df$los_class)-1
  
)


test_pool <- catboost.load_pool(
  
  test_x,
  
  label =
    as.numeric(test_df$los_class)-1
  
)



# Train final CatBoost model using optimal parameters

cat_final <- catboost.train(
  
  learn_pool = train_pool,
  
  params = list(
    
    loss_function = "MultiClass",
    
    eval_metric = "MultiClass",
    
    
    # Parameters selected from CV
    iterations =
      best_row$iterations,
    
    depth =
      best_row$depth,
    
    learning_rate =
      best_row$learning_rate,
    
    l2_leaf_reg =
      best_row$l2_leaf_reg,
    
    
    random_seed = 123,
    
    verbose = 100
    
  )
  
)

##############################################################################
# 6. GENERATE PREDICTED PROBABILITIES
##############################################################################

# Obtain probability for each LOS class

cat_prob <- catboost.predict(
  
  cat_final,
  
  test_pool,
  
  prediction_type =
    "Probability"
  
)


# Convert output into probability matrix:
# Rows = observations
# Columns = LOS classes

cat_prob <- matrix(
  
  cat_prob,
  
  ncol =
    length(levels(test_df$los_class)),
  
  byrow = TRUE
  
)



##############################################################################
# 7. CONVERT PROBABILITIES INTO FINAL CLASS PREDICTIONS
##############################################################################

# Select class with highest predicted probability

pred_cat <- factor(
  
  levels(test_df$los_class)[
    max.col(cat_prob)
  ],
  
  levels =
    levels(test_df$los_class)
  
)

##############################################################################
# 8. CONFUSION MATRIX AND OVERALL PERFORMANCE
##############################################################################

library(caret)
library(MLmetrics)


cm_cat <- confusionMatrix(
  pred_cat,
  test_df$los_class
)


print(cm_cat)



##############################################################################
# 9. ACCURACY AND KAPPA
##############################################################################

accuracy_cat <- cm_cat$overall["Accuracy"]

kappa_cat <- cm_cat$overall["Kappa"]

##############################################################################
# 10. MACRO-AVERAGED PRECISION, RECALL AND F1 SCORE
##############################################################################

# Extract class-specific precision and recall

precision <- cm_cat$byClass[, "Pos Pred Value"]

recall <- cm_cat$byClass[, "Sensitivity"]


# Replace undefined values with zero

precision[is.na(precision)] <- 0

recall[is.na(recall)] <- 0



# Calculate class-level F1 score

f1 <- ifelse(
  
  precision + recall == 0,
  
  0,
  
  2 * precision * recall /
    (precision + recall)
  
)



# Average across all LOS classes

macro_precision <- mean(precision)

macro_recall <- mean(recall)

macro_f1 <- mean(f1)

##############################################################################
# 11. MULTICLASS LOG LOSS
##############################################################################

# Convert actual labels into one-hot encoded matrix

actual_matrix <- model.matrix(
  ~ test_df$los_class - 1
)


# Avoid numerical problems from log(0)

eps <- 1e-15


cat_prob <- pmin(
  pmax(cat_prob, eps),
  1 - eps
)



# Calculate multiclass log loss

logloss <- MLmetrics::MultiLogLoss(
  y_pred = cat_prob,
  y_true = actual_matrix
)

##############################################################################
# 12. RMSE OF CLASS PREDICTIONS
##############################################################################

rmse <- sqrt(
  mean(
    (
      as.numeric(pred_cat) -
        as.numeric(test_df$los_class)
    )^2
  )
)

##############################################################################
# 13. FINAL PERFORMANCE TABLE
##############################################################################

results <- data.frame(
  
  Model = "CatBoost",
  
  Accuracy = round(accuracy_cat,3),
  
  Kappa = round(kappa_cat,3),
  
  Macro_Precision = round(macro_precision,3),
  
  Macro_Recall = round(macro_recall,3),
  
  Macro_F1 = round(macro_f1,3),
  
  LogLoss = round(logloss,3),
  
  RMSE = round(rmse,3)
  
)


print(results)

##############################################################################
# 14. ADDITIONAL CHECK OF MACRO METRICS FROM CARET
##############################################################################

macro_precision <- mean(
  cm_cat$byClass[, "Pos Pred Value"],
  na.rm = TRUE
)


macro_recall <- mean(
  cm_cat$byClass[, "Sensitivity"],
  na.rm = TRUE
)


macro_f1 <- mean(
  cm_cat$byClass[, "F1"],
  na.rm = TRUE
)



cat(
  "\nMacro Precision =", round(macro_precision, 3),
  "\nMacro Recall    =", round(macro_recall, 3),
  "\nMacro F1        =", round(macro_f1, 3), 
  "\n"
)

##############################################################################
# 15. MANUAL VERIFICATION OF LOG LOSS
##############################################################################

# Generate probabilities again

prob <- catboost.predict(
  cat_final,
  test_pool,
  prediction_type = "Probability"
)


dim(prob)

head(prob)


# Check that probabilities sum to 1

rowSums(prob)[1:10]



# Convert true labels to zero-based indexing

actual <- as.integer(test_df$los_class) - 1



# Prevent log(0)

eps <- 1e-15


prob_safe <- pmax(
  pmin(prob, 1 - eps),
  eps
)



# Manual multiclass log loss calculation

logloss_cat <- -mean(
  log(
    prob_safe[
      cbind(
        seq_along(actual),
        actual + 1
      )
    ]
  )
)



logloss_cat

##############################################################################
# 16. SAVE COMPLETE R ENVIRONMENT
##############################################################################

save.image(
  file = "CatBoost.RData"
)

save(
  cat_final,
  predictors,
  classes,
  file = "CatBoost_model.RData"
)
