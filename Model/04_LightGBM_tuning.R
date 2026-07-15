##############################################################################
# 04_LightGBM_tuning.R
# LightGBM
# Hyperparameter tuning + Final model
# For demonstration purposes, synthetic data approximating the distributional 
# properties of the original administrative dataset is used in place of the real data.
##############################################################################

library(lightgbm)
library(caret)

##############################################################################
# LOAD DATA
##############################################################################

train_df <- readRDS("synthetic_train.rds")
test_df  <- readRDS("synthetic_test.rds.rds")

##############################################################################
# 10% TUNING SAMPLE
##############################################################################

set.seed(123)

sample_idx <- createDataPartition(
  train_df$los_class,
  p = 0.10,
  list = FALSE
)

train_tune <- train_df[sample_idx, ]

##############################################################################
# TUNING GRID
##############################################################################

lgb_grid <- expand.grid(
  num_leaves = c(31, 63),
  learning_rate = c(0.05, 0.10),
  feature_fraction = c(0.8),
  bagging_fraction = c(0.8)
)

##############################################################################
# MACRO F1
##############################################################################

macro_f1 <- function(actual, predicted){
  
  cm <- confusionMatrix(predicted, actual)
  
  precision <- cm$byClass[, "Pos Pred Value"]
  recall <- cm$byClass[, "Sensitivity"]
  
  precision[is.na(precision)] <- 0
  recall[is.na(recall)] <- 0
  
  f1 <- ifelse(
    precision + recall == 0,
    0,
    2 * precision * recall /
      (precision + recall)
  )
  
  mean(f1)
}

##############################################################################
# CROSS VALIDATION
##############################################################################

cv_lgb <- function(
    num_leaves,
    learning_rate,
    feature_fraction,
    bagging_fraction,
    data,
    k=5
){
  
  folds <- createFolds(data$los_class,k=k)
  
  scores <- c()
  
  for(i in seq_along(folds)){
    
    cat("Fold",i,"of",k,"\n")
    
    test_idx <- folds[[i]]
    
    cv_train <- data[-test_idx,]
    cv_test <- data[test_idx,]
    
    X_train <- as.matrix(
      cv_train[,setdiff(names(cv_train),"los_class")]
    )
    
    X_test <- as.matrix(
      cv_test[,setdiff(names(cv_test),"los_class")]
    )
    
    y_train <- as.numeric(cv_train$los_class)-1
    
    dtrain <- lgb.Dataset(
      data=X_train,
      label=y_train
    )
    
    params <- list(
      objective="multiclass",
      num_class=5,
      metric="multi_logloss",
      num_leaves=num_leaves,
      learning_rate=learning_rate,
      feature_fraction=feature_fraction,
      bagging_fraction=bagging_fraction,
      bagging_freq=5,
      verbosity=-1
    )
    
    model <- lightgbm(
      data=dtrain,
      params=params,
      nrounds=200
    )
    
    pred_prob <- predict(
      model,
      X_test
    )
    
    # pred_prob <- matrix(
    #   pred_prob,
    #   ncol=5,
    #   byrow=TRUE
    # )
    
    pred_class <- factor(
      as.character(max.col(pred_prob)-1),
      levels=levels(data$los_class)
    )
    
    scores <- c(
      scores,
      macro_f1(
        cv_test$los_class,
        pred_class
      )
    )
  }
  
  mean(scores)
  
}

##############################################################################
# GRID SEARCH
##############################################################################

results_lgb <- data.frame()

for(i in 1:nrow(lgb_grid)){
  
  cat(
    "\nTesting model",
    i,
    "of",
    nrow(lgb_grid),
    "\n"
  )
  
  score <- cv_lgb(
    num_leaves=lgb_grid$num_leaves[i],
    learning_rate=lgb_grid$learning_rate[i],
    feature_fraction=lgb_grid$feature_fraction[i],
    bagging_fraction=lgb_grid$bagging_fraction[i],
    data=train_tune
  )
  
  results_lgb <- rbind(
    results_lgb,
    data.frame(
      num_leaves=lgb_grid$num_leaves[i],
      learning_rate=lgb_grid$learning_rate[i],
      feature_fraction=lgb_grid$feature_fraction[i],
      bagging_fraction=lgb_grid$bagging_fraction[i],
      MacroF1=score
    )
  )
  
  print(results_lgb)
  
}

##############################################################################
# BEST MODEL
##############################################################################

best_row <- results_lgb[
  which.max(results_lgb$MacroF1),
]

best_row

##############################################################################
# TRAIN FINAL MODEL
##############################################################################

X_train <- as.matrix(
  train_df[,setdiff(names(train_df),"los_class")]
)

y_train <- as.numeric(train_df$los_class)-1

dtrain <- lgb.Dataset(
  data=X_train,
  label=y_train
)

params <- list(
  objective="multiclass",
  num_class=5,
  metric="multi_logloss",
  num_leaves=best_row$num_leaves,
  learning_rate=best_row$learning_rate,
  feature_fraction=best_row$feature_fraction,
  bagging_fraction=best_row$bagging_fraction,
  bagging_freq=5,
  verbosity=-1
)

lgb_final <- lightgbm(
  data=dtrain,
  params=params,
  nrounds=300
)

##############################################################################
# TEST SET PREDICTIONS
##############################################################################

X_test <- as.matrix(
  test_df[,setdiff(names(test_df),"los_class")]
)

lgb_prob <- predict(
  lgb_final,
  X_test
)

# lgb_prob <- matrix(
#   lgb_prob,
#   ncol=5,
#   byrow=TRUE
# )

pred_lgb <- factor(
  as.character(max.col(lgb_prob)-1),
  levels=levels(test_df$los_class)
)

cm_lgb <- confusionMatrix(
  pred_lgb,
  test_df$los_class
)

print(cm_lgb)

##############################################################################
# FEATURE IMPORTANCE
##############################################################################

importance_lgb <- lgb.importance(
  lgb_final,
  percentage = TRUE
)

print(importance_lgb)

##############################################################################
# SAVE RESULTS
##############################################################################
save.image(file = "LightGBM.RData")
