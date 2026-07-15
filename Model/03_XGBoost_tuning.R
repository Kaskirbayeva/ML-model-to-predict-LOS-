##############################################################################
# 03_XGBoost_tuning.R
# XGBoost 
# Hyperparameter tuning + Final model
# For demonstration purposes, synthetic data approximating the distributional 
# properties of the original administrative dataset is used in place of the real data.
##############################################################################

rm(list = ls())

library(caret)
library(xgboost)

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
# DESIGN MATRICES
##############################################################################

X_tune <- model.matrix(
  los_class ~ . - 1,
  data = train_tune
)

y_tune <- as.numeric(train_tune$los_class) - 1

##############################################################################
# TUNING GRID
##############################################################################

xgb_grid <- expand.grid(
  max_depth = c(4,6,8),
  eta = c(0.03,0.05,0.10),
  subsample = c(0.8),
  colsample_bytree = c(0.8)
)

print(xgb_grid)

##############################################################################
# MACRO F1 FUNCTION
##############################################################################

macro_f1 <- function(actual,predicted){
  
  cm <- confusionMatrix(
    predicted,
    actual
  )
  
  precision <- cm$byClass[,"Pos Pred Value"]
  recall <- cm$byClass[,"Sensitivity"]
  
  precision[is.na(precision)] <- 0
  recall[is.na(recall)] <- 0
  
  f1 <- ifelse(
    precision + recall == 0,
    0,
    2*precision*recall/(precision+recall)
  )
  
  mean(f1)
  
}

##############################################################################
# 5-FOLD CROSS-VALIDATION
##############################################################################

folds <- createFolds(
  train_tune$los_class,
  k=5
)

results_xgb <- data.frame()

for(i in 1:nrow(xgb_grid)){
  
  cat("\n")
  cat("=====================================\n")
  cat("Model",i,"of",nrow(xgb_grid),"\n")
  cat("=====================================\n")
  
  fold_scores <- c()
  
  for(j in seq_along(folds)){
    
    cat(" Fold",j,"\n")
    
    test_idx <- folds[[j]]
    
    cv_train <- train_tune[-test_idx,]
    cv_test <- train_tune[test_idx,]
    
    X_train_cv <- model.matrix(
      los_class~.-1,
      data=cv_train
    )
    
    X_test_cv <- model.matrix(
      los_class~.-1,
      data=cv_test
    )
    
    y_train_cv <- as.numeric(cv_train$los_class)-1
    
    dtrain <- xgb.DMatrix(
      data=X_train_cv,
      label=y_train_cv
    )
    
    model <- xgb.train(
      
      params=list(
        
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        
        max_depth=xgb_grid$max_depth[i],
        eta=xgb_grid$eta[i],
        subsample=xgb_grid$subsample[i],
        colsample_bytree=xgb_grid$colsample_bytree[i]
        
      ),
      
      data=dtrain,
      
      nrounds=200,
      
      verbose=0
      
    )
    
    pred_prob <- predict(
      model,
      X_test_cv
    )
    
    pred_prob <- matrix(
      pred_prob,
      ncol=5,
      byrow=TRUE
    )
    
    pred_class <- factor(
      
      levels(train_df$los_class)[max.col(pred_prob)],
      
      levels=levels(train_df$los_class)
      
    )
    
    fold_scores <- c(
      
      fold_scores,
      
      macro_f1(
        cv_test$los_class,
        pred_class
      )
      
    )
    
  }
  
  results_xgb <- rbind(
    
    results_xgb,
    
    data.frame(
      
      max_depth=xgb_grid$max_depth[i],
      
      eta=xgb_grid$eta[i],
      
      subsample=xgb_grid$subsample[i],
      
      colsample_bytree=xgb_grid$colsample_bytree[i],
      
      MacroF1=mean(fold_scores)
      
    )
    
  )
  
  print(results_xgb)
  
  saveRDS(
    results_xgb,
    "results_xgb_partial.rds"
  )
  
}

##############################################################################
# BEST MODEL
##############################################################################

best_row <- results_xgb[
  which.max(results_xgb$MacroF1),
]

print(best_row)

best_depth <- best_row$max_depth
best_eta <- best_row$eta
best_subsample <- best_row$subsample
best_colsample <- best_row$colsample_bytree

##############################################################################
# TRAIN FINAL MODEL
##############################################################################

X_train <- model.matrix(
  los_class~.-1,
  data=train_df
)

X_test <- model.matrix(
  los_class~.-1,
  data=test_df
)

y_train <- as.numeric(train_df$los_class)-1

dtrain <- xgb.DMatrix(
  X_train,
  label=y_train
)

xgb_final <- xgb.train(
  
  params=list(
    
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    
    max_depth=best_depth,
    eta=best_eta,
    subsample=best_subsample,
    colsample_bytree=best_colsample
    
  ),
  
  data=dtrain,
  
  nrounds=300,
  
  verbose=0
  
)

##############################################################################
# TEST SET PREDICTIONS
##############################################################################

xgb_prob <- predict(
  xgb_final,
  X_test
)

xgb_prob <- matrix(
  
  xgb_prob,
  
  ncol=5,
  
  byrow=TRUE
  
)

pred_xgb <- factor(
  
  levels(train_df$los_class)[max.col(xgb_prob)],
  
  levels=levels(train_df$los_class)
  
)

##############################################################################
# PERFORMANCE
##############################################################################

cm_xgb <- confusionMatrix(
  pred_xgb,
  test_df$los_class
)

print(cm_xgb)

##############################################################################
# SAVE RESULTS
##############################################################################

save.image(file = "XGB.RData")
