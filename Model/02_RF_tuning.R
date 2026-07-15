##############################################################################
# 02_RF_tuning.R
# RANDOM FOREST
# Hyperparameter tuning + Final model
# For demonstration purposes, synthetic data approximating the distributional 
# properties of the original administrative dataset is used in place of the real data.
##############################################################################

rm(list = ls())

##############################################################################
# PACKAGES
##############################################################################

library(caret)
library(ranger)

##############################################################################
# LOAD DATA
##############################################################################

train_df <- readRDS("synthetic_train.rds")
test_df  <- readRDS("synthetic_test.rds.rds")

##############################################################################
# VERIFY DATASET
##############################################################################

cat("Training observations :", nrow(train_df), "\n")
cat("Testing observations  :", nrow(test_df), "\n")
cat("Predictors            :", ncol(train_df)-1, "\n")

stopifnot(
  identical(names(train_df), names(test_df))
)

stopifnot(
  !"death" %in% names(train_df),
  !"discharged" %in% names(train_df),
  !"self_discharge" %in% names(train_df),
  !"los" %in% names(train_df),
  !"id" %in% names(train_df)
)

##############################################################################
# 10% SAMPLE FOR TUNING
##############################################################################

set.seed(123)

sample_idx <- createDataPartition(
  train_df$los_class,
  p = 0.10,
  list = FALSE
)

train_tune <- train_df[sample_idx, ]

##############################################################################
# GRID
##############################################################################

p <- ncol(train_tune)-1

rf_grid <- expand.grid(
  
  mtry = c(
    floor(sqrt(p)),
    floor(p/4),
    floor(p/3)
  ),
  
  min.node.size = c(
    1,
    5,
    10
  )
  
)

rf_grid

##############################################################################
# MACRO F1 FUNCTION
##############################################################################

macro_f1 <- function(actual, predicted){
  
  cm <- confusionMatrix(predicted, actual)
  
  precision <- cm$byClass[, "Pos Pred Value"]
  recall    <- cm$byClass[, "Sensitivity"]
  
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

cv_rf <- function(
    
  mtry,
  min.node.size,
  data,
  k=5
  
){
  
  folds <- createFolds(
    data$los_class,
    k=k
  )
  
  scores <- c()
  
  for(i in seq_along(folds)){
    
    test_idx <- folds[[i]]
    
    cv_train <- data[-test_idx,]
    cv_test  <- data[test_idx,]
    
    model <- ranger(
      
      los_class~.,
      
      data=cv_train,
      
      num.trees=300,
      
      mtry=mtry,
      
      min.node.size=min.node.size,
      
      probability=TRUE,
      
      classification=TRUE
      
    )
    
    pred_prob <- predict(
      model,
      cv_test
    )$predictions
    
    pred_class <- factor(
      
      colnames(pred_prob)[max.col(pred_prob)],
      
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

results_rf <- data.frame()

for(i in 1:nrow(rf_grid)){
  
  cat(
    "\nTesting model",
    i,
    "of",
    nrow(rf_grid),
    "\n"
  )
  
  score <- cv_rf(
    
    mtry = rf_grid$mtry[i],
    
    min.node.size = rf_grid$min.node.size[i],
    
    data = train_tune
    
  )
  
  results_rf <- rbind(
    
    results_rf,
    
    data.frame(
      
      mtry = rf_grid$mtry[i],
      
      min.node.size = rf_grid$min.node.size[i],
      
      MacroF1 = score
      
    )
    
  )
  
  print(results_rf)
  
}

##############################################################################
# BEST PARAMETERS
##############################################################################

best_row <- results_rf[
  which.max(results_rf$MacroF1),
]

best_mtry <- best_row$mtry
best_node <- best_row$min.node.size

cat("\nBest mtry =",best_mtry,"\n")
cat("Best min.node.size =",best_node,"\n")

##############################################################################
# FINAL MODEL
##############################################################################
library(ranger)
rf_final <- ranger(
  
  los_class~.,
  
  data=train_df,
  
  num.trees=500,
  
  mtry=best_mtry,
  
  min.node.size=best_node,
  
  probability=TRUE,
  
  classification=TRUE,
  
  importance="impurity",
  
  seed=123
  
)

##############################################################################
# TEST PREDICTIONS
##############################################################################

rf_prob <- predict(
  rf_final,
  test_df
)$predictions

pred_rf <- factor(
  
  colnames(rf_prob)[max.col(rf_prob)],
  
  levels=levels(test_df$los_class)
  
)

##############################################################################
# CONFUSION MATRIX
##############################################################################

cm_rf <- confusionMatrix(
  
  pred_rf,
  
  test_df$los_class
  
)

print(cm_rf)

##############################################################################
# VARIABLE IMPORTANCE
##############################################################################

importance_rf <- sort(
  
  rf_final$variable.importance,
  
  decreasing=TRUE
  
)

importance_rf

##############################################################################
# SAVE RESULTS
##############################################################################
save.image(file = "RF.RData")

