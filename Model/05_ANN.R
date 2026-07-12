##############################################################################
# ANN
# ANN (nnet) Hyperparameter Optimization and Final Model
##############################################################################

rm(list = ls())

library(caret)
library(nnet)

##############################################################################
# LOAD DATA
##############################################################################

train_df <- readRDS("train_df.rds")
test_df  <- readRDS("test_df.rds")

# Objects loaded:
# train_df
# test_df

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
# DESIGN MATRICES (SCALED — REQUIRED FOR ANN CONVERGENCE)
##############################################################################

X_tune_raw <- model.matrix(
  los_class ~ . - 1,
  data = train_tune
)


preproc_tune <- preProcess(
  X_tune_raw,
  method = c("center", "scale")
)

X_tune <- predict(preproc_tune, X_tune_raw)

y_tune <- train_tune$los_class

##############################################################################
# TUNING GRID
##############################################################################

ann_grid <- expand.grid(
  size  = c(5, 10, 15),
  decay = c(0.001, 0.01, 0.1)
)

print(ann_grid)

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

results_ann <- data.frame()

for(i in 1:nrow(ann_grid)){
  
  cat("\n")
  cat("=====================================\n")
  cat("Model",i,"of",nrow(ann_grid),"\n")
  cat("=====================================\n")
  
  fold_scores <- c()
  
  for(j in seq_along(folds)){
    
    cat(" Fold",j,"\n")
    
    test_idx <- folds[[j]]
    
    cv_train <- train_tune[-test_idx,]
    cv_test <- train_tune[test_idx,]
    
    X_train_cv_raw <- model.matrix(
      los_class~.-1,
      data=cv_train
    )
    
    X_test_cv_raw <- model.matrix(
      los_class~.-1,
      data=cv_test
    )
    
    preproc_cv <- preProcess(
      X_train_cv_raw,
      method = c("center","scale")
    )
    
    X_train_cv <- predict(preproc_cv, X_train_cv_raw)
    X_test_cv  <- predict(preproc_cv, X_test_cv_raw)
    
    y_train_cv <- cv_train$los_class
    
    set.seed(123)
    
    model <- nnet(
      
      x = X_train_cv,
      y = class.ind(y_train_cv),
      
      size  = ann_grid$size[i],
      decay = ann_grid$decay[i],
      
      softmax = TRUE,
      maxit   = 300,
      trace   = FALSE,
      MaxNWts = 10000
      
    )
    
    pred_prob <- predict(
      model,
      X_test_cv,
      type = "raw"
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
  
  results_ann <- rbind(
    
    results_ann,
    
    data.frame(
      
      size  = ann_grid$size[i],
      decay = ann_grid$decay[i],
      
      MacroF1=mean(fold_scores)
      
    )
    
  )
  
  print(results_ann)
  
  saveRDS(
    results_ann,
    "results_ann_partial.rds"
  )
  
}

##############################################################################
# BEST MODEL
##############################################################################

best_row <- results_ann[
  which.max(results_ann$MacroF1),
]

print(best_row)

best_size  <- best_row$size
best_decay <- best_row$decay

##############################################################################
# TRAIN FINAL MODEL
##############################################################################

X_train_raw <- model.matrix(
  los_class~.-1,
  data=train_df
)

X_test_raw <- model.matrix(
  los_class~.-1,
  data=test_df
)

preproc_final <- preProcess(
  X_train_raw,
  method = c("center","scale")
)

X_train <- predict(preproc_final, X_train_raw)
X_test  <- predict(preproc_final, X_test_raw)

y_train <- train_df$los_class

set.seed(123)

ann_final <- nnet(
  
  x = X_train,
  y = class.ind(y_train),
  
  size  = best_size,
  decay = best_decay,
  
  softmax = TRUE,
  maxit   = 500,
  trace   = FALSE,
  MaxNWts = 10000
  
)

##############################################################################
# TEST SET PREDICTIONS
##############################################################################

ann_prob <- predict(
  ann_final,
  X_test,
  type = "raw"
)

colnames(ann_prob) <- levels(train_df$los_class)

pred_ann <- factor(
  
  levels(train_df$los_class)[max.col(ann_prob)],
  
  levels=levels(train_df$los_class)
  
)

##############################################################################
# PERFORMANCE
##############################################################################

cm_ann <- confusionMatrix(
  pred_ann,
  test_df$los_class
)

print(cm_ann)

##############################################################################
# SAVE OUTPUTS
##############################################################################

save.image(file = "ANN.RData")
