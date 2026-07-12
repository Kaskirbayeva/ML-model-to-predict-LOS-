##############################################################################
# 08_External_Validation_2023.R
# External Temporal Validation — All Models, 2023 Dataset
##############################################################################

rm(list = ls())

library(caret)
library(pROC)
library(nnet)

##############################################################################
# 1. LOAD AND PREPARE 2023 (EXTERNAL) DATA
##############################################################################

df2023 <- readRDS(df2023.rds")
# Apply same LOS class derivation as 2022
df2023$los_class <- cut(
  df2023$los,
  breaks = c(0, 5, 10, 20, 30, 90),
  labels = c("0", "1", "2", "3", "4"),
  right = FALSE,
  include.lowest = TRUE
)

# Apply same exclusions as 2022
drop_vars <- c(
  "id",
  "los",
  "death",
  "discharged",
  "self_discharge"
)

validation_df <- df2023[
  , !(names(df2023) %in% drop_vars)
]

##############################################################################
# 2. LOAD 2022 TRAINING DATA 
##############################################################################

train_df <- readRDS("train_df.rds")
test_df  <- readRDS("test_df.rds")

##############################################################################
# 3. SEPARATE PREDICTORS AND OUTCOME
##############################################################################

X_val <- as.matrix(validation_df[, names(validation_df) != "los_class"])
y_val <- validation_df$los_class

classes <- levels(train_df$los_class)

##############################################################################
# 4. LOAD FINAL TUNED MODELS
##############################################################################

load("RF.RData")   # rf_final
load("XGB.RData")        # xgb_final
load("LightGBM.RData")       # lgb_final
load("ANN.RData")            # ann_final, preproc_final
load("MultiLR.RData")  # multinom_model

##############################################################################
# 5. GENERATE PREDICTIONS ON EXTERNAL (2023) SET
##############################################################################

## ---------- Random Forest ----------

prob_rf_ext <- predict(rf_final, data = X_val, type = "response")$predictions
colnames(prob_rf_ext) <- classes
pred_rf_ext <- factor(classes[max.col(prob_rf_ext)], levels = classes)## ---------- XGBoost ----------

## ---------- XGBoost ----------
prob_xgb_ext <- predict(xgb_final, X_val)
prob_xgb_ext <- matrix(prob_xgb_ext, ncol = length(classes), byrow = TRUE)
colnames(prob_xgb_ext) <- classes
pred_xgb_ext <- factor(classes[max.col(prob_xgb_ext)], levels = classes)

## ---------- LightGBM ----------

prob_lgb_ext <- predict(lgb_final, X_val)
prob_lgb_ext <- matrix(prob_lgb_ext, ncol = length(classes), byrow = TRUE)
colnames(prob_lgb_ext) <- classes
pred_lgb_ext <- factor(classes[max.col(prob_lgb_ext)], levels = classes)

## ---------- ANN ----------

X_val_scaled <- predict(preproc_final, X_val)
prob_ann_ext <- predict(ann_final, X_val_scaled, type = "raw")
colnames(prob_ann_ext) <- classes
pred_ann_ext <- factor(classes[max.col(prob_ann_ext)], levels = classes)

## ---------- Multinomial Logistic Regression ----------

prob_multinom_ext <- predict(multinom_model, newdata = validation_df, type = "probs")
colnames(prob_multinom_ext) <- classes
pred_multinom_ext <- factor(classes[max.col(prob_multinom_ext)], levels = classes)

##############################################################################
# 6. MODEL REGISTRY (EXTERNAL SET)
##############################################################################

models_ext <- list(
  "Random Forest"                    = list(pred = pred_rf_ext,       prob = prob_rf_ext),
  "XGBoost"                          = list(pred = pred_xgb_ext,      prob = prob_xgb_ext),
  "LightGBM"                         = list(pred = pred_lgb_ext,      prob = prob_lgb_ext),
  "Artificial Neural Network (ANN)"  = list(pred = pred_ann_ext,      prob = prob_ann_ext),
  "Multinomial Logistic Regression"  = list(pred = pred_multinom_ext, prob = prob_multinom_ext)
)

##############################################################################
# 7. CLASS-SPECIFIC METRICS WITH 95% BOOTSTRAP CI — EXTERNAL SET
##############################################################################

set.seed(123)
B <- 1000

results_ext <- data.frame()

for(model_name in names(models_ext)){
  
  cat("\n")
  cat("=====================================\n")
  cat("External validation:", model_name, "\n")
  cat("=====================================\n")
  
  pred_model <- models_ext[[model_name]]$pred
  prob_model <- models_ext[[model_name]]$prob
  
  for(k in seq_along(classes)){
    
    positive <- classes[k]
    
    ## ---------- Point estimates ----------
    
    obs_bin <- factor(
      ifelse(y_val == positive, positive, "Other"),
      levels = c(positive, "Other")
    )
    
    pred_bin <- factor(
      ifelse(pred_model == positive, positive, "Other"),
      levels = c(positive, "Other")
    )
    
    cm <- confusionMatrix(
      pred_bin,
      obs_bin,
      positive = positive
    )
    
    precision <- as.numeric(cm$byClass["Pos Pred Value"])
    recall <- as.numeric(cm$byClass["Sensitivity"])
    
    if(is.na(precision)) precision <- 0
    if(is.na(recall)) recall <- 0
    
    f1 <- ifelse(
      precision + recall == 0,
      0,
      2 * precision * recall / (precision + recall)
    )
    
    auc_est <- tryCatch({
      
      roc_obj <- roc(
        response = obs_bin,
        predictor = prob_model[,positive],
        levels = c("Other", positive),
        quiet = TRUE
      )
      
      as.numeric(auc(roc_obj))
      
    }, error=function(e) NA)
    
    ## ---------- Bootstrap ----------
    
    boot <- matrix(NA, B, 4)
    
    for(b in 1:B){
      
      idx <- sample(
        seq_len(nrow(validation_df)),
        replace = TRUE
      )
      
      obs_b <- factor(
        ifelse(y_val[idx] == positive, positive, "Other"),
        levels = c(positive, "Other")
      )
      
      pred_b <- factor(
        ifelse(pred_model[idx] == positive, positive, "Other"),
        levels = c(positive, "Other")
      )
      
      cm_b <- tryCatch(
        
        confusionMatrix(
          pred_b,
          obs_b,
          positive = positive
        ),
        
        error=function(e) NULL
        
      )
      
      if(!is.null(cm_b)){
        
        p <- as.numeric(cm_b$byClass["Pos Pred Value"])
        r <- as.numeric(cm_b$byClass["Sensitivity"])
        
        if(is.na(p)) p <- 0
        if(is.na(r)) r <- 0
        
        f <- ifelse(
          p+r==0,
          0,
          2*p*r/(p+r)
        )
        
        a <- tryCatch({
          
          roc_obj <- roc(
            response = obs_b,
            predictor = prob_model[idx,positive],
            levels = c("Other",positive),
            quiet=TRUE
          )
          
          as.numeric(auc(roc_obj))
          
        }, error=function(e) NA)
        
        boot[b,] <- c(p,r,f,a)
        
      }
      
    }
    
    results_ext <- rbind(
      
      results_ext,
      
      data.frame(
        
        Model=model_name,
        
        Class=positive,
        
        Precision=sprintf(
          "%.3f (%.3f\u2013%.3f)",
          precision,
          quantile(boot[,1],0.025,na.rm=TRUE),
          quantile(boot[,1],0.975,na.rm=TRUE)
        ),
        
        Recall=sprintf(
          "%.3f (%.3f\u2013%.3f)",
          recall,
          quantile(boot[,2],0.025,na.rm=TRUE),
          quantile(boot[,2],0.975,na.rm=TRUE)
        ),
        
        F1=sprintf(
          "%.3f (%.3f\u2013%.3f)",
          f1,
          quantile(boot[,3],0.025,na.rm=TRUE),
          quantile(boot[,3],0.975,na.rm=TRUE)
        ),
        
        AUC=sprintf(
          "%.3f (%.3f\u2013%.3f)",
          auc_est,
          quantile(boot[,4],0.025,na.rm=TRUE),
          quantile(boot[,4],0.975,na.rm=TRUE)
        ),
        
        row.names = NULL
        
      )
      
    )
    
  }
  
}

print(results_ext)

write.csv(
  results_ext,
  "External_2023_Class_Metrics_95CI.csv",
  row.names = FALSE
)

##############################################################################
# 8. OVERALL METRICS PER MODEL — EXTERNAL SET
##############################################################################

log_loss <- function(actual, predicted_probs, eps = 1e-15){
  
  actual_matrix <- model.matrix(~ actual - 1)
  colnames(actual_matrix) <- levels(actual)
  
  predicted_probs <- predicted_probs[, colnames(actual_matrix)]
  predicted_probs <- pmin(pmax(predicted_probs, eps), 1 - eps)
  
  -mean(rowSums(actual_matrix * log(predicted_probs)))
  
}

overall_ext <- data.frame()

for(model_name in names(models_ext)){
  
  pred_model <- models_ext[[model_name]]$pred
  prob_model <- models_ext[[model_name]]$prob
  
  cm <- confusionMatrix(pred_model, y_val)
  
  precision_per_class <- cm$byClass[, "Pos Pred Value"]
  recall_per_class    <- cm$byClass[, "Sensitivity"]
  
  precision_per_class[is.na(precision_per_class)] <- 0
  recall_per_class[is.na(recall_per_class)] <- 0
  
  f1_per_class <- ifelse(
    precision_per_class + recall_per_class == 0,
    0,
    2 * precision_per_class * recall_per_class /
      (precision_per_class + recall_per_class)
  )
  
  overall_ext <- rbind(
    
    overall_ext,
    
    data.frame(
      Model           = model_name,
      Accuracy        = round(as.numeric(cm$overall["Accuracy"]), 3),
      Kappa           = round(as.numeric(cm$overall["Kappa"]), 3),
      MacroPrecision  = round(mean(precision_per_class), 3),
      MacroRecall     = round(mean(recall_per_class), 3),
      MacroF1         = round(mean(f1_per_class), 3),
      LogLoss         = round(log_loss(y_val, prob_model), 3)
    )
    
  )
  
}

overall_ext <- overall_ext[order(-overall_ext$MacroF1), ]

print(overall_ext)

write.csv(
  overall_ext,
  "External_2023_Overall_Metrics.csv",
  row.names = FALSE
)

##############################################################################
# 9. INTERNAL (2022 TEST) vs EXTERNAL (2023) PERFORMANCE COMPARISON
##############################################################################
# Requires the internal test-set overall metrics table to already exist
# (e.g. produced by evaluation_bootstrap_CI.R / overall metrics scripts
# and saved as Model_Comparison_Overall_Metrics.csv). Adjust the file
# name/model labels below if yours differ.

overall_int <- read.csv("Model_Comparison_Overall_Metrics.csv")

performance_comparison <- merge(
  overall_int[, c("Model","MacroF1")],
  overall_ext[, c("Model","MacroF1")],
  by = "Model",
  suffixes = c("_Internal","_External")
)

performance_comparison$F1_Delta <-
  performance_comparison$MacroF1_External - performance_comparison$MacroF1_Internal

performance_comparison <- performance_comparison[
  order(performance_comparison$F1_Delta),
]

print(performance_comparison)

write.csv(
  performance_comparison,
  "Internal_vs_External_Performance.csv",
  row.names = FALSE
)

##############################################################################
# 10. SAVE ALL EXTERNAL VALIDATION OUTPUTS
##############################################################################

save(
  validation_df,
  X_val,
  y_val,
  models_ext,
  results_ext,
  overall_ext,
  performance_comparison,
  missing_cols,
  extra_cols,
  file = "08_External_Validation_2023_results.RData"
)

cat("\n")
cat("=====================================\n")
cat("External validation (2023) complete.\n")
cat("=====================================\n")
