##############################################################################
# evaluation_bootstrap_CI.R
# Class-Specific Metrics with 95% Bootstrap CI — ALL MODELS
##############################################################################
# This script assumes each model script (02_RandomForest_tuning.R through
# 06_MultinomialLR.R) has already been run and has saved its predictions
# and predicted probabilities as .rds files, e.g.:
#
#   pred_rf.rds        / prob_rf.rds
#   pred_xgb.rds       / prob_xgb.rds     (already saved in 03_XGBoost_tuning.R)
#   pred_lgb.rds       / prob_lgb.rds
#   pred_ann.rds       / prob_ann.rds
#   pred_multinom.rds  / prob_multinom.rds (already saved in 06_MultinomialLR.R)
#
# Adjust the file paths below if your naming convention differs.
##############################################################################

rm(list = ls())

library(caret)
library(pROC)

##############################################################################
# LOAD TEST DATA
##############################################################################

load("01_data_preparation.RData")

# Objects loaded:
# train_df
# test_df

##############################################################################
# LOAD MODEL PREDICTIONS
##############################################################################

pred_rf        <- readRDS("pred_rf.rds")
prob_rf        <- readRDS("prob_rf.rds")

pred_xgb       <- readRDS("pred_xgb.rds")
prob_xgb       <- readRDS("prob_xgb.rds")

pred_lgb       <- readRDS("pred_lgb.rds")
prob_lgb       <- readRDS("prob_lgb.rds")

pred_ann       <- readRDS("pred_ann.rds")
prob_ann       <- readRDS("prob_ann.rds")

pred_multinom  <- readRDS("pred_multinom.rds")
prob_multinom  <- readRDS("prob_multinom.rds")

##############################################################################
# STANDARDISE PROBABILITY MATRICES
##############################################################################
# Ensure every probability matrix has column names matching the class
# levels, so it can be indexed by name (prob[,positive]) regardless of
# how each model script originally constructed it.

classes <- levels(test_df$los_class)

colnames(prob_rf)       <- classes
colnames(prob_xgb)      <- classes
colnames(prob_lgb)      <- classes
colnames(prob_ann)      <- classes
colnames(prob_multinom) <- classes

##############################################################################
# MODEL REGISTRY
##############################################################################
# Add or remove models here — the evaluation loop below is fully generic
# and will iterate over every entry in this list.

models <- list(
  "Random Forest"                    = list(pred = pred_rf,       prob = prob_rf),
  "XGBoost"                          = list(pred = pred_xgb,      prob = prob_xgb),
  "LightGBM"                         = list(pred = pred_lgb,      prob = prob_lgb),
  "Artificial Neural Network (ANN)"  = list(pred = pred_ann,      prob = prob_ann),
  "Multinomial Logistic Regression"  = list(pred = pred_multinom, prob = prob_multinom)
)

##############################################################################
# CLASS-SPECIFIC METRICS WITH 95% BOOTSTRAP CI
##############################################################################

set.seed(123)
B <- 1000

results_all <- data.frame()

for(model_name in names(models)){
  
  cat("\n")
  cat("=====================================\n")
  cat("Evaluating:", model_name, "\n")
  cat("=====================================\n")
  
  pred_model <- models[[model_name]]$pred
  prob_model <- models[[model_name]]$prob
  
  for(k in seq_along(classes)){
    
    positive <- classes[k]
    
    ## ---------- Point estimates ----------
    
    obs_bin <- factor(
      ifelse(test_df$los_class == positive, positive, "Other"),
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
        seq_len(nrow(test_df)),
        replace = TRUE
      )
      
      obs_b <- factor(
        ifelse(test_df$los_class[idx] == positive,
               positive,
               "Other"),
        levels = c(positive, "Other")
      )
      
      pred_b <- factor(
        ifelse(pred_model[idx] == positive,
               positive,
               "Other"),
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
    
    results_all <- rbind(
      
      results_all,
      
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

##############################################################################
# RESULTS
##############################################################################

print(results_all)

write.csv(
  results_all,
  "All_Models_Class_Metrics_95CI.csv",
  row.names = FALSE
)

##############################################################################
# MACRO-AVERAGED SUMMARY PER MODEL (OPTIONAL)
##############################################################################
# Quick reference table: mean point-estimate F1 and AUC per model,
# averaged across all LOS classes (useful for a single-row model comparison).

extract_point_estimate <- function(x){
  as.numeric(sub("^([0-9.]+).*", "\\1", x))
}

summary_all <- aggregate(
  cbind(
    F1  = extract_point_estimate(results_all$F1),
    AUC = extract_point_estimate(results_all$AUC)
  ) ~ Model,
  data = results_all,
  FUN = mean
)

summary_all <- summary_all[order(-summary_all$F1), ]

print(summary_all)

write.csv(
  summary_all,
  "All_Models_MacroAveraged_Summary.csv",
  row.names = FALSE
)

cat("\n")
cat("=====================================\n")
cat("Bootstrap CI evaluation complete.\n")
cat("=====================================\n")
