##############################################################################
# evaluation_bootstrap_CI.R
# Class-Specific Metrics with 95% Bootstrap CI — ALL MODELS
##############################################################################
#   RF.RData            -> pred_rf,       rf_prob
#   XGB.RData           -> pred_xgb,      xgb_prob
#   LightGBM.RData      -> pred_lgb,      lgb_prob
#   ANN.RData           -> pred_ann,      ann_prob
#   MultiLR.RData       -> pred_multinom, multinom_prob
#
# Each load() call brings its objects into the global environment under
# their original names
##############################################################################

rm(list = ls())

library(caret)
library(pROC)

##############################################################################
# LOAD TEST DATA
##############################################################################

train_df <- readRDS("train_df.rds")
test_df  <- readRDS("test_df.rds")

##############################################################################
# LOAD MODEL RESULTS (.RData FILES)
##############################################################################

load("Rf.RData")   
load("XGB.RData")       
load("LightGBM.RData")       
load("ANN.RData")            
load("MultiLR.RData") 

##############################################################################
# STANDARDISE PROBABILITY MATRICES
##############################################################################

classes <- levels(test_df$los_class)

colnames(rf_prob)        <- classes
colnames(xgb_prob)       <- classes
colnames(lgb_prob)       <- classes
colnames(ann_prob)       <- classes
colnames(multinom_prob)  <- classes

##############################################################################
# MODEL REGISTRY
##############################################################################

models <- list(
  "Random Forest"                    = list(pred = pred_rf,       prob = rf_prob),
  "XGBoost"                          = list(pred = pred_xgb,      prob = xgb_prob),
  "LightGBM"                         = list(pred = pred_lgb,      prob = lgb_prob),
  "Artificial Neural Network (ANN)"  = list(pred = pred_ann,      prob = ann_prob),
  "Multinomial Logistic Regression"  = list(pred = pred_multinom, prob = multinom_prob)
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

##############################################################################
# SAVE COMBINED RESULTS OBJECT
##############################################################################

save(
  results_all,
  summary_all,
  file = "evaluation_bootstrap_CI_results.RData"
)

cat("\n")
cat("=====================================\n")
cat("Bootstrap CI evaluation complete.\n")
cat("=====================================\n")
