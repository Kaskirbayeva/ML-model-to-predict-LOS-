library(tidyverse)
library(caret)
library(pROC)
library(ggplot2)
library(patchwork)

models <- list(
  RF = list(prob = rf_prob, pred = pred_rf),
  XGBoost = list(prob = xgb_prob, pred = pred_xgb),
  LightGBM = list(prob = lgb_prob, pred = pred_lgb),
  CatBoost = list(prob = cat_prob, pred = pred_cat),
  ANN = list(prob = ann_prob, pred = pred_ann)
)
actual <- factor(
  test_df$los_class,
  levels = c("0", "1", "2", "3", "4")
)
classes <- levels(actual)
###
roc_df <- data.frame()

for(model in names(models)){
  
  probs <- models[[model]]$prob
  
  for(i in seq_along(classes)){
    
    truth <- ifelse(actual==classes[i],1,0)
    
    roc_obj <- roc(
      truth,
      probs[,i],
      quiet=TRUE
    )
    
    tmp <- data.frame(
      
      FPR = 1-roc_obj$specificities,
      TPR = roc_obj$sensitivities,
      Model = model,
      Class = classes[i],
      AUC = round(as.numeric(auc(roc_obj)),3)
      
    )
    
    roc_df <- rbind(roc_df,tmp)
    
  }
  
}
###

library(pROC)

classes <- levels(actual)

# Line types (black & white)
ltys <- c(1, 2, 3, 4, 5)

# Start empty plot using first class
roc1 <- roc(actual == classes[1], rf_prob[,1], quiet = TRUE)

plot(
  roc1,
  legacy.axes = TRUE,
  col = "black",
  lwd = 2,
  lty = ltys[1],
  main = "Random Forest",
  xlab = "Specificity",
  ylab = "Sensitivity"
)

# Add remaining classes
aucs <- numeric(length(classes))
aucs[1] <- auc(roc1)

for(i in 2:length(classes)){
  
  roc_i <- roc(
    actual == classes[i],
    rf_prob[,i],
    quiet = TRUE
  )
  
  lines(
    roc_i,
    col = "black",
    lwd = 2,
    lty = ltys[i]
  )
  
  aucs[i] <- auc(roc_i)
}

# Diagonal reference line
abline(a = 1, b = -1, col = "grey80", lwd = 1)

legend(
  "bottomright",
  legend = paste0(
    "LOS ",
    classes,
    ": AUC = ",
    sprintf("%.3f", aucs)
  ),
  lty = ltys,
  lwd = 2,
  bty = "n",
  col = "black",
  cex = 0.9
)
####
plot_multiclass_roc <- function(prob, actual, model_name){
  
  classes <- levels(actual)
  ltys <- c(1,2,3,4,5)
  
  roc1 <- roc(actual == classes[1], prob[,1], quiet=TRUE)
  
  plot(
    roc1,
    legacy.axes = TRUE,
    col = "black",
    lwd = 2,
    lty = ltys[1],
    main = model_name,
    xlab = "Specificity",
    ylab = "Sensitivity"
  )
  
  aucs <- numeric(length(classes))
  aucs[1] <- auc(roc1)
  
  for(i in 2:length(classes)){
    
    r <- roc(actual == classes[i], prob[,i], quiet=TRUE)
    
    lines(
      r,
      col="black",
      lwd=2,
      lty=ltys[i]
    )
    
    aucs[i] <- auc(r)
  }
  
  abline(a=1,b=-1,col="grey80")
  
  legend(
    "bottomright",
    legend=paste0(
      "LOS ",
      classes,
      " (AUC=",
      sprintf("%.3f",aucs),
      ")"
    ),
    lty=ltys,
    lwd=2,
    col="black",
    bty="n",
    cex=0.9
  )
}
###

plot_multiclass_roc(rf_prob, actual, "Random Forest")

plot_multiclass_roc(xgb_prob, actual, "XGBoost")

plot_multiclass_roc(lgb_prob, actual, "LightGBM")

plot_multiclass_roc(cat_prob, actual, "CatBoost")

plot_multiclass_roc(ann_prob, actual, "ANN")

plot_multiclass_roc(multinom_prob, actual, "Multinomial LR")
####
##############################################################################
# Figure: Per-class Precision, Recall and F1-score
##############################################################################

library(tidyverse)
library(caret)

classes <- levels(actual)

#---------------------------------------------------
# Function
#---------------------------------------------------

get_metrics <- function(pred, truth){
  
  cm <- confusionMatrix(pred, truth)
  
  by <- as.data.frame(cm$byClass, check.names = FALSE)
  
  data.frame(
    Class = c("0–4","5–9","10–19","20–29","30–90"),
    Precision = by[["Pos Pred Value"]],
    Recall    = by[["Sensitivity"]],
    F1        = by[["F1"]]
  )
}
#---------------------------------------------------
# Calculate metrics
#---------------------------------------------------

metric_df <- bind_rows(
  
  get_metrics(pred_rf, actual)  %>% mutate(Model="RF"),
  
  get_metrics(pred_xgb, actual) %>% mutate(Model="XGBoost"),
  
  get_metrics(pred_lgb, actual) %>% mutate(Model="LightGBM"),
  
  get_metrics(pred_cat, actual) %>% mutate(Model="CatBoost"),
  
  get_metrics(pred_ann, actual) %>% mutate(Model="ANN")
  
)

metric_long <-
  
  metric_df %>%
  
  pivot_longer(
    
    Precision:F1,
    
    names_to="Metric",
    
    values_to="Value"
    
  )

metric_long$Metric <-
  
  factor(
    
    metric_long$Metric,
    
    levels=c(
      
      "Precision",
      
      "Recall",
      
      "F1"
      
    )
    
  )

metric_long$Model <-
  
  factor(
    
    metric_long$Model,
    
    levels=c(
      
      "RF",
      
      "XGBoost",
      
      "LightGBM",
      
      "CatBoost",
      
      "ANN"
      
    )
    
  )

#---------------------------------------------------
# Figure
#---------------------------------------------------

ggplot(
  
  metric_long,
  
  aes(
    
    Class,
    
    Value,
    
    fill=Model
    
  )
  
)+
  
  geom_col(
    
    position=position_dodge(0.8),
    
    width=0.72,
    
    colour="black",
    
    linewidth=.25
    
  )+
  
  facet_wrap(
    
    ~Metric,
    
    nrow=1
    
  )+
  
  scale_fill_grey(
    
    start=.2,
    
    end=.85
    
  )+
  
  coord_cartesian(
    
    ylim=c(0,1)
    
  )+
  
  labs(
    
    x="Length of stay class (days)",
    
    y="Score",
    
    fill="Model"
    
  )+
  
  theme_bw(base_size=13)+
  
  theme(
    
    panel.grid.major.y=element_line(
      
      colour="grey90"
      
    ),
    
    panel.grid.minor=element_blank(),
    
    strip.background=element_blank(),
    
    strip.text=element_text(
      
      face="bold",
      
      size=13
      
    ),
    
    axis.title=element_text(
      
      face="bold"
      
    ),
    
    legend.position="bottom",
    
    legend.title=element_text(
      
      face="bold"
      
    )
    
  )

ggsave(
  
  "Figure_PerClass_Metrics.tiff",
  
  width=11,
  
  height=4.8,
  
  dpi=600,
  
  compression="lzw"
  
)
