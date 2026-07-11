##############################################################################
# 06_MultinomialLR.R
# Multinomial Logistic Regression
##############################################################################

library(nnet)
library(caret)

##############################################################################
# LOAD DATA
##############################################################################

train_df <- readRDS("train_df.rds")
test_df  <- readRDS("test_df.rds")

##############################################################################
# TRAIN MULTINOMIAL LOGISTIC REGRESSION
##############################################################################

set.seed(123)

multinom_model <- multinom(
  los_class ~ .,
  data = train_df,
  trace = FALSE,
  MaxNWts = 10000,
  maxit = 300
)

##############################################################################
# PREDICT CLASS PROBABILITIES
##############################################################################

multinom_prob <- predict(
  multinom_model,
  newdata = test_df,
  type = "probs"
)

multinom_prob <- as.matrix(multinom_prob)

##############################################################################
# PREDICT CLASSES
##############################################################################

pred_multinom <- factor(
  colnames(multinom_prob)[max.col(multinom_prob)],
  levels = levels(test_df$los_class)
)

##############################################################################
# CONFUSION MATRIX
##############################################################################

cm_multinom <- confusionMatrix(
  pred_multinom,
  test_df$los_class
)

print(cm_multinom)

##############################################################################
# VARIABLE IMPORTANCE (OPTIONAL)
##############################################################################

coef_multinom <- coef(multinom_model)

importance_multinom <- apply(
  abs(coef_multinom),
  2,
  mean
)

importance_multinom <- sort(
  importance_multinom,
  decreasing = TRUE
)

print(importance_multinom)

##############################################################################
# SAVE MODEL
##############################################################################

saveRDS(
  multinom_model,
  "multinom_model.rds"
)

saveRDS(
  importance_multinom,
  "importance_multinom.rds"
)

##############################################################################
# SAVE RESULTS
##############################################################################
save.image(file = "MultiLR.RData")
