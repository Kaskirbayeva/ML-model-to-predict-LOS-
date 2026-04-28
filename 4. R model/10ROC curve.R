library(pROC)
library(ggplot2)

# Initialize list to hold ROC objects
roc_list <- list()
auc_values <- c()

# Loop over each class for one-vs-all ROC
for (k in 0:4) {
  # Create binary actual: 1 for class k, 0 otherwise
  actual_binary <- as.numeric(y_test == k)
  
  # Ensure it's a factor with two levels: 0, 1
  if (length(unique(actual_binary)) < 2) next
  actual_binary <- factor(actual_binary, levels = c(0, 1))
  
  # Predicted probability for class k
  probs_k <- pred_matrix[, k + 1]  # +1 since R is 1-indexed
  
  # Compute ROC curve
  roc_k <- roc(actual_binary, probs_k, legacy.axes = TRUE, quiet = TRUE)
  auc_k <- auc(roc_k)
  
  roc_list[[paste0("Class_", k)]] <- roc_k
  auc_values <- c(auc_values, auc_k)
}

# Plot all ROC curves in one plot
plot(roc_list[[1]], col = 1, lwd = 2,
     main = "Class-specific ROC Curves",
     legacy.axes = TRUE)
for (i in 2:length(roc_list)) {
  plot(roc_list[[i]], col = i, lwd = 2, add = TRUE)
}

legend("bottomright",
       legend = paste0(names(roc_list), " (AUC = ", round(auc_values, 3), ")"),
       col = 1:length(roc_list),
       lwd = 2,
       cex = 0.4,
       bty = "n")
