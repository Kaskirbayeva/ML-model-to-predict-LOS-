# Load 8XGboost.RData
library(shapviz)
# 1 Prepare your test data as a matrix (required)
X_test_matrix <- as.matrix(X_test_top)  # your top features subset
# Compute SHAP values
sv <- shapviz(final_model_top, X_pred = data.matrix(X_test_matrix), X = X_test_matrix)

# Two types of visualizations
sv_waterfall(sv, row_id = 1)
sv_force(sv, row_id = 1)

# Three types of variable importance plots
sv_importance(sv) +  
  theme_minimal()
sv_importance(sv, kind = "bar")
sv_importance(sv, kind = "both", alpha = 0.2, width = 0.2)



sv_importance(sv)+
  ggplot2::theme_classic(base_size = 12)

sv_importance(sv, kind = "beeswarm")

###
shp_i <- shapviz(
  final_model_top, X_pred = data.matrix(X_test_matrix), X = X_test_matrix, interactions = TRUE
)
sv_dependence(shp_i, v = "log_carat", color_var = xvars, interactions = TRUE)