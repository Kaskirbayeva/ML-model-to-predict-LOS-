############################################################
# Shiny EHR Simulation App with XGBoost LOS Predictions
# and Dynamic Impact Metrics
############################################################

# 1. Load libraries
library(shiny)
library(DT)
library(dplyr)
library(xgboost)

# 2. Load external dataset
setwd("~/Jasgalym/1.Data preparation")
load("df_all_23.RData")  # external dataset
df_2023 <- df_all_23

# Create LOS classes
df_2023$los_class <- cut(
  df_2023$los,
  breaks = c(0,5,10,20,30,90),
  labels = 0:4,
  right = FALSE
)
df_2023$los_class <- as.factor(df_2023$los_class)

# Remove unnecessary columns and NAs
df_2023_clean <- df_2023 %>% select(-id, -los, -'NA') %>% drop_na()

# 3. Prepare feature matrix
X_ext <- model.matrix(los_class ~ . -1, df_2023_clean)

# 4. Load trained XGBoost model
setwd("~/Jasgalym/4. R model")
xgb_model <- readRDS("xgb_model.rds")
train_features <- xgb_model$feature_names

# Align external features with training features
ext_features <- colnames(X_ext)

# Add missing columns with zeros
missing_cols <- setdiff(train_features, ext_features)
for (col in missing_cols) {
  X_ext <- cbind(X_ext, rep(0, nrow(X_ext)))
  colnames(X_ext)[ncol(X_ext)] <- col
}

# Remove extra columns
extra_cols <- setdiff(ext_features, train_features)
if(length(extra_cols) > 0){
  X_ext <- X_ext[, !(colnames(X_ext) %in% extra_cols)]
}

# Reorder columns to match training
X_ext <- X_ext[, train_features]

# Convert to DMatrix
dext <- xgb.DMatrix(data = X_ext)

# 5. Predict LOS classes with XGBoost
pred_prob_ext <- predict(xgb_model, dext)
pred_matrix <- matrix(pred_prob_ext, ncol = 5, byrow = TRUE)
pred_xgb_class <- max.col(pred_matrix) - 1

# Map predicted classes to readable labels
patients <- df_2023_clean
patients$LOS_Pred <- factor(pred_xgb_class,
                            labels = c("0-5 days","5-10 days","10-20 days","20-30 days","30-90 days"))

# Optional: Priority flag based on predicted LOS
patients$Priority_Flag <- ifelse(patients$LOS_Pred %in% c("20-30 days","30-90 days"), "High",
                                 ifelse(patients$LOS_Pred %in% c("10-20 days"), "Moderate", "Low"))

# Map LOS classes to numeric midpoints
patients$Predicted_LOS_days <- case_when(
  trimws(as.character(patients$LOS_Pred)) == "0-5 days"   ~ 3,
  trimws(as.character(patients$LOS_Pred)) == "5-10 days"  ~ 7,
  trimws(as.character(patients$LOS_Pred)) == "10-20 days" ~ 15,
  trimws(as.character(patients$LOS_Pred)) == "20-30 days" ~ 25,
  trimws(as.character(patients$LOS_Pred)) == "30-90 days" ~ 60,
  TRUE ~ NA_real_
)

patients$Baseline_LOS_days <- case_when(
  df_2023_clean$los_class == 0 ~ 3,
  df_2023_clean$los_class == 1 ~ 7,
  df_2023_clean$los_class == 2 ~ 15,
  df_2023_clean$los_class == 3 ~ 25,
  df_2023_clean$los_class == 4 ~ 60,
  TRUE ~ NA_real_
)

patients <- patients %>% filter(!is.na(Predicted_LOS_days) & !is.na(Baseline_LOS_days))

############################################################
# 6. Shiny UI
############################################################
ui <- fluidPage(
  titlePanel("Simulated EHR: ML-Based LOS Predictions & Impact"),
  
  sidebarLayout(
    sidebarPanel(
      helpText("Interactive EHR interface showing predicted LOS and efficiency impact."),
      
      selectInput(
        "filterCategory",
        "Filter by LOS Category:",
        choices = c("All", levels(patients$LOS_Pred)),
        selected = "All"
      ),
      
      checkboxInput(
        "showImpact",
        "Show estimated bed occupancy impact",
        value = TRUE
      )
    ),
    
    mainPanel(
      DTOutput("patientTable"),
      br(),
      tableOutput("impactSummary")
    )
  )
)

############################################################
# 7. Shiny Server
############################################################
server <- function(input, output) {
  
  # Reactive filtered dataset
  filtered_patients <- reactive({
    data <- patients
    if (input$filterCategory != "All") {
      data <- data %>% filter(LOS_Pred == input$filterCategory)
    }
    data
  })
  
  # Render patient table
  output$patientTable <- renderDT({
    data_to_show <- filtered_patients()
    
    if (input$showImpact) {
      data_to_show <- data_to_show %>%
        mutate(BedImpact = case_when(
          Priority_Flag == "High" ~ "Reduce delay by 1-2 days",
          Priority_Flag == "Moderate" ~ "Reduce delay by 0-1 day",
          TRUE ~ "Minimal impact"
        ))
    } else {
      data_to_show <- data_to_show %>% select(-BedImpact)
    }
    
    datatable(data_to_show, rownames = FALSE, options = list(pageLength = 10)) %>%
      formatStyle(
        'LOS_Pred',
        target = 'row',
        backgroundColor = styleEqual(
          c("0-5 days","5-10 days","10-20 days","20-30 days","30-90 days"),
          c('#d4edda','#fff3cd','#ffe0b2','#ffcccb','#f8d7da')
        )
      ) %>%
      formatStyle(
        'Priority_Flag',
        target = 'row',
        fontWeight = styleEqual(c("High","Moderate","Low"), c("bold","normal","normal"))
      )
  })
  
  # Render impact summary table
  output$impactSummary <- renderTable({
    if(!input$showImpact) return(NULL)
    data <- filtered_patients()
    
    total_beddays_baseline <- sum(data$Baseline_LOS_days)
    total_beddays_predicted <- sum(data$Predicted_LOS_days)
    total_beddays_improvement <- (total_beddays_baseline - total_beddays_predicted)/total_beddays_baseline * 100
    
    baseline_delays <- sum(data$Baseline_LOS_days > 10)
    predicted_delays <- sum(data$Predicted_LOS_days > 10)
    reduction_delays <- ifelse(baseline_delays>0,
                               (baseline_delays - predicted_delays)/baseline_delays * 100, 0)
    
    avg_bed_turnover_baseline <- mean(data$Baseline_LOS_days)
    avg_bed_turnover_predicted <- mean(data$Predicted_LOS_days)
    bed_turnover_improvement <- (avg_bed_turnover_baseline - avg_bed_turnover_predicted)/avg_bed_turnover_baseline * 100
    
    data.frame(
      Metric = c("Total Bed-Days", "Delayed Discharges (>10d)", "Average Bed Turnover (days)"),
      Baseline = c(total_beddays_baseline, baseline_delays, round(avg_bed_turnover_baseline,2)),
      Intervention = c(total_beddays_predicted, predicted_delays, round(avg_bed_turnover_predicted,2)),
      Improvement_Percent = c(
        round(total_beddays_improvement,1),
        round(reduction_delays,1),
        round(bed_turnover_improvement,1)
      )
    )
  })
  
}

############################################################
# 8. Run the Shiny App
############################################################
shinyApp(ui = ui, server = server)
