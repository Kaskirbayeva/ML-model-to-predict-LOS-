############################################################
# Shiny EHR Simulation App for XGBoost LOS Predictions
############################################################
# Load model and external data
setwd("~/Jasgalym/4. R model")
xgb_model <- readRDS("xgb_model.rds")
patients <- readRDS("patients.rds")  # prepared dataset

library(shiny)
library(DT)
library(dplyr)

# 1. UI

ui <- fluidPage(
  titlePanel("Simulated EHR: ML-Based LOS Predictions"),
  
  sidebarLayout(
    sidebarPanel(
      helpText("Interactive EHR interface showing predicted Length of Stay (LOS) and priority alerts."),
      
      selectInput(
        "filterCategory",
        "Filter by LOS Category:",
        choices = c("All", levels(factor(patients$LOS_Category))),
        selected = "All"
      ),
      
      checkboxInput(
        "showImpact",
        "Show estimated bed occupancy impact",
        value = TRUE
      )
    ),
    
    mainPanel(
      DTOutput("patientTable")
    )
  )
)


# 2. Server

server <- function(input, output) {
  
  output$patientTable <- renderDT({
    
    data_to_show <- patients
    
    # Filter by LOS category if selected
    if (input$filterCategory != "All") {
      data_to_show <- data_to_show %>% filter(LOS_Category == input$filterCategory)
    }
    
    # Add simulated bed impact
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
    
    # Render datatable with color-coded rows
    datatable(data_to_show, rownames = FALSE, options = list(pageLength = 10)) %>%
      formatStyle(
        'LOS_Category',
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
  
}


# 3. Run App

shinyApp(ui = ui, server = server)
