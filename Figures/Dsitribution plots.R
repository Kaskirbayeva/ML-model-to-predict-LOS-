##############################################################################
# LOS CLASS DISTRIBUTION (2022 vs 2023)
##############################################################################

library(dplyr)
library(ggplot2)
library(scales)

#----------------------------------------------------------
# Create LOS classes
#----------------------------------------------------------

df2022 <- df2022 %>%
  mutate(
    LOS_class = cut(
      los,
      breaks = c(0, 5, 10, 20, 30, 90),
      right = FALSE,
      include.lowest = TRUE,
      labels = c("0–4", "5–9", "10–19", "20–29", "30–89")
    )
  )

df2023 <- df2023 %>%
  mutate(
    LOS_class = cut(
      los,
      breaks = c(0, 5, 10, 20, 30, 90),
      right = FALSE,
      include.lowest = TRUE,
      labels = c("0–4", "5–9", "10–19", "20–29", "30–89")
    )
  )

#----------------------------------------------------------
# Count observations
#----------------------------------------------------------

dist2022 <- df2022 %>%
  count(LOS_class) %>%
  mutate(Dataset = "2022")

dist2023 <- df2023 %>%
  count(LOS_class) %>%
  mutate(Dataset = "2023")

los_dist <- bind_rows(dist2022, dist2023)

library(scales)

# Plot
#----------------------------------------------------------

p <- ggplot(los_dist,
            aes(x = LOS_class,
                y = n,
                fill = Dataset)) +
  
  geom_col(
    position = position_dodge(width = 0.75),
    width = 0.65,
    colour = "black",
    linewidth = 0.3
  ) +
  
  geom_text(
    aes(label = comma(n)),
    position = position_dodge(width = 0.75),
    vjust = -0.35,
    size = 3.3
  ) +
  
  scale_fill_grey(start = 0.35, end = 0.75) +
  
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult = c(0, 0.08))
  ) +
  
  labs(
    x = "LOS days",
    y = "Number of hospitalisations",
    fill = "Dataset"
  ) +
  
  theme_bw(base_size = 14) +
  
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold")
  )

p

##############################################################################
# Distribution of Length of Stay (Histogram)
##############################################################################

library(ggplot2)

ppp <- ggplot(df2023, aes(x = los)) +
  
  geom_histogram(
    binwidth = 0.5,
    colour = "black",
    fill = "grey70",
    linewidth = 0.2
  ) +
  
  coord_cartesian(xlim = c(0, 50)) +
  
  scale_x_continuous(
    breaks = seq(0, 50, 5)
  ) +
  
  labs(
    title = "Distribution of hospital length of stay",
    x = "LOS days",
    y = "Number of hospitalisations"
  ) +
  
  theme_bw(base_size = 14) +
  
  theme(
    panel.grid.major = element_line(colour = "grey90"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold")
  )

ppp

ggsave(
  "LOS_distribution_histogram.tiff",
  p,
  width = 7,
  height = 5,
  dpi = 600,
  compression = "lzw"
)

##############################################################################
# Boxplot of LOS by Admission Outcome
##############################################################################

library(ggplot2)
library(dplyr)

library(dplyr)

df_plot <- df2022 %>%
  mutate(
    admission_outcome = case_when(
      death == 1 ~ "Death",
      discharged == 1 ~ "Discharged",
      referred == 1 ~ "Referred",
      self_discharge == 1 ~ "Self-discharge",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(admission_outcome))

df_plot$admission_outcome <- factor(
  df_plot$admission_outcome,
  levels = c("Death",
             "Discharged",
             "Referred",
             "Self-discharge")
)

p <- ggplot(df_plot,
            aes(x = admission_outcome,
                y = los)) +
  
  geom_boxplot(
    fill = "grey90",
    colour = "black",
    linewidth = 0.6,
    outlier.shape = 1,
    outlier.size = 0.8
  ) +
  
  coord_cartesian(ylim = c(0, 30)) +
  
  labs(
    x = "Admission outcome",
    y = "LOS days"
  ) +
  
  theme_bw(base_size = 14) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    ),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

p

ggsave(
  "LOS_Boxplot_AdmissionOutcome.tiff",
  p,
  width = 7,
  height = 5,
  dpi = 600,
  compression = "lzw"
)


##############################################################################
# Mean Length of Stay by Age Group
##############################################################################

library(dplyr)
library(ggplot2)

#------------------------------------------------------------
# Reconstruct age groups
#------------------------------------------------------------

df_age <- df2022 %>%
  mutate(
    AgeGroup = case_when(
      child == 1         ~ "0–4",
      newborn == 1       ~ "0–4",      # Remove this line if newborn is a separate category
      person == 1        ~ "5–17",
      young_adult == 1   ~ "18–44",
      middle_adult == 1  ~ "45–71",
      senior == 1        ~ "72+",
      TRUE               ~ NA_character_
    )
  ) %>%
  filter(!is.na(AgeGroup))

#------------------------------------------------------------
# Mean LOS
#------------------------------------------------------------

age_summary <- df_age %>%
  group_by(AgeGroup) %>%
  summarise(
    Mean_LOS = mean(los, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

age_summary$AgeGroup <- factor(
  age_summary$AgeGroup,
  levels = c("0–4", "5–17", "18–44", "45–71", "72+")
)

#------------------------------------------------------------
# Plot
#------------------------------------------------------------

p <- ggplot(age_summary,
            aes(x = AgeGroup,
                y = Mean_LOS)) +
  
  geom_col(
    fill = "grey70",
    colour = "black",
    width = 0.65
  ) +
  
  geom_text(
    aes(label = sprintf("%.2f", Mean_LOS)),
    vjust = -0.35,
    size = 4
  ) +
  
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.08))
  ) +
  
  labs(
    x = "Age group",
    y = "Mean LOS days"
  ) +
  
  theme_bw(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

p
