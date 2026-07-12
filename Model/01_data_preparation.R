df2022_clean <- readRDS("C:/Users/Daliy/OneDrive/Documents/Jasgalym/5. Publications/Data/df2022_clean.rds")
df2022_clean$los_class <- cut(
  df2022_clean$los,
  breaks = c(0, 5, 10, 20, 30, 90),
  labels = c("0", "1", "2", "3", "4"),
  right = FALSE,
  include.lowest = TRUE
)

drop_vars <- c(
  "id",
  "los",
  "death",
  "discharged",
  "self_discharge"
)

model_df <- df2022_clean[
  ,
  !(names(df2022_clean) %in% drop_vars)
]

set.seed(123)

library(caret)

train_index <- createDataPartition(
  model_df$los_class,
  p = 0.80,
  list = FALSE
)

train_df <- model_df[train_index, ]
test_df  <- model_df[-train_index, ]

saveRDS(train_df,"train_df.rds")
saveRDS(test_df,"test_df.rds")

