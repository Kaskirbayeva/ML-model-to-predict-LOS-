
############################################################
# PREPARE HOSPITAL ADMISSIONS DATA: 2022-2023
############################################################
# FUNCTION FOR DATA PREPARATION
############################################################

prepare_hospital_data <- function(df, sur_merged) {
  
  library(tidyverse)
  library(stringr)
  
  ##############################################################################
  # BASE SETUP
  ##############################################################################
  
  db <- df %>%
    mutate(id = as.character(rownames(.)))
  
  ##############################################################################
  # STEP I: ICD10
  ##############################################################################
  
  db <- db %>%
    mutate(
      icd10 = substr(ICD10_final, 1, 3),
      icd_gr = case_when(
        icd10 >= "A00" & icd10 <= "B99" ~ "I",
        icd10 >= "C00" & icd10 <= "D49" ~ "II",
        icd10 >= "D50" & icd10 <= "D89" ~ "III",
        icd10 >= "E00" & icd10 <= "E90" ~ "IV",
        icd10 >= "F00" & icd10 <= "F99" ~ "V",
        icd10 >= "G00" & icd10 <= "G99" ~ "VI",
        icd10 >= "H00" & icd10 <= "H95" ~ "VII",
        icd10 >= "I00" & icd10 <= "I99" ~ "IX",
        icd10 >= "J00" & icd10 <= "J99" ~ "X",
        icd10 >= "K00" & icd10 <= "K94" ~ "XI",
        icd10 >= "L00" & icd10 <= "L99" ~ "XII",
        icd10 >= "M00" & icd10 <= "M99" ~ "XIII",
        icd10 >= "N00" & icd10 <= "N99" ~ "XIV",
        icd10 >= "O00" & icd10 <= "O99" ~ "XV",
        icd10 >= "P00" & icd10 <= "P99" ~ "XVI",
        icd10 >= "Q00" & icd10 <= "Q99" ~ "XVII",
        icd10 >= "R00" & icd10 <= "R99" ~ "XVIII",
        icd10 >= "S00" & icd10 <= "T99" ~ "XIX",
        icd10 >= "V01" & icd10 <= "Y98" ~ "XX",
        icd10 >= "Z00" & icd10 <= "Z99" ~ "XXI",
        icd10 >= "U00" & icd10 <= "U99" ~ "XXII",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(icd_gr))
  
  icd_map <- c(
    I="infectious", II="neoplasms", III="blood", IV="endocrine",
    V="mental", VI="nervous", VII="eye", IX="circulatory",
    X="respiratory", XI="digestive", XII="skin",
    XIII="musculoskeletal", XIV="genitourinary",
    XV="pregnancy", XVI="perinatal", XVII="congenital",
    XVIII="clinical_findings", XIX="injury",
    XX="external", XXI="health_services", XXII="special"
  )
  
  db$diag <- icd_map[db$icd_gr]
  
  hdiag_item <- db %>%
    group_by(id) %>%
    summarise(diag = list(diag), .groups = "drop") %>%
    unnest(diag) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = diag, values_from = value, values_fill = 0)
  
  ##############################################################################
  # STEP II: AGE + GENDER
  ##############################################################################
  
  db <- db %>%
    mutate(
      age_cat = case_when(
        age < 5 ~ "newborn",
        age < 18 ~ "child",
        age < 45 ~ "young_adult",
        age < 72 ~ "middle_adult",
        TRUE ~ "senior"
      ),
      gender = ifelse(sex == "Мужской", "M", "F")
    )
  
  hage_item <- db %>%
    group_by(id) %>%
    summarise(age_cat = list(age_cat), .groups = "drop") %>%
    unnest(age_cat) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = age_cat, values_from = value, values_fill = 0)
  
  hgen_item <- db %>%
    group_by(id) %>%
    summarise(gender = list(gender), .groups = "drop") %>%
    unnest(gender) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = gender, values_from = value, values_fill = 0)
  
  ##############################################################################
  # STEP III: PAYMENT TYPE
  ##############################################################################
  
  db <- db %>%
    mutate(mshi = ifelse(str_detect(employment, "FT"), "person", "gov"))
  
  hosms_item <- db %>%
    group_by(id) %>%
    summarise(mshi = list(mshi), .groups = "drop") %>%
    unnest(mshi) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = mshi, values_from = value, values_fill = 0)
  
  ##############################################################################
  # STEP IV: SURGERY
  ##############################################################################
  
  db$main_sur <- ifelse(str_detect(db$benefit_category, "main"), 1, 0)
  
  ##############################################################################
  # STEP V: ADMISSION TYPE
  ##############################################################################
  
  db <- db %>%
    mutate(adm_type = ifelse(adm_type2 == "Planned",
                             "planned", "emergency"))
  
  hadm_item <- db %>%
    group_by(id) %>%
    summarise(adm_type = list(adm_type), .groups = "drop") %>%
    unnest(adm_type) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = adm_type, values_from = value, values_fill = 0)
  
  ##############################################################################
  # STEP VI: OUTCOME
  ##############################################################################
  
  db <- db %>%
    mutate(adm_outcome = case_when(
      admission_outcome == "disc" ~ "discharged",
      admission_outcome == "ref" ~ "referred",
      admission_outcome == "dead" ~ "death",
      admission_outcome == "self_disc" ~ "self_discharge",
      TRUE ~ NA_character_
    ))
  
  houtcome_item <- db %>%
    group_by(id) %>%
    summarise(adm_outcome = list(adm_outcome), .groups = "drop") %>%
    unnest(adm_outcome) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = adm_outcome, values_from = value, values_fill = 0)
  
  ##############################################################################
  # STEP VII: COMPLICATION
  ##############################################################################
  
  db <- db %>%
    mutate(compl = ifelse(complication %in% c("NULL", "not_specified"),
                          "complication", "no_complication"))
  
  hcompl_item <- db %>%
    group_by(id) %>%
    summarise(compl = list(compl), .groups = "drop") %>%
    unnest(compl) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = compl, values_from = value, values_fill = 0)
  
  ##############################################################################
  # STEP VIII: SPECIALTY
  ##############################################################################
  
  specialty_to_department <- c(
    "ObstetricsGynecology",
    "ObstetricsGynecology",
    "Pediatrics",
    "InternalMedicine",
    "NeurologyNeurosurgery",
    "Cardiology_CardiovascularSurgery",
    "Oncology_Hematology",
    "Pulmonology_InfectiousDiseases",
    "Surgery",
    "Orthopedics_Trauma",
    "ENT_Ophthalmology_Dental",
    "ENT_Ophthalmology_Dental",
    "Critical_Care",
    "Dermatology",
    "Nephrology"
  )
  
  db$specialty <- specialty_to_department[as.character(db$profile)]
  
  hspec_item <- db %>%
    group_by(id) %>%
    summarise(specialty = list(specialty), .groups = "drop") %>%
    unnest(specialty) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = specialty, values_from = value, values_fill = 0)
  
  ##############################################################################
  # STEP IX: HOSPITAL DATA (LEVEL + OWNERSHIP)
  ##############################################################################
  
  dff <- db %>%
    left_join(sur_merged, by = "mo_name")
  
  dff <- dff %>%
    mutate(
      # LEVEL
      level = case_when(
        Level == "Regional" ~ "regional",
        Level == "City" ~ "city",
        Level %in% c("Rayon", "Village") ~ "rural",
        TRUE ~ "republican"
      ),
      
      # OWNERSHIP
      public = ifelse(Ownership == "priv", "private", "public"),
      
  hlevel_item <- dff %>%
    group_by(id) %>%
    summarise(level = list(level), .groups = "drop") %>%
    unnest(level) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = level, values_from = value, values_fill = 0)
  
  hpublic_item <- dff %>%
    group_by(id) %>%
    summarise(public = list(public), .groups = "drop") %>%
    unnest(public) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = public, values_from = value, values_fill = 0)
  
  hgeo_item <- dff %>%
    group_by(id) %>%
    summarise(geo = list(geo), .groups = "drop") %>%
    unnest(geo) %>%
    mutate(value = 1) %>%
    pivot_wider(names_from = geo, values_from = value, values_fill = 0)
  
  ##############################################################################
  # FINAL TARGET
  ##############################################################################
  
  df_los <- dff %>% select(id, los)
  
  ##############################################################################
  # FINAL MERGE → SINGLE DATAFRAME
  ##############################################################################
  
  final_df <- reduce(
    list(
      hadm_item,
      hage_item,
      hdiag_item,
      hgen_item,
      hosms_item,
      hcompl_item,
      hspec_item,
      hlevel_item,
      hpublic_item,
      hgeo_item,
      houtcome_item,
      df_los
    ),
    full_join,
    by = "id"
  )
  
  return(final_df)
}

############################################################
# APPLY FUNCTION FOR DATA PREPARATION
############################################################

df2022 <- prepare_hospital_data(df2022_synthetic, merged)
df2023 <- prepare_hospital_data(df2023_synthetic, merged)

##############################################################################
# STEP 1: CREATE GLOBAL FEATURE SPACE
##############################################################################

all_cols <- union(colnames(df2022), colnames(df2023))

##############################################################################
# STEP 2: REMOVE PROBLEMATIC COLUMNS
##############################################################################

bad_cols <- c("NA", "NA.x", "NA.y")
all_cols <- setdiff(all_cols, bad_cols)

##############################################################################
# STEP 3: ALIGN FUNCTION
##############################################################################

align_schema <- function(df, template_cols) {
  
  # add missing columns with 0
  missing <- setdiff(template_cols, colnames(df))
  df[missing] <- 0
  
  # ensure correct order + drop extras
  df <- df[, template_cols]
  
  return(df)
}

##############################################################################
# STEP 4: APPLY TO BOTH YEARS
##############################################################################

df2022 <- align_schema(df2022, all_cols)
df2023 <- align_schema(df2023, all_cols)

##############################################################################
# STEP 5: REMOVE REMAINING NA ISSUES (SAFEGUARD)
##############################################################################

df2022d[is.na(df2022)] <- 0
df2023[is.na(df2023)] <- 0

##############################################################################
# STEP 6: FINAL CHECK
##############################################################################

cat("Same columns:", identical(colnames(df2022_final_aligned),
                               colnames(df2023_final_aligned)), "\n")

cat("2022 rows:", nrow(df2022), "\n")
cat("2023 rows:", nrow(df2023), "\n")
cat("Final features:", length(colnames(df2022)), "\n")

identical(colnames(df2022), colnames(df2023))

df2022 <- df2022 [, sort(colnames(df2022))]
df2023 <- df2023 [, sort(colnames(df2023))]

############################################################
# SAVE DATAFRAMES
############################################################
saveRDS(df2022, "df2022.rds")
saveRDS(df2023, "df2023.rds")

############################################################
# PREPARE 2022 DATA
############################################################

df2022$los_class <- cut(
  df2022$los,
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

model_df <- df2022[
  ,
  !(names(df2022) %in% drop_vars)
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

synthetic_train <- train_df
synthetic_test <- test_df
############################################################
# SAVE DATAFRAMES
############################################################

saveRDS(synthetic_train ,"synthetic_train.rds")
saveRDS(synthetic_test,"synthetic_test.rds")

