##### setup #####

library(targets)
source("R/functions.R")
tar_option_set(packages = c("tidyverse", "tidymodels", "magrittr", "xgboost", "ranger", "e1071", "klaR"))

##### parallel computing #####
cl <- parallel::makeCluster(4)
doParallel::registerDoParallel(cl = cl)
parallel::clusterSetRNGStream(cl, 312)

### targets ###
list(
  # prep data 
  tar_target(diab_data_raw, read_diab_data(file.dir = "~/Downloads/")), 
  tar_target(diab_data_clean, clean_diab_data(diab.df = diab_data_raw)), 
  # set up train / test / validation sets as well as 10-fold cross-validation folds
  tar_target(diab_val_split, validation_split(data = diab_data_clean, prop = 0.85, strata = DIAGNOSIS)), 
  tar_target(diab_val_data, assessment(diab_val_split)), 
  tar_target(diab_analysis_data, analysis(diab_val_split)), 
  tar_target(diab_tt_split, initial_split(data = diab_analysis_data, prop = 0.6, strata = DIAGNOSIS)), 
  tar_target(diab_train, training(diab_tt_split)),
  tar_target(diab_test, testing(diab_tt_split)),
  tar_target(diab_train_folds, vfold_cv(diab_train, v = 10, strata = DIAGNOSIS))
)
