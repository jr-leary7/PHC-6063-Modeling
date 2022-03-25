##### functions #####
read_diab_data <- function(file.dir = "./") {
  filepath <- paste0(file.dir, "diabetes_012_health_indicators_BRFSS2015.csv")
  if (!file.exists(filepath)) {
    stop("File does not exist in the provided directory.")
  } else {
    dat <- readr::read_csv(filepath)
    dat <- dat %>% dplyr::with_groups(Diabetes_012, dplyr::slice_sample, n = 200)
  }
  return(dat)
}

clean_diab_data <- function(diab.df = NULL) {
  # check input 
  
  # clean data 
  diab.df <- diab.df %>% 
             dplyr::rename(DIAGNOSIS = Diabetes_012, 
                           HIGH_BP_IND = HighBP, 
                           HIGH_CHOL_IND = HighChol, 
                           CHOL_CHECK_IND = CholCheck, 
                           SMOKING_IND = Smoker, 
                           STROKE_IND = Stroke, 
                           CHD_OR_MI_IND = HeartDiseaseorAttack, 
                           PHYS_ACTIVITY_IND = PhysActivity, 
                           EAT_FRUIT_IND = Fruits, 
                           EAT_VEG_IND = Veggies, 
                           HEAVY_ALC_IND = HvyAlcoholConsump, 
                           HEALTH_INS_IND = AnyHealthcare, 
                           AVOID_DOC_COST_IND = NoDocbcCost, 
                           GENERAL_HEALTH = GenHlth, 
                           MENTAL_HEALTH_BAD_LAST_30_DAYS = MentHlth, 
                           PHYS_HEALTH_BAD_LAST_30_DAYS = PhysHlth, 
                           WALK_DIFFICULTY_IND = DiffWalk, 
                           SEX = Sex, 
                           AGE_CATEGORY = Age, 
                           EDU_CATEGORY = Education, 
                           INCOME_CATEGORY = Income) %>% 
             dplyr::mutate(DIAGNOSIS = factor(DIAGNOSIS, labels = c("None", "Prediabetes", "Diabetes"), levels = c("0", "1", "2")), 
                           SEX = factor(SEX, labels = c("Female", "Male"), levels = c("0", "1")), 
                           dplyr::across(dplyr::contains("_IND|_CATEGORY"), as.factor), 
                           BMI = as.numeric(BMI), 
                           MENTAL_HEALTH_BAD_LAST_30_DAYS = as.numeric(MENTAL_HEALTH_BAD_LAST_30_DAYS), 
                           PHYS_HEALTH_BAD_LAST_30_DAYS = as.numeric(PHYS_HEALTH_BAD_LAST_30_DAYS))
  return(diab.df)
}
