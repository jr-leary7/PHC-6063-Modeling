##### functions #####
read_diab_data <- function(file.dir = "./") {
  filepath <- paste0(file.dir, "diabetes_012_health_indicators_BRFSS2015.csv")
  if (!file.exists(filepath)) {
    stop("File does not exist in the provided directory.")
  } else {
    dat <- readr::read_csv(filepath)
  }
  return(dat)
}

clean_diab_data <- function(diab.df = NULL) {
  # check input 
  if (is.null(diab.df)) { stop("clean_diab_data() needs non-NULL arguments.") }
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

tune_model <- function(mod.workflow = NULL, cv.folds = NULL, mod.type = NULL) {
  # check inputs 
  if (is.null(mod.workflow) | is.null(cv.folds)) { stop("tune_model() needs non-NULL arguments.") }
  mod.type <- tolower(mod.type)
  # tune model 
  if (mod.type == "") {
    # comment
  } else if (mod.type == "rf") {
    param_grid <- grid_latin_hypercube(list(mtry = mtry(c(1, floor(ncol(cv.folds$splits[[1]]$data) / 2))), 
                                            trees = trees(), 
                                            min_n = min_n()), 
                                       size = 15)
  } else if (mod.type == "xgboost") {
    param_grid <- grid_latin_hypercube(list(trees = trees(), 
                                            tree_depth = tree_depth(), 
                                            min_n = min_n(), 
                                            loss_reduction = loss_reduction(), 
                                            sample_size = sample_prop(c(0.1, 0.6)), 
                                            mtry = mtry(c(1, floor(ncol(cv.folds$splits[[1]]$data) / 2))), 
                                            stop_iter = stop_iter(), 
                                            learn_rate = learn_rate()), 
                                       size = 15)
  } else if (mod.type == "svm_poly") {
    param_grid <- grid_latin_hypercube(list(cost = cost(),
                                            degree = degree(),
                                            scale_factor = scale_factor(),
                                            margin = svm_margin()), 
                                       size = 15)
  } else if (mod.type == "svm_linear") {
    param_grid <- grid_latin_hypercube(list(cost = cost(), margin = svm_margin()), 
                                       size = 15)
  } else if (mod.type == "svm_radial") {
    param_grid <- grid_latin_hypercube(list(cost = cost(), 
                                            rbf_sigma = rbf_sigma(),
                                            margin = svm_margin()), 
                                       size = 15)
  } else if (mod.type == "knn") {
    param_grid <- grid_latin_hypercube(list(neighbors = neighbors(), 
                                            weight_func = weight_func(), 
                                            dist_power = dist_power()), 
                                       size = 15)
  } else if (mod.type == "nn") {
    param_grid <- grid_latin_hypercube(list(hidden_units = hidden_units(), 
                                            penalty = penalty(), 
                                            epochs = epochs()))
  }
  
  tuning_metrics <- metric_set(roc_auc, 
                               precision, 
                               recall, 
                               pr_auc, 
                               kap, 
                               sens, 
                               spec, 
                               ppv, 
                               npv, 
                               f_meas, 
                               bal_accuracy)
  tune_controls <- control_grid(save_pred = TRUE, 
                                parallel_over = "resamples", 
                                allow_par = TRUE)
  if (mod.type == "naivebayes") {
    tuning_res <- tune_grid(object = mod.workflow, 
                            resamples = cv.folds, 
                            metrics = tuning_metrics, 
                            control = tune_controls)
  } else {
    tuning_res <- tune_grid(object = mod.workflow, 
                            resamples = cv.folds, 
                            grid = param_grid, 
                            metrics = tuning_metrics, 
                            control = tune_controls)
  }
  
  return(tuning_res)
}

fit_best_train <- function(mod.workflow = NULL, best.params = NULL, train.dat = NULL) {
  # check inputs 
  if (is.null(mod.workflow) | is.null(best.params) | is.null(train.dat)) { stop("Arguments to fit_best_train() must be non-NULL.") }
  # fit
  training_res <- mod.workflow %>% 
                  finalize_workflow(parameters = best.params) %>% 
                  fit(data = train.dat)
  return(training_res)
}

get_var_imp <- function(mod.fit = NULL, mod.type = "xgboost") {
  # check inputs 
  if (is.null(mod.fit)) { stop("mod.fit must be non-NULL.") }
  mod.type <- tolower(mod.type)
  # feature importance 
  if (mod.type == "xgboost") {
    var_imp <- xgboost::xgb.importance(model = mod.fit$fit$fit$fit)
  } else {
    var_imp <- ranger::importance(mod.fit$fit$fit$fit)
  }
  return(var_imp)
}
