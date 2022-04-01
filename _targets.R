##### setup #####

library(targets)
source("R/functions.R")
tar_option_set(packages = c("tidyverse", "tidymodels", "magrittr", "xgboost", "ranger", "kernlab", "klaR", "kknn", "discrim", "nnet"))

##### parallel computing #####
cl <- parallel::makeCluster(8)
doParallel::registerDoParallel(cl = cl)
parallel::clusterSetRNGStream(cl, 312)

### targets ###
list(
  # prep data 
  tar_target(diab_data_raw, read_diab_data(file.dir = "~/repos/PHC-6063-Modeling/data/"), cue = tar_cue(mode = "never")), 
  tar_target(diab_data_clean, clean_diab_data(diab.df = diab_data_raw), cue = tar_cue(mode = "never")), 
  # set up train / test / validation sets as well as 10-fold cross-validation folds
  tar_target(diab_val_split, validation_split(data = diab_data_clean, prop = 0.85, strata = DIAGNOSIS), cue = tar_cue(mode = "never")), 
  tar_target(diab_val_data, assessment(diab_val_split$splits[[1]]), cue = tar_cue(mode = "never")), 
  tar_target(diab_analysis_data, analysis(diab_val_split$splits[[1]]), cue = tar_cue(mode = "never")), 
  tar_target(diab_tt_split, initial_split(data = diab_analysis_data, prop = 0.82, strata = DIAGNOSIS), cue = tar_cue(mode = "never")),  # ensures 70 / 15 / 15 overall split
  tar_target(diab_train, training(diab_tt_split), cue = tar_cue(mode = "never")),
  tar_target(diab_test, testing(diab_tt_split), cue = tar_cue(mode = "never")),
  tar_target(diab_train_folds, vfold_cv(diab_train, v = 10, strata = DIAGNOSIS), cue = tar_cue(mode = "never")), 
  
  # build multinomial naive Bayes model
  tar_target(naive_bayes_mod, naive_Bayes() %>%
                              set_mode("classification") %>%
                              set_engine("klaR"), cue = tar_cue(mode = "never")),
  tar_target(naive_bayes_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>%
                                 step_nzv(all_predictors(), freq_cut = 99/1) %>%
                                 step_normalize(all_numeric_predictors()), cue = tar_cue(mode = "never")),
  tar_target(naive_bayes_workflow, workflow() %>%
                                   add_model(naive_bayes_mod) %>%
                                   add_recipe(naive_bayes_recipe), cue = tar_cue(mode = "never")),
  # train multinomial naive Bayes model & generate predictions (no hyperparameters for this one)
  tar_target(naive_bayes_fit, naive_bayes_workflow %>% fit(data = diab_train), cue = tar_cue(mode = "never")), 
  tar_target(naive_bayes_train_preds, predict(naive_bayes_fit, new_data = diab_train, type = "prob"), cue = tar_cue(mode = "never")),
  tar_target(naive_bayes_test_preds, predict(naive_bayes_fit, new_data = diab_test, type = "prob"), cue = tar_cue(mode = "never")),
  
  # build KNN model
  tar_target(knn_mod, nearest_neighbor(neighbors = tune(), 
                                       weight_func = tune(), 
                                       dist_power = tune()) %>% 
                      set_mode("classification") %>% 
                      set_engine("kknn"), cue = tar_cue(mode = "never")), 
  tar_target(knn_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>% 
                         step_dummy(all_nominal_predictors()) %>%
                         step_nzv(all_predictors(), freq_cut = 99/1) %>%
                         step_normalize(all_numeric_predictors()), cue = tar_cue(mode = "never")),
  tar_target(knn_workflow, workflow() %>%
                           add_model(knn_mod) %>%
                           add_recipe(knn_recipe), cue = tar_cue(mode = "never")), 
  # train knn model & generate predictions
  tar_target(knn_tuning_res, tune_model(mod.workflow = knn_workflow, 
                                        cv.folds = diab_train_folds,
                                        mod.type = "knn"), cue = tar_cue(mode = "never")), 
  tar_target(knn_best_params, select_best(knn_tuning_res, "roc_auc")),
  tar_target(knn_fit, fit_best_train(mod.workflow = knn_workflow,
                                     best.params = knn_best_params,
                                     train.dat = diab_train)),
  tar_target(knn_train_preds, predict(knn_fit, new_data = diab_train, type = "prob"), cue = tar_cue(mode = "never")),
  tar_target(knn_test_preds, predict(knn_fit, new_data = diab_test, type = "prob"), cue = tar_cue(mode = "never")), 
  
  # # build linear SVM model
  tar_target(svm_linear_mod, svm_linear(cost = tune(), margin = tune()) %>%
                             set_mode("classification") %>%
                             set_engine("kernlab", scaled = FALSE)),
  tar_target(svm_linear_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>%
                                step_dummy(all_nominal_predictors()) %>%
                                step_nzv(all_predictors(), freq_cut = 99/1) %>%
                                step_normalize(all_numeric_predictors())),
  tar_target(svm_linear_workflow, workflow() %>%
                                  add_model(svm_linear_mod) %>%
                                  add_recipe(svm_linear_recipe)),
  # train linear SVM model & generate predictions
  tar_target(svm_linear_tuning_res, tune_model(mod.workflow = svm_linear_workflow,
                                               cv.folds = diab_train_folds,
                                               mod.type = "svm_linear")),
  tar_target(svm_linear_best_params, select_best(svm_linear_tuning_res, "roc_auc")),
  tar_target(svm_linear_fit, fit_best_train(mod.workflow = svm_linear_workflow,
                                            best.params = svm_linear_best_params,
                                            train.dat = diab_train)),
  tar_target(svm_linear_train_preds, predict(svm_linear_fit, new_data = diab_train, type = "prob")),
  tar_target(svm_linear_test_preds, predict(svm_linear_fit, new_data = diab_test, type = "prob")),
  # 
  # # build polynomial SVM model
  # tar_target(svm_poly_mod, svm_poly(cost = tune(),
  #                                   degree = tune(),
  #                                   scale_factor = tune(),
  #                                   margin = tune()) %>%
  #                          set_mode("classification") %>%
  #                          set_engine("kernlab", scaled = FALSE)),
  # tar_target(svm_poly_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>% 
  #                             step_dummy(all_nominal_predictors()) %>%
  #                             step_nzv(all_predictors(), freq_cut = 99/1) %>%
  #                             step_normalize(all_numeric_predictors())),
  # tar_target(svm_poly_workflow, workflow() %>%
  #                               add_model(svm_poly_mod) %>%
  #                               add_recipe(svm_poly_recipe)),
  # # train polynomial SVM model & generate predictions
  # tar_target(svm_poly_tuning_res, tune_model(mod.workflow = svm_poly_workflow, 
  #                                            cv.folds = diab_train_folds,
  #                                            mod.type = "svm_poly")), 
  # tar_target(svm_poly_best_params, select_best(svm_poly_tuning_res, "roc_auc")),
  # tar_target(svm_poly_fit, fit_best_train(mod.workflow = svm_poly_workflow,
  #                                   best.params = svm_poly_best_params,
  #                                   train.dat = diab_train)),
  # tar_target(svm_poly_train_preds, predict(svm_poly_fit, new_data = diab_train, type = "prob")),
  # tar_target(svm_poly_test_preds, predict(svm_poly_fit, new_data = diab_test, type = "prob")), 
  # 
  # # build radial SVM model
  # tar_target(svm_radial_mod, svm_rbf(cost = tune(), 
  #                                    rbf_sigma = tune(), 
  #                                    margin = tune()) %>%
  #                            set_mode("classification") %>%
  #                            set_engine("kernlab", scaled = FALSE)),
  # tar_target(svm_radial_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>% 
  #                               step_dummy(all_nominal_predictors()) %>%
  #                               step_nzv(all_predictors(), freq_cut = 99/1) %>%
  #                               step_normalize(all_numeric_predictors())),
  # tar_target(svm_radial_workflow, workflow() %>%
  #                                 add_model(svm_radial_mod) %>%
  #                                 add_recipe(svm_radial_recipe)),
  # # train radial SVM model & generate predictions
  # tar_target(svm_radial_tuning_res, tune_model(mod.workflow = svm_radial_workflow, 
  #                                              cv.folds = diab_train_folds,
  #                                              mod.type = "svm_radial")), 
  # tar_target(svm_radial_best_params, select_best(svm_radial_tuning_res, "roc_auc")),
  # tar_target(svm_radial_fit, fit_best_train(mod.workflow = svm_radial_workflow,
  #                                           best.params = svm_radial_best_params,
  #                                           train.dat = diab_train)),
  # tar_target(svm_radial_train_preds, predict(svm_radial_fit, new_data = diab_train, type = "prob")),
  # tar_target(svm_radial_test_preds, predict(svm_radial_fit, new_data = diab_test, type = "prob")), 
  
  # build random forest model
  tar_target(rf_mod, rand_forest(mtry = tune(), 
                                 trees = tune(),
                                 min_n = tune()) %>%
                     set_mode("classification") %>%
                     set_engine("ranger", importance = "impurity"), cue = tar_cue(mode = "never")), 
  tar_target(rf_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>% 
                        step_dummy(all_nominal_predictors()) %>%
                        step_nzv(all_predictors(), freq_cut = 99/1) %>%
                        step_normalize(all_numeric_predictors()), cue = tar_cue(mode = "never")), 
  tar_target(rf_workflow, workflow() %>% 
                          add_model(rf_mod) %>% 
                          add_recipe(rf_recipe), cue = tar_cue(mode = "never")), 
  # train random forest model & generate predictions
  tar_target(rf_tuning_res, tune_model(mod.workflow = rf_workflow,
                                       cv.folds = diab_train_folds,
                                       mod.type = "rf")),
  tar_target(rf_best_params, select_best(rf_tuning_res, "roc_auc")),
  tar_target(rf_fit, fit_best_train(mod.workflow = rf_workflow,
                                    best.params = rf_best_params,
                                    train.dat = diab_train)),
  tar_target(rf_train_preds, predict(rf_fit, new_data = diab_train, type = "prob")),
  tar_target(rf_test_preds, predict(rf_fit, new_data = diab_test, type = "prob")),
  tar_target(rf_var_imp, get_var_imp(mod.fit = rf_fit, mod.type = "rf")),
  
  # build xgboost model 
  tar_target(xgboost_mod, boost_tree(trees = tune(),
                                     tree_depth = tune(),
                                     min_n = tune(),
                                     loss_reduction = tune(),
                                     sample_size = tune(),
                                     mtry = tune(),
                                     stop_iter = tune(),
                                     learn_rate = tune()) %>% 
                          set_mode("classification") %>% 
                          set_engine("xgboost"), cue = tar_cue(mode = "never")), 
  tar_target(xgboost_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>% 
                             step_dummy(all_nominal_predictors()) %>%
                             step_nzv(all_predictors(), freq_cut = 99/1) %>%
                             step_normalize(all_numeric_predictors()), cue = tar_cue(mode = "never")), 
  tar_target(xgboost_workflow, workflow() %>% 
                               add_model(xgboost_mod) %>% 
                               add_recipe(xgboost_recipe), cue = tar_cue(mode = "never")), 
  # train xgboost model & generate predictions
  tar_target(xgboost_tuning_res, tune_model(mod.workflow = xgboost_workflow, 
                                            cv.folds = diab_train_folds, 
                                            mod.type = "xgboost"), cue = tar_cue(mode = "never")), 
  tar_target(xgboost_best_params, select_best(xgboost_tuning_res, "roc_auc"), cue = tar_cue(mode = "never")),
  tar_target(xgboost_fit, fit_best_train(mod.workflow = xgboost_workflow,
                                         best.params = xgboost_best_params,
                                         train.dat = diab_train), cue = tar_cue(mode = "never")),
  tar_target(xgboost_train_preds, predict(xgboost_fit, new_data = diab_train, type = "prob"), cue = tar_cue(mode = "never")),
  tar_target(xgboost_test_preds, predict(xgboost_fit, new_data = diab_test, type = "prob"), cue = tar_cue(mode = "never")), 
  tar_target(xgboost_var_imp, get_var_imp(mod.fit = xgboost_fit, mod.type = "xgboost"), cue = tar_cue(mode = "never")), 
  
  # build neural network model 
  tar_target(nn_mod, mlp(hidden_units = tune(), 
                         penalty = tune(), 
                         epochs = tune()) %>% 
                     set_mode("classification") %>% 
                     set_engine("nnet"), cue = tar_cue(mode = "never")), 
  tar_target(nn_recipe, recipe(DIAGNOSIS ~., data = diab_train) %>% 
                        step_dummy(all_nominal_predictors()) %>%
                        step_nzv(all_predictors(), freq_cut = 99/1) %>%
                        step_normalize(all_numeric_predictors()), cue = tar_cue(mode = "never")), 
  tar_target(nn_workflow, workflow() %>% 
                          add_model(nn_mod) %>% 
                          add_recipe(nn_recipe), cue = tar_cue(mode = "never")), 
  # train neural network model & generate predictions
  tar_target(nn_tuning_res, tune_model(mod.workflow = nn_workflow, 
                                       cv.folds = diab_train_folds, 
                                       mod.type = "nn"), cue = tar_cue(mode = "never")), 
  tar_target(nn_best_params, select_best(nn_tuning_res, "roc_auc")),
  tar_target(nn_fit, fit_best_train(mod.workflow = nn_workflow,
                                    best.params = nn_best_params,
                                    train.dat = diab_train), cue = tar_cue(mode = "never")),
  tar_target(nn_train_preds, predict(nn_fit, new_data = diab_train, type = "prob"), cue = tar_cue(mode = "never")),
  tar_target(nn_test_preds, predict(nn_fit, new_data = diab_test, type = "prob"), cue = tar_cue(mode = "never"))
)
