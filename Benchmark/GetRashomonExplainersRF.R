get_Rashomon_models_exp_rf <- function(tuner_history, task, n){
  
  top_params_df <- tuner_history %>%
    arrange(-classif.auc) %>%
    dplyr::select(num.trees, mtry, classif.auc) %>%
    dplyr::top_n(n, classif.auc) %>%
    dplyr::select(-classif.auc)
  
  top_params_ls <- split(top_params_df, seq(nrow(top_params_df)))
  top_params_ls <- lapply(top_params_ls, as.list)
  
  
  rf_models_ls <- list()
  for(params_conf in top_params_ls){
    rf_learner_cardio = LearnerClassifRanger$new()
    rf_learner_cardio$predict_type <- 'prob'
    rf_learner_cardio$param_set$values <- params_conf
    rf_learner_cardio$train(task)
    rf_models_ls <- c(rf_models_ls, rf_learner_cardio)
  }
  
  rf_exp_ls <- list()
  for (i in 1:length(rf_models_ls)) {
    rf_exp_ls[[i]] = explain_mlr3(rf_models_ls[[i]],
                                   data = task$data(),
                                   y = (task$data()[[task$target_names]]) == "1",
                                   label = paste('RF', i))
  }
  
  return(list(modelsrf = rf_models_ls, explainersrf = rf_exp_ls))
  
}
