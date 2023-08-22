get_Rashomon_models <- function(tuner_history, task, n){
  
  top_params_df <- tuner_history %>%
    arrange(-classif.auc) %>%
    dplyr::select(n.trees, interaction.depth,  shrinkage, classif.auc) %>%
    dplyr::top_n(n, classif.auc) %>%
    dplyr::select(-classif.auc)
  
  top_params_ls <- split(top_params_df, seq(nrow(top_params_df)))
  top_params_ls <- lapply(top_params_ls, as.list)
  
  
  gbm_models_ls <- list()
  for(params_conf in top_params_ls){
    gbm_learner = LearnerClassifGBM$new()
    gbm_learner$predict_type <- 'prob'
    gbm_learner$param_set$values <- params_conf
    gbm_learner$train(task)
    gbm_models_ls <- c(gbm_models_ls, gbm_learner)
  }
  
  gbm_exp_ls <- list()
  for (i in 1:length(gbm_models_ls)) {
    gbm_exp_ls[[i]] = explain_mlr3(gbm_models_ls[[i]],
                             data = task$data(),
                             y = as.numeric(as.character(task$data()[[task$target_names]])),
                             label = paste('GBM', i))
  }
  

  return(list(models = gbm_models_ls, explainers = gbm_exp_ls))
}
