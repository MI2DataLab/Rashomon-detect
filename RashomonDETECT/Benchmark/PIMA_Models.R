source("model_selection.R")
source("pdi_measures.R")
source("performance_calculation.R")
source("profile_dissimilarity_calculation.R")
source("plot.R")
source("rashomon_detect.R")

source("GetRashomonExplainers.R")
source("GetRashomonExplainersRF.R")

# Data set

PIMA <- read.csv("12_Data.csv", dec = ".")

PIMA <- PIMA %>% 
  mutate(Outcome = factor(Outcome))

PIMA <- PIMA[,-c(1)]

# Modelling

set.seed(125)

taskDataPIMA_all = TaskClassif$new("DataPIMA", PIMA, target = "Outcome")

#### mlr_learners

# Gradient Boosting Models 

gbm_learner_DataPIMA = LearnerClassifGBM$new()
gbm_learner_DataPIMA$predict_type <- 'prob'
gbm_learner_DataPIMA
gbm_learner_DataPIMA$param_set

# Random search 

random_tuner_DataPIMA = tnr("grid_search", resolution = 5)
evals20_DataPIMA = trm("evals", n_evals = 50)
cv_DataPIMA = rsmp("repeated_cv", folds = 5, repeats = 5)
measure_DataPIMA = msr("classif.auc")

params_set_range_DataPIMA = ParamSet$new(list(
  ParamInt$new("n.trees", lower = 50, upper = 700),
  ParamInt$new("interaction.depth", lower = 1, upper = 10),
  # ParamInt$new("n.minobsinnode", lower = 1, upper = 500),
  ParamDbl$new("shrinkage", lower = 0.1, upper = 0.5)
  #ParamDbl$new("bag.fraction", lower = min(gbm_params$bag.fraction), upper = max(gbm_params$bag.fraction))
  
))

search_space_DataPIMA = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

random_tuner_config_DataPIMA = TuningInstanceSingleCrit$new(
  task = taskDataPIMA_all,
  learner = gbm_learner_DataPIMA,
  resampling = cv_DataPIMA,
  measure = measure_DataPIMA,
  search_space = params_set_range_DataPIMA,
  terminator = evals20_DataPIMA
)


random_tuner_DataPIMA$optimize(random_tuner_config_DataPIMA)

### best results
random_tuner_config_DataPIMA$result_learner_param_vals

## all history
random_tuner_config_DataPIMA$archive


# Random Forest

rf_learner_DataPIMA = LearnerClassifRanger$new()
rf_learner_DataPIMA$predict_type <- 'prob'
rf_learner_DataPIMA
rf_learner_DataPIMA$param_set

random_tuner_DataPIMA = tnr("grid_search", resolution = 5)
evals20_DataPIMA = trm("evals", n_evals = 50)
cv_DataPIMA = rsmp("repeated_cv", folds = 5, repeats = 5)
measure_DataPIMA = msr("classif.auc")

params_set_range_rf_DataPIMA = ParamSet$new(list(
  ParamInt$new("num.trees", lower = 50, upper = 500),
  ParamInt$new("mtry", lower = 1, upper = 8),
  ParamInt$new("max.depth", lower = 2, upper = 5)
  #ParamInt$new("min.node.size", lower = 1, upper = 500)
  #ParamDbl$new("bag.fraction", lower = min(gbm_params$bag.fraction), upper = max(gbm_params$bag.fraction))
  
))

search_space_rf_DataPIMA = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

search_space_rf_DataPIMA

random_tuner_config_rf_DataPIMA = TuningInstanceSingleCrit$new(
  task = taskDataPIMA_all,
  learner = rf_learner_DataPIMA,
  resampling = cv_DataPIMA,
  measure = measure_DataPIMA,
  search_space = params_set_range_rf_DataPIMA,
  terminator = evals20_DataPIMA
)
random_tuner_DataPIMA$optimize(random_tuner_config_rf_DataPIMA)

random_tuner_config_rf_DataPIMA$result_learner_param_vals

data.frame(random_tuner_config_rf_DataPIMA$archive$data) %>%
  arrange(classif.auc) %>%
  mutate(x = row_number()) %>%
  ggplot(aes(x = x, y = classif.auc))+
  geom_line()+
  theme_minimal()

data.frame(random_tuner_config_rf_DataPIMA$archive$data) %>%
  arrange(-classif.auc) 

#get rashomon model with given tolerance

DataPIMA_models_rf <- random_tuner_config_rf_DataPIMA$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(num.trees, mtry,  max.depth, classif.auc) 

DataPIMA_models <- random_tuner_config_DataPIMA$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(n.trees, interaction.depth,  shrinkage, classif.auc) 

best_result_rf_PIMA = DataPIMA_models_rf[1,]
best_result_PIMA = DataPIMA_models[1,]

best_performance_PIMA = max(best_result_PIMA$classif.auc, best_result_rf_PIMA$classif.auc)

tolerance = 0.006 

rashomon_models_rf_PIMA = list() #RF

for (i in 1:dim(DataPIMA_models_rf)[1]) {
  if (abs(DataPIMA_models_rf$classif.auc[i] - best_performance_PIMA) < tolerance) {
    rashomon_models_rf_PIMA = list(rashomon_models_rf_PIMA, DataPIMA_models_rf[i])
  }
}

rashomon_models_PIMA = list() #GBM

for (i in 1:dim(DataPIMA_models)[1]) {
  if (abs(DataPIMA_models$classif.auc[i] - best_performance_PIMA) < tolerance) {
    rashomon_models_PIMA = list(rashomon_models_PIMA, DataPIMA_models[i])
  }
}


rf_explainers_PIMA <- get_Rashomon_models_exp_rf(random_tuner_config_rf_DataPIMA$archive$data, taskDataPIMA_all, 24 )
gbm_explainers_PIMA <- get_Rashomon_models(random_tuner_config_DataPIMA$archive$data, taskDataPIMA_all, 1)

###______________________

source("rashomon_detect.R")

explainers <- list(rf_explainers_PIMA$explainers[[1]],rf_explainers_PIMA$explainers[[2]],rf_explainers_PIMA$explainers[[3]],
                        rf_explainers_PIMA$explainers[[4]],
                        rf_explainers_PIMA$explainers[[5]],rf_explainers_PIMA$explainers[[6]],rf_explainers_PIMA$explainers[[7]],
                        rf_explainers_PIMA$explainers[[8]],
                        rf_explainers_PIMA$explainers[[9]],rf_explainers_PIMA$explainers[[10]],rf_explainers_PIMA$explainers[[11]],
                        rf_explainers_PIMA$explainers[[12]],
                        rf_explainers_PIMA$explainers[[13]],rf_explainers_PIMA$explainers[[14]],rf_explainers_PIMA$explainers[[15]],
                        rf_explainers_PIMA$explainers[[16]],
                        rf_explainers_PIMA$explainers[[17]],rf_explainers_PIMA$explainers[[18]],rf_explainers_PIMA$explainers[[19]],
                        rf_explainers_PIMA$explainers[[20]],
                        rf_explainers_PIMA$explainers[[21]],rf_explainers_PIMA$explainers[[22]],rf_explainers_PIMA$explainers[[23]],
                        rf_explainers_PIMA$explainers[[24]], gbm_explainers_PIMA$explainers[[1]])

res_PIMA <- rashomon_detect(explainers, k=3)
res_PIMA

res_PIMA_eucl <- rashomon_detect(explainers_PIMA, k=3, pdi_method_numerical= derivative_euclidean_distance)
res_PIMA_eucl

res_PIMA_der_eucl <- rashomon_detect(explainers_PIMA, pdi_method_numerical = derivative_euclidean_distance, k=3)
res_PIMA_der_eucl

plot_summary_matrix(res_PIMA)

