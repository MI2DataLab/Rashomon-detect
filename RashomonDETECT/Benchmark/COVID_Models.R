source("model_selection.R")
source("pdi_measures.R")
source("performance_calculation.R")
source("profile_dissimilarity_calculation.R")
source("plot.R")
source("rashomon_detect.R")

source("GetRashomonExplainers.R")
source("GetRashomonExplainersRF.R")

# Data set

Data <- read.csv("2_data.csv", dec = ".")

Data <- Data %>% 
 mutate(SARS.Cov.2.exam.result  = factor(SARS.Cov.2.exam.result),
        Patient.addmited.to.intensive.care.unit..1.yes..0.no. = factor(Patient.addmited.to.intensive.care.unit..1.yes..0.no.),
        Patient.addmited.to.semi.intensive.unit..1.yes..0.no. = factor(Patient.addmited.to.semi.intensive.unit..1.yes..0.no.), 
        Patient.addmited.to.regular.ward..1.yes..0.no. = factor(Patient.addmited.to.regular.ward..1.yes..0.no.))


Data <- Data[,-c(1)]

#library(ROSE)

Data_model <- ovun.sample(SARS.Cov.2.exam.result ~., data = Data, method = "over")$data

# Modelling

set.seed(125)

taskdata_all = TaskClassif$new("data", Data_model, target = "SARS.Cov.2.exam.result")

#### mlr_learners

# Gradient Boosting Models 

gbm_learner_data = LearnerClassifGBM$new()
gbm_learner_data$predict_type <- 'prob'
gbm_learner_data
gbm_learner_data$param_set


# Random search 

random_tuner_data = tnr("grid_search", resolution = 10)
evals20_data = trm("evals", n_evals = 50)
cv_data = rsmp("repeated_cv", folds = 5, repeats = 10)
measure_data = msr("classif.auc")

params_set_range_data = ParamSet$new(list(
  ParamInt$new("n.trees", lower = 50, upper = 700),
  ParamInt$new("interaction.depth", lower = 1, upper = 10),
  ParamDbl$new("shrinkage", lower = 0.1, upper = 0.5)
))

search_space_data = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

random_tuner_config_data = TuningInstanceSingleCrit$new(
  task = taskdata_all,
  learner = gbm_learner_data,
  resampling = cv_data,
  measure = measure_data,
  search_space = params_set_range_data,
  terminator = evals20_data
)


random_tuner_data$optimize(random_tuner_config_data)

# best results
random_tuner_config_data$result_learner_param_vals

# all history
random_tuner_config_data$archive

# Random Forest

rf_learner_data = LearnerClassifRanger$new()
rf_learner_data$predict_type <- 'prob'
rf_learner_data
rf_learner_data$param_set

as.data.table(rf_learner_data$param_set)[, list(id, class, lower, upper, nlevels)]

random_tuner_data = tnr("grid_search", resolution = 5)
evals20_data = trm("evals", n_evals = 50)
cv_data = rsmp("repeated_cv", folds = 5, repeats = 5)
measure_data = msr("classif.auc")

params_set_range_rf_data = ParamSet$new(list(
  ParamInt$new("num.trees", lower = 50, upper = 500),
  ParamInt$new("mtry", lower = 1, upper = 10),
  ParamInt$new("max.depth", lower = 2, upper = 5)
  #ParamInt$new("min.node.size", lower = 1, upper = 500)
  #ParamDbl$new("bag.fraction", lower = min(gbm_params$bag.fraction), upper = max(gbm_params$bag.fraction))
  
))

search_space_rf_data = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

search_space_rf_data

random_tuner_config_rf_data = TuningInstanceSingleCrit$new(
  task = taskdata_all,
  learner = rf_learner_data,
  resampling = cv_data,
  measure = measure_data,
  search_space = params_set_range_rf_data,
  terminator = evals20_data
)
random_tuner_data$optimize(random_tuner_config_rf_data)

random_tuner_config_rf_data$result_learner_param_vals

#get rashomon model with given tolerance

data_models_rf <- random_tuner_config_rf_data$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(num.trees, mtry,  max.depth, classif.auc) 

data_models <- random_tuner_config_data$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(n.trees, interaction.depth,  shrinkage, classif.auc) 

best_result_rf = data_models_rf[1,]
best_result = data_models[1,]

best_performance = max(best_result$classif.auc, best_result_rf$classif.auc)

tolerance = 0.001 

rashomon_models_rf = list() #RF

for (i in 1:dim(data_models_rf)[1]) {
  if (abs(data_models_rf$classif.auc[i] - best_performance) < tolerance) {
    rashomon_models_rf = list(rashomon_models_rf, data_models_rf[i])
  }
}

rashomon_models = list() #GBM

for (i in 1:dim(data_models)[1]) {
  if (abs(data_models$classif.auc[i] - best_performance) < tolerance) {
    rashomon_models = list(rashomon_models, data_models[i])
  }
}


#rf_explainers <- get_Rashomon_models_exp_rf(random_tuner_config_rf_data$archive$data, taskdata_all, 0 )
gbm_explainers <- get_Rashomon_models(random_tuner_config_data$archive$data, taskdata_all, 8)

# Rashomon_DETECT

source("rashomon_detect.R")

explainers <- list(gbm_explainers$explainers[[1]], gbm_explainers$explainers[[2]], gbm_explainers$explainers[[3]], 
                   gbm_explainers$explainers[[4]], gbm_explainers$explainers[[5]], gbm_explainers$explainers[[6]],
                   gbm_explainers$explainers[[7]], gbm_explainers$explainers[[8]])

res <- rashomon_detect(explainers, sorted=TRUE, k=3)
res

res_covid_eucl <- rashomon_detect(explainers, pdi_method_numerical= euclidean_distance, sorted=TRUE, k=3)
res_covid_eucl

res_covid_der_eucl <- rashomon_detect(explainers, pdi_method_numerical= derivative_euclidean_distance, k=3, sorted=TRUE)

plot_summary_matrix(res)
