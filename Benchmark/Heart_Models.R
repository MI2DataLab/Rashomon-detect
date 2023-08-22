source("model_selection.R")
source("pdi_measures.R")
source("performance_calculation.R")
source("profile_dissimilarity_calculation.R")
source("plot.R")
source("rashomon_detect.R")

source("GetRashomonExplainers.R")
source("GetRashomonExplainersRF.R")

# Data set

Heart <- read.csv("8_Data.csv", dec = ".")

Heart <- Heart %>% 
  mutate(target  = factor(target),
         sex = factor(sex), 
         chest_pain_type = factor(chest_pain_type),
         fasting_blood_sugar = factor(fasting_blood_sugar),
         resting_ecg = factor(resting_ecg),
         exercise_angina = factor(exercise_angina), 
         ST_slope = factor(ST_slope))

Heart <- Heart[,-c(1)]

Heart <- Heart[-which(Heart$ST_slope=="0"),]

Heart <- Heart %>% 
  mutate(ST_slope = factor(ST_slope, levels = c("1", "2", "3")))

DataH_model <- Heart

# Modelling

set.seed(125)
taskDataH_all = TaskClassif$new("DataH", DataH_model, target = "target")

# mlr_learners

# Gradient Boosting Models 

gbm_learner_DataH = LearnerClassifGBM$new()
gbm_learner_DataH$predict_type <- 'prob'
gbm_learner_DataH
gbm_learner_DataH$param_set

# Random search -----------------------------------------------------------

random_tuner_DataH = tnr("grid_search", resolution = 5)
evals20_DataH = trm("evals", n_evals = 50)
cv_DataH = rsmp("repeated_cv", folds = 5, repeats = 5)
measure_DataH = msr("classif.auc")

params_set_range_DataH = ParamSet$new(list(
  ParamInt$new("n.trees", lower = 50, upper = 700),
  ParamInt$new("interaction.depth", lower = 1, upper = 10),
  # ParamInt$new("n.minobsinnode", lower = 1, upper = 500),
  ParamDbl$new("shrinkage", lower = 0.1, upper = 0.5)
  #ParamDbl$new("bag.fraction", lower = min(gbm_params$bag.fraction), upper = max(gbm_params$bag.fraction))
  
))

search_space_DataH = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

random_tuner_config_DataH = TuningInstanceSingleCrit$new(
  task = taskDataH_all,
  learner = gbm_learner_DataH,
  resampling = cv_DataH,
  measure = measure_DataH,
  search_space = params_set_range_DataH,
  terminator = evals20_DataH
)

random_tuner_DataH$optimize(random_tuner_config_DataH)

# best results
random_tuner_config_DataH$result_learner_param_vals

# all history
random_tuner_config_DataH$archive

# Random Forest

rf_learner_DataH = LearnerClassifRanger$new()
rf_learner_DataH$predict_type <- 'prob'
rf_learner_DataH
rf_learner_DataH$param_set

random_tuner_DataH = tnr("grid_search", resolution = 5)
evals20_DataH = trm("evals", n_evals = 50)
cv_DataH = rsmp("repeated_cv", folds = 5, repeats = 5)
measure_DataH = msr("classif.auc")

params_set_range_rf_DataH = ParamSet$new(list(
  ParamInt$new("num.trees", lower = 50, upper = 500),
  ParamInt$new("mtry", lower = 1, upper = 10),
  ParamInt$new("max.depth", lower = 2, upper = 5)
  #ParamInt$new("min.node.size", lower = 1, upper = 500)
  #ParamDbl$new("bag.fraction", lower = min(gbm_params$bag.fraction), upper = max(gbm_params$bag.fraction))
  
))

search_space_rf_DataH = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

search_space_rf_DataH

random_tuner_config_rf_DataH = TuningInstanceSingleCrit$new(
  task = taskDataH_all,
  learner = rf_learner_DataH,
  resampling = cv_DataH,
  measure = measure_DataH,
  search_space = params_set_range_rf_DataH,
  terminator = evals20_DataH
)
random_tuner_DataH$optimize(random_tuner_config_rf_DataH)

#get rashomon model with given tolerance

DataH_models_rf <- random_tuner_config_rf_DataH$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(num.trees, mtry,  max.depth, classif.auc) 

DataH_models <- random_tuner_config_DataH$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(n.trees, interaction.depth,  shrinkage, classif.auc) 

best_result_rf_Heart = DataH_models_rf[1,]
best_result_Heart = DataH_models[1,]

best_performance_Heart = max(best_result_Heart$classif.auc, best_result_rf_Heart$classif.auc)

tolerance = 0.023

rashomon_models_rf_Heart = list() #RF

for (i in 1:dim(DataH_models_rf)[1]) {
  if (abs(DataH_models_rf$classif.auc[i] - best_performance_Heart) < tolerance) {
    rashomon_models_rf_Heart = list(rashomon_models_rf_Heart, DataH_models_rf[i])
  }
}

rashomon_models_Heart = list() #GBM

for (i in 1:dim(DataH_models)[1]) {
  if (abs(DataH_models$classif.auc[i] - best_performance_Heart) < tolerance) {
    rashomon_models_Heart = list(rashomon_models_Heart, DataH_models[i])
  }
}


rf_explainers_Heart <- get_Rashomon_models_exp_rf(random_tuner_config_rf_DataH$archive$data, taskDataH_all, 5 )
gbm_explainers_Heart <- get_Rashomon_models(random_tuner_config_DataH$archive$data, taskDataH_all, 38)

# Rashomon_DETECT

source("rashomon_detect.R")

explainers <- list(gbm_explainers_Heart$explainers[[1]], gbm_explainers_Heart$explainers[[2]],
                          gbm_explainers_Heart$explainers[[3]], gbm_explainers_Heart$explainers[[4]],
                          gbm_explainers_Heart$explainers[[5]], gbm_explainers_Heart$explainers[[6]],
                          gbm_explainers_Heart$explainers[[7]], gbm_explainers_Heart$explainers[[8]],
                          gbm_explainers_Heart$explainers[[9]], gbm_explainers_Heart$explainers[[10]],
                          gbm_explainers_Heart$explainers[[11]], gbm_explainers_Heart$explainers[[12]],
                         gbm_explainers_Heart$explainers[[13]], gbm_explainers_Heart$explainers[[14]],
                         gbm_explainers_Heart$explainers[[15]], gbm_explainers_Heart$explainers[[16]],
                         gbm_explainers_Heart$explainers[[17]], gbm_explainers_Heart$explainers[[18]],
                         gbm_explainers_Heart$explainers[[19]], gbm_explainers_Heart$explainers[[20]],
                         gbm_explainers_Heart$explainers[[21]], gbm_explainers_Heart$explainers[[22]],
                         gbm_explainers_Heart$explainers[[23]],gbm_explainers_Heart$explainers[[24]],
                         gbm_explainers_Heart$explainers[[25]],gbm_explainers_Heart$explainers[[26]],
                         gbm_explainers_Heart$explainers[[27]],gbm_explainers_Heart$explainers[[28]],
                         gbm_explainers_Heart$explainers[[29]],gbm_explainers_Heart$explainers[[30]],
                         gbm_explainers_Heart$explainers[[31]],gbm_explainers_Heart$explainers[[32]],
                         gbm_explainers_Heart$explainers[[33]],gbm_explainers_Heart$explainers[[34]],
                         gbm_explainers_Heart$explainers[[35]],gbm_explainers_Heart$explainers[[36]],
                         gbm_explainers_Heart$explainers[[37]],gbm_explainers_Heart$explainers[[38]],
                         rf_explainers_Heart$explainers[[1]], rf_explainers_Heart$explainers[[2]],
                         rf_explainers_Heart$explainers[[3]],rf_explainers_Heart$explainers[[4]],
                         rf_explainers_Heart$explainers[[5]])

res_Heart <- rashomon_detect(explainers_list = explainers, sorted=TRUE, k=3)
res_Heart

res_Heart_euc <- rashomon_detect(explainers_list = explainers, pdi_method_numerical= euclidean_distance, sorted=TRUE, k=3)
res_Heart_der_euc <- rashomon_detect(explainers_list = explainers, pdi_method_numerical= derivative_euclidean_distance, sorted=TRUE, k=3)

plot_summary_matrix(res_Heart)


