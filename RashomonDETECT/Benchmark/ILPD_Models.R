source("model_selection.R")
source("pdi_measures.R")
source("performance_calculation.R")
source("profile_dissimilarity_calculation.R")
source("plot.R")
source("rashomon_detect.R")

source("GetRashomonExplainers.R")
source("GetRashomonExplainersRF.R")

# Data set

ILPD <- read.csv("13_Data.csv", dec = ".")

ILPD <- ILPD %>% 
  mutate(target = factor(target),
         Gender = factor(Gender))


DataILPD_model <- ILPD[,-c(1, 5, 11)]

DataILPD_model <- DataILPD_model %>%
  mutate(target = as.factor(target))

# Modelling

set.seed(125)

taskDataILPD_all = TaskClassif$new("DataILPD", DataILPD_model, target = "target")

#### mlr_learners

# Gradient Boosting Models 

gbm_learner_DataILPD = LearnerClassifGBM$new()
gbm_learner_DataILPD$predict_type <- 'prob'
gbm_learner_DataILPD
gbm_learner_DataILPD$param_set


# Random search 

random_tuner_DataILPD = tnr("grid_search", resolution = 5)
evals20_DataILPD = trm("evals", n_evals = 50)
cv_DataILPD = rsmp("repeated_cv", folds = 5, repeats = 5)
measure_DataILPD = msr("classif.auc")

params_set_range_DataILPD = ParamSet$new(list(
  ParamInt$new("n.trees", lower = 50, upper = 700),
  ParamInt$new("interaction.depth", lower = 1, upper = 10),
  ParamDbl$new("shrinkage", lower = 0.1, upper = 0.5)
))

search_space_DataILPD = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

random_tuner_config_DataILPD = TuningInstanceSingleCrit$new(
  task = taskDataILPD_all,
  learner = gbm_learner_DataILPD,
  resampling = cv_DataILPD,
  measure = measure_DataILPD,
  search_space = params_set_range_DataILPD,
  terminator = evals20_DataILPD
)

random_tuner_DataILPD$optimize(random_tuner_config_DataILPD)

# best results
random_tuner_config_DataILPD$result_learner_param_vals

# all history
random_tuner_config_DataILPD$archive

# Random Forest

rf_learner_DataILPD = LearnerClassifRanger$new()
rf_learner_DataILPD$predict_type <- 'prob'
rf_learner_DataILPD
rf_learner_DataILPD$param_set

random_tuner_DataILPD = tnr("grid_search", resolution = 5)
evals20_DataILPD = trm("evals", n_evals = 50)
cv_DataILPD = rsmp("repeated_cv", folds = 5, repeats = 5)
measure_DataILPD = msr("classif.auc")

params_set_range_rf_DataILPD = ParamSet$new(list(
  ParamInt$new("num.trees", lower = 50, upper = 500),
  ParamInt$new("mtry", lower = 1, upper = 8),
  ParamInt$new("max.depth", lower = 2, upper = 5)
  #ParamInt$new("min.node.size", lower = 1, upper = 500)
  #ParamDbl$new("bag.fraction", lower = min(gbm_params$bag.fraction), upper = max(gbm_params$bag.fraction))
  
))

search_space_rf_DataILPD = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  minsplit = p_int(lower = 1, upper = 10)
)

search_space_rf_DataILPD

random_tuner_config_rf_DataILPD = TuningInstanceSingleCrit$new(
  task = taskDataILPD_all,
  learner = rf_learner_DataILPD,
  resampling = cv_DataILPD,
  measure = measure_DataILPD,
  search_space = params_set_range_rf_DataILPD,
  terminator = evals20_DataILPD
)

random_tuner_DataILPD$optimize(random_tuner_config_rf_DataILPD)

random_tuner_config_rf_DataILPD$result_learner_param_vals

data.frame(random_tuner_config_rf_DataILPD$archive$data) %>%
  arrange(classif.auc) %>%
  mutate(x = row_number()) %>%
  ggplot(aes(x = x, y = classif.auc))+
  geom_line()+
  theme_minimal()

data.frame(random_tuner_config_rf_DataILPD$archive$data) %>%
  arrange(-classif.auc) 

#get rashomon model with given tolerance

DataILPD_models_rf <- random_tuner_config_rf_DataILPD$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(num.trees, mtry,  max.depth, classif.auc) 

DataILPD_models <- random_tuner_config_DataILPD$archive$data %>%
  arrange(-classif.auc) %>%
  dplyr::select(n.trees, interaction.depth,  shrinkage, classif.auc) 

best_result_rf_ILPD = DataILPD_models_rf[1,]
best_result_ILPD = DataILPD_models[1,]

best_performance_ILPD = max(best_result_ILPD$classif.auc, best_result_rf_ILPD$classif.auc)

tolerance = 0.01

rashomon_models_rf_ILPD = list() #RF

for (i in 1:dim(DataILPD_models_rf)[1]) {
  if (abs(DataILPD_models_rf$classif.auc[i] - best_performance_ILPD) < tolerance) {
    rashomon_models_rf_ILPD = list(rashomon_models_rf_ILPD, DataILPD_models_rf[i])
  }
}

rashomon_models_ILPD = list() #GBM

for (i in 1:dim(DataILPD_models)[1]) {
  if (abs(DataILPD_models$classif.auc[i] - best_performance_ILPD) < tolerance) {
    rashomon_models_ILPD = list(rashomon_models_ILPD, DataILPD_models[i])
  }
}

rf_explainers_ILPD <- get_Rashomon_models_exp_rf(random_tuner_config_rf_DataILPD$archive$data, taskDataILPD_all, 31 )
gbm_explainers_ILPD <- get_Rashomon_models(random_tuner_config_DataILPD$archive$data, taskDataILPD_all, 1)

# Rashomon_DETECT

explainers <- list(rf_explainers_ILPD$explainers[[1]],rf_explainers_ILPD$explainers[[2]],rf_explainers_ILPD$explainers[[3]],
                        rf_explainers_ILPD$explainers[[4]],rf_explainers_ILPD$explainers[[5]],rf_explainers_ILPD$explainers[[6]],
                        rf_explainers_ILPD$explainers[[7]],rf_explainers_ILPD$explainers[[8]],rf_explainers_ILPD$explainers[[9]],
                        rf_explainers_ILPD$explainers[[10]],rf_explainers_ILPD$explainers[[11]],rf_explainers_ILPD$explainers[[12]],
                        rf_explainers_ILPD$explainers[[13]],rf_explainers_ILPD$explainers[[14]],rf_explainers_ILPD$explainers[[15]],
                        rf_explainers_ILPD$explainers[[16]],rf_explainers_ILPD$explainers[[17]],rf_explainers_ILPD$explainers[[18]],
                        rf_explainers_ILPD$explainers[[19]],rf_explainers_ILPD$explainers[[20]],rf_explainers_ILPD$explainers[[21]],
                        rf_explainers_ILPD$explainers[[22]],rf_explainers_ILPD$explainers[[23]],rf_explainers_ILPD$explainers[[24]],
                        rf_explainers_ILPD$explainers[[25]],rf_explainers_ILPD$explainers[[26]],rf_explainers_ILPD$explainers[[27]],
                        rf_explainers_ILPD$explainers[[28]],rf_explainers_ILPD$explainers[[29]],rf_explainers_ILPD$explainers[[30]],
                        rf_explainers_ILPD$explainers[[31]],
                       gbm_explainers_ILPD$explainers[[1]])

res_ILPD <- rashomon_detect(explainers, sorted =TRUE, k=3) #done 1
res_ILPD

res_ILPD_eucl <- rashomon_detect(explainers_ILPD, pdi_method_numerical = euclidean_distance, sorted=TRUE, k=3)
res_ILPD_eucl

res_ILPD_der_eucl <- rashomon_detect(explainers_ILPD, pdi_method_numerical = derivative_euclidean_distance, sorted=TRUE, k=3)
res_ILPD_der_eucl

plot_summary_matrix(res_ILPD)