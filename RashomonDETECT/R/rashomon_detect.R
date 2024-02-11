#' Detect `k` Most Different Models in Provided Set
#'
#' This function performs a `Rashomon_DETECT` algorithm on a given set of models,
#' aiming to identify the `k` most different models based on selected parameters.
#' It returns a list containing performance values, model profiles, selected models,
#' and calculated dissimilarities.
#'
#' @export
#'
#' @param explainers_list A list of models wrapped in explainers using the `DALEX::explain()` function (`make_explainer_list()` function can be helpful). All models should be classification or regression models.
#' @param performance_measure Performance measure used to evaluate model performance on data in explainers. For regression, options include `"mse"`, `"rmse"`, `"r2"`, or `"mad"`. For classification, options include `"auc"`, `"accuracy"`, `"f1"`, `"recall"`, or `"precision"`. Defaults are `"mse"` and `"auc"`.
#' @param sorted Whether models are sorted by performance. If not, sorting will be performed based on performance on data in explainers.
#' @param k Number of models to be detected as most different.
#' @param profile_type Method used for generating profiles. Options are `"pdp"` for Partial Dependence Plots and `"ale"` for Accumulated Local Effects. 
#' @param pdi_method_numerical Method for comparing profiles for numerical variables. Options are `"pdi"`, `"derivative l2"`, `"l2"`, or a custom function that takes three numerical vectors (profile 1, profile 2, and variable values) and returns a distance.
#' @param pdi_method_categorical Method for comparing profiles for categorical variables. Options are `"vector_distance"` or a custom function that takes three numerical vectors (profile 1, profile 2, and variable values) and returns a distance.
#' @param comparison Comparison method: `"all"` if the next model should be selected based on comparison with all previously selected models; `"last"` if it should be compared only with the last selected model.
#' @param include_categorical_variables Whether to include categorical variables for comparison.
#' @param N Number of data points used for calculating profiles, passed to the `DALEX::model_profile()` function.
#' @param variable_splits Variable splits used for calculating profiles, passed to the `DALEX::model_profile()` function.
#'
#' @importFrom DALEX model_profile
#'
#' @return A list containing explainers, calculated performances, profiles, dissimilarities (distances) and indices of selected models.
#'
#' @examples
#' # train models
#' library(DALEX)
#' library(ranger)
#' library(randomForest)
#' library(gbm)
#' model1 <- ranger(survived ~
#'                  gender + age + class + embarked + fare + sibsp + parch,
#'                  data = titanic_imputed, classification = TRUE)
#' model2 <- glm(survived ~
#'               gender + age + class + embarked + fare + sibsp + parch,
#'               data = titanic_imputed, family = binomial)
#' model3 <- randomForest(factor(survived) ~
#'                        gender + age + class + embarked + fare + sibsp + parch,
#'                        data = titanic_imputed, ntree=100)
#' model4 <- gbm(survived ~
#'               gender + age + class + embarked + fare + sibsp + parch,
#'               data = titanic_imputed, distribution = "bernoulli")
#' explainers <- make_explainer_list(model1, model2, model3, model4,
#'               data = titanic_imputed[, -8], y = titanic_imputed$survived)
#' res <- rashomon_detect(explainers, k=3, comparison = "last")
#' res
#'
#' plot_summary_matrix(res)
#'
rashomon_detect <- function(explainers_list,
                            performance_measure = NULL,
                            sorted = FALSE,
                            k = 2,
                            profile_type = "pdp",
                            pdi_method_numerical = "pdi",
                            pdi_method_categorical = "vector_distance",
                            comparison = "last",
                            include_categorical_variables = TRUE,
                            N = NULL,
                            variable_splits = NULL) {
  
  task_type <- unique(sapply(explainers_list, function(x) x$model_info$type))
  stopifnot("There are models of different task type in the list of explainers" = length(task_type) == 1)
  
  if (is.null(performance_measure)) {
    performance_measure <-
      ifelse(task_type == "classification", "auc", "mse")
  }

  stopifnot(length(explainers_list) > k)
  stopifnot(profile_type %in% c("pdp", "ale"))
  profile_type <- ifelse(profile_type == "pdp", "partial", "accumulated")

  performance_vals <-
    get_performance_vals(explainers_list, performance_measure)
  best_performance_val <-
    get_best_performance_val(performance_vals, performance_measure)
  best_performing_model_index <- ifelse(sorted == FALSE,
    get_best_performing_model_index(performance_vals, best_performance_val),
    1)

  res <- list()
  res$explainers <- explainers_list
  res$performances <- performance_vals

  profiles_numerical <- lapply(
    explainers_list,
    model_profile,
    N = N,
    variable_splits = variable_splits,
    type = profile_type
  )
  res$profiles_numerical <- profiles_numerical

  profiles_categorical <- NULL
  if (include_categorical_variables & !all(sapply(explainers_list[[1]]$data, is.numeric))){
    profiles_categorical <- lapply(
      explainers_list,
      model_profile,
      N = N,
      variable_type = "categorical",
      type = profile_type
    )
    res$profiles_categorical <- profiles_categorical
  }

  pdi_method_numerical <- switch(pdi_method_numerical,
         "pdi" = derivative_fraction_sign_difference,
         "derivative l2" = derivative_euclidean_distance,
         "l2" = euclidean_distance)

  if (pdi_method_categorical == "vector_distance")
    pdi_method_categorical <- vector_distance

  if (comparison == "all") {
    distances <- calculate_all_distances(profiles_numerical,
                                         profiles_categorical,
                                         pdi_method_numerical,
                                         pdi_method_categorical)
    selected_models_indices <-
      select_most_different_models_all(distances,
                                       k,
                                       best_performing_model_index)
  } else if (comparison == "last") {
    tmp <- select_most_different_models_last(profiles_numerical,
                                             profiles_categorical,
                                             pdi_method_numerical,
                                             pdi_method_categorical,
                                             k,
                                             best_performing_model_index)
    distances <- tmp$distances
    selected_models_indices <- tmp$selected_models_indices
  } else {
    stop("Only comparison = 'last' and comparison = 'all' are implemented.")
  }

  res$distances <- distances
  res$selected_models_indices <- selected_models_indices
  res$selected_models_distance_matrix <-
    get_final_distance_matrix(distances, selected_models_indices)
  res
}
