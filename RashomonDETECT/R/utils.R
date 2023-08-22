#' Create a List of Explainers from Models
#'
#' This function creates a list of model explainers using the `DALEX::explain()` function.
#'
#' @export
#' @importFrom DALEX explain
#'
#' @param model1 The primary model for which an explainer will be created.
#' @param ... Additional models for which explainers will be created.
#' @param data The dataset used that will be used for explaining the models (creating profiles).
#' @param y Response variable in the dataset.
#' @param verbose If TRUE, verbose output will be printed during explanation.
#'
#' @return A list of model explainers for the provided models.
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
#' explainers
#'
make_explainer_list <-
  function(model1, ..., data, y, verbose = FALSE) {
    model_list <- list(model1, ...)
    lapply(model_list,
           explain,
           data = data,
           y = y,
           verbose = verbose)
  }
