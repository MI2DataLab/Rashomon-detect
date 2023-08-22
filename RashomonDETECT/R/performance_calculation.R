#' @importFrom DALEX model_performance
get_performance_vals <-
  function(explainers_list, performance_measure = NULL) {
    performances <- lapply(explainers_list, model_performance)
    performance_vals <-
      sapply(performances, function(x)
        x[["measures"]][[performance_measure]])
    performance_vals
  }

get_best_performance_val <-
  function(performance_vals, performance_measure) {
    best_performance_val <-
      ifelse(
        performance_measure %in% c("mse", "rmse", "mad"),
        min(performance_vals),
        max(performance_vals)
      )
  }

get_best_performing_model_index <-
  function(performance_vals, best_performance) {
    which(performance_vals == best_performance)[1]
  }
