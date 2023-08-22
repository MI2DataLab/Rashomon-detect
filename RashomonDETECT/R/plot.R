#' Plot Summary Matrix
#'
#' This function generates a summary matrix plot that visualizes various aspects of the model comparison results. It includes bar plots for pairwise distances, dissimilarity plots, and performance plots.
#'
#' @export
#'
#' @param res A list containing the results obtained from the `rashomon_detect()` function.
#'
#' @import ggplot2
#' @import reshape2
#' @import GGally
#'
#' @return A matrix plot visualizing `Rashomon_DETECT` results.
plot_summary_matrix <- function(res){
  plots_list <- list()

  melted_data <- melt(res$distances, id.vars = c("model_ind_1", "model_ind_2"))
  max_y_value <- max(melted_data$value)
  fill_limits_dissimilarity <- range(res$selected_models_distance_matrix[res$selected_models_distance_matrix != 0])
  fill_limits_performance <- range(res$performances[res$selected_models_indices])
  nmodels <- length(res$selected_models_indices)

  counter <- 1
  for (i in 1:nmodels){
    model_index_i <- res$selected_models_indices[i]
    for (j in 1:nmodels){
      model_index_j <- res$selected_models_indices[j]
      if (i > j){
        model_pair_data <- melted_data[(
          melted_data$model_ind_1 == model_index_i &
            melted_data$model_ind_2 == model_index_j
        ) |
          (
            melted_data$model_ind_1 == model_index_j &
              melted_data$model_ind_2 == model_index_i
          ),]
        plots_list[[counter]] <- create_bar_plot(model_pair_data, max_y_value)
      } else if (i < j) {
        distance <- res$selected_models_distance_matrix[i, j]
        plots_list[[counter]] <- create_dissimilarity_plot(distance, fill_limits_dissimilarity)
      } else {
        performance <- res$performances[model_index_i]
        plots_list[[counter]] <- create_performance_plot(performance, fill_limits_performance)
      }
      counter <- counter + 1
    }
  }

  model_labels <- sapply(res$explainers, function(x) x$label)
  selected_models_labels <- paste("Model", res$selected_models_indices, "-", model_labels[res$selected_models_indices])

  ggmatrix(plots_list, nrow = nmodels, ncol = nmodels,
           xAxisLabels = selected_models_labels,
           yAxisLabels = selected_models_labels)
}


create_bar_plot <- function(model_pair_data, max_y_value) {
  main_df <- model_pair_data[model_pair_data$variable != "avg_pdi", ]
  ggplot(main_df, aes_string(x = "variable", y = "value")) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_hline(yintercept = model_pair_data$value[model_pair_data$variable == "avg_pdi"],
               color = "red") +
    theme_minimal() +
    labs(x = "Variable",
         y = "Distance",
         fill = "Variable") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_y_continuous(limits=c(0, max_y_value), expand = c(0, 0))
}

create_dissimilarity_plot <- function(distance, fill_limits){
  ggplot() +
    geom_tile(aes(x = 0, y = 0, fill = distance)) +
    scale_fill_distiller(palette="Blues",
                         direction=1,
                         limits=fill_limits) +
    geom_label(aes(x = 0, y = 0),
               label = paste("Dissimilarity =", as.character(round(distance, 3))),
               color = "black") +
    theme_void()
}

create_performance_plot <- function(performance, fill_limits){
  ggplot() +
    geom_tile(aes(x = 0, y = 0, fill = performance)) +
    scale_fill_distiller(palette="Greens",
                         direction=1,
                         limits=fill_limits) +
    geom_label(aes(x = 0, y = 0),
               label = paste("Performance =", as.character(round(performance, 3))),
               color = "black") +
    theme_void()
}


