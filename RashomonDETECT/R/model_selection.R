select_most_different_models_all <- function(dist_matrix,
                                             k,
                                             best_performing_model_index) {
  selected_models_indices <- best_performing_model_index
  counter <- 1
  
  unique_models <-
    unique(c(dist_matrix$model_ind_1, dist_matrix$model_ind_2))
  
  while (counter < k) {
    next_model_index <-
      find_next_model_index_all(dist_matrix, unique_models, selected_models_indices)
    selected_models_indices <-
      c(selected_models_indices, next_model_index)
    counter <- counter + 1
  }
  selected_models_indices
}

find_next_model_index_all <-
  function(dist_matrix,
           unique_models,
           selected_models_indices) {
    available_models <- setdiff(unique_models, selected_models_indices)
    average_distances <-
      sapply(available_models, function(model_index) {
        avg_distance <-
          mean(dist_matrix$avg_pdi[(
            dist_matrix$model_ind_1 %in% selected_models_indices &
              dist_matrix$model_ind_2 == model_index
          ) |
            (
              dist_matrix$model_ind_2 %in% selected_models_indices &
                dist_matrix$model_ind_1 == model_index
            )])
        return(avg_distance)
      })
    available_models[which.max(average_distances)]
  }


select_most_different_models_last <- function(profiles_numerical,
                                              profiles_categorical,
                                              pdi_method_numerical,
                                              pdi_method_categorical,
                                              k,
                                              best_performing_model_index) {
  selected_models_indices <- best_performing_model_index
  counter <- 1
  
  unique_models <- 1:length(profiles_numerical)
  
  all_vnames_numerical <- unique(unlist(
      lapply(profiles_numerical, function(df)
        unique(df$agr_profiles$`_vname_`))
    ))
  
  all_vnames_categorical <- unique(unlist(
      lapply(profiles_categorical, function(df)
        unique(df$agr_profiles$`_vname_`))
    ))
  
  distances <- matrix(
    0,
    nrow = 0,
    ncol = 2 + length(all_vnames_numerical) + length(all_vnames_categorical),
    dimnames = list(NULL, c(
      "model_ind_1", "model_ind_2", all_vnames_numerical, all_vnames_categorical
    ))
  )
  
  while (counter < k) {
    new_dist <- calculate_distances_to_last_model(
      profiles_numerical,
      profiles_categorical,
      pdi_method_numerical,
      pdi_method_categorical,
      all_vnames_numerical, 
      all_vnames_categorical,
      unique_models,
      selected_models_indices
    )
    distances <- rbind(distances, new_dist)
    
    next_model_index <- find_next_model_index_last(new_dist)
    selected_models_indices <-
      c(selected_models_indices, next_model_index)
    counter <- counter + 1
  }
  
  list(distances = distances,
       selected_models_indices = selected_models_indices)
}


find_next_model_index_last <- function(new_dist) {
  new_dist$model_ind_2[which.max(new_dist$avg_pdi)]
}
