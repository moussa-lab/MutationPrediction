library(tensorflow)
library(keras3)
library(R.matlab)
library(pheatmap)
library(parallel)
library(future)
library(furrr)
library(progressr)

handlers(global = TRUE)
handlers(list(handler_txtprogressbar()))

# ---------------------
# Load Data
# ---------------------

set.seed(123)

mat_train_data <- readMat("study_data/COAD2.mat")
COAD2_unshuffled <- as.matrix(mat_train_data$COAD2[[2]]) # Training data
#COAD2 <- COAD2_unshuffled[, sample(ncol(COAD2_unshuffled))]

DFCI2_test_data <- readMat("study_data/DFCI2.mat")  # Test data
DFCI2_unshuffled <- as.matrix(DFCI2_test_data$DFCI2[[3]])  # Test mutation matrix
#DFCI2 <- DFCI2_unshuffled[, sample(ncol(DFCI2_unshuffled))]

MGI2_test_data <- readMat("study_data/MGI2.mat")
MGI2_unshuffled <- as.matrix(MGI2_test_data$MGI2[[3]])
#MGI2 <- MGI2_unshuffled[, sample(ncol(MGI2_unshuffled))]

combined_data_unshuffled <- cbind(COAD2_unshuffled, DFCI2_unshuffled, MGI2_unshuffled)
combined_data <- combined_data_unshuffled[, sample(ncol(combined_data_unshuffled))]

total_samples <- ncol(combined_data)
train_sample_size <- floor(0.9 * total_samples)

train_indices <- sample(seq_len(total_samples), size = train_sample_size)
test_indices <- setdiff(seq_len(total_samples), train_indices)

TrainingSet <- combined_data[, train_indices]
TestSet <- combined_data[, test_indices]

#TrainingSet <- DFCI2_unshuffled[, sample(ncol(DFCI2_unshuffled))]
#TestSet <- COAD2_unshuffled[, sample(ncol(COAD2_unshuffled))]

if (!dir.exists("new4_trained_models")) dir.create("new4_trained_models")

# ---------------------
# Train LSTM Function
# ---------------------
TrainLSTM <- function(t, data, labels) {
  clear_session()
  # Prepare data using mutations from t+1 to the last mutation (rows above t)
  data <- data[1:(t - 1), ]
  if (is.vector(data)) data <- matrix(data, nrow = 1)
  
  n <- nrow(data)  # Number of mutations
  m <- ncol(data)  # Number of samples
  
  # Build 3D array for Keras
  sequences <- lapply(seq_len(m), function(j) data[, j])
  reshaped_data <- array(unlist(sequences), dim = c(m, n, 1))
  
  # Define and compile model
  model <- keras_model_sequential() %>%
    layer_lstm(units = 5, input_shape = c(n, 1), return_sequences = FALSE) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    optimizer = "adam",
    loss = "binary_focal_crossentropy",
    metrics = list("accuracy", "recall", "precision")
  )
  
  # Params from Auslander paper
  model %>% fit(
    x = reshaped_data,
    y = labels,
    epochs = 10,
    batch_size = 27,
    verbose = 1
  )
  
  model |> save_model(paste0("new4_trained_models/model_t_", t, ".keras"), overwrite = TRUE)
  
  rm(model, reshaped_data, sequences, data, early_stopping)
  gc()
}

# ---------------------
# Train Models in Parallel Batches
# ---------------------
train_all_models_in_batches <- function(TrainingSet, num_mutations, retrain,
                                        batch_size = 32) {
  # We'll do parallel training in batches of batch_size
  # to avoid huge memory usage.
  
  all_ts <- 2:num_mutations
  batch_indices <- split(all_ts, ceiling(seq_along(all_ts) / batch_size))
  
  for (batch_vec in batch_indices) {
    # Start a fresh plan for each batch to ensure memory cleanup
    plan(sequential)
    plan(multisession, workers = min(detectCores() - 1, 24))
    on.exit(plan(sequential), add = TRUE)
    
    with_progress({
      p <- progressor(steps = length(batch_vec))
      future_map(batch_vec, ~{
        p(sprintf("Training model for mutation %d", .x))
        model_file <- paste0("new4_trained_models/model_t_", .x, ".keras")
        training_labels <- TrainingSet[.x, ]
        
        if (retrain || !file.exists(model_file)) {
          TrainLSTM(.x, TrainingSet, training_labels)
        } else {
          message(sprintf("Skipping retrain for mutation %d (model exists)", .x))
        }
      }, .options = furrr_options(seed = TRUE, globals = TRUE))
    })
    
    plan(sequential)  # Release workers between batches
    gc()
  }
}

# Parallelized Predictions with Retraining if Necessary
compute_accuracy_matrix_parallel <- function(num_mutations, TrainingSet, TestSet, batch_size = 32) {
  num_samples <- ncol(TestSet)
  all_ts <- 2:num_mutations
  batch_indices <- split(all_ts, ceiling(seq_along(all_ts) / batch_size))
  
  # Initialize accuracy matrix
  prediction_accuracy <- matrix(0, nrow = num_mutations, ncol = num_samples)
  
  with_progress({
    p <- progressor(steps = length(batch_indices))
    for (batch_vec in batch_indices) {
      plan(sequential)
      plan(multisession, workers = min(detectCores() - 1, 24))
      on.exit(plan(sequential), add = TRUE)
      
      batch_results <- future_map(batch_vec, function(t) {
        clear_session()
        model_file <- paste0("new4_trained_models/model_t_", t, ".keras")
        accuracy <- NULL
        
        # Ensure the model is valid and retrain if necessary
        if (file.exists(model_file)) {
          tryCatch({
            # Attempt to load and predict
            model <- load_model(model_file)
            input_sequences <- array(TestSet[1:(t - 1), ], dim = c(num_samples, t - 1, 1))
            predictions <- predict(model, input_sequences)
            predicted_classes <- ifelse(predictions > 0.5, 1, 0)
            actual_classes <- TestSet[t, ]
            accuracy <- as.numeric(predicted_classes == actual_classes)
            rm(model, input_sequences, predictions, predicted_classes, actual_classes)
            gc()
          }, error = function(e) {
            message(sprintf("Error predicting for mutation %d: %s. Retrying with retraining...", t, e$message))
            accuracy <- NULL  # Signal retraining is needed
          })
        } else {
          message(sprintf("Model missing for mutation %d. Retraining...", t))
        }
        
        # If accuracy is NULL, retrain and retry predictions
        if (is.null(accuracy)) {
          tryCatch({
            training_labels <- TrainingSet[t, ]
            TrainLSTM(t, TrainingSet, training_labels)  # Retrain model
            model <- load_model(model_file)  # Load the retrained model
            input_sequences <- array(TestSet[1:(t - 1), ], dim = c(num_samples, t - 1, 1))
            predictions <- predict(model, input_sequences)
            predicted_classes <- ifelse(predictions > 0.5, 1, 0)
            actual_classes <- TestSet[t, ]
            accuracy <- as.numeric(predicted_classes == actual_classes)
            rm(model, input_sequences, predictions, predicted_classes, actual_classes)
            gc()
          }, error = function(e) {
            message(sprintf("Failed to retrain or predict for mutation %d: %s", t, e$message))
            accuracy <- rep(NA, num_samples)  # Return NA for this mutation
          })
        }
        
        return(accuracy)
      }, .options = furrr_options(seed = TRUE, globals = TRUE))
      
      # Update accuracy matrix with batch results
      for (i in seq_along(batch_vec)) {
        prediction_accuracy[batch_vec[i], ] <- batch_results[[i]]
      }
      
      p()
      plan(sequential)  # Release workers between batches
      gc()
    }
  })
  
  return(prediction_accuracy)
}

# ---------------------
# Main Execution
# ---------------------
apply_models_to_test_set_batch <- function(retrain = FALSE) {
  num_mutations <- 1000  # for testing
  #num_mutations <- nrow(COAD2)
  
  # Train all models in parallel batches
  train_all_models_in_batches(TrainingSet, num_mutations, retrain, batch_size = 48)
  
  # Compute accuracy matrix
  accuracy_matrix <- compute_accuracy_matrix_parallel(num_mutations, TrainingSet, TestSet, batch_size = 32)
  return(accuracy_matrix)
}

# ---------------------
# Run and Generate Plots
# ---------------------
accuracy_matrix <- apply_models_to_test_set_batch(retrain = TRUE)

pheatmap(
  accuracy_matrix,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  color = c("red", "green"),
  legend_breaks = c(0, 1),
  legend_labels = c("Incorrect", "Correct"),
  main = "Prediction Accuracy Heatmap"
)

# Generate confusion matrix from accuracy_matrix
generate_confusion_matrix <- function(accuracy_matrix, TestSet) {
  num_mutations <- nrow(accuracy_matrix)
  num_samples <- ncol(accuracy_matrix)
  
  # Initialize confusion matrix counts
  TP <- 0
  FP <- 0
  TN <- 0
  FN <- 0
  
  for (t in 2:num_mutations) {
    for (s in 1:num_samples) {
      predicted <- accuracy_matrix[t, s]  # 1 = Correct, 0 = Incorrect
      actual <- TestSet[t, s]            # Ground truth from test set
      
      if (predicted == 1 && actual == 1) {
        TP <- TP + 1
      } else if (predicted == 1 && actual == 0) {
        TN <- TN + 1
      } else if (predicted == 0 && actual == 0) {
        FP <- FP + 1
      } else if (predicted == 0 && actual == 1) {
        FN <- FN + 1
      }
    }
  }
  
  # Create confusion matrix
  confusion_matrix <- matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE,
                             dimnames = list("Actual" = c("Positive", "Negative"),
                                             "Predicted" = c("Positive", "Negative")))
  
  # Print confusion matrix
  print("Confusion Matrix:")
  print(confusion_matrix)
  
  return(confusion_matrix)
}

# Function to calculate metrics from confusion matrix
generate_metrics_table <- function(confusion_matrix) {
  # Extract confusion matrix values
  TP <- confusion_matrix[1, 1]
  FN <- confusion_matrix[1, 2]
  FP <- confusion_matrix[2, 1]
  TN <- confusion_matrix[2, 2]
  
  # Total positives and negatives
  P <- TP + FN  # Total actual positives
  N <- TN + FP  # Total actual negatives
  total <- P + N  # Total population
  
  # Metrics calculations
  TPR <- TP / P  # True Positive Rate (Recall/Sensitivity)
  FNR <- FN / P  # False Negative Rate
  FPR <- FP / N  # False Positive Rate
  TNR <- TN / N  # True Negative Rate (Specificity)
  
  PPV <- TP / (TP + FP)  # Positive Predictive Value (Precision)
  FOR <- FN / (FN + TN)  # False Omission Rate
  FDR <- FP / (TP + FP)  # False Discovery Rate
  NPV <- TN / (TN + FN)  # Negative Predictive Value
  
  ACC <- (TP + TN) / total  # Accuracy
  BA <- (TPR + TNR) / 2     # Balanced Accuracy
  
  F1 <- 2 * (PPV * TPR) / (PPV + TPR)  # F1 Score
  #MCC <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews Correlation Coefficient
  MCC <- sqrt(TPR * TNR * PPV * NPV) - sqrt(FNR * FPR * FOR * FDR)
  TS <- TP / (TP + FN + FP)  # Threat Score (CSI/Jaccard Index)
  
  LR_plus <- TPR / FPR  # Positive Likelihood Ratio
  LR_minus <- FNR / TNR  # Negative Likelihood Ratio
  
  # Diagnostic Odds Ratio (DOR)
  DOR <- LR_plus / LR_minus
  
  # Compile into a table
  metrics_table <- data.frame(
    Metric = c("True Positive Rate (Sensitivity, Recall)",
               "False Negative Rate",
               "True Negative Rate (Specificity)",
               "False Positive Rate",
               "Positive Predictive Value (Precision)",
               "Negative Predictive Value",
               "False Omission Rate",
               "False Discovery Rate",
               "Accuracy",
               "Balanced Accuracy",
               "F1 Score",
               "Matthews Correlation Coefficient",
               "Threat Score (CSI, Jaccard Index)",
               "Positive Likelihood Ratio",
               "Negative Likelihood Ratio",
               "Diagnostic Odds Ratio"),
    Value = c(TPR, FNR, TNR, FPR, PPV, NPV, FOR, FDR, ACC, BA, F1, MCC, TS, LR_plus, LR_minus, DOR)
  )
  
  return(metrics_table)
}

# Generate confusion matrix
confusion_matrix <- generate_confusion_matrix(accuracy_matrix, TestSet)

metrics_table <- generate_metrics_table(confusion_matrix)
print(metrics_table)