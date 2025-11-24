library(tensorflow)
library(keras3)
library(R.matlab)
library(pheatmap)
library(parallel)
library(future)
library(furrr)
library(progressr)

TAG <- "COAD2_metric-improvement_thr0p02_min1_cyclefeedback-arc-set" # set per run

BASE_DIR <- file.path("outputs", TAG)
MODELS_DIR <- file.path(BASE_DIR, "models")
LP_PATH <- file.path(BASE_DIR, sprintf("longest_path_edges_%s.rds", TAG))
# LP_PATH <- file.path("MRM_order.txt")

if (!dir.exists(BASE_DIR)) dir.create(BASE_DIR, recursive = TRUE)
if (!dir.exists(MODELS_DIR)) dir.create(MODELS_DIR, recursive = TRUE)

handlers(global = TRUE)
handlers(list(handler_txtprogressbar()))

set.seed(123)

# helpers
read_named_mat <- function(mat_path, ds_name, data_genes_samples = c(1, 2, 3)) {
  x <- readMat(mat_path)
  ds <- x[[ds_name]]
  if (is.null(ds)) stop("Dataset name '", ds_name, "' not found in ", mat_path)
  mat <- as.matrix(ds[[data_genes_samples[1]]])
  rownames(mat) <- as.character(Reduce(c, ds[[data_genes_samples[2]]]))
  colnames(mat) <- as.character(Reduce(c, ds[[data_genes_samples[3]]]))
  mat
}

# load longest-path ordering
lp <- readRDS(LP_PATH) # from graph_generation.R output (saved in outputs/<TAG>/)
# lp <- readLines(LP_PATH)
# shuffle lp
# lp <- sample(lp)
print(head(lp))
lp_meta <- attr(lp, "params")
print(lp_meta)

# load matrices
COAD2 <- read_named_mat("study_data/COAD2.mat", "COAD2", c(2, 1, 3))
DFCI2 <- read_named_mat("study_data/DFCI2.mat", "DFCI2", c(3, 2, 1))
MGI2 <- read_named_mat("study_data/MGI2.mat", "MGI2", c(3, 2, 1))

combined_data_unshuffled <- cbind(COAD2, DFCI2, MGI2)
rm(COAD2, DFCI2, MGI2)
gc()

# subset to longest path and preserve order
path_genes_in_data <- intersect(lp, rownames(combined_data_unshuffled))
if (length(path_genes_in_data) < 2) stop("Not enough genes from longest path in data")

ordered_combined <- combined_data_unshuffled[path_genes_in_data, , drop = FALSE]
rm(combined_data_unshuffled)
gc()

# shuffle samples only
combined_data <- ordered_combined[, sample(ncol(ordered_combined)), drop = FALSE]
rm(ordered_combined)
gc()

# split
total_samples <- ncol(combined_data)
train_sample_size <- floor(0.8 * total_samples)
train_indices <- sample(seq_len(total_samples), size = train_sample_size, replace = FALSE)
test_indices <- setdiff(seq_len(total_samples), train_indices)

TrainingSet <- combined_data[, train_indices, drop = FALSE]
TestSet <- combined_data[, test_indices, drop = FALSE]
rm(combined_data)
gc()

print(head(TrainingSet)[, 1:5])
print(head(TestSet)[, 1:5])

# LSTM + batching
TrainLSTM <- function(t, data, labels) {
  clear_session()
  # Prepare data using mutations from t+1 to the last mutation (rows above t)
  data <- data[1:(t - 1), ]
  if (is.vector(data)) data <- matrix(data, nrow = 1)

  n <- nrow(data) # Number of mutations
  m <- ncol(data) # Number of samples

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

  save_path <- file.path(MODELS_DIR, sprintf("model_t_%d.keras", t))
  model |> save_model(save_path, overwrite = TRUE)

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
    plan(multisession, workers = min(detectCores() - 1, 4))
    on.exit(plan(sequential), add = TRUE)

    with_progress({
      p <- progressor(steps = length(batch_vec))
      future_map(batch_vec, ~ {
        p(sprintf("Training model for mutation %d", .x))
        model_file <- file.path(MODELS_DIR, sprintf("model_t_%d.keras", .x))
        training_labels <- TrainingSet[.x, ]

        if (retrain || !file.exists(model_file)) {
          TrainLSTM(.x, TrainingSet, training_labels)
        } else {
          message(sprintf("Skipping retrain for mutation %d (model exists)", .x))
        }
      }, .options = furrr_options(seed = TRUE, globals = TRUE))
    })

    plan(sequential) # Release workers between batches
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
      plan(multisession, workers = min(detectCores() - 1, 4))
      on.exit(plan(sequential), add = TRUE)

      batch_results <- future_map(batch_vec, function(t) {
        clear_session()
        model_file <- file.path(MODELS_DIR, sprintf("model_t_%d.keras", t))
        accuracy <- NULL

        # Ensure the model is valid and retrain if necessary
        if (file.exists(model_file)) {
          tryCatch(
            {
              # Attempt to load and predict
              model <- load_model(model_file)
              input_sequences <- array(TestSet[1:(t - 1), ], dim = c(num_samples, t - 1, 1))
              predictions <- predict(model, input_sequences)
              predicted_classes <- ifelse(predictions > 0.5, 1, 0)
              actual_classes <- TestSet[t, ]
              accuracy <- as.numeric(predicted_classes == actual_classes)
              rm(model, input_sequences, predictions, predicted_classes, actual_classes)
              gc()
            },
            error = function(e) {
              message(sprintf("Error predicting for mutation %d: %s. Retrying with retraining...", t, e$message))
              accuracy <- NULL # Signal retraining is needed
            }
          )
        } else {
          message(sprintf("Model missing for mutation %d. Retraining...", t))
        }

        # If accuracy is NULL, retrain and retry predictions
        if (is.null(accuracy)) {
          tryCatch(
            {
              training_labels <- TrainingSet[t, ]
              TrainLSTM(t, TrainingSet, training_labels) # Retrain model
              model <- load_model(model_file) # Load the retrained model
              input_sequences <- array(TestSet[1:(t - 1), ], dim = c(num_samples, t - 1, 1))
              predictions <- predict(model, input_sequences)
              predicted_classes <- ifelse(predictions > 0.5, 1, 0)
              actual_classes <- TestSet[t, ]
              accuracy <- as.numeric(predicted_classes == actual_classes)
              rm(model, input_sequences, predictions, predicted_classes, actual_classes)
              gc()
            },
            error = function(e) {
              message(sprintf("Failed to retrain or predict for mutation %d: %s", t, e$message))
              accuracy <- rep(NA, num_samples) # Return NA for this mutation
            }
          )
        }

        return(accuracy)
      }, .options = furrr_options(seed = TRUE, globals = TRUE))

      # Update accuracy matrix with batch results
      for (i in seq_along(batch_vec)) {
        prediction_accuracy[batch_vec[i], ] <- batch_results[[i]]
      }

      p()
      plan(sequential) # Release workers between batches
      gc()
    }
  })

  return(prediction_accuracy)
}

# ---------------------
# Main Execution
# ---------------------
apply_models_to_test_set_batch <- function(retrain = FALSE) {
  # num_mutations <- 1000  # for testing
  num_mutations <- nrow(TrainingSet)

  # Train all models in parallel batches
  train_all_models_in_batches(TrainingSet, num_mutations, retrain, batch_size = 8)

  # Compute accuracy matrix
  accuracy_matrix <- compute_accuracy_matrix_parallel(num_mutations, TrainingSet, TestSet, batch_size = 8)
  saveRDS(accuracy_matrix, file.path(BASE_DIR, "accuracy_matrix.rds"))
  return(accuracy_matrix)
}

# ---------------------
# Run and Generate Plots
# ---------------------
accuracy_matrix <- apply_models_to_test_set_batch(retrain = TRUE)

png(file.path(BASE_DIR, "accuracy_heatmap.png"), width = 1200, height = 12000, res = 180)
pheatmap(
  accuracy_matrix,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  color = c("red", "green"),
  legend_breaks = c(0, 1),
  legend_labels = c("Incorrect", "Correct"),
  main = "Prediction Accuracy Heatmap"
)
dev.off()

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
      predicted <- accuracy_matrix[t, s] # 1 = Correct, 0 = Incorrect
      actual <- TestSet[t, s] # Ground truth from test set

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
  confusion_matrix <- matrix(c(TP, FN, FP, TN),
    nrow = 2, byrow = TRUE,
    dimnames = list(
      "Actual" = c("Positive", "Negative"),
      "Predicted" = c("Positive", "Negative")
    )
  )

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
  P <- TP + FN # Total actual positives
  N <- TN + FP # Total actual negatives
  total <- P + N # Total population

  # Metrics calculations
  TPR <- TP / P # True Positive Rate (Recall/Sensitivity)
  FNR <- FN / P # False Negative Rate
  FPR <- FP / N # False Positive Rate
  TNR <- TN / N # True Negative Rate (Specificity)

  PPV <- TP / (TP + FP) # Positive Predictive Value (Precision)
  FOR <- FN / (FN + TN) # False Omission Rate
  FDR <- FP / (TP + FP) # False Discovery Rate
  NPV <- TN / (TN + FN) # Negative Predictive Value

  ACC <- (TP + TN) / total # Accuracy
  BA <- (TPR + TNR) / 2 # Balanced Accuracy

  F1 <- 2 * (PPV * TPR) / (PPV + TPR) # F1 Score
  # MCC <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # Matthews Correlation Coefficient
  MCC <- sqrt(TPR * TNR * PPV * NPV) - sqrt(FNR * FPR * FOR * FDR)
  TS <- TP / (TP + FN + FP) # Threat Score (CSI/Jaccard Index)

  LR_plus <- TPR / FPR # Positive Likelihood Ratio
  LR_minus <- FNR / TNR # Negative Likelihood Ratio

  # Diagnostic Odds Ratio (DOR)
  DOR <- LR_plus / LR_minus

  # Compile into a table
  metrics_table <- data.frame(
    Metric = c(
      "True Positive Rate (Sensitivity, Recall)",
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
      "Diagnostic Odds Ratio"
    ),
    Value = c(TPR, FNR, TNR, FPR, PPV, NPV, FOR, FDR, ACC, BA, F1, MCC, TS, LR_plus, LR_minus, DOR)
  )

  return(metrics_table)
}

# Generate confusion matrix
confusion_matrix <- generate_confusion_matrix(accuracy_matrix, TestSet)

metrics_table <- generate_metrics_table(confusion_matrix)
print(metrics_table)

write.csv(confusion_matrix, file.path(BASE_DIR, "confusion_matrix.csv"), row.names = TRUE)
write.csv(metrics_table, file.path(BASE_DIR, "metrics_table.csv"), row.names = FALSE)

summary_path <- file.path(BASE_DIR, "run_summary.txt")
con <- file(summary_path, "wt")
writeLines(c(
  sprintf("Run Summary for TAG: %s", TAG),
  sprintf("Timestamp: %s", Sys.time()),
  "",
  "=== Graph Meta (from longest_path_genes) ==="
), con)

# If meta exists from graph step
if (!is.null(lp_meta)) {
  for (nm in names(lp_meta)) {
    writeLines(sprintf("%s: %s", nm, lp_meta[[nm]]), con)
  }
  if (!is.null(lp_meta$cycle_break_method)) {
    writeLines(sprintf("Cycle breaking method: %s", lp_meta$cycle_break_method), con)
  }
} else {
  writeLines("No meta info found in longest_path_genes RDS.", con)
}

writeLines(c(
  "",
  "=== LSTM Evaluation ===",
  sprintf("Num mutations (longest path): %d", nrow(TrainingSet)),
  sprintf("Num training samples: %d", ncol(TrainingSet)),
  sprintf("Num test samples: %d", ncol(TestSet)),
  "",
  "Confusion Matrix:"
), con)

capture.output(print(confusion_matrix), file = con)

writeLines(c("", "Metrics Table:"), con)
capture.output(print(metrics_table), file = con)

close(con)

message(sprintf("Saved run summary to %s", summary_path))

# ====== RUN SUMMARY (JSON sidecar) ======
# Ensure jsonlite is available (lightweight; only used here)
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}

# Helper: encode a matrix (with dimnames) as a JSON-friendly list
matrix_to_list <- function(mat) {
  list(
    dim = dim(mat),
    dimnames = dimnames(mat),
    data_byrow = split(as.vector(t(mat)), rep(seq_len(nrow(mat)), each = ncol(mat)))
  )
}

json_payload <- list(
  tag = TAG,
  timestamp = as.character(Sys.time()),
  graph_meta = if (!is.null(lp_meta)) lp_meta else NULL,
  cycle_break_method = if (!is.null(lp_meta$cycle_break_method)) lp_meta$cycle_break_method else NA,
  lstm = list(
    num_mutations = nrow(TrainingSet),
    num_train_samples = ncol(TrainingSet),
    num_test_samples = ncol(TestSet)
  ),
  results = list(
    confusion_matrix = matrix_to_list(confusion_matrix),
    metrics_table = list(
      columns = colnames(metrics_table),
      rows = lapply(seq_len(nrow(metrics_table)), function(i) as.list(metrics_table[i, , drop = FALSE]))
    )
  )
)

jsonlite::write_json(
  json_payload,
  path = file.path(BASE_DIR, "run_summary.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  na = "null",
  digits = NA
)

message(sprintf("Saved JSON sidecar to %s", file.path(BASE_DIR, "run_summary.json")))
