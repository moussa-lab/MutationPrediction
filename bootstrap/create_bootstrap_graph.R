# Authors:
#   Ekene Okeke, Department of Computer Science, Georgia State University, Atlanta, GA, USA
#   Alexander Zelikovsky, Department of Computer Science, Georgia State University, Atlanta, GA, USA 

# PIPELINE PHASE: Graph Consensus Edge List Generation 
# ==============================================================================

library(R.matlab)
library(R.matlab)
library(pheatmap)
library(parallel)
library(data.table)
library(future)
library(progressr)
library(igraph) 
library(dplyr)
library(utils)  
library(pbapply)       
library(ggplot2) 

handlers(global = TRUE)
handlers(list(handler_txtprogressbar()))

# ==========================================
# 1.CONFIGURATION & PARAMETERS
# ==========================================

# Relative Base Paths (run from repository root)
BOOTSTRAP_DIR       <- "bootstrap"
DATA_DIR            <- file.path(BOOTSTRAP_DIR, "data")
STUDY_DATA_DIR      <- file.path(BOOTSTRAP_DIR, "bootstrap_datasets")
EDGE_OUTPUT_DIR     <- file.path(DATA_DIR, "edges")
DEBUG_OUTPUT_DIR    <- file.path(DATA_DIR, "conditional_probability")

# Input File Specifications
FILE_PREFIX         <- "dataset_" 
FILE_SUFFIX         <- ".csv"
TEMP_FILE_SUFFIX    <- "_edges.csv" 

# Thresholds & Constraints
N_DATASETS          <- 200 
PROB_THRESHOLD      <- 0.05  
DAG_FREQUENCY_THRESHOLD <- 103 

# Batch Settings for Memory Management
BATCH_1_RANGE       <- 1:40
BATCH_2_RANGE       <- 41:80
BATCH_3_RANGE       <- 81:120
BATCH_4_RANGE       <- 121:150
BATCH_5_RANGE       <- 151:170
BATCH_6_RANGE       <- 171:181
BATCH_7_RANGE       <- 182:200

# Batch Checkpoint Paths
BATCH_1_CONSENSUS_PATH <- file.path(DATA_DIR, "batch1_consensus_edges.csv")
BATCH_2_CONSENSUS_PATH <- file.path(DATA_DIR, "batch2_consensus_edges.csv")
BATCH_3_CONSENSUS_PATH <- file.path(DATA_DIR, "batch3_consensus_edges.csv")
BATCH_4_CONSENSUS_PATH <- file.path(DATA_DIR, "batch4_consensus_edges.csv")
BATCH_5_CONSENSUS_PATH <- file.path(DATA_DIR, "batch5_consensus_edges.csv")
BATCH_6_CONSENSUS_PATH <- file.path(DATA_DIR, "batch6_consensus_edges.csv")
BATCH_7_CONSENSUS_PATH <- file.path(DATA_DIR, "batch7_consensus_edges.csv")

CHECKPOINT_PATHS       <- c(BATCH_1_CONSENSUS_PATH, BATCH_2_CONSENSUS_PATH, 
                            BATCH_3_CONSENSUS_PATH, BATCH_4_CONSENSUS_PATH)

# Output Paths
INTERMEDIATE_CONSENSUS_PATH <- file.path(DATA_DIR, "Beforecycles_removed.csv")
FINAL_OUTPUT_PATH           <- file.path(DATA_DIR, "final_consensus_dag.csv")

# --- Parallel Settings ---
N_CORES             <- 1

#-------------------------------------------------
# 2. CORE ALGORITHMIC FUNCTIONS

build_conditional_prob_matrix <- function(gene_matrix) {
  G <- gene_matrix
  G_not <- 1 - G
  n_genes <- nrow(G)
  
  # Matrix multiplication to get all pairwise counts
  c11 <- G %*% t(G)         # both are 1
  c10 <- G %*% t(G_not)     # g1 is 1, g2 is 0
  c01 <- G_not %*% t(G)     # g1 is 0, g2 is 1
  
  # Initialize result matrix
  P <- matrix(NA_real_, nrow = n_genes, ncol = n_genes)
  
  # Compute conditional probabilities:
  # P[i,j] = P(gene i = 1 | gene j = 1) = c11[i,j] / (c11[i,j] + c01[i,j])
  denom_ij = c11 + c01
  P_ij <- ifelse(denom_ij > 0, c11 / denom_ij, NA)
  
  # P[j,i] = P(gene j = 1 | gene i = 1) = c11[i,j] / (c11[i,j] + c10[i,j])
  denom_ji = c11 + c10
  P_ji <- ifelse(denom_ji > 0, c11 / denom_ji, NA)
  
  # Fill P matrix: P[i,j] from P_ij, P[j,i] from P_ji
  upper_idx <- upper.tri(P)
  lower_idx <- lower.tri(P)
  
  P[upper_idx] <- P_ij[upper_idx]
  P[lower_idx] <- t(P_ji)[lower_idx]
  
  diag(P) <- NA  
  
  return(P)
}

build_asymmetric_graph <- function(W) {
  stopifnot(is.matrix(W), nrow(W) == ncol(W))
  
  node_names <- rownames(W)
  if (is.null(node_names)) {
    node_names <- as.character(seq_len(nrow(W)))
  }
  
  # Get all off-diagonal indices (i, j)
  idx <- which(row(W) != col(W), arr.ind = TRUE)
  
  # W[i, j] = P(j|i) (effect|cause)
  wij <- W[idx]
  
  # W[j, i] = P(i|j) (cause|effect)
  wji <- W[cbind(idx[, 2], idx[, 1])]
  
  # Calculate the edge weight (absolute difference)
  diffs <- abs(wij - wji)
  
  # Identify edges where P(j|i) > P(i|j) -> direction is i -> j
  mask <- wij > wji
  
  # Correctly map indices for i -> j direction
  from_idx <- idx[mask, 1] # i (The cause)
  to_idx <- idx[mask, 2]   # j (The effect)
  
  weights <- diffs[mask]
  
  from_names <- as.character(node_names[from_idx])
  to_names <- as.character(node_names[to_idx])
  
  edges_dt <- data.table::data.table(     #edges instead of graph for memory
    from = from_names,  
    to = to_names,                        #I saved the graph ;later 
    weight = weights
  )
  
  edges_dt <- edges_dt[complete.cases(edges_dt), ]
  
  return(edges_dt) # Return data.table, not igraph object
}

cleanup_temp_files <- function(batch_range, file_prefix, temp_file_suffix, edge_output_dir) {
  temp_files <- file.path(edge_output_dir, paste0(file_prefix, batch_range, temp_file_suffix))
  existing_files <- temp_files[file.exists(temp_files)]
  if (length(existing_files) > 0) {
    unlink(existing_files)
  }
  invisible(existing_files)
}

process_single_dataset <- function(i) {
  file_name        <- paste0(FILE_PREFIX, i, FILE_SUFFIX)
  current_data_path <- file.path(STUDY_DATA_DIR, file_name)
  
  cat(paste("Processing dataset:", file_name, "\n"))
  
  output_file_name <- paste0(FILE_PREFIX, i, TEMP_FILE_SUFFIX)
  output_file_path <- file.path(EDGE_OUTPUT_DIR, output_file_name)
  
  data_dt              <- data.table::fread(current_data_path, header = FALSE)
  gene_names_from_file <- as.character(data_dt[[1]][-1])
  data_cols            <- data_dt[-1, -1, with = FALSE] 
  
  data_int_list <- lapply(data_cols, as.integer)
  gene_matrix   <- do.call(cbind, data_int_list)
  gene_matrix[is.na(gene_matrix)] <- 0
  rownames(gene_matrix)           <- gene_names_from_file
  
  P_g1_given_g2           <- build_conditional_prob_matrix(gene_matrix)
  rownames(P_g1_given_g2) <- gene_names_from_file
  colnames(P_g1_given_g2) <- gene_names_from_file
  
  P_g1_given_g2[P_g1_given_g2 < PROB_THRESHOLD] <- 0
  edges_dt <- build_asymmetric_graph(P_g1_given_g2)
  num_edges <- nrow(edges_dt)
  
  data.table::fwrite(edges_dt, output_file_path, row.names = FALSE)
  cat(paste0("File saved: ", output_file_path, " (", num_edges, " edges)\n"))
  
  rm(data_dt, gene_matrix, P_g1_given_g2, edges_dt, data_cols, data_int_list)
  gc()
  
  return(output_file_path) 
}

aggregate_and_cleanup_batch <- function(batch_range, output_path, edge_output_dir, file_prefix, temp_file_suffix) {
  cat(paste0("\n--- Starting ITERATIVE aggregation for batch ", min(batch_range), " - ", max(batch_range), " ---\n"))
  
  expected_files <- paste0(file_prefix, batch_range, temp_file_suffix)
  temp_files     <- file.path(edge_output_dir, expected_files)
  existing_files <- temp_files[file.exists(temp_files)]
  
  if (length(existing_files) == 0) {
    cat("Warning: No temporary files found for this batch. Skipping aggregation.\n")
    return(NULL)
  }
  
  master_aggregate_dt <- data.table::data.table(from = character(), to = character(), frequency = integer(), total_weight_sum = numeric())
  
  for (file_path in existing_files) {
    new_edges_dt <- data.table::fread(file_path)
    if (nrow(new_edges_dt) == 0) {
      rm(new_edges_dt)
      next
    }
    
    new_aggregate_dt <- new_edges_dt[, list(
      frequency        = .N, 
      total_weight_sum = sum(weight)
    ), by = list(from, to)]
    
    master_aggregate_dt <- data.table::rbindlist(list(master_aggregate_dt, new_aggregate_dt))
    master_aggregate_dt <- master_aggregate_dt[, list(
      frequency        = sum(frequency),
      total_weight_sum = sum(total_weight_sum)
    ), by = list(from, to)]
    
    rm(new_edges_dt, new_aggregate_dt)
    gc() 
  }
  
  if (nrow(master_aggregate_dt) > 0) {
    master_aggregate_dt[, mean_weight := total_weight_sum / frequency]
    consensus_edges_dt <- master_aggregate_dt[, list(from, to, mean_weight, frequency)]
  } else {
    consensus_edges_dt <- data.table::data.table(from = character(), to = character(), mean_weight = numeric(), frequency = integer())
  }
  
  data.table::fwrite(consensus_edges_dt, output_path, row.names = FALSE)
  cat(paste("Batch consensus saved to:", output_path, " (", nrow(consensus_edges_dt), " unique edges)\n"))
  
  cleanup_temp_files(batch_range, file_prefix, temp_file_suffix, edge_output_dir)
  rm(master_aggregate_dt, consensus_edges_dt)
  gc()
  
  return(output_path)
}

initialize_master_consensus <- function(first_checkpoint_path, master_path) {
  if (file.exists(first_checkpoint_path)) {
    file.copy(first_checkpoint_path, master_path, overwrite = TRUE)
    cat(paste("Master consensus initialized to:", master_path, " (Copied from", basename(first_checkpoint_path), ")\n"))
    return(TRUE)
  } else {
    stop("Error: First checkpoint file not found. Cannot initialize master consensus.")
  }
}

update_master_consensus_iterative <- function(master_path, new_checkpoint_path) {
  cat(paste0(" -> Combining: ", basename(new_checkpoint_path), " into master...\n"))
  
  master_aggregate_dt <- data.table::fread(master_path)
  master_aggregate_dt[, total_weight_sum := frequency * mean_weight]
  
  dt_to_combine <- data.table::fread(new_checkpoint_path)
  dt_to_combine[, total_weight_sum := frequency * mean_weight]
  
  temp_combined_dt            <- data.table::rbindlist(list(master_aggregate_dt, dt_to_combine))
  master_aggregate_dt_updated <- temp_combined_dt[, list(
    frequency        = sum(frequency),
    total_weight_sum = sum(total_weight_sum)
  ), by = list(from, to)]
  
  master_aggregate_dt_updated[, mean_weight := total_weight_sum / frequency]
  final_dt <- master_aggregate_dt_updated[, list(from, to, mean_weight, frequency)]
  
  data.table::fwrite(final_dt, master_path, row.names = FALSE)
  rm(master_aggregate_dt, dt_to_combine, temp_combined_dt, master_aggregate_dt_updated, final_dt)
  gc()
  
  return(nrow(data.table::fread(master_path))) 
}

# 3. RUNTIME PIPELINE EXECUTION
# ==============================================================================
# Ensure directory boundaries exist locally
if (!dir.exists(EDGE_OUTPUT_DIR))  { dir.create(EDGE_OUTPUT_DIR, recursive = TRUE) }
if (!dir.exists(DEBUG_OUTPUT_DIR)) { dir.create(DEBUG_OUTPUT_DIR, recursive = TRUE) }

cat(paste0("\nStarting sequential processing for ", N_DATASETS, " datasets in 7 sub-batches.\n"))

# --- BATCH 1 ---
cat("\n--- Running Batch 1/7 (Datasets 1 - 40) ---\n")
pb <- txtProgressBar(min = 1, max = length(BATCH_1_RANGE), style = 3)
for (i_index in seq_along(BATCH_1_RANGE)) {
  process_single_dataset(BATCH_1_RANGE[i_index])
  setTxtProgressBar(pb, i_index) 
}
close(pb)
aggregate_and_cleanup_batch(BATCH_1_RANGE, BATCH_1_CONSENSUS_PATH, EDGE_OUTPUT_DIR, FILE_PREFIX, TEMP_FILE_SUFFIX)

# --- BATCH 2 ---
cat("\n--- Running Batch 2/7 (Datasets 41 - 80) ---\n")
pb <- txtProgressBar(min = 1, max = length(BATCH_2_RANGE), style = 3)
for (i_index in seq_along(BATCH_2_RANGE)) {
  process_single_dataset(BATCH_2_RANGE[i_index])
  setTxtProgressBar(pb, i_index) 
}
close(pb)
aggregate_and_cleanup_batch(BATCH_2_RANGE, BATCH_2_CONSENSUS_PATH, EDGE_OUTPUT_DIR, FILE_PREFIX, TEMP_FILE_SUFFIX)

# --- BATCH 3 ---
cat("\n--- Running Batch 3/7 (Datasets 81 - 120) ---\n")
pb <- txtProgressBar(min = 1, max = length(BATCH_3_RANGE), style = 3)
for (i_index in seq_along(BATCH_3_RANGE)) {
  process_single_dataset(BATCH_3_RANGE[i_index])
  setTxtProgressBar(pb, i_index) 
}
close(pb)
aggregate_and_cleanup_batch(BATCH_3_RANGE, BATCH_3_CONSENSUS_PATH, EDGE_OUTPUT_DIR, FILE_PREFIX, TEMP_FILE_SUFFIX)

# --- BATCH 4 ---
cat("\n--- Running Batch 4/7 (Datasets 121 - 150) ---\n")
pb <- txtProgressBar(min = 1, max = length(BATCH_4_RANGE), style = 3)
for (i_index in seq_along(BATCH_4_RANGE)) {
  process_single_dataset(BATCH_4_RANGE[i_index])
  setTxtProgressBar(pb, i_index) 
}
close(pb)
aggregate_and_cleanup_batch(BATCH_4_RANGE, BATCH_4_CONSENSUS_PATH, EDGE_OUTPUT_DIR, FILE_PREFIX, TEMP_FILE_SUFFIX)

# --- BATCH 5 ---
cat("\n--- Running Batch 5/7 (Datasets 151 - 170) ---\n")
pb <- txtProgressBar(min = 1, max = length(BATCH_5_RANGE), style = 3)
for (i_index in seq_along(BATCH_5_RANGE)) {
  process_single_dataset(BATCH_5_RANGE[i_index])
  setTxtProgressBar(pb, i_index) 
}
close(pb)
aggregate_and_cleanup_batch(BATCH_5_RANGE, BATCH_5_CONSENSUS_PATH, EDGE_OUTPUT_DIR, FILE_PREFIX, TEMP_FILE_SUFFIX) 

# --- BATCH 6 ---
cat("\n--- Running Batch 6/7 (Datasets 171 - 181) ---\n")
pb <- txtProgressBar(min = 1, max = length(BATCH_6_RANGE), style = 3)
for (i_index in seq_along(BATCH_6_RANGE)) {
  process_single_dataset(BATCH_6_RANGE[i_index])
  setTxtProgressBar(pb, i_index) 
}
close(pb)
aggregate_and_cleanup_batch(BATCH_6_RANGE, BATCH_6_CONSENSUS_PATH, EDGE_OUTPUT_DIR, FILE_PREFIX, TEMP_FILE_SUFFIX)

# --- BATCH 7 ---
cat("\n--- Running Batch 7/7 (Datasets 182 - 200) ---\n")
pb <- txtProgressBar(min = 1, max = length(BATCH_7_RANGE), style = 3)
for (i_index in seq_along(BATCH_7_RANGE)) {
  process_single_dataset(BATCH_7_RANGE[i_index])
  setTxtProgressBar(pb, i_index) 
}
close(pb)
aggregate_and_cleanup_batch(BATCH_7_RANGE, BATCH_7_CONSENSUS_PATH, EDGE_OUTPUT_DIR, FILE_PREFIX, TEMP_FILE_SUFFIX)

# 4. INTER-BATCH CONSOLIDATION
# ==============================================================================
checkpoint_paths_list <- c(
  BATCH_1_CONSENSUS_PATH, BATCH_2_CONSENSUS_PATH, BATCH_3_CONSENSUS_PATH,
  BATCH_4_CONSENSUS_PATH, BATCH_5_CONSENSUS_PATH, BATCH_6_CONSENSUS_PATH,
  BATCH_7_CONSENSUS_PATH
)

cat("\nCompiling global structural consensus across batch files...\n")
initialize_master_consensus(checkpoint_paths_list[1], INTERMEDIATE_CONSENSUS_PATH)

pb <- txtProgressBar(min = 2, max = length(checkpoint_paths_list), style = 3)
for (i in 2:length(checkpoint_paths_list)) {
  new_count <- update_master_consensus_iterative(
    master_path         = INTERMEDIATE_CONSENSUS_PATH,
    new_checkpoint_path = checkpoint_paths_list[i]
  )
  setTxtProgressBar(pb, i)
}
close(pb)

# Instantiating final igraph structural definitions
final_aggregate_dt <- data.table::fread(INTERMEDIATE_CONSENSUS_PATH)
edge_list_matrix1  <- as.matrix(final_aggregate_dt[, list(from, to)])
temp_g1            <- igraph::graph_from_edgelist(el = edge_list_matrix1, directed = TRUE)

cat("Pipeline completed successfully. Network file saved to Beforecycles_removed.csv.\n")
