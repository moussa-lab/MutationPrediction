# Authors:
#   Ekene Okeke, Department of Computer Science, Georgia State University, Atlanta, GA, USA
#   Alexander Zelikovsky, Department of Computer Science, Georgia State University, Atlanta, GA, USA 

# ==============================================================================
# PIPELINE PHASE: Binary Search Optimization for T_min Structural Discovery
# ==============================================================================

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


# 1. CONFIGURATION & PARAMETERS
# ==============================================================================

BOOTSTRAP_DIR               <- "bootstrap"
DATA_DIR                    <- file.path(BOOTSTRAP_DIR, "data")
INTERMEDIATE_CONSENSUS_PATH <- file.path(DATA_DIR, "Beforecycles_removed.csv")
FINAL_RDS_PATH              <- file.path(DATA_DIR, "t_min_dag_graph.rds")
TOPO_ORDER_PATH             <- file.path(DATA_DIR, "topological_order_103.txt")
LONGEST_PATH_PATH           <- file.path(DATA_DIR, "longest_path_unweighted_103.txt")
WEIGHTED_PATH_PATH          <- file.path(DATA_DIR, "longest_path_weighted_103.txt")


# 2. CORE ALGORITHMIC FUNCTIONS
# ==============================================================================

find_t_min_dag <- function(data) {
  
  # 1. Preprocessing and Search Space definition
  all_freqs    <- sort(unique(data$frequency))
  search_space <- c(0, all_freqs) 
  
  low_idx  <- 1
  high_idx <- length(search_space)
  t_min    <- search_space[high_idx] 
  
  message("--- Starting Binary Search for T_min (Directed Acyclic Graph) ---")
  message(paste("Total unique thresholds to check:", length(search_space)))
  
  # 2. Binary Search Optimization Loop
  while (low_idx <= high_idx) {
    mid_idx <- low_idx + floor((high_idx - low_idx) / 2)
    T       <- search_space[mid_idx] 
    
    # Filter using optimized data.table syntax
    filtered_data <- data[frequency >= T, ]
    
    # Evaluate cycle presence in the remaining subset edge list
    if (nrow(filtered_data) == 0) {
      is_dag <- TRUE 
      g      <- igraph::make_empty_graph(n = 0, directed = TRUE)
    } else {
      g      <- igraph::graph_from_data_frame(d = filtered_data, directed = TRUE)
      is_dag <- igraph::is_dag(g) 
    }
    
    cat(sprintf("Testing T = %-5d | Edges Remaining = %-5d | Is DAG: %s\n", 
                T, igraph::ecount(g), is_dag))
    
    if (is_dag) {
      t_min    <- T
      high_idx <- mid_idx - 1 # Tighten the filter to locate minimal threshold boundary
    } else {
      low_idx  <- mid_idx + 1 # Loops back to elevate the threshold requirement
    }
  }
  
  # 3. Final Result Compilation
  message("\n--- Analysis Complete ---")
  final_data_edges <- data[frequency >= t_min, ]
  
  if (nrow(final_data_edges) == 0) {
    final_dag_graph <- igraph::make_empty_graph(n = 0, directed = TRUE)
  } else {
    final_dag_graph <- igraph::graph_from_data_frame(d = final_data_edges, directed = TRUE)
  }
  
  message("\n--- Final DAG Structure Metrics (Edges >= T_min) ---")
  message(paste0("  T_min (Threshold): ", t_min))
  message(paste0("  Vertices (Nodes):  ", igraph::vcount(final_dag_graph)))
  message(paste0("  Edges Remaining:   ", igraph::ecount(final_dag_graph)))
  message("-------------------------------------------------")
  stopifnot(igraph::is_dag(final_dag_graph))
  
  return(list(
    T_min          = t_min,
    dag_graph      = final_dag_graph,
    final_edges_df = final_data_edges
  ))
}

longest_path_dag <- function(g, weight_attr = NULL) {
  stopifnot(igraph::is_dag(g))

  topo <- as.integer(igraph::topo_sort(g, mode = "out"))
  n <- igraph::vcount(g)
  dist <- rep(-Inf, n)
  pred <- rep(NA_integer_, n)

  for (v in topo) {
    if (!is.finite(dist[v])) {
      dist[v] <- 0
    }

    out_edges <- igraph::incident(g, v, mode = "out")
    if (length(out_edges) == 0) {
      next
    }

    for (edge_id in as.integer(out_edges)) {
      target <- as.integer(igraph::head_of(g, edge_id))
      edge_weight <- if (is.null(weight_attr)) {
        1
      } else {
        as.numeric(igraph::edge_attr(g, weight_attr, index = edge_id))
      }
      if (!is.finite(edge_weight)) {
        edge_weight <- 0
      }

      candidate <- dist[v] + edge_weight
      if (candidate > dist[target]) {
        dist[target] <- candidate
        pred[target] <- v
      }
    }
  }

  end_node <- which.max(dist)
  path <- integer()
  while (!is.na(end_node)) {
    path <- c(end_node, path)
    end_node <- pred[end_node]
  }

  igraph::V(g)$name[path]
}


# 3. DATA IMPORT & WORKSPACE INITIALIZATION
# ==============================================================================

if (!file.exists(INTERMEDIATE_CONSENSUS_PATH)) {
  stop(paste("Error: Source data file not found at", INTERMEDIATE_CONSENSUS_PATH))
}

final_aggregate_dt <- data.table::fread(INTERMEDIATE_CONSENSUS_PATH)

# Instantiating the baseline unweighted graph tracking topology
edge_list_matrix1 <- as.matrix(final_aggregate_dt[, list(from, to)])
temp_g1           <- igraph::graph_from_edgelist(el = edge_list_matrix1, directed = TRUE)
MAX_FREQUENCY     <- max(final_aggregate_dt$frequency)


# 4. EXECUTION PIPELINE
# ==============================================================================

cat("\n--- EXECUTION: Finding T_min Boundary ---\n")
dag_results <- find_t_min_dag(final_aggregate_dt)

# Extract the calculated minimal consensus graph object
temp_g <- dag_results$dag_graph
edges_after_filter <- igraph::ecount(temp_g)


# 5. LONGEST PATH DETECTION & EXPORT
# ==============================================================================

if (edges_after_filter > 0) {
  longest_path <- longest_path_dag(temp_g)
  weighted_path <- longest_path_dag(temp_g, weight_attr = "mean_weight")

  cat("\nUnweighted longest path (", length(longest_path) - 1, " edges):\n", sep = "")
  print(longest_path)
  cat("\nWeighted longest path (", length(weighted_path) - 1, " edges):\n", sep = "")
  print(weighted_path)

  writeLines(longest_path, LONGEST_PATH_PATH)
  writeLines(weighted_path, WEIGHTED_PATH_PATH)
} else {
  cat("\nT_min search resulted in an empty edge list topology; longest path skipped.\n")
}

topo_order <- igraph::topo_sort(temp_g, mode = "out")
writeLines(V(temp_g)[topo_order]$name, TOPO_ORDER_PATH)

# Serialize model graph to local storage
saveRDS(temp_g, FINAL_RDS_PATH)
cat(paste0("\nSerialized pipeline graph successfully saved to: ", FINAL_RDS_PATH, "\n"))
cat(paste0("Topological order saved to: ", TOPO_ORDER_PATH, "\n"))
cat(paste0("Unweighted longest path saved to: ", LONGEST_PATH_PATH, "\n"))
cat(paste0("Weighted longest path saved to: ", WEIGHTED_PATH_PATH, "\n"))
