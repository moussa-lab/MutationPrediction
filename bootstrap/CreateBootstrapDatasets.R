# Authors:
#   Ekene Okeke, Department of Computer Science, Georgia State University, Atlanta, GA, USA
#   Alexander Zelikovsky, Department of Computer Science, Georgia State University, Atlanta, GA, USA 

# Create 200 Bootstrap Datasets from bootstrap/coad2(in).csv
library(tidyverse)


# 1. PARAMETERS & DIRECTORY SETUP

input_path <- file.path("bootstrap", "coad2(in).csv")
bootstrap_dir <- file.path("bootstrap", "bootstrap_datasets")

n_bootstraps <- 200
random_seed <- 42

# Ensure output directory exists
if (!dir.exists(bootstrap_dir)) {
  dir.create(bootstrap_dir, recursive = TRUE)
}


# 2. DATA LOADING

# Read the dataset 
data <- read_csv(input_path, show_col_types = FALSE)

# Preserve the gene IDs/names from the first column
gene_ids <- data[[1]]

# Define dimensions based on the data matrix
# Exclude the first column (gene IDs) to get actual sample columns
n_samples <- ncol(data) - 1
n_genes <- nrow(data)

cat(sprintf("Number of genes (rows): %d\n", n_genes))
cat(sprintf("Number of samples (columns): %d\n", n_samples))


# 3.  BOOTSTRAP RESAMPLING



cat(sprintf("Executing %d bootstrap iterations...\n", n_bootstraps))

for (b in 1:n_bootstraps) {
  # Set a reproducible seed for each iteration matching the Python framework
  set.seed(random_seed + b)
  
  # Resample the sample columns with replacement (columns 2 through end)
  # Column 1 (Gene IDs) stays fixed in place
  sampled_indices <- sample(2:ncol(data), size = n_samples, replace = TRUE)
  sampled_df <- data[, c(1, sampled_indices)]
  
  # Save tracking dataset slice
  dataset_path <- file.path(bootstrap_dir, sprintf("dataset_%d.csv", b))
  write_csv(sampled_df, dataset_path)
  
}

cat("Pipeline completed successfully.\n")
