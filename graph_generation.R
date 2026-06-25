# Authors:
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA

# Graph generation (COAD / DFCI / MGI / CSV)
## - COAD (from Auslander et al. github repo) uses (data, genes, samples) = (2,1,3)
## - DFCI, MGI (from Auslander et al. github repo) use (3,2,1)
## - Modes: "coad", "coad_dfci_mgi", "csv"

# CONFIG
USE_MODE <- "csv" # "coad" | "coad_dfci_mgi" | "csv"

# .mat inputs
COAD_MAT_PATH <- "study_data/COAD2.mat"
COAD_DS <- "COAD2"

DFCI_MAT_PATH <- "study_data/DFCI2.mat"
DFCI_DS <- "DFCI2"

MGI_MAT_PATH <- "study_data/MGI2.mat"
MGI_DS <- "MGI2"

# CSV input (already genes x samples)
# This is our dataset created as described in the paper
CSV_PATH <- "study_data/combined_union.csv"

# Graph params
BLOCK_SIZE <- NA_integer_ # if NA, auto = ceiling(n_genes/6)
TAG_COMMENT <- "[[]]" # used to add whatever bit to the tag you want for the output files
THRESHOLD_DIFF <- 0.02 # weight threshold on |P(i|j)-P(j|i)|
MIN_COOCC <- 1 # min c11 co-occurrences
PROB_MIN <- 0.02 # floor: values below are set to 0
WEIGHT_MIN <- THRESHOLD_DIFF
DROP_MIN_ONES <- 0 # drop genes with <= this many 1s
USE_SUBSAMPLE <- FALSE # subsampling was used for testing during iterative part of development
SUBSAMPLE_SIZE <- 1000
CYCLE_BREAK_METHOD <- "feedback_arc_set" # "none" | "feedback_arc_set" | "feedback_vertex_set"
# feedback vertex set probably won't work. it's still experimental in igraph (you'll need to
# install the specific experimental build) and we didn't manage to make it work. best to keep
# it as feedback arc set.
PATH_WEIGHT_ATTR <- "w_diff" # edge attr used for weighted longest path

set.seed(123)

suppressPackageStartupMessages({
    library(R.matlab)
    library(Matrix)
    library(igraph)
})

# Helpers
fmt_num <- function(x, digits = 6) {
    s <- formatC(x, format = "f", digits = digits, drop0trailing = TRUE)
    s <- sub("\\.", "p", s)
    gsub("\\s+", "", s)
}
fmt_id <- function(s) {
    tolower(gsub("[^A-Za-z0-9]+", "-", s))
}

read_mat_dataset <- function(mat_path, ds_name, dgs) {
    # dgs = integer vector length 3 giving positions for (data, genes, samples) in the MATLAB struct
    md <- readMat(mat_path)
    ds <- md[[ds_name]]
    if (is.null(ds)) stop("Dataset name '", ds_name, "' not found in ", mat_path)

    dat <- as.matrix(ds[[dgs[1]]])
    genes <- as.character(Reduce(c, ds[[dgs[2]]]))
    sams <- as.character(Reduce(c, ds[[dgs[3]]]))
    rownames(dat) <- genes
    colnames(dat) <- sams
    storage.mode(dat) <- "numeric"
    dat[is.na(dat)] <- 0
}

combine_union_by_genes <- function(mats, prefixes = NULL) {
    # mats: list of matrices (genes x samples)
    # prefixes: optional vector of same length to prefix sample names
    stopifnot(length(mats) >= 1)
    all_genes <- unique(unlist(lapply(mats, rownames), use.names = FALSE))
    all_genes <- all_genes[!is.na(all_genes)]
    out_list <- vector("list", length(mats))

    for (i in seq_along(mats)) {
        M <- mats[[i]]
        # align to union of genes
        miss <- setdiff(all_genes, rownames(M))
        if (length(miss)) {
            Z <- Matrix(0,
                nrow = length(miss), ncol = ncol(M), sparse = FALSE,
                dimnames = list(miss, colnames(M))
            )
            M <- rbind(M, Z)
        }
        M <- M[all_genes, , drop = FALSE]
        # prefix sample names to avoid collisions
        if (!is.null(prefixes) && !is.na(prefixes[i])) {
            colnames(M) <- paste0(prefixes[i], "||", colnames(M))
        }
        out_list[[i]] <- M
    }
    do.call(cbind, out_list)
}

set_path_weights <- function(g, mode = PATH_WEIGHT_ATTR) {
    w <- switch(mode,
        w_diff = E(g)$w_diff,
        stop("unknown PATH_WEIGHT_ATTR: ", mode)
    )
    w[!is.finite(w)] <- 0
    E(g)$w_path <- pmax(w, 0) + 1e-12
    g
}

make_dag <- function(g) {
    if (is_dag(g)) {
        message("Graph is already a DAG; no cycles to break.")
        return(g)
    }
    if (CYCLE_BREAK_METHOD == "feedback_arc_set") {
        fas <- feedback_arc_set(g, weights = E(g)$w_diff)
        message(sprintf("Removing %d edges via feedback_arc_set to break cycles.", length(fas)))
        delete_edges(g, fas)
    } else if (CYCLE_BREAK_METHOD == "feedback_vertex_set") {
        fvs <- feedback_vertex_set(g)
        writeLines(fvs, "feedback_vertex_set.txt")
        message(sprintf("Removing %d vertices via feedback_vertex_set to break cycles.", length(fvs)))
        delete_vertices(g, fvs)
    } else if (CYCLE_BREAK_METHOD == "none") {
        message("CYCLE_BREAK_METHOD is 'none'; returning original graph (not a DAG).")
        g
    } else {
        stop("unknown CYCLE_BREAK_METHOD: ", CYCLE_BREAK_METHOD)
    }
}

longest_path_dag <- function(g) {
    stopifnot(igraph::is.dag(g))
    topo <- as.integer(topo_sort(g, mode = "out"))
    n <- vcount(g)
    pred <- rep(NA_integer_, n)
    dist <- rep(-Inf, n)
    for (v in topo) {
        if (!is.finite(dist[v])) dist[v] <- 0L
        nbrs <- as.integer(igraph::neighbors(g, v, mode = "out"))
        if (length(nbrs) == 0) next
        for (w in nbrs) {
            alt <- dist[v] + 1L
            if (dist[w] < alt) {
                dist[w] <- alt
                pred[w] <- v
            }
        }
    }
    j <- which.max(dist)
    if (!length(j) || !is.finite(dist[j])) {
        return(igraph::V(g)$name[0])
    }
    path <- integer()
    while (!is.na(j)) {
        path <- c(j, path)
        j <- pred[j]
    }
    igraph::V(g)$name[path]
}

longest_path_dag_weighted <- function(g, weight_attr = "w_path") {
    stopifnot(igraph::is.dag(g))
    topo <- as.integer(topo_sort(g, mode = "out"))
    n <- vcount(g)
    eids <- seq_len(ecount(g))
    src <- as.integer(tail_of(g, eids))
    dst <- as.integer(head_of(g, eids))
    w <- edge_attr(g, weight_attr)
    ord <- order(src)
    src <- src[ord]
    dst <- dst[ord]
    w <- w[ord]
    deg <- tabulate(src, nbins = n)
    first <- match(seq_len(n), src)
    pred <- rep(NA_integer_, n)
    dist <- rep(-Inf, n)
    for (v in topo) {
        if (!is.finite(dist[v])) dist[v] <- 0
        k0 <- first[v]
        if (is.na(k0)) next
        idx <- k0:(k0 + deg[v] - 1L)
        targets <- dst[idx]
        cand <- dist[v] + w[idx]
        improve <- cand > dist[targets]
        if (any(improve)) {
            tt <- targets[improve]
            dist[tt] <- cand[improve]
            pred[tt] <- v
        }
    }
    j <- which.max(dist)
    if (!length(j) || !is.finite(dist[j])) {
        return(igraph::V(g)$name[0])
    }
    path <- integer()
    while (!is.na(j)) {
        path <- c(j, path)
        j <- pred[j]
    }
    igraph::V(g)$name[path]
}

sum_path_weight <- function(g, path_names, weight_attr = "w_path") {
    if (length(path_names) < 2) {
        return(0)
    }
    sum(edge_attr(g, weight_attr, index = E(g, path = path_names)))
}

# Save topological order of DAG
save_topo <- function(g, tag) {
    stopifnot(is.dag(g))
    topo_idx <- as.integer(topo_sort(g, mode = "out"))
    topo_names <- V(g)$name[topo_idx]
    writeLines(topo_names, file.path(sprintf("outputs/%s", tag), sprintf("topo_order_%s.txt", tag)))
    saveRDS(topo_names, file.path(sprintf("outputs/%s", tag), sprintf("topo_order_%s.rds", tag)))
    topo_names
}

# Load data per mode
dataset_label <- switch(USE_MODE,
    "coad" = COAD_DS,
    "coad_dfci_mgi" = paste(COAD_DS, DFCI_DS, MGI_DS, sep = "+"),
    "csv" = "CSV",
    stop("Unknown USE_MODE: ", USE_MODE)
)

message("Mode: ", USE_MODE)
t0 <- Sys.time()

if (USE_MODE == "csv") {
    message("Loading CSV: ", CSV_PATH)
    gene_matrix <- as.matrix(read.csv(CSV_PATH, row.names = 1, check.names = FALSE))
    storage.mode(gene_matrix) <- "numeric"
    gene_matrix[is.na(gene_matrix)] <- 0
} else if (USE_MODE == "coad") {
    message("Loading COAD (2,1,3): ", COAD_MAT_PATH, " / ", COAD_DS)
    gene_matrix <- read_mat_dataset(COAD_MAT_PATH, COAD_DS, c(2, 1, 3))
} else if (USE_MODE == "coad_dfci_mgi") {
    message("Loading COAD (2,1,3), DFCI (3,2,1), MGI (3,2,1)")
    coad <- read_mat_dataset(COAD_MAT_PATH, COAD_DS, c(2, 1, 3))
    dfci <- read_mat_dataset(DFCI_MAT_PATH, DFCI_DS, c(3, 2, 1))
    mgi <- read_mat_dataset(MGI_MAT_PATH, MGI_DS, c(3, 2, 1))
    # union by genes, prefix sample names to keep unique
    gene_matrix <- combine_union_by_genes(
        list(coad, dfci, mgi),
        prefixes = c(COAD_DS, DFCI_DS, MGI_DS)
    )
} else {
    stop("Unhandled USE_MODE.")
}

if (USE_SUBSAMPLE) {
    gene_matrix <- gene_matrix[sample(1:nrow(gene_matrix), min(SUBSAMPLE_SIZE, nrow(gene_matrix)), replace = FALSE), , drop = FALSE]
}

# Build sparse logical matrix
M <- Matrix::Matrix(gene_matrix != 0, sparse = TRUE)
rm(gene_matrix)
gc()

n <- nrow(M)
m <- ncol(M)
message(sprintf("Matrix size: %d genes x %d samples (sparse)", n, m))

n1 <- Matrix::rowSums(M)
keep <- which(n1 > 0L & n1 > DROP_MIN_ONES)
if (length(keep) < n) {
    message(sprintf("Dropping %d genes with n1 <= %d.", n - length(keep), DROP_MIN_ONES))
    M <- M[keep, , drop = FALSE]
    n1 <- n1[keep]
}
gene_names <- rownames(M)
n <- nrow(M)

# Dynamic block size if not specified
if (is.na(BLOCK_SIZE) || BLOCK_SIZE <= 0) {
    BLOCK_SIZE <- max(1L, ceiling(n / 6))
}
idx <- split(seq_len(n), ceiling(seq_len(n) / BLOCK_SIZE))
tM <- t(M)

message("Using asymmetric conditional-prob metric")
edges_list <- vector("list", length(idx))

# Build edges
for (b in seq_along(idx)) {
    I <- idx[[b]]
    C11_I <- M[I, , drop = FALSE] %*% tM
    if (length(C11_I@x) == 0L) next

    Tdg <- as(C11_I, "dgTMatrix")
    ii <- I[Tdg@i + 1L]
    jj <- Tdg@j + 1L
    x <- Tdg@x

    Pij <- x / n1[jj] # P(i=1 | j=1)
    Pji <- x / n1[ii] # P(j=1 | i=1)

    if (PROB_MIN > 0) {
        Pij[Pij < PROB_MIN] <- 0
        Pji[Pji < PROB_MIN] <- 0
    }

    dir_mask <- Pij > Pji
    w_abs <- abs(Pij - Pji)
    sel <- which(dir_mask & (w_abs >= WEIGHT_MIN) & (x >= MIN_COOCC))
    if (!length(sel)) next

    edges_list[[b]] <- data.frame(
        from = gene_names[ii[sel]], # i -> j
        to = gene_names[jj[sel]], # We chose direction where Pij > Pji, i.e., i -> j
        w_diff = w_abs[sel],
        support = x[sel],
        stringsAsFactors = FALSE
    )
}

parts <- Filter(Negate(is.null), edges_list)
edges_df <- if (length(parts)) {
    do.call(rbind, parts)
} else {
    data.frame(from = character(), to = character(), w_diff = double(), support = double(), stringsAsFactors = FALSE)
}

if (nrow(edges_df) == 0L) stop("no edges passed the thresholds")
message(sprintf("Built %d candidate directed edges.", nrow(edges_df)))

g <- graph_from_data_frame(edges_df, directed = TRUE, vertices = gene_names)

cat(sprintf("\n[time] initial graph: %.2f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
t1 <- Sys.time()

g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)

key_g <- paste0(tail_of(g, E(g))$name, "->", head_of(g, E(g))$name)
key_df <- paste0(edges_df$from, "->", edges_df$to)
E(g)$w_diff <- edges_df$w_diff[match(key_g, key_df)]
E(g)$support <- edges_df$support[match(key_g, key_df)]
E(g)$w_plot <- pmax(E(g)$w_diff, 1e-6)

g <- set_path_weights(g)

message(sprintf("Graph: %d vertices, %d edges (after simplify).", vcount(g), ecount(g)))

g_dag <- make_dag(g)
message(sprintf("DAG result: %d vertices, %d edges.", vcount(g_dag), ecount(g_dag)))

# Tagging & output (saving graphs, edges, longest paths, metadata for reference)
tag <- sprintf(
    "%s_comment-%s_thr%s_min%d_cycle%s",
    dataset_label, fmt_id(TAG_COMMENT), fmt_num(THRESHOLD_DIFF),
    as.integer(MIN_COOCC), fmt_id(CYCLE_BREAK_METHOD)
)
outdir <- file.path("outputs", tag)
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

run_meta <- list(
    dataset = dataset_label,
    n_genes = n,
    n_samples = m,
    drop_min_ones = DROP_MIN_ONES,
    block_size = BLOCK_SIZE,
    comment = TAG_COMMENT,
    threshold = THRESHOLD_DIFF,
    min_coocc = MIN_COOCC,
    prob_min = PROB_MIN,
    cycle_break_method = CYCLE_BREAK_METHOD,
    timestamp = as.character(Sys.time()),
    n_vertices_raw = vcount(g),
    n_edges_raw = ecount(g),
    n_vertices_dag = vcount(g_dag),
    n_edges_dag = ecount(g_dag),
    mode = USE_MODE
)

attr(g, "params") <- modifyList(run_meta, list(object = "graph_raw"))
attr(g_dag, "params") <- modifyList(run_meta, list(object = "graph_dag"))

saveRDS(g, file = file.path(outdir, sprintf("graph_raw_%s.rds", tag)))
saveRDS(g_dag, file = file.path(outdir, sprintf("graph_dag_%s.rds", tag)))
write.csv(edges_df, file = file.path(outdir, sprintf("edges_%s.csv", tag)), row.names = FALSE)

# Longest paths
message("Calculating longest paths in DAG (unweighted)...")
lp_edges <- longest_path_dag(g_dag)
message(sprintf(
    "LP (edges): %d edges, %d vertices",
    max(0, length(lp_edges) - 1), length(lp_edges)
))

message("Calculating longest paths in DAG (weighted)...")
lp_weighted <- longest_path_dag_weighted(g_dag, "w_path")
message(sprintf(
    "LP (weighted): %.4f total weight over %d edges",
    sum_path_weight(g_dag, lp_weighted, "w_path"),
    max(0, length(lp_weighted) - 1)
))

if (length(lp_edges)) {
    attr(lp_edges, "params") <- modifyList(run_meta, list(object = "longest_path_edges"))
    saveRDS(lp_edges, file.path(outdir, sprintf("longest_path_edges_%s.rds", tag)))
    writeLines(lp_edges, file.path(outdir, sprintf("longest_path_edges_%s.txt", tag)))
}
if (length(lp_weighted)) {
    attr(lp_weighted, "params") <- modifyList(run_meta, list(
        object = "longest_path_weighted",
        weight_attr = "w_path",
        total_weight = sum_path_weight(g_dag, lp_weighted)
    ))
    saveRDS(lp_weighted, file.path(outdir, sprintf("longest_path_weighted_%s.rds", tag)))
    writeLines(lp_weighted, file.path(outdir, sprintf("longest_path_weighted_%s.txt", tag)))
}

writeLines(tag, file.path(outdir, sprintf("tag_%s.txt", tag)))
writeLines(tag, file.path("outputs", sprintf("latest_tag_%s.txt", fmt_id(dataset_label))))
write.csv(run_meta, file.path("outputs", sprintf("latest_meta_%s.csv", fmt_id(dataset_label))), row.names = FALSE)
write.csv(run_meta, file.path(outdir, sprintf("meta_%s.csv", tag)), row.names = FALSE)

message("Done.")
cat(sprintf("\n[time] longest path calc: %.2f s\n", as.numeric(difftime(Sys.time(), t1, units = "secs"))))
cat(sprintf("\n[time] total: %.2f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

save_topo(g_dag, tag)
