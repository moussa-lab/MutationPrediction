MAT_PATH <- "study_data/COAD2.mat"
DATASET_NAME <- "COAD2"
BLOCK_SIZE <- ceiling(12323 / 6)
METRIC <- "classic_inv" # "improvement" or "classic"
THRESHOLD_DIFF <- 0.5
MIN_COOCC <- 1
DROP_MIN_ONES <- 0
# PLOT_ENABLE <- TRUE
# PLOT_MAX_VERTS <- 400
USE_SUBSAMPLE <- FALSE
SUBSAMPLE_SIZE <- 1000
CYCLE_BREAK_METHOD <- "feedback_arc_set" # "none" or "feedback_arc_set" or "feedback_vertex_set"
PATH_WEIGHT_ATTR <- "w_diff" # edge attribute to use for path weights

suppressPackageStartupMessages({
    library(R.matlab)
    library(Matrix)
    library(igraph)
})

set.seed(123)

message("Loading .mat: ", MAT_PATH)
mat_train_data <- readMat(MAT_PATH)

ds <- mat_train_data[[DATASET_NAME]]
if (is.null(ds)) {
    stop(
        "Dataset name '", DATASET_NAME,
        "' not found in ", MAT_PATH
    )
}

# COAD: 213, DFCI 321 (data, genes, samples)
gene_matrix <- as.matrix(ds[[2]])
rownames(gene_matrix) <- as.character(Reduce(c, ds[[1]]))
colnames(gene_matrix) <- as.character(Reduce(c, ds[[3]]))

if (USE_SUBSAMPLE) {
    gene_matrix <- gene_matrix[sample(1:dim(gene_matrix)[1], SUBSAMPLE_SIZE, replace = FALSE), ]
}

t0 <- Sys.time()

M <- as(gene_matrix != 0, "lgCMatrix")
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
invn1 <- 1 / n1

message(sprintf("Kept %d genes with n1 > %d.", n, DROP_MIN_ONES))

idx <- split(seq_len(n), ceiling(seq_len(n) / BLOCK_SIZE))

tM <- t(M)

edges_list <- vector("list", length(idx))

if (METRIC == "improvement") {
    message("Using metric: improvement")
    dval_metric <- function(x, ii, jj) {
        x * (invn1[jj] - invn1[ii]) - (n1[ii] - n1[jj]) / as.numeric(m)
    }
} else if (METRIC == "classic") {
    message("Using metric: classic")
    dval_metric <- function(x, ii, jj) {
        x * (invn1[jj] - invn1[ii])
    }
} else if (METRIC == "classic_inv") {
    message("Using metric: classic")
    dval_metric <- function(x, ii, jj) {
        x * (invn1[ii] - invn1[jj])
    }
} else {
    stop("unknown metric: ", METRIC)
}

for (b in seq_along(idx)) {
    I <- idx[[b]]

    C11_I <- M[I, , drop = FALSE] %*% tM
    if (length(C11_I@x) == 0L) next

    T <- as(C11_I, "dgTMatrix")
    ii <- I[T@i + 1L]
    jj <- T@j + 1L
    x <- T@x

    # (P(i|j) - P(i)) - (P(j|i) - P(j))
    dval <- dval_metric(x, ii, jj)
    sel <- which(dval >= THRESHOLD_DIFF & x >= MIN_COOCC)
    if (!length(sel)) next

    edges_list[[b]] <- data.frame(
        from = gene_names[jj[sel]],
        to = gene_names[ii[sel]],
        w_diff = dval[sel],
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

if (nrow(edges_df) == 0L) {
    stop("no edges passed the thresholds")
}

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

set_path_weights <- function(g, mode = PATH_WEIGHT_ATTR) {
    w <- switch(mode,
        w_diff = E(g)$w_diff,
        stop("unknown PATH_WEIGHT_ATTR: ", mode)
    )
    w[!is.finite(w)] <- 0
    E(g)$w_path <- pmax(w, 0) + 1e-12
    g
}

g <- set_path_weights(g)

message(sprintf("Graph: %d vertices, %d edges (after simplify).", vcount(g), ecount(g)))

make_dag <- function(g) {
    if (is_dag(g)) {
        message("Graph is already a DAG; no cycles to break.")
        return(g)
    }

    if (CYCLE_BREAK_METHOD == "feedback_arc_set") {
        fas <- feedback_arc_set(g, weights = E(g)$w_diff)

        message(sprintf("Removing %d edges via feedback_arc_set to break cycles.", length(fas)))
        g2 <- delete_edges(g, fas)
        return(g2)
    } else if (CYCLE_BREAK_METHOD == "feedback_vertex_set") {
        fvs <- feedback_vertex_set(g)
        writeLines(fvs, "feedback_vertex_set.txt")


        message(sprintf("Removing %d vertices via feedback_vertex_set to break cycles.", length(fvs)))
        g2 <- delete_vertices(g, fvs)
        return(g2)
    } else if (CYCLE_BREAK_METHOD == "none") {
        message("CYCLE_BREAK_METHOD is 'none'; returning original graph (not a DAG).")
        return(g)
    } else {
        stop("unknown CYCLE_BREAK_METHOD: ", CYCLE_BREAK_METHOD)
    }

    g
}

g_dag <- make_dag(g)
message(sprintf("DAG result: %d vertices, %d edges.", vcount(g_dag), ecount(g_dag)))

fmt_num <- function(x, digits = 6) {
    s <- formatC(x, format = "f", digits = digits, drop0trailing = TRUE)
    s <- sub("\\.", "p", s)
    gsub("\\s+", "", s)
}
fmt_id <- function(s) {
    tolower(gsub("[^A-Za-z0-9]+", "-", s))
}

tag <- sprintf("%s_metric-%s_thr%s_min%d_cycle%s", DATASET_NAME, fmt_id(METRIC), fmt_num(THRESHOLD_DIFF), as.integer(MIN_COOCC), fmt_id(CYCLE_BREAK_METHOD))

if (!dir.exists(sprintf("outputs/%s", tag))) dir.create(sprintf("outputs/%s", tag), recursive = TRUE)

run_meta <- list(
    dataset = DATASET_NAME,
    n_genes = n,
    n_samples = m,
    drop_min_ones = DROP_MIN_ONES,
    block_size = BLOCK_SIZE,
    metric = METRIC,
    threshold = THRESHOLD_DIFF,
    min_coocc = MIN_COOCC,
    cycle_break_method = CYCLE_BREAK_METHOD,
    timestamp = as.character(Sys.time()),
    n_vertices_raw = vcount(g),
    n_edges_raw = ecount(g),
    n_vertices_dag = vcount(g_dag),
    n_edges_dag = ecount(g_dag)
)

attr(g, "params") <- modifyList(run_meta, list(object = "graph_raw"))
attr(g_dag, "params") <- modifyList(run_meta, list(object = "graph_dag"))

saveRDS(g, file = file.path(sprintf("outputs/%s", tag), sprintf("graph_raw_%s.rds", tag)))
saveRDS(g_dag, file = file.path(sprintf("outputs/%s", tag), sprintf("graph_dag_%s.rds", tag)))
if (exists("edges_df")) {
    write.csv(edges_df, file = file.path(sprintf("outputs/%s", tag), sprintf("edges_%s.csv", tag)), row.names = FALSE)
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

save_topo <- function(g, tag) {
    stopifnot(is.dag(g))
    topo_idx <- as.integer(topo_sort(g, mode = "out"))
    topo_names <- V(g)$name[topo_idx]
    writeLines(topo_names, file.path(sprintf("outputs/%s", tag), sprintf("topo_order_%s.txt", tag)))
    saveRDS(topo_names, file.path(sprintf("outputs/%s", tag), sprintf("topo_order_%s.rds", tag)))
    topo_names
}

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

# Save both
if (length(lp_edges)) {
    attr(lp_edges, "params") <- modifyList(run_meta, list(object = "longest_path_edges"))
    saveRDS(lp_edges, file.path(sprintf("outputs/%s", tag), sprintf("longest_path_edges_%s.rds", tag)))
    writeLines(lp_edges, file.path(sprintf("outputs/%s", tag), sprintf("longest_path_edges_%s.txt", tag)))
}
if (length(lp_weighted)) {
    attr(lp_weighted, "params") <- modifyList(run_meta, list(
        object = "longest_path_weighted",
        weight_attr = "w_path",
        total_weight = sum_path_weight(g_dag, lp_weighted)
    ))
    saveRDS(lp_weighted, file.path(sprintf("outputs/%s", tag), sprintf("longest_path_weighted_%s.rds", tag)))
    writeLines(lp_weighted, file.path(sprintf("outputs/%s", tag), sprintf("longest_path_weighted_%s.txt", tag)))
}

writeLines(tag, file.path(sprintf("outputs/%s", tag), sprintf("tag_%s.txt", tag)))

# also save newest tag to top level of outputs directory
writeLines(tag, file.path("outputs", sprintf("latest_tag_%s.txt", DATASET_NAME)))
# and save metadata to csv (both latest and per-run)
write.csv(run_meta, file.path("outputs", sprintf("latest_meta_%s.csv", DATASET_NAME)), row.names = FALSE)
write.csv(run_meta, file.path(sprintf("outputs/%s", tag), sprintf("meta_%s.csv", tag)), row.names = FALSE)

message("Done.")

cat(sprintf("\n[time] longest path calc: %.2f s\n", as.numeric(difftime(Sys.time(), t1, units = "secs"))))
cat(sprintf("\n[time] total: %.2f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

save_topo(g_dag, tag)
