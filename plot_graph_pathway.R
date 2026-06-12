suppressPackageStartupMessages({
    library(igraph)
    library(ggplot2)
    library(ggraph)
    library(ggrepel)
    library(dplyr)
    library(RColorBrewer)
    library(grid)
})

# Update this TAG to match your specific run output
TAG <- "CSV_metric-mrm_thr0p02_min1_cyclefeedback-arc-set"

BASE_DIR <- file.path("outputs", TAG)
GRAPH_FILE <- file.path(BASE_DIR, sprintf("graph_dag_%s.rds", TAG))
LP_UNWEIGHTED_FILE <- file.path(BASE_DIR, sprintf("longest_path_edges_%s.rds", TAG))
LP_WEIGHTED_FILE <- file.path(BASE_DIR, sprintf("longest_path_weighted_%s.rds", TAG))

# Output settings
OUTPUT_STEM <- file.path(BASE_DIR, "figure_pathway_graph")

WIDTH_IN <- 14
HEIGHT_IN <- 17

# Save PDF/EPS/TIFF at all three resolutions
OUTPUT_DPIS <- c(1200, 600, 350)
# Save JPG only at this resolution
JPG_DPI <- 350

# Font family
PLOT_FONT_FAMILY <- "Helvetica"

# Main graph layout
GRAPH_LAYOUT <- "sugiyama"

# Label placement
TOP_ALTERNATING_LABEL_N <- 18

## Small manual nudges for nodes that Sugiyama places nearly on top of each other
MANUAL_NODE_NUDGES <- data.frame(
    name = c("TP53", "TTN"),
    x_nudge = c(-0.018, 0.018),
    y_nudge = c(0.006, 0.012),
    stringsAsFactors = FALSE
)

# Visualization constraints
# 0 = only show the paths themselves
# 1 = show paths + immediate neighbors
NEIGHBOR_ORDER <- 0

# Genes to ALWAYS label if present in the plot.
IMPORTANT_GENES <- c()

safe_dev_off <- function() {
    if (dev.cur() > 1) {
        try(dev.off(), silent = TRUE)
    }
}

find_ghostscript <- function() {
    candidates <- c(
        Sys.getenv("R_GSCMD"),
        Sys.which("gs"),
        Sys.which("gswin64c"),
        Sys.which("gswin32c")
    )

    candidates <- candidates[nzchar(candidates)]

    if (length(candidates) == 0) {
        return("")
    }

    candidates[[1]]
}

embed_fonts_if_possible <- function(path) {
    gs <- find_ghostscript()

    if (!nzchar(gs)) {
        message("  [font] Ghostscript not found; skipping font embedding for: ", path)
        return(path)
    }

    old_gs <- Sys.getenv("R_GSCMD")
    Sys.setenv(R_GSCMD = gs)

    tmp <- tempfile(
        pattern = paste0(tools::file_path_sans_ext(basename(path)), "_embedded_"),
        fileext = paste0(".", tools::file_ext(path))
    )

    result <- tryCatch(
        {
            grDevices::embedFonts(file = path, outfile = tmp)

            if (file.exists(tmp) && file.info(tmp)$size > 0) {
                file.copy(tmp, path, overwrite = TRUE)
                unlink(tmp)
                message("  [font] Embedded fonts in: ", path)
            } else {
                message("  [font] Font embedding produced no output for: ", path)
            }

            path
        },
        error = function(e) {
            message(
                "  [font] Could not embed fonts in ",
                path,
                ": ",
                conditionMessage(e)
            )
            path
        }
    )

    Sys.setenv(R_GSCMD = old_gs)
    result
}

save_one_output <- function(path, device_fun, device_args, embed_fonts = FALSE) {
    tryCatch(
        {
            do.call(device_fun, c(list(filename = path), device_args))
            grid.newpage()
            print(p, newpage = FALSE)
            dev.off()

            if (embed_fonts) {
                embed_fonts_if_possible(path)
            }

            path
        },
        error = function(e) {
            safe_dev_off()

            fallback_path <- sub(
                "\\.([^.]+)$",
                paste0("_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".\\1"),
                path
            )

            message("Could not write ", path, ": ", conditionMessage(e))
            message("Trying fallback output: ", fallback_path)

            do.call(device_fun, c(list(filename = fallback_path), device_args))
            grid.newpage()
            print(p, newpage = FALSE)
            dev.off()

            if (embed_fonts) {
                embed_fonts_if_possible(fallback_path)
            }

            fallback_path
        }
    )
}

# Load graph and paths
if (!file.exists(GRAPH_FILE)) {
    stop("Graph file not found: ", GRAPH_FILE)
}

message("Loading graph...")
g <- readRDS(GRAPH_FILE)

message("Loading paths...")
path_unweighted <- if (file.exists(LP_UNWEIGHTED_FILE)) {
    readRDS(LP_UNWEIGHTED_FILE)
} else {
    character(0)
}

path_weighted <- if (file.exists(LP_WEIGHTED_FILE)) {
    readRDS(LP_WEIGHTED_FILE)
} else {
    character(0)
}

message(sprintf("Graph size: %d nodes, %d edges", vcount(g), ecount(g)))

# Identify nodes to plot

nodes_in_path_u <- path_unweighted
nodes_in_path_w <- path_weighted
all_path_nodes <- unique(c(nodes_in_path_u, nodes_in_path_w))

if (length(all_path_nodes) == 0) {
    stop("No paths found to plot!")
}

if (NEIGHBOR_ORDER > 0) {
    message("Expanding subgraph to neighbors order ", NEIGHBOR_ORDER)

    sub_vids <- unlist(
        ego(
            g,
            order = NEIGHBOR_ORDER,
            nodes = all_path_nodes,
            mode = "all"
        )
    )

    subgraph_nodes <- V(g)$name[unique(sub_vids)]
} else {
    subgraph_nodes <- all_path_nodes
}

message(sprintf("Subgraph size: %d nodes", length(subgraph_nodes)))

# create subgraph

sg <- induced_subgraph(g, subgraph_nodes)

# annotate

V(sg)$type <- "Background"
V(sg)$type[V(sg)$name %in% nodes_in_path_u] <- "Unweighted Path"
V(sg)$type[V(sg)$name %in% nodes_in_path_w] <- "Weighted Path"
V(sg)$type[V(sg)$name %in% intersect(nodes_in_path_u, nodes_in_path_w)] <- "Both"

V(sg)$label_text <- NA

label_mask <- (
    (V(sg)$name %in% IMPORTANT_GENES) |
        (V(sg)$name %in% nodes_in_path_w) |
        (V(sg)$name %in% nodes_in_path_u)
)

V(sg)$label_text[label_mask] <- V(sg)$name[label_mask]

V(sg)$size <- 2.6
V(sg)$size[V(sg)$type != "Background"] <- 4.4

# Annotate edges

E(sg)$edge_type <- "Background"
E(sg)$width <- 0.25

# Avoid transparency for EPS compatibility
E(sg)$alpha <- 1.0

mark_path_edges <- function(graph, path_nodes) {
    if (length(path_nodes) < 2) {
        return(integer(0))
    }

    path_edges_keys <- paste(
        path_nodes[-length(path_nodes)],
        path_nodes[-1],
        sep = "|"
    )

    graph_edges <- as_edgelist(graph)
    graph_edges_keys <- paste(graph_edges[, 1], graph_edges[, 2], sep = "|")

    which(graph_edges_keys %in% path_edges_keys)
}

idx_u <- mark_path_edges(sg, path_unweighted)
idx_w <- mark_path_edges(sg, path_weighted)

E(sg)$edge_type[idx_u] <- "Unweighted Path"
E(sg)$edge_type[idx_w] <- "Weighted Path"
E(sg)$edge_type[intersect(idx_u, idx_w)] <- "Both"

E(sg)$width[E(sg)$edge_type != "Background"] <- 1.15

# layout

message("Generating layout...")

layout <- create_layout(sg, layout = GRAPH_LAYOUT)
layout_df <- as.data.frame(layout)

layout_x_span <- diff(range(layout_df$x, na.rm = TRUE))
layout_y_span <- diff(range(layout_df$y, na.rm = TRUE))

if (!is.finite(layout_x_span) || layout_x_span == 0) {
    layout_x_span <- 1
}

if (!is.finite(layout_y_span) || layout_y_span == 0) {
    layout_y_span <- 1
}

for (i in seq_len(nrow(MANUAL_NODE_NUDGES))) {
    node_idx <- which(layout$name == MANUAL_NODE_NUDGES$name[i])

    if (length(node_idx) == 1) {
        layout$x[node_idx] <- layout$x[node_idx] +
            MANUAL_NODE_NUDGES$x_nudge[i] * layout_x_span

        layout$y[node_idx] <- layout$y[node_idx] +
            MANUAL_NODE_NUDGES$y_nudge[i] * layout_y_span
    }
}

layout_df <- as.data.frame(layout)

edge_palette <- c(
    "Background" = "grey92",
    "Unweighted Path" = "#E69F00",
    "Weighted Path" = "#56B4E9",
    "Both" = "#009E73"
)

node_palette <- c(
    "Background" = "grey60",
    "Unweighted Path" = "#E69F00",
    "Weighted Path" = "#56B4E9",
    "Both" = "#009E73"
)

label_df <- layout_df %>%
    filter(!is.na(label_text), label_text != "")

x_span <- diff(range(layout_df$x, na.rm = TRUE))
y_span <- diff(range(layout_df$y, na.rm = TRUE))

if (!is.finite(x_span) || x_span == 0) {
    x_span <- 1
}

if (!is.finite(y_span) || y_span == 0) {
    y_span <- 1
}

if (nrow(label_df) > 0) {
    x_mid <- median(layout_df$x, na.rm = TRUE)

    top_alternating_genes <- label_df %>%
        arrange(desc(y), x) %>%
        slice_head(n = TOP_ALTERNATING_LABEL_N) %>%
        mutate(top_side = if_else(row_number() %% 2 == 1, -1, 1)) %>%
        select(name, top_side)

    label_df <- label_df %>%
        mutate(
            label_side = if_else(x <= x_mid, -1, 1),
            label_hjust = if_else(label_side < 0, 1, 0),
            label_nudge = label_side * 0.07 * x_span
        ) %>%
        left_join(top_alternating_genes, by = "name") %>%
        mutate(
            label_side = coalesce(top_side, label_side),
            label_hjust = if_else(label_side < 0, 1, 0),
            label_nudge = label_side * 0.08 * x_span
        )
}

left_label_df <- filter(label_df, label_side < 0)
right_label_df <- filter(label_df, label_side >= 0)

# plot

p <- ggraph(layout) +
    # background edges
    geom_edge_link(
        aes(
            filter = edge_type == "Background",
            color = edge_type,
            width = width,
            alpha = alpha
        ),
        arrow = arrow(length = unit(2.2, "mm"), type = "closed"),
        end_cap = circle(2.2, "mm")
    ) +

    # highlighted path edges
    geom_edge_link(
        aes(
            filter = edge_type != "Background",
            color = edge_type,
            width = width,
            alpha = alpha
        ),
        arrow = arrow(length = unit(2.2, "mm"), type = "closed"),
        end_cap = circle(2.2, "mm")
    ) +
    scale_edge_width_identity() +
    scale_edge_alpha_identity() +
    scale_edge_color_manual(
        values = edge_palette,
        name = "Interaction Type"
    ) +

    # nodes
    geom_node_point(aes(color = type, size = size)) +
    scale_size_identity() +
    scale_color_manual(
        values = node_palette,
        name = "Gene Type"
    ) +

    # left labels
    geom_label_repel(
        data = left_label_df,
        aes(x = x, y = y, label = label_text, color = type),
        inherit.aes = FALSE,
        nudge_x = -0.08 * x_span,
        direction = "y",
        hjust = 1,
        fill = "white",
        linewidth = 0.12,
        size = 4,
        family = PLOT_FONT_FAMILY,
        fontface = "bold",
        box.padding = 0.18,
        point.padding = 0,
        segment.color = "grey20",
        segment.size = 0.26,
        segment.alpha = 1.0,
        min.segment.length = 0,
        max.overlaps = Inf,
        seed = 103,
        show.legend = FALSE
    ) +

    # right labels
    geom_label_repel(
        data = right_label_df,
        aes(x = x, y = y, label = label_text, color = type),
        inherit.aes = FALSE,
        nudge_x = 0.08 * x_span,
        direction = "y",
        hjust = 0,
        fill = "white",
        linewidth = 0.12,
        size = 4,
        family = PLOT_FONT_FAMILY,
        fontface = "bold",
        box.padding = 0.18,
        point.padding = 0,
        segment.color = "grey20",
        segment.size = 0.26,
        segment.alpha = 1.0,
        min.segment.length = 0,
        max.overlaps = Inf,
        seed = 103,
        show.legend = FALSE
    ) +

    coord_cartesian(
        xlim = range(layout_df$x, na.rm = TRUE) + c(-0.09, 0.09) * x_span,
        ylim = range(layout_df$y, na.rm = TRUE) + c(-0.045, 0.035) * y_span,
        clip = "off"
    ) +

    theme_void(base_family = PLOT_FONT_FAMILY) +
    theme(
        legend.position = "bottom",
        legend.text = element_text(family = PLOT_FONT_FAMILY),
        legend.title = element_text(family = PLOT_FONT_FAMILY),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(2, 12, 20, 12)
    ) +
    ggtitle(NULL)

# save

save_pathway_outputs_for_dpi <- function(output_stem, dpi) {
    output_pdf <- sprintf("%s_%ddpi.pdf", output_stem, dpi)
    output_eps <- sprintf("%s_%ddpi.eps", output_stem, dpi)
    output_tiff <- sprintf("%s_%ddpi.tif", output_stem, dpi)

    message("Saving PDF at fallback/raster DPI = ", dpi, "...")
    pdf_out <- save_one_output(
        path = output_pdf,
        device_fun = grDevices::cairo_pdf,
        device_args = list(
            width = WIDTH_IN,
            height = HEIGHT_IN,
            family = PLOT_FONT_FAMILY,
            fallback_resolution = dpi
        ),
        embed_fonts = TRUE
    )

    message("Saving EPS at fallback/raster DPI = ", dpi, "...")
    eps_out <- save_one_output(
        path = output_eps,
        device_fun = grDevices::cairo_ps,
        device_args = list(
            width = WIDTH_IN,
            height = HEIGHT_IN,
            family = PLOT_FONT_FAMILY,
            onefile = FALSE,
            fallback_resolution = dpi
        ),
        embed_fonts = TRUE
    )

    message("Saving TIFF at ", dpi, " dpi...")
    tiff_out <- save_one_output(
        path = output_tiff,
        device_fun = grDevices::tiff,
        device_args = list(
            width = WIDTH_IN,
            height = HEIGHT_IN,
            units = "in",
            res = dpi,
            compression = "lzw",
            type = "cairo",
            family = PLOT_FONT_FAMILY
        ),
        embed_fonts = FALSE
    )

    list(
        dpi = dpi,
        pdf = pdf_out,
        eps = eps_out,
        tiff = tiff_out
    )
}

save_jpg_output <- function(output_stem, dpi) {
    output_jpg <- sprintf("%s_%ddpi.jpg", output_stem, dpi)

    message("Saving JPG at ", dpi, " dpi...")
    jpg_out <- save_one_output(
        path = output_jpg,
        device_fun = grDevices::jpeg,
        device_args = list(
            width = WIDTH_IN,
            height = HEIGHT_IN,
            units = "in",
            res = dpi,
            quality = 95,
            type = "cairo",
            bg = "white"
        ),
        embed_fonts = FALSE
    )

    jpg_out
}

outputs <- lapply(
    OUTPUT_DPIS,
    function(dpi) save_pathway_outputs_for_dpi(OUTPUT_STEM, dpi)
)

jpg_output <- save_jpg_output(OUTPUT_STEM, JPG_DPI)

message("Done. Saved outputs:")
for (out in outputs) {
    message("  ", out$dpi, " dpi PDF:  ", out$pdf)
    message("  ", out$dpi, " dpi EPS:  ", out$eps)
    message("  ", out$dpi, " dpi TIFF: ", out$tiff)
}
message("  ", JPG_DPI, " dpi JPG:  ", jpg_output)