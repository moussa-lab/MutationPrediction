# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

import argparse
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import scipy.cluster.hierarchy as sch

from pathlib import Path
from matplotlib.colors import ListedColormap


sys.setrecursionlimit(10000)

DATASET_LABELS = {}

OUTPUT_DPIS = [1200, 600, 350]
JPG_DPI = 350

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def load_gene_list(path):
    if not path:
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def clean_heatmap(data, **kwargs):
    ax = kwargs.get("ax")

    hm = sns.heatmap(
        data,
        linewidths=0,
        linecolor=None,
        rasterized=True,
        antialiased=False,
        **kwargs,
    )

    if ax is not None and ax.collections:
        mesh = ax.collections[0]
        mesh.set_edgecolor("face")
        mesh.set_linewidth(0)
        mesh.set_antialiased(False)

    return hm


def get_dataset_annotation(df, all_columns_ref):
    datasets = [
        col.split("||")[0] if "||" in col else "Unknown"
        for col in df.columns
    ]
    all_datasets = [
        c.split("||")[0] if "||" in c else "Unknown"
        for c in all_columns_ref
    ]

    unique_datasets = sorted(list(set(all_datasets)))
    dataset_counts = {d: all_datasets.count(d) for d in unique_datasets}

    palette = sns.color_palette("tab20", n_colors=len(unique_datasets))
    dataset_cmap = ListedColormap(palette)
    dataset_map = {d: i for i, d in enumerate(unique_datasets)}

    numeric_row = np.array([dataset_map[d] for d in datasets]).reshape(1, -1)

    patches = []
    for i, d in enumerate(unique_datasets):
        clean_name = DATASET_LABELS.get(d, d)
        count = dataset_counts.get(d, 0)
        label_str = f"{clean_name} (n={count})"
        patches.append(mpatches.Patch(color=palette[i], label=label_str))

    return numeric_row, dataset_cmap, patches


def calculate_global_order(df_full):
    if df_full.shape[0] > 20:
        print("   -> Optimizing: Clustering on top 20 genes...")
        gene_freq = df_full.sum(axis=1)
        top_genes = gene_freq.sort_values(ascending=False).head(20).index
        df_cluster_input = df_full.loc[top_genes]
    else:
        df_cluster_input = df_full

    try:
        Z = sch.linkage(df_cluster_input.T, method="average", metric="euclidean")
        dendro = sch.dendrogram(Z, no_plot=True)
        col_indices = dendro["leaves"]
        ordered_samples = df_full.columns[col_indices]
        return ordered_samples
    except Exception as e:
        print(f"   [warn] Global clustering failed ({e}). Using default order.")
        return df_full.columns


def process_panel(
    df_full,
    gene_list_path,
    top_n,
    global_sample_order,
    selection_mode="prevalence",
):
    print(f"[info] Processing panel for list: {gene_list_path} | Mode: {selection_mode}")

    # load path genes
    if gene_list_path:
        path_genes = load_gene_list(gene_list_path)
        valid_genes_ordered = [
            g for g in path_genes
            if g in df_full.index and g != "Unknown"
        ]
        if not valid_genes_ordered:
            raise ValueError(f"No valid genes found in data for list: {gene_list_path}")
    else:
        valid_genes_ordered = df_full.index.tolist()

    # select genes
    if selection_mode == "strict_order":
        #strict: just take the first N genes found in the list
        #used in fig 3 to preserve order of paths
        final_gene_order = valid_genes_ordered[:top_n]
        print(f"   -> strict_order: Selected first {len(final_gene_order)} genes from file.")
    else:
        # prevalance, used in fig 1
        df_temp = df_full.loc[valid_genes_ordered]
        gene_freq = df_temp.sum(axis=1)
        final_gene_order = (
            gene_freq
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
        print(f"   -> prevalence: Selected top {len(final_gene_order)} genes by mutation count.")

    df = df_full.loc[final_gene_order, global_sample_order]

    return df


def output_stem_from_path(output_path):
    # remove file extension if present since we append dpi and stuff
    p = Path(output_path)
    if p.suffix:
        return str(p.with_suffix(""))
    return str(p)


def save_all_outputs(fig, output_stem):
    # save PDF/EPS/TIFF at 1200, 600, and 350 dpi.
    # save JPG at 350 dpi only.
    saved = []

    for dpi in OUTPUT_DPIS:
        pdf_path = f"{output_stem}_{dpi}dpi.pdf"
        eps_path = f"{output_stem}_{dpi}dpi.eps"
        tif_path = f"{output_stem}_{dpi}dpi.tif"

        print(f"[info] Saving PDF at dpi={dpi}: {pdf_path}")
        fig.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            metadata={
                "Creator": "matplotlib",
                "Producer": "matplotlib",
            },
        )
        saved.append(pdf_path)

        print(f"[info] Saving EPS at dpi={dpi}: {eps_path}")
        fig.savefig(
            eps_path,
            format="eps",
            bbox_inches="tight",
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
        )
        saved.append(eps_path)

        print(f"[info] Saving TIFF at dpi={dpi}: {tif_path}")
        fig.savefig(
            tif_path,
            format="tiff",
            bbox_inches="tight",
            dpi=dpi,
            facecolor="white",
            edgecolor="none",
            pil_kwargs={
                "compression": "tiff_lzw",
            },
        )
        saved.append(tif_path)

        if dpi == JPG_DPI:
            jpg_path = f"{output_stem}_{dpi}dpi.jpg"

            print(f"[info] Saving JPG at dpi={dpi}: {jpg_path}")
            fig.savefig(
                jpg_path,
                format="jpg",
                bbox_inches="tight",
                dpi=dpi,
                facecolor="white",
                edgecolor="none",
                pil_kwargs={
                    "quality": 95,
                    "subsampling": 0,
                },
            )
            saved.append(jpg_path)

    return saved


def build_figure(args, df_master, global_order):
    mm_to_inch = 1 / 25.4
    heatmap_cmap = ListedColormap(["#f0f0f0", "#0072B2"])

    if args.lp_list2:
        # dual
        print("[info] Dual Path Mode Detected.")

        # strict order
        df_a = process_panel(
            df_master,
            args.lp_list,
            args.top_genes,
            global_order,
            selection_mode="strict_order",
        )
        df_b = process_panel(
            df_master,
            args.lp_list2,
            args.top_genes,
            global_order,
            selection_mode="strict_order",
        )

        annot_a, cmap_a, legend_a = get_dataset_annotation(df_a, df_master.columns)
        annot_b, cmap_b, legend_b = get_dataset_annotation(df_b, df_master.columns)

        fig = plt.figure(
            figsize=(
                args.width_mm * mm_to_inch,
                args.height_mm * mm_to_inch,
            )
        )

        gs = gridspec.GridSpec(
            2,
            5,
            width_ratios=[1, 0.3, 1, 0.05, 0.05],
            height_ratios=[20, 1],
            wspace=0.0,
            hspace=0.02,
        )

        # Plot A
        ax_a = plt.subplot(gs[0, 0])
        clean_heatmap(
            df_a,
            cmap=heatmap_cmap,
            cbar=False,
            ax=ax_a,
            yticklabels=True,
            xticklabels=False,
        )
        ax_a.set_title("Unweighted Longest Path", fontsize=10)
        ax_a.tick_params(axis="y", labelsize=6)
        ax_a.set_ylabel("")

        ax_a_bar = plt.subplot(gs[1, 0])
        clean_heatmap(
            annot_a,
            cmap=cmap_a,
            cbar=False,
            ax=ax_a_bar,
            xticklabels=False,
            yticklabels=False,
        )
        ax_a_bar.set_xlabel(f"Samples (n={df_a.shape[1]})", fontsize=8)

        # Plot B
        ax_b = plt.subplot(gs[0, 2])
        cbar_ax = plt.subplot(gs[0, 4])
        clean_heatmap(
            df_b,
            cmap=heatmap_cmap,
            cbar=True,
            ax=ax_b,
            cbar_ax=cbar_ax,
            yticklabels=True,
            xticklabels=False,
            cbar_kws={
                "label": "Mutation Status",
                "ticks": [0, 1],
            },
        )
        ax_b.set_title("Weighted Longest Path", fontsize=10)
        ax_b.tick_params(axis="y", labelsize=6)
        ax_b.set_ylabel("")

        ax_b_bar = plt.subplot(gs[1, 2])
        clean_heatmap(
            annot_b,
            cmap=cmap_b,
            cbar=False,
            ax=ax_b_bar,
            xticklabels=False,
            yticklabels=False,
        )
        ax_b_bar.set_xlabel(f"Samples (n={df_b.shape[1]})", fontsize=8)

        # Legend
        fig.legend(
            handles=legend_a,
            title="Dataset Source",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            fontsize=6,
            title_fontsize=7,
        )

    else:
        # single
        print("[info] Single Path Mode.")

        # prevalance
        df_a = process_panel(
            df_master,
            args.lp_list,
            args.top_genes,
            global_order,
            selection_mode="prevalence",
        )
        annot_a, cmap_a, legend_a = get_dataset_annotation(df_a, df_master.columns)

        fig = plt.figure(
            figsize=(
                args.width_mm * mm_to_inch,
                args.height_mm * mm_to_inch,
            )
        )

        gs = gridspec.GridSpec(
            2,
            3,
            width_ratios=[1, 0.03, 0.05],
            height_ratios=[30, 1],
            wspace=0.02,
            hspace=0.01,
        )

        ax_main = plt.subplot(gs[0, 0])
        cbar_ax = plt.subplot(gs[0, 2])

        clean_heatmap(
            df_a,
            cmap=heatmap_cmap,
            ax=ax_main,
            cbar_ax=cbar_ax,
            cbar_kws={
                "label": "Mutation Status",
                "ticks": [0, 1],
            },
            xticklabels=False,
            yticklabels=True,
        )

        ax_bar = plt.subplot(gs[1, 0])
        clean_heatmap(
            annot_a,
            cmap=cmap_a,
            ax=ax_bar,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )

        ax_main.set_xlabel("")
        ax_main.tick_params(axis="y", labelsize=8)
        ax_bar.set_xlabel(f"Samples (n={df_a.shape[1]})", fontsize=9)
        ax_main.set_ylabel("")

        fig.legend(
            handles=legend_a,
            title="Dataset Source",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=2,
            fontsize=6,
            title_fontsize=7,
        )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--lp_list", help="Path A gene list")
    parser.add_argument(
        "--lp_list2",
        default=None,
        help="Path B gene list (Triggers Dual Mode)",
    )
    parser.add_argument(
        "--top_genes",
        type=int,
        default=20,
        help="Max genes per panel",
    )
    parser.add_argument(
        "--output",
        default="mutation_heatmap.pdf",
        help=(
            "Output filename or stem. Extension is ignored for multi-format output. "
            "Example: mutation_heatmap.pdf becomes mutation_heatmap_600dpi.pdf, etc."
        ),
    )
    parser.add_argument("--width_mm", type=float, default=180)
    parser.add_argument("--height_mm", type=float, default=120)

    args = parser.parse_args()

    # load data
    print(f"[info] Loading master dataset: {args.data_csv}")
    df_master = pd.read_csv(args.data_csv, index_col=0)
    df_master = (df_master > 0).astype(int)

    # drop unnamed mutation records
    if "Unknown" in df_master.index:
        df_master = df_master.drop("Unknown")

    global_order = calculate_global_order(df_master)

    fig = build_figure(args, df_master, global_order)

    output_stem = output_stem_from_path(args.output)
    output_dir = os.path.dirname(output_stem)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    saved_paths = save_all_outputs(fig, output_stem)

    plt.close(fig)

    print("[done] Saved outputs:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()