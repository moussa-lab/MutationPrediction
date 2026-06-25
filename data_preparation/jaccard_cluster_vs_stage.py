# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


STAGED_CSV = Path("clustered_datasets/by_stage/staged_only.csv")
LABELS_CSV = Path("clustered_datasets/sample_labels.csv")
N_CLUSTERS = 4
LINKAGE_METHOD = "ward"
RANDOM_SEED = 42
RUNS = [
    ("unfiltered", 0.0, Path("clustered_datasets/jaccard_vs_stage_nofilter")),
    ("filtered_5pct", 0.05, Path("clustered_datasets/jaccard_vs_stage")),
]
STAGE_ORDER = ["I", "II", "III", "IV"]
STAGE_COLORS = {"I": "#2ecc71", "II": "#3498db", "III": "#e67e22", "IV": "#e74c3c"}


sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11})


def load_data():
    if not STAGED_CSV.exists():
        raise SystemExit(f"{STAGED_CSV} not found. Run create_clustered_datasets.py first.")
    mat = pd.read_csv(STAGED_CSV, index_col=0)
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[labels["sample"].isin(mat.columns) & (labels["stage"] != "Unknown")]
    labels = labels.set_index("sample").reindex(mat.columns).reset_index().rename(columns={"index": "sample"})
    return mat, labels


def score(stages, cluster_ids):
    stage_ids = np.array([STAGE_ORDER.index(s) for s in stages])
    crosstab = pd.crosstab(
        pd.Series(cluster_ids, name="cluster"),
        pd.Series(stages, name="stage"),
    ).reindex(columns=STAGE_ORDER, fill_value=0)
    return {
        "ari": adjusted_rand_score(stage_ids, cluster_ids),
        "nmi": normalized_mutual_info_score(stage_ids, cluster_ids, average_method="arithmetic"),
        "purity": crosstab.max(axis=1).sum() / len(stages),
        "crosstab": crosstab,
    }


def plot_confusion(crosstab, out_path):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.7 * len(crosstab))))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Samples"})
    ax.set_xlabel("Stage")
    ax.set_ylabel("Cluster")
    ax.set_title("Jaccard clusters vs AJCC stage")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_pca(x, cluster_ids, stages, out_path):
    coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cluster_palette = sns.color_palette("Set2", len(set(cluster_ids)))
    cluster_colors = {c: cluster_palette[i] for i, c in enumerate(sorted(set(cluster_ids)))}

    for cluster in sorted(set(cluster_ids)):
        mask = cluster_ids == cluster
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.65,
                        c=[cluster_colors[cluster]], label=f"C{cluster} (n={mask.sum()})")
    axes[0].set_title("Cluster")
    axes[0].legend(fontsize=8)

    for stage in STAGE_ORDER:
        mask = stages == stage
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.65,
                        c=[STAGE_COLORS[stage]], label=f"Stage {stage} (n={mask.sum()})")
    axes[1].set_title("AJCC stage")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def run_analysis(mat, labels, label, gene_frequency_threshold, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    freq = mat.sum(axis=1) / mat.shape[1]
    filtered = mat.loc[freq >= gene_frequency_threshold]
    x = filtered.T.values.astype(bool)
    stages = labels["stage"].values

    distances = pdist(x, metric="jaccard")
    z = linkage(distances, method=LINKAGE_METHOD)
    cluster_ids = fcluster(z, t=N_CLUSTERS, criterion="maxclust")
    run_labels = labels.copy()
    run_labels["cluster"] = cluster_ids

    metrics = score(stages, cluster_ids)
    metrics["crosstab"].to_csv(out_dir / "cluster_vs_stage_summary.csv")
    run_labels[["sample", "dataset", "stage", "cluster"]].to_csv(
        out_dir / "sample_cluster_labels.csv", index=False
    )

    metrics_row = pd.DataFrame([{
        "analysis": label,
        "method": LINKAGE_METHOD,
        "n_samples": len(run_labels),
        "n_genes": filtered.shape[0],
        "gene_frequency_threshold": gene_frequency_threshold,
        "n_clusters": N_CLUSTERS,
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "purity": metrics["purity"],
    }])
    metrics_row.to_csv(out_dir / "metrics.csv", index=False)
    metrics_row.to_string(out_dir / "metrics.txt", index=False, float_format=lambda x: f"{x:.4f}")

    plot_confusion(metrics["crosstab"], out_dir / "confusion_matrix.png")
    plot_pca(x, cluster_ids, stages, out_dir / "pca_comparison.png")

    return metrics_row

def main():
    mat, labels = load_data()
    rows = []
    for label, threshold, out_dir in RUNS:
        row = run_analysis(mat, labels, label, threshold, out_dir)
        rows.append(row)
        print(row.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"Wrote {out_dir}")

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv("clustered_datasets/jaccard_vs_stage_summary.csv", index=False)


if __name__ == "__main__":
    main()
