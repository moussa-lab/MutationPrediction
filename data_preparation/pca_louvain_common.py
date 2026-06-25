# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

from pathlib import Path

import community as community_louvain
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import kneighbors_graph


STAGED_CSV = Path("clustered_datasets/by_stage/staged_only.csv")
LABELS_CSV = Path("clustered_datasets/sample_labels.csv")
STAGE_ORDER = ["I", "II", "III", "IV"]
STAGE_COLORS = {"I": "#2ecc71", "II": "#3498db", "III": "#e67e22", "IV": "#e74c3c"}
K_NEIGHBORS = 15
RANDOM_SEED = 42


sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11})


def find_elbow(values):
    x = np.linspace(0, 1, len(values))
    y = (values - values.min()) / (values.max() - values.min() + 1e-12)
    points = np.column_stack([x, y])
    start, end = points[0], points[-1]
    line = end - start
    line = line / np.linalg.norm(line)
    distances = []
    for point in points:
        projection = np.dot(point - start, line)
        closest = start + projection * line
        distances.append(np.linalg.norm(point - closest))
    return int(np.argmax(distances))


def score_clusters(stage_labels, cluster_labels):
    stage_ids = np.array([STAGE_ORDER.index(s) for s in stage_labels])
    crosstab = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series(stage_labels, name="stage"),
    ).reindex(columns=STAGE_ORDER, fill_value=0)
    return {
        "ari": adjusted_rand_score(stage_ids, cluster_labels),
        "nmi": normalized_mutual_info_score(stage_ids, cluster_labels, average_method="arithmetic"),
        "purity": crosstab.max(axis=1).sum() / len(stage_labels),
        "crosstab": crosstab,
        "n_clusters": len(set(cluster_labels)),
    }


def plot_scree(var_explained, cumulative, n_pcs, out_path):
    max_pcs = len(var_explained)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, min(51, max_pcs + 1)), var_explained[:50] * 100, color="#4C78A8")
    axes[0].axvline(n_pcs, color="#D62728", linestyle="--", label=f"PC{n_pcs}")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("Scree")
    axes[0].legend()

    axes[1].plot(range(1, max_pcs + 1), cumulative * 100, "o-", markersize=3, color="#2C3E50")
    axes[1].axvline(n_pcs, color="#D62728", linestyle="--", label=f"PC{n_pcs}")
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative variance")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_silhouette(k_values, scores, best_k, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, scores, "o-", color="#2C3E50")
    ax.axvline(best_k, color="#D62728", linestyle="--", label=f"k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Ward silhouette scan")
    ax.set_xticks(k_values)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_clusters(coords, cluster_ids, stages, ev_pct, title, out_path):
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
        ax.set_xlabel(f"PC1 ({ev_pct[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev_pct[1]:.1f}%)")
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion(crosstab, method, out_path):
    fig, ax = plt.subplots(figsize=(8, max(4, 0.7 * len(crosstab))))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Samples"},
    )
    ax.set_title(method)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def load_staged_data():
    if not STAGED_CSV.exists():
        raise FileNotFoundError(f"{STAGED_CSV} not found. Run create_clustered_datasets.py first.")
    mat = pd.read_csv(STAGED_CSV, index_col=0)
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[labels["sample"].isin(mat.columns) & (labels["stage"] != "Unknown")]
    labels = labels.set_index("sample").reindex(mat.columns).reset_index().rename(columns={"index": "sample"})
    return mat, labels


def run_analysis(gene_frequency_threshold, output_dir, label, pca_random_state=RANDOM_SEED):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat, labels = load_staged_data()
    if gene_frequency_threshold > 0:
        freq = mat.sum(axis=1) / mat.shape[1]
        mat = mat.loc[freq >= gene_frequency_threshold]

    x = mat.T.values
    stages = labels["stage"].values
    max_pcs = min(100, x.shape[0], x.shape[1])
    pca = PCA(n_components=max_pcs, random_state=pca_random_state)
    x_pca = pca.fit_transform(x)
    var_explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_explained)
    n_pcs = find_elbow(var_explained) + 1
    x_reduced = x_pca[:, :n_pcs]

    z = linkage(x_reduced, method="ward")
    k_values = list(range(2, 16))
    silhouette_scores = []
    for k in k_values:
        ids = fcluster(z, t=k, criterion="maxclust")
        silhouette_scores.append(
            silhouette_score(
                x_reduced,
                ids,
                metric="euclidean",
                sample_size=min(5000, len(x_reduced)),
                random_state=RANDOM_SEED,
            )
        )
    best_k = k_values[int(np.argmax(silhouette_scores))]
    best_silhouette = max(silhouette_scores)

    knn = kneighbors_graph(x_reduced, n_neighbors=K_NEIGHBORS, mode="connectivity", include_self=False)
    knn = knn + knn.T
    knn[knn > 1] = 1

    graph_nx = nx.from_scipy_sparse_array(knn)
    louvain_partition = community_louvain.best_partition(graph_nx, random_state=RANDOM_SEED)
    louvain_ids = np.array([louvain_partition[i] for i in range(len(x_reduced))])

    graph_ig = ig.Graph(n=len(x_reduced), edges=list(graph_nx.edges()), directed=False)
    leiden_partition = leidenalg.find_partition(graph_ig, leidenalg.ModularityVertexPartition, seed=RANDOM_SEED)
    leiden_ids = np.array(leiden_partition.membership)

    ward_ids = fcluster(z, t=best_k, criterion="maxclust")
    clusterings = {
        "louvain": louvain_ids,
        "leiden": leiden_ids,
        "ward": ward_ids,
    }

    rows = []
    for method, ids in clusterings.items():
        scores = score_clusters(stages, ids)
        rows.append({
            "analysis": label,
            "method": method,
            "n_genes": mat.shape[0],
            "n_samples": mat.shape[1],
            "gene_frequency_threshold": gene_frequency_threshold,
            "n_pcs": n_pcs,
            "cumvar_at_elbow": cumulative[n_pcs - 1],
            "pc1_tmb_pearson": pearsonr(x_pca[:, 0], labels["tmb"])[0],
            "best_ward_k": best_k,
            "best_ward_silhouette": best_silhouette,
            "n_clusters": scores["n_clusters"],
            "ari": scores["ari"],
            "nmi": scores["nmi"],
            "purity": scores["purity"],
        })
        scores["crosstab"].to_csv(out_dir / f"{method}_stage_crosstab.csv")
        plot_clusters(
            x_pca[:, :2],
            ids,
            stages,
            var_explained[:2] * 100,
            f"{method.title()} ({label})",
            out_dir / f"{method}_pca.png",
        )
        plot_confusion(scores["crosstab"], f"{method.title()} vs stage", out_dir / f"confusion_{method}.png")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    metrics.to_string(out_dir / "metrics.txt", index=False, float_format=lambda x: f"{x:.4f}")
    plot_scree(var_explained, cumulative, n_pcs, out_dir / "scree_elbow.png")
    plot_silhouette(k_values, silhouette_scores, best_k, out_dir / "silhouette_k_scan.png")

    print(f"{label}: {mat.shape[1]} samples, {mat.shape[0]} genes, {n_pcs} PCs")
    print(metrics[["method", "n_clusters", "ari", "nmi", "purity"]].to_string(index=False))
    print(f"Wrote {out_dir}")
