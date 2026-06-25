# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import community as community_louvain
import networkx as nx

RANDOM_SEED = 42

def main():
    COMBINED_CSV = Path("combined_union.csv")
    GENE_FREQ_THRESHOLD = 0.05

    print(f"Loading {COMBINED_CSV}...")
    mat = pd.read_csv(COMBINED_CSV, index_col=0)
    
    print("Filtering genes...")
    freq = mat.sum(axis=1) / mat.shape[1]
    keep_genes = freq[freq >= GENE_FREQ_THRESHOLD].index
    mat_filtered = mat.loc[keep_genes]
    X = mat_filtered.T.values  # (samples, genes)
    sample_names = mat_filtered.columns.tolist()

    def find_elbow(var_explained):
        n = len(var_explained)
        x = np.linspace(0, 1, n)
        y = (var_explained - var_explained.min()) / (var_explained.max() - var_explained.min() + 1e-12)
        coords = np.column_stack([x, y])
        line_start = coords[0]
        line_end = coords[-1]
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        line_unit = line_vec / line_len
        dists = []
        for pt in coords:
            vec = pt - line_start
            proj = np.dot(vec, line_unit)
            closest = line_start + proj * line_unit
            dists.append(np.linalg.norm(pt - closest))
        return int(np.argmax(dists))

    print("Running PCA...")
    pca_full = PCA(n_components=min(100, X.shape[0], X.shape[1]), random_state=RANDOM_SEED)
    X_pca_full = pca_full.fit_transform(X)
    var_explained = pca_full.explained_variance_ratio_
    elbow_idx = find_elbow(var_explained)
    n_pcs = elbow_idx + 1

    X_reduced = X_pca_full[:, :n_pcs]

    print(f"Running Louvain community detection on {n_pcs} PCs...")
    k_neighbors = 15
    knn = kneighbors_graph(X_reduced, n_neighbors=k_neighbors, mode="connectivity", include_self=False)
    knn_sym = knn + knn.T
    knn_sym[knn_sym > 1] = 1

    G_nx = nx.from_scipy_sparse_array(knn_sym)
    partition = community_louvain.best_partition(G_nx, random_state=RANDOM_SEED)
    louvain_ids = np.array([partition[i] for i in range(len(X_reduced))])

    unique_comms, counts = np.unique(louvain_ids, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    largest_comm = unique_comms[sorted_indices[0]]
    second_largest_comm = unique_comms[sorted_indices[1]]

    samples_largest = [sample_names[i] for i, comm in enumerate(louvain_ids) if comm == largest_comm]
    samples_second = [sample_names[i] for i, comm in enumerate(louvain_ids) if comm == second_largest_comm]

    print(f"Largest community: {largest_comm} with {len(samples_largest)} samples.")
    print(f"Second largest community: {second_largest_comm} with {len(samples_second)} samples.")

    print("Extracting datasets...")
    largest_df = mat[samples_largest]
    second_df = mat[samples_second]

    out1 = "louvain_largest_community.csv"
    out2 = "louvain_second_largest_community.csv"
    
    print(f"Saving to {out1}...")
    largest_df.to_csv(out1)
    print(f"Saving to {out2}...")
    second_df.to_csv(out2)
    
    print("Done!")

if __name__ == "__main__":
    main()
