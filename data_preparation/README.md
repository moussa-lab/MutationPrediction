# Data Preparation

Staging bundle for the mutation matrix, stage subsets, Louvain/Leiden
clustering, and clustering statistics used in the paper methods/results.

Run commands from this directory. Probably easiest to use the same venv
from the repo root.

## Contents

- `data/`: raw cBioPortal study folders. Download these separately; see below.
- `binarize_mutations.py`: converts each `data/*/data_mutations.txt` file into a
  binary gene-by-sample mutation matrix.
- `combine_matrices.py`: combines per-study matrices into `combined_union.csv`
  and writes `sample_to_dataset.csv`.
- `create_clustered_datasets.py`: uses clinical metadata to create stage and TMB
  subsets under `clustered_datasets/`.
- `jaccard_cluster_vs_stage.py`: compares full-gene and 5% filtered Jaccard
  hierarchical clusters to AJCC stage.
- `pca_louvain_cluster.py`: computes filtered PCA, Louvain, Leiden, and Ward
  clustering statistics for staged samples.
- `pca_louvain_nofilter.py`: repeats the PCA/Louvain/Leiden/Ward analysis using
  all genes.
- `pca_louvain_common.py`: shared implementation for the filtered and unfiltered
  PCA/community-detection runs.
- `extract_louvain.py`: extracts the two largest Louvain community matrices.

## Raw Data

The raw cBioPortal study archives are not included in this repository. Download
each study archive from cBioPortal and extract it into `data_preparation/data/`.
The archives should unpack directly into the study folders listed here:

- [`coad_cptac_2019`](https://www.cbioportal.org/study/summary?id=coad_cptac_2019)
- [`coad_cptac_gdc`](https://www.cbioportal.org/study/summary?id=coad_cptac_gdc)
- [`coad_silu_2022`](https://www.cbioportal.org/study/summary?id=coad_silu_2022)
- [`coad_tcga_gdc`](https://www.cbioportal.org/study/summary?id=coad_tcga_gdc)
- [`coadread_cass_2020`](https://www.cbioportal.org/study/summary?id=coadread_cass_2020)
- [`coadread_dfci_2016`](https://www.cbioportal.org/study/summary?id=coadread_dfci_2016)
- [`coadread_genentech`](https://www.cbioportal.org/study/summary?id=coadread_genentech)
- [`coadread_tcga_pan_can_atlas_2018`](https://www.cbioportal.org/study/summary?id=coadread_tcga_pan_can_atlas_2018)
- [`coadread_tcga_pub`](https://www.cbioportal.org/study/summary?id=coadread_tcga_pub)

For example (running from this directory):

```bash
mkdir -p data
tar -xzf ~/Downloads/coad_cptac_2019.tar.gz -C data
```

After extraction, `data/` should contain one directory per study, and each study
directory should include `data_mutations.txt` plus the clinical metadata files.

## Rebuild Order

```bash
python binarize_mutations.py
python combine_matrices.py
python create_clustered_datasets.py
python jaccard_cluster_vs_stage.py
python pca_louvain_cluster.py
python pca_louvain_nofilter.py
python extract_louvain.py
```

`extract_louvain.py` writes `louvain_largest_community.csv` and
`louvain_second_largest_community.csv`.

`jaccard_cluster_vs_stage.py` writes separate full-gene and 5% filtered
statistics under `clustered_datasets/`.
