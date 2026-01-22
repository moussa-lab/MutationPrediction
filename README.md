# Mutation Prediction

This project implements mutation prediction using sequence models (LSTM and dilated CNN) on directed acyclic graphs (DAGs) constructed from gene mutation co-occurrence patterns.

## Overview

The pipeline consists of two main stages:

1. **Graph Generation** (`graph_generation.R`), which constructs a DAG from mutation data using conditional probability-based edge weights
2. **Mutation Prediction** (`mutation_prediction.py`), which trains sequence models to predict gene mutations based on predecessor mutations in the DAG

---

## Data

### Combined Union Dataset

The primary dataset used in this project is `combined_union.csv`, which contains mutation data with genes as rows and samples as columns. This dataset consolidates samples from 9 major colorectal cancer studies.

> Note: `combined_union.csv` is too large to include directly in the repository. It is provided as a compressed file (`combined_union.csv.zip`). You should unzip it before running the pipeline.

**Datasets included in `combined_union.csv`:**

- coadread_dfci_2016 - Giannakis et al. (2016). Genomic Correlates of Immune-Cell Infiltrates in Colorectal Carcinoma. Cell Reports.
- coad_tcga_gdc - Heath et al. (2021). The NCI Genomic Data Commons. Nature Genetics.
- coad_cptac_gdc - Heath et al. (2021). The NCI Genomic Data Commons. Nature Genetics.
- coadread_tcga_pan_can_atlas_2018 - Heath et al. (2021). The NCI Genomic Data Commons. Nature Genetics.
- coad_silu_2022 - Roelands et al. (2023). An integrated tumor, immune and microbiome atlas of colon cancer. Nature Medicine.
- coadread_cass_2020 - Li et al. (2020). Integrated Omics of Metastatic Colorectal Cancer. Cancer Cell.
- coad_cptac_2019 - Vasaikar et al. (2019). Proteogenomic Analysis of Human Colon Cancer. Cell.
- coad_tcga_pub - Cancer Genome Atlas Network (2012). Comprehensive molecular characterization of human colon and rectal cancer. Nature.
- coadread_genentech - Seshagiri et al. (2012). Recurrent R-spondin fusions in colon cancer. Nature.

### COAD2, DFCI2, MGI2 (.mat files)

The `.mat` files located in `study_data/` are pre-processed matrices derived from the study by Auslander et al. (2019):

- Auslander, N., Wolf, Y. I., & Koonin, E. V. (2019). In silico learning of tumor evolution through mutational time series. Proceedings of the National Academy of Sciences.

---

## Graph Generation (`graph_generation.R`)

Constructs a directed graph from mutation co-occurrence data using asymmetric conditional probabilities.

### Key Parameters

| Parameter        | Default | Description                                          |
| ---------------- | ------- | ---------------------------------------------------- |
| `USE_MODE`       | `"csv"` | Data source: `"coad"`, `"coad_dfci_mgi"`, or `"csv"` |
| `THRESHOLD_DIFF` | `0.02`  | Minimum \|P(i\|j) - P(j\|i)\| for edge inclusion     |
| `MIN_COOCC`      | `1`     | Minimum co-occurrence count (c11)                    |
| `PROB_MIN`       | `0.02`  | Floor for conditional probabilities                  |

### Outputs

Results are saved to `outputs/<TAG>/`:

- `graph_raw_<TAG>.rds` — Original directed graph (may contain cycles)
- `graph_dag_<TAG>.rds` — DAG after cycle removal
- `edges_<TAG>.csv` — Edge list with weights
- `longest_path_edges_<TAG>.txt` — Longest path (unweighted, by edge count)
- `longest_path_weighted_<TAG>.txt` — Longest path (weighted by `w_diff`)
- `topo_order_<TAG>.txt` — Topological ordering of the DAG

### Usage

```r
# Edit configuration at top of script, then:
source("graph_generation.R")
```

Ensure necessary packages are installed in your environment.

---

## Mutation Prediction (`mutation_prediction.py`)

Trains LSTM or dilated CNN models to predict whether a gene is mutated, given the mutation status of its predecessors in the DAG.

### Key Arguments

| Argument               | Default              | Description                                                        |
| ---------------------- | -------------------- | ------------------------------------------------------------------ |
| `--tag`                | (required)           | Run identifier; outputs saved to `outputs/<TAG>/`                  |
| `--lp_txt`             | (required)           | Path to longest path genes file                                    |
| `--topo_txt`           | (required)           | Path to topological order file                                     |
| `--combined_csv`       | `combined_union.csv` | Mutation matrix (genes × samples)                                  |
| `--mode`               | `coad_vs_dfci_mgi`   | Data mode: `coad_vs_dfci_mgi`, `old_80_20`, `newcsv_80_20`         |
| `--context`            | `full`               | `full` (all TOPO predecessors) or `lp_only` (LP predecessors only) |
| `--model_type`         | `lstm`               | Model architecture: `lstm` or `dilated_cnn`                        |
| `--lstm_units`         | `5`                  | Number of LSTM units                                               |
| `--cnn_filters`        | `32`                 | Number of CNN filters                                              |
| `--cnn_dilation_rates` | `1,2,4`              | Dilation rates for CNN                                             |
| `--epochs`             | `10`                 | Training epochs                                                    |
| `--batch_size`         | `27`                 | Batch size                                                         |
| `--seed`               | `123`                | Random seed for reproducibility                                    |
| `--tune_threshold`     | `False`              | Tune classification threshold per gene                             |

### Outputs

Results are saved to `outputs/<TAG>/`:

- `models/model_t_<k>.keras` — Trained model for each LP gene
- `accuracy_matrix.npy` — Per-gene, per-sample prediction correctness
- `confusion_matrix.csv` / `standard_confusion_matrix.csv` — Confusion matrices
- `metrics_table.csv` / `standard_metrics_table.csv` — Performance metrics including AUC
- `run_summary.json` — Full run configuration and results
- `split_indices.pkl` — Train/test split indices

### Example Usage

```bash
# Basic LSTM run with CSV data
python mutation_prediction.py \
    --tag CSV_* \
    --lp_txt outputs/CSV_*/longest_path_weighted_*.txt \
    --topo_txt outputs/CSV_*/topo_order_*.txt \
    --combined_csv combined_union.csv \
    --mode newcsv_80_20 \
    --context full

# Dilated CNN
python mutation_prediction.py \
    --tag cnn_experiment \
    --lp_txt outputs/CSV_*/longest_path_weighted_*.txt \
    --topo_txt outputs/CSV_*/topo_order_*.txt \
    --model_type dilated_cnn \
    --cnn_filters 64 \
    --epochs 50

# Evaluation only (no training)
python mutation_prediction.py \
    --tag eval_run \
    --lp_txt ... \
    --topo_txt ... \
    --eval_only
```

---

## Dependencies

### R

- `R.matlab`
- `Matrix`
- `igraph`

### Python

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `tensorflow` / `keras`

Install Python dependencies:

```bash
pip install numpy pandas scipy scikit-learn tensorflow
```

(Plus whatever else may come up if you still encounter errors. We'd recommend [doing the python parts of this in a venv](https://docs.python.org/3/library/venv.html), though you know what python workflow works best for you.)

---

## Project Structure

```
MutationPrediction/
├── mutation_prediction.py # Main prediction script
├── graph_generation.R # Graph construction script
├── study_data/ # .mat files (COAD2, DFCI2, MGI2) and CSV file
└── outputs/ # Generated outputs by tag
    └── <TAG>/
        ├── models/
        ├── graph_*.rds
        ├── edges_*.csv
        ├── longest_path_*.txt
        ├── topo_order_*.txt
        └── run_summary.json
```
