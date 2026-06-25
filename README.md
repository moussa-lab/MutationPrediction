# Mutation Prediction

This repository contains the raw-data preparation, graph construction, bootstrap consensus graph, sequence-model evaluation, and figure-generation code used for the paper.

Run commands from the repository root unless a script says otherwise.

The full path is raw cBioPortal study data -> binarized per-study matrices -> `study_data/combined_union.csv` -> graph construction -> mutation-prediction models -> figures. The bootstrap consensus graph uses `bootstrap/coad2(in).csv` and the scripts in `bootstrap/`.

## Data

`data_preparation/` contains the raw cBioPortal study folders and the scripts needed to regenerate the combined and subset matrices.

`study_data/` contains the prepared matrices used by the graph and model pipeline:

- `COAD2.mat`, `DFCI2.mat`, `MGI2.mat`: Auslander-style matrices.
- `combined_union.csv.zip`: compressed 9-study mutation matrix. Unzip this to create `study_data/combined_union.csv` before primary graph/model runs.
- `stage_i.csv`, `stage_ii.csv`, `stage_iii.csv`, `stage_iv.csv`, `stage_unknown.csv`, `staged_only.csv`: stage-stratified matrices.
- `louvain_largest_community.csv`, `louvain_second_largest_community.csv`: community subset matrices.

Unzip `study_data/combined_union.csv.zip` before running workflows that read `study_data/combined_union.csv`.

## Raw Data Preparation

Skip this section if using the prepared files already in `study_data/`. To rebuild from the raw study folders, run the preprocessing scripts from `data_preparation/`:

```bash
cd data_preparation
python3 binarize_mutations.py
python3 combine_matrices.py
python3 create_clustered_datasets.py
python3 jaccard_cluster_vs_stage.py
python3 pca_louvain_cluster.py
python3 pca_louvain_nofilter.py
python3 extract_louvain.py
cd ..
```

This produces:

- `data_preparation/combined_union.csv`
- `data_preparation/clustered_datasets/by_stage/*.csv`
- `data_preparation/louvain_largest_community.csv`
- `data_preparation/louvain_second_largest_community.csv`
- clustering/statistics outputs under `data_preparation/clustered_datasets/`

`jaccard_cluster_vs_stage.py` writes both the full-gene and 5% gene-frequency
Jaccard stage-clustering summaries.

To run the graph and model code from the regenerated files, place the generated matrices in `study_data/`:

```bash
cp data_preparation/combined_union.csv study_data/combined_union.csv
cp data_preparation/clustered_datasets/by_stage/*.csv study_data/
cp data_preparation/louvain_largest_community.csv study_data/
cp data_preparation/louvain_second_largest_community.csv study_data/
```

## Primary Graph

Edit the configuration block at the top of `graph_generation.R`, then run:

```bash
Rscript graph_generation.R
```

Important settings:

- `USE_MODE <- "csv"` for `study_data/combined_union.csv`.
- `THRESHOLD_DIFF <- 0.02`
- `MIN_COOCC <- 1`
- `PROB_MIN <- 0.02`
- `CYCLE_BREAK_METHOD <- "feedback_arc_set"`

Primary outputs are written under `outputs/<TAG>/`:

- `graph_raw_<TAG>.rds`
- `graph_dag_<TAG>.rds`
- `edges_<TAG>.csv`
- `longest_path_edges_<TAG>.txt`
- `longest_path_weighted_<TAG>.txt`
- `topo_order_<TAG>.txt`
- `meta_<TAG>.csv`

Large graph objects and edge lists are generated outputs.

## Bootstrap T=103 Graph

The bootstrap workflow is documented in `bootstrap/README.md`. The source scripts are:

- `bootstrap/CreateBootstrapDatasets.R`
- `bootstrap/create_bootstrap_graph.R`
- `bootstrap/minimum_t.R`

Run:

```bash
Rscript bootstrap/CreateBootstrapDatasets.R
Rscript bootstrap/create_bootstrap_graph.R
Rscript bootstrap/minimum_t.R
```

The bootstrap scripts generate resampled datasets, consensus edges, RDS graph files, and exported path files.

## Mutation Prediction

`mutation_prediction.py` trains and evaluates the LSTM, dilated CNN, and shared LSTM-attention models.

Core arguments:

- `--tag`: output tag under `outputs/<TAG>/`
- `--lp_txt`: longest-path gene list (path from repo root to file)
- `--topo_txt`: DAG topological order (path from repo root to file)
- `--combined_csv`: gene x sample matrix for CSV mode (path from repo root to file)
- `--mode newcsv_80_20`: 80/20 split on the supplied CSV (path from repo root to file)
- `--context lp_only` or `--context full`
- `--model_type lstm`, `dilated_cnn`, or `shared_lstm_attn`
- `--tune_threshold --tune_metric f1`: validation-set threshold tuning
- omit `--tune_threshold` or use eval-only copied models for fixed `0.5` threshold runs

Example:

```bash
python mutation_prediction.py \
  --tag paper_example_w_fullcontext_lstm_seed1 \
  --lp_txt outputs/<GRAPH_TAG>/longest_path_weighted_<GRAPH_TAG>.txt \
  --topo_txt outputs/<GRAPH_TAG>/topo_order_<GRAPH_TAG>.txt \
  --combined_csv study_data/combined_union.csv \
  --mode newcsv_80_20 \
  --context full \
  --model_type lstm \
  --lstm_units 5 \
  --epochs 10 \
  --batch_size 27 \
  --seed 1 \
  --macro_metrics \
  --tune_threshold \
  --tune_metric f1 \
  --val_frac 0.2 \
  --th_grid 0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75
```

Model files and metrics are generated under `outputs/<TAG>/`.

The paper tables report mean and standard deviation across 15 random seeds.
Run the same graph/model configuration once per seed, then summarize the
per-seed metrics for the reported table values.

## Figures

Publication figure scripts:

- `plot_graph_pathway.R`: graph/pathway visualization from saved graph RDS and path RDS/TXT artifacts.
- `plot_mutation_heatmap.py`: mutation heatmap panels for path gene lists across the mutation matrix.

Example heatmap command:

```bash
python plot_mutation_heatmap.py \
  --data_csv study_data/combined_union.csv \
  --lp_list outputs/<GRAPH_TAG>/longest_path_edges_<GRAPH_TAG>.txt \
  --lp_list2 outputs/<GRAPH_TAG>/longest_path_weighted_<GRAPH_TAG>.txt \
  --output outputs/<GRAPH_TAG>/figure_mutation_heatmap.pdf
```

The figure scripts save publication formats at multiple resolutions.

## Dependencies

R packages:

- `R.matlab`
- `Matrix`
- `data.table`
- `future`
- `progressr`
- `igraph`
- `dplyr`
- `pbapply`
- `ggplot2`
- `ggraph`
- `ggrepel`
- `pheatmap`
- `RColorBrewer`
- `tidyverse`

Python packages:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `tensorflow` / `keras`
- `matplotlib`
- `seaborn`
- `networkx`
- `python-louvain`
- `igraph`
- `leidenalg`

Install Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
