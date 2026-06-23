# Mutation Prediction

This repository contains the graph construction, bootstrap consensus graph, sequence-model evaluation, and figure-generation code used for the paper.

Run commands from the repository root unless a script says otherwise.

The primary graph-derived results use `study_data/combined_union.csv` after unzipping `study_data/combined_union.csv.zip`. The bootstrap consensus graph uses `bootstrap/coad2(in).csv` and the scripts in `bootstrap/`.

## Data

`study_data/` contains the source matrices and subset matrices used by the paper pipeline:

- `COAD2.mat`, `DFCI2.mat`, `MGI2.mat`: Auslander-style matrices.
- `combined_union.csv.zip`: compressed 9-study mutation matrix. Unzip this to create `study_data/combined_union.csv` before primary graph/model runs.
- `stage_i.csv`, `stage_ii.csv`, `stage_iii.csv`, `stage_iv.csv`, `stage_unknown.csv`, `staged_only.csv`: stage-stratified matrices.
- `louvain_largest_community.csv`, `louvain_second_largest_community.csv`: community subset matrices.

Unzip `study_data/combined_union.csv.zip` before running workflows that read `study_data/combined_union.csv`.

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
- `igraph`
- `data.table`
- `dplyr`
- `ggplot2`
- `ggraph`
- `ggrepel`
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

Install Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn tensorflow matplotlib seaborn
```
