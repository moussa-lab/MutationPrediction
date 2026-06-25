# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

from pca_louvain_common import run_analysis


if __name__ == "__main__":
    run_analysis(
        gene_frequency_threshold=0.0,
        output_dir="clustered_datasets/pca_louvain_nofilter",
        label="unfiltered",
    )
