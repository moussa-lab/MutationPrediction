# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
GENE_COL = "Hugo_Symbol"
SAMPLE_COL = "Tumor_Sample_Barcode"


def mutation_matrix(path):
    df = pd.read_csv(path, sep="\t", dtype=str, comment="#", low_memory=False)
    missing = {GENE_COL, SAMPLE_COL} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    pairs = df[[GENE_COL, SAMPLE_COL]].copy()
    pairs[GENE_COL] = pairs[GENE_COL].astype(str).str.strip()
    pairs[SAMPLE_COL] = pairs[SAMPLE_COL].astype(str).str.strip()
    pairs = pairs[(pairs[GENE_COL] != "") & (pairs[SAMPLE_COL] != "")]
    pairs = pairs.drop_duplicates()

    mat = pd.crosstab(index=pairs[GENE_COL], columns=pairs[SAMPLE_COL])
    mat = (mat > 0).astype("uint8")
    return mat.sort_index().sort_index(axis=1)


def main():
    mutation_files = sorted(DATA_DIR.glob("*/data_mutations.txt"))
    if not mutation_files:
        raise SystemExit("No data/*/data_mutations.txt files found.")

    for path in mutation_files:
        dataset = path.parent.name
        try:
            mat = mutation_matrix(path)
        except Exception as exc:
            print(f"{dataset}: skipped ({exc})")
            continue
        out_path = Path(f"{dataset}.csv")
        mat.to_csv(out_path)
        print(f"{dataset}: {mat.shape[0]} genes x {mat.shape[1]} samples -> {out_path}")


if __name__ == "__main__":
    main()
