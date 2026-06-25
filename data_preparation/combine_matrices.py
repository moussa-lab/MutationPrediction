# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


GENERATED = {
    "combined_union.csv",
    "sample_to_dataset.csv",
    "dropped_duplicates.csv",
}


def read_matrix(path):
    df = pd.read_csv(path, index_col=0)
    if not df.index.is_unique:
        df = df.groupby(level=0).max()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return pd.DataFrame((df.values > 0).astype(np.uint8), index=df.index, columns=df.columns)


def input_csvs(top, pattern, output_path):
    return sorted(
        p for p in top.iterdir()
        if p.is_file()
        and p.match(pattern)
        and p.name not in GENERATED
        and p.resolve() != output_path.resolve()
    )


def choose_duplicate_samples(datasets):
    counts = {}
    sample_to_dataset = {}
    sample_to_barcode = {}

    for dataset, df, mapping in datasets:
        col_counts = df.sum(axis=0).astype(int)
        for col in df.columns:
            counts[col] = int(col_counts[col])
            sample_to_dataset[col] = dataset
            sample_to_barcode[col] = mapping[col]

    barcode_groups = defaultdict(list)
    for internal, barcode in sample_to_barcode.items():
        barcode_groups[barcode].append(internal)

    keep = set()
    dropped = []
    for barcode, internals in barcode_groups.items():
        if len(internals) == 1:
            keep.add(internals[0])
            continue
        ranked = sorted(internals, key=lambda c: (-counts[c], sample_to_dataset[c], c))
        keep.add(ranked[0])
        for col in ranked[1:]:
            dropped.append({
                "barcode": barcode,
                "kept_internal": ranked[0],
                "kept_dataset": sample_to_dataset[ranked[0]],
                "kept_mut_count": counts[ranked[0]],
                "dropped_internal": col,
                "dropped_dataset": sample_to_dataset[col],
                "dropped_mut_count": counts[col],
            })
    return keep, dropped


def main():
    parser = argparse.ArgumentParser(description="Build a union gene-by-sample mutation matrix.")
    parser.add_argument("--glob", default="*.csv")
    parser.add_argument("--out", default="combined_union.csv")
    parser.add_argument("--top", default=".")
    parser.add_argument("--no-prefix", action="store_true")
    args = parser.parse_args()

    top = Path(args.top)
    out_path = Path(args.out)
    csvs = input_csvs(top, args.glob, out_path)
    if not csvs:
        raise SystemExit("No per-study CSVs found.")

    datasets = []
    for path in csvs:
        df = read_matrix(path)
        internal_cols = [f"{path.stem}||{col}" for col in df.columns]
        mapping = dict(zip(internal_cols, df.columns))
        df.columns = internal_cols
        datasets.append((path.stem, df, mapping))
        print(f"{path.stem}: {df.shape[0]} genes x {df.shape[1]} samples")

    keep, dropped = choose_duplicate_samples(datasets)
    if dropped:
        pd.DataFrame(dropped).sort_values(["barcode", "dropped_dataset"]).to_csv(
            "dropped_duplicates.csv", index=False
        )

    parts = []
    sample_rows = []
    for dataset, df, mapping in datasets:
        cols = [col for col in df.columns if col in keep]
        if not cols:
            continue
        parts.append(df.loc[:, cols])
        for col in cols:
            sample_rows.append({
                "sample": mapping[col] if args.no_prefix else col,
                "dataset": dataset,
            })

    combined = pd.concat(parts, axis=1, join="outer").fillna(0).astype(np.uint8)
    if args.no_prefix:
        rename = {
            col: mapping[col]
            for _, _, mapping in datasets
            for col in mapping
            if col in combined.columns
        }
        combined = combined.rename(columns=rename)

    zero_rows = combined.sum(axis=1) == 0
    dropped_zero_rows = int(zero_rows.sum())
    if dropped_zero_rows:
        combined = combined.loc[~zero_rows]

    combined = combined.sort_index(axis=0).reindex(sorted(combined.columns), axis=1)
    combined.to_csv(out_path)
    pd.DataFrame(sample_rows).to_csv("sample_to_dataset.csv", index=False)

    print(f"combined: {combined.shape[0]} genes x {combined.shape[1]} samples -> {out_path}")
    if dropped_zero_rows:
        print(f"dropped all-zero genes after duplicate sample resolution: {dropped_zero_rows}")
    if dropped:
        print(f"dropped duplicate sample columns: {len(dropped)}")


if __name__ == "__main__":
    main()
