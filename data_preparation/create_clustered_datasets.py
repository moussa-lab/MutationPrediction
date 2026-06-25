# Authors:
#   Collin Sumrell, School of Computer Science, University of Oklahoma, Norman, OK, USA
#   Marmar Moussa, School of Computer Science and Stephenson School of 
#       Biomedical Engineering, University of Oklahoma, Norman, OK, USA

"""Create stage and TMB subsets from the combined mutation matrix."""

import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
COMBINED_CSV = Path("combined_union.csv")
SAMPLE_MAP_CSV = Path("sample_to_dataset.csv")
OUT_DIR = Path("clustered_datasets")

# Dataset (relative clinical patient file, stage column name)
# None for col means no usable stage info
STAGE_SOURCES = {
    "coad_tcga_gdc": ("coad_tcga_gdc/data_clinical_patient.txt", "PATH_STAGE"),
    "coad_cptac_2019": ("coad_cptac_2019/data_clinical_patient.txt", "STAGE"),
    "coad_silu_2022": ("coad_silu_2022/data_clinical_patient.txt", "AJCC_PATH_STAGE"),
    "coadread_cass_2020": ("coadread_cass_2020/data_clinical_patient.txt", "STAGE"),
    "coadread_tcga_pan_can_atlas_2018": (
        "coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
    ),
    "coadread_tcga_pub": ("coadread_tcga_pub/data_clinical_patient.txt", None),
    "coad_cptac_gdc": None,
    "coadread_dfci_2016": None,
    "coadread_genentech": None,
}


def normalize_stage(raw) -> str | None:
    """Convert heterogeneous stage representations to I / II / III / IV."""
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in ("", "NA", "nan", "[Not Available]", "[Not Applicable]"):
        return None

    if s in ("1", "1.0"):
        return "I"
    if s in ("2", "2.0"):
        return "II"
    if s in ("3", "3.0"):
        return "III"
    if s in ("4", "4.0"):
        return "IV"

    s_upper = s.upper()
    if re.search(r"STAGE\s*IV|^IV", s_upper):
        return "IV"
    if re.search(r"STAGE\s*III|^III", s_upper):
        return "III"
    if re.search(r"STAGE\s*II|^II", s_upper):
        return "II"
    if re.search(r"STAGE\s*I\b|^I\b|^IA\b|^IB\b", s_upper):
        return "I"

    return None


def patient_id_from_sample(sample_key: str, dataset: str) -> list[str]:
    # Generate candidate patient IDs from a sample key.
    # Returns a list of IDs to try
    bare = sample_key.split("||", 1)[1] if "||" in sample_key else sample_key
    candidates = [bare]

    if bare.startswith("TCGA-"):
        candidates.append(bare[:12])

    if "silu" in dataset.lower():
        parts = bare.rsplit("-", 2)
        if len(parts) >= 3:
            candidates.append(parts[0])

    return candidates


def load_stage_map() -> dict[str, str]:
    # Parse clinical files and return patient_id to stage mapping.
    patient_stages: dict[str, str] = {}

    for dataset, source_info in STAGE_SOURCES.items():
        if source_info is None:
            continue
        rel_path, col = source_info
        if col is None:
            continue

        clin_path = DATA_DIR / rel_path
        if not clin_path.exists():
            print(f"  [WARN] Clinical file not found: {clin_path}")
            continue

        try:
            df = pd.read_csv(clin_path, sep="\t", comment="#")
        except Exception as e:
            print(f"  [WARN] Cannot read {clin_path}: {e}")
            continue

        if col not in df.columns:
            print(f"  [WARN] Column '{col}' not in {clin_path.name}. "
                  f"Available: {[c for c in df.columns if 'STAGE' in c.upper()]}")
            continue

        count = 0
        for _, row in df.iterrows():
            pid = str(row["PATIENT_ID"]).strip()
            stage = normalize_stage(row[col])
            if stage:
                patient_stages[pid] = stage
                count += 1

        print(f"  [{dataset}] {count} patients with stage info (col={col})")

    return patient_stages


def assign_stages(
    s2d: pd.DataFrame, patient_stages: dict[str, str]
) -> pd.Series:
    # Return a Series (indexed like s2d) with stage labels or 'Unknown'.
    stages = []
    for _, row in s2d.iterrows():
        sample_key = row["sample"]
        dataset = row["dataset"]
        candidates = patient_id_from_sample(sample_key, dataset)

        stage = None
        for cand in candidates:
            if cand in patient_stages:
                stage = patient_stages[cand]
                break

        stages.append(stage if stage else "Unknown")

    return pd.Series(stages, index=s2d.index, name="stage")


def assign_tmb_tertiles(combined: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # Compute TMB per sample and assign tertile labels.
    tmb = combined.sum(axis=0).astype(int)
    tmb.name = "tmb"

    t33 = tmb.quantile(1 / 3)
    t67 = tmb.quantile(2 / 3)

    def tertile_label(val):
        if val <= t33:
            return "Low"
        elif val <= t67:
            return "Medium"
        else:
            return "High"

    tmb_group = tmb.map(tertile_label)
    tmb_group.name = "tmb_group"

    print(f"\n  TMB tertile boundaries: Low ≤ {t33:.0f} < Medium ≤ {t67:.0f} < High")
    print(f"  TMB stats: min={tmb.min()}, median={tmb.median():.0f}, "
          f"mean={tmb.mean():.0f}, max={tmb.max()}")

    return tmb, tmb_group


def write_submatrix(combined: pd.DataFrame, sample_keys: list[str],
                    out_path: Path, label: str):
    # Write a genes x samples sub-matrix CSV.
    if not sample_keys:
        print(f"    [{label}] 0 samples - skipped")
        return
    sub = combined[sample_keys]
    # Drop genes that are all-zero in this subset (optional, reduces file size)
    sub = sub.loc[sub.sum(axis=1) > 0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path)
    print(f"    [{label}] {sub.shape[1]} samples, {sub.shape[0]} genes -> {out_path.name}")


def plot_composition(labels: pd.DataFrame, group_col: str, title: str, out_path: Path):
    # Bar chart showing dataset composition within each group.
    ct = pd.crosstab(labels[group_col], labels["dataset"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ct.plot(kind="bar", stacked=True, ax=axes[0], colormap="tab20", width=0.8)
    axes[0].set_title(f"{title} - Sample Counts", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_xlabel(group_col.replace("_", " ").title())
    axes[0].legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(True, alpha=0.3, axis="y")

    ct_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20", width=0.8)
    axes[1].set_title(f"{title} - Dataset Composition (%)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Percentage of Group")
    axes[1].set_xlabel(group_col.replace("_", " ").title())
    axes[1].legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved plot: {out_path.name}")


def main():
    if not COMBINED_CSV.exists():
        print(f"ERROR: {COMBINED_CSV} not found. Run combine_matrices.py first.")
        sys.exit(1)

    print("Creating stage and TMB subsets")
    combined = pd.read_csv(COMBINED_CSV, index_col=0)
    print(f"  combined: {combined.shape[0]} genes x {combined.shape[1]} samples")

    s2d = pd.read_csv(SAMPLE_MAP_CSV)
    s2d = s2d[s2d["sample"].isin(combined.columns)].reset_index(drop=True)

    print("Parsing clinical stage metadata")
    patient_stages = load_stage_map()
    print(f"  patient-stage mappings: {len(patient_stages)}")

    stages = assign_stages(s2d, patient_stages)
    s2d["stage"] = stages

    print("Stage distribution:")
    for stage_label in ["I", "II", "III", "IV", "Unknown"]:
        n = (stages == stage_label).sum()
        pct = 100 * n / len(stages)
        print(f"  {stage_label:>7s}: {n:>5d} ({pct:5.1f}%)")

    print("Computing TMB tertiles")
    tmb, tmb_group = assign_tmb_tertiles(combined)
    s2d["tmb"] = s2d["sample"].map(tmb).astype(int)
    s2d["tmb_group"] = s2d["sample"].map(tmb_group)

    print("TMB distribution:")
    for grp in ["Low", "Medium", "High"]:
        n = (s2d["tmb_group"] == grp).sum()
        tmb_range = s2d.loc[s2d["tmb_group"] == grp, "tmb"]
        print(f"  {grp:>6s}: {n:>5d} (TMB range: {tmb_range.min()}-{tmb_range.max()})")

    print("Writing subset matrices")
    for stage_label in ["I", "II", "III", "IV", "Unknown"]:
        samples = s2d.loc[s2d["stage"] == stage_label, "sample"].tolist()
        fname = f"stage_{stage_label.lower() if stage_label != 'Unknown' else 'unknown'}.csv"
        write_submatrix(combined, samples, OUT_DIR / "by_stage" / fname, f"Stage {stage_label}")

    staged_samples = s2d.loc[s2d["stage"] != "Unknown", "sample"].tolist()
    write_submatrix(combined, staged_samples, OUT_DIR / "by_stage" / "staged_only.csv", "Staged Only")

    for grp in ["Low", "Medium", "High"]:
        samples = s2d.loc[s2d["tmb_group"] == grp, "sample"].tolist()
        write_submatrix(combined, samples, OUT_DIR / "by_tmb" / f"tmb_{grp.lower()}.csv", f"TMB {grp}")

    labels_path = OUT_DIR / "sample_labels.csv"
    s2d.to_csv(labels_path, index=False)
    print(f"  labels: {labels_path}")

    summary_rows = []
    for stage_label in ["I", "II", "III", "IV", "Unknown"]:
        n = (s2d["stage"] == stage_label).sum()
        summary_rows.append({"axis": "stage", "group": stage_label, "n_samples": n})
    for grp in ["Low", "Medium", "High"]:
        n = (s2d["tmb_group"] == grp).sum()
        tmb_range = s2d.loc[s2d["tmb_group"] == grp, "tmb"]
        summary_rows.append({
            "axis": "tmb",
            "group": grp,
            "n_samples": n,
            "tmb_min": int(tmb_range.min()),
            "tmb_max": int(tmb_range.max()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "clustering_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  summary: {summary_path}")

    plot_composition(s2d, "stage", "Stage-Based Clustering", OUT_DIR / "composition_by_stage.png")
    plot_composition(s2d, "tmb_group", "TMB-Based Clustering", OUT_DIR / "composition_by_tmb.png")

    print(f"Wrote {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
