import os, json, time, argparse, random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

import tensorflow as tf
import keras
from keras import layers, models, initializers, losses, metrics, callbacks

# Determinism defaults
SEED_DEFAULT = 123
os.environ.setdefault("PYTHONHASHSEED", str(SEED_DEFAULT))
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# I/O helpers
def read_longest_path_txt(path_txt: str) -> list[str]:
    with open(path_txt, "r") as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    return genes

# RDS helpers to get some metadata from the R graphs/objects.
# optional, and also didn't end up using it a ton so not sure how robust this is.
def try_read_rds_meta(path_rds: str):
    if not path_rds or not Path(path_rds).exists():
        return None
    try:
        import pyreadr
    except Exception:
        return None
    try:
        res = pyreadr.read_r(path_rds)
        r_obj = list(res.values())[0] if len(res.values()) else None
        if r_obj is None:
            return None
        if isinstance(r_obj, dict):
            if "meta" in r_obj and isinstance(r_obj["meta"], dict):
                return r_obj["meta"]
            if "params" in r_obj and isinstance(r_obj["params"], dict):
                return r_obj["params"]
            return {k: v for k, v in r_obj.items() if isinstance(v, (str, int, float))}
        meta = None
        if hasattr(r_obj, "attrs") and isinstance(r_obj.attrs, dict) and len(r_obj.attrs):
            meta = r_obj.attrs
        if (not meta) and hasattr(r_obj, "_metadata"):
            try:
                meta = {name: getattr(r_obj, name, None) for name in r_obj._metadata}
            except Exception:
                pass
        if (not meta) and hasattr(r_obj, "params"):
            try:
                maybe = getattr(r_obj, "params")
                if isinstance(maybe, dict):
                    meta = maybe
            except Exception:
                pass
        return meta or None
    except Exception:
        return None

# MAT helpers
def _to_str_list(x) -> List[str]:
    arr = np.asarray(x)
    if arr.dtype.kind in ("U", "S"):
        if arr.ndim == 2:
            return ["".join(row).strip() for row in arr]
        return arr.astype(str).ravel().tolist()
    if arr.dtype == object:
        out = []
        for el in arr.ravel(order="K"):
            if isinstance(el, np.ndarray) and el.dtype.kind in ("U", "S"):
                out.append("".join(el).strip())
            else:
                out.append(str(el))
        return out
    return [str(z) for z in arr.ravel(order="K")]

# read a dataset from a matlab file
# again, didn't end up using this a ton so not super robust
def read_named_mat(
    mat_path: str,
    ds_name: str,
    data_genes_samples: Tuple[int, int, int] = (1, 2, 3),
) -> Tuple[np.ndarray, List[str], List[str]]:
    md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if ds_name not in md:
        raise ValueError(f"Dataset '{ds_name}' not found in {mat_path}")
    ds = md[ds_name]
    fields = list(getattr(ds, "_fieldnames", []) or list(ds.__dict__.keys()))
    if len(fields) < 3:
        raise ValueError(f"Unexpected struct for '{ds_name}' in {mat_path}: fields={fields}")

    def get_by_1based(idx: int):
        name = fields[idx - 1]
        return name, getattr(ds, name)

    (f_data, raw_data) = get_by_1based(data_genes_samples[0])
    (_, raw_genes) = get_by_1based(data_genes_samples[1])
    (_, raw_samps) = get_by_1based(data_genes_samples[2])

    try:
        data = np.array(raw_data, dtype=np.float32)
    except Exception as e:
        raise TypeError(f"Field '{f_data}' not numeric") from e
    if data.ndim != 2:
        data = np.atleast_2d(data)

    genes = _to_str_list(raw_genes)
    samples = _to_str_list(raw_samps)

    # orientation fix
    if not (data.shape[0] == len(genes) and data.shape[1] == len(samples)):
        if data.shape[1] == len(genes) and data.shape[0] == len(samples):
            data = data.T
        else:
            raise ValueError(
                f"Shape mismatch in '{ds_name}': data {data.shape}, genes {len(genes)}, samples {len(samples)}"
            )
    return data, genes, samples


# Model helpers
def array3_from_2d(seq2d: np.ndarray) -> np.ndarray:
    # (timesteps, samples) -> (samples, timesteps, 1)
    return np.expand_dims(seq2d.T.astype(np.float32, copy=False), axis=-1)

def build_lstm_model(n_timesteps, lstm_units=5, seed=123):
    xavier = initializers.GlorotUniform(seed=seed)
    ortho = initializers.Orthogonal(seed=seed)
    zeros = initializers.Zeros()
    inputs = layers.Input(shape=(n_timesteps, 1))
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        kernel_initializer=xavier,
        recurrent_initializer=ortho,
        bias_initializer=zeros,
    )(inputs)
    outputs = layers.Dense(
        1, activation="sigmoid", kernel_initializer=xavier, bias_initializer=zeros
    )(x)
    model = models.Model(inputs, outputs)
    try:
        loss = losses.BinaryFocalCrossentropy()
    except Exception:
        loss = losses.BinaryCrossentropy()
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=["accuracy", metrics.Recall(name="recall"), metrics.Precision(name="precision")],
    )
    return model

def build_dilated_cnn_model(n_timesteps, filters=32, kernel_size=3, dilation_rates=(1, 2, 4), seed=123):
    xavier = initializers.GlorotUniform(seed=seed)
    zeros = initializers.Zeros()
    inputs = layers.Input(shape=(n_timesteps, 1))
    x = inputs
    # layer for each dilation rate
    for i, d in enumerate(dilation_rates):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=int(d),
            padding="causal",
            activation="relu",
            kernel_initializer=xavier,
            bias_initializer=zeros,
            name=f"dilated_conv_{i+1}",
        )(x)
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3, seed=seed)(x)
    outputs = layers.Dense(
        1, activation="sigmoid", kernel_initializer=xavier, bias_initializer=zeros
    )(x)
    model = models.Model(inputs, outputs)
    try:
        loss = losses.BinaryFocalCrossentropy()
    except Exception:
        loss = losses.BinaryCrossentropy()
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=["accuracy", metrics.Recall(name="recall"), metrics.Precision(name="precision")],
    )
    return model

def build_model(
    n_timesteps,
    model_type="lstm",
    lstm_units=5,
    cnn_filters=32,
    cnn_kernel_size=3,
    cnn_dilation_rates=(1, 2, 4),
    seed=123,
):
    if model_type == "lstm":
        return build_lstm_model(n_timesteps, lstm_units=lstm_units, seed=seed)
    if model_type == "dilated_cnn":
        return build_dilated_cnn_model(
            n_timesteps,
            filters=cnn_filters,
            kernel_size=cnn_kernel_size,
            dilation_rates=cnn_dilation_rates,
            seed=seed,
        )
    raise ValueError(f"Unknown model_type: {model_type}")

# Metrics helpers
# first one was an older style of calculating it from an accuracy matrix rather than direct preds
# not used much anymore but keeping for reference and to keep everything working
def confusion_from_accuracy_matrix(acc_mat: np.ndarray, test_set: np.ndarray) -> np.ndarray:
    """
    acc_mat[r, s] = 1 if prediction for row r was correct on sample s.
    Skip row 0 (no predecessors)
    Returns [[TP, FN],[FP, TN]] where "pred positive" means "pred was correct".
    """
    T, S = acc_mat.shape
    TP = TN = FP = FN = 0
    for r in range(1, T):
        for s in range(S):
            correctness = int(acc_mat[r, s])
            actual = int(test_set[r, s])
            if correctness == 1 and actual == 1:
                TP += 1
            elif correctness == 1 and actual == 0:
                TN += 1
            elif correctness == 0 and actual == 0:
                FP += 1
            elif correctness == 0 and actual == 1:
                FN += 1
    return np.array([[TP, FN], [FP, TN]], dtype=np.int64)

# does it in a normal way from predicted classes and true classes
def standard_confusion_from_preds(all_pred_cls: np.ndarray, all_true_cls: np.ndarray) -> np.ndarray:
    pred = all_pred_cls.astype(int).ravel()
    true = all_true_cls.astype(int).ravel()
    TP = int(((pred == 1) & (true == 1)).sum())
    TN = int(((pred == 0) & (true == 0)).sum())
    FP = int(((pred == 1) & (true == 0)).sum())
    FN = int(((pred == 0) & (true == 1)).sum())
    return np.array([[TP, FN], [FP, TN]], dtype=np.int64)

def metrics_table_from_confusion(cm: np.ndarray) -> pd.DataFrame:
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    P = TP + FN
    N = TN + FP
    total = (P + N) or 1
    TPR = TP / P if P else 0.0
    FNR = FN / P if P else 0.0
    TNR = TN / N if N else 0.0
    FPR = FP / N if N else 0.0
    PPV = TP / (TP + FP) if (TP + FP) else 0.0
    NPV = TN / (TN + FN) if (TN + FN) else 0.0
    FOR = FN / (FN + TN) if (FN + TN) else 0.0
    FDR = FP / (TP + FP) if (TP + FP) else 0.0
    ACC = (TP + TN) / total
    BA = (TPR + TNR) / 2.0
    F1 = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) else 0.0
    MCC = np.sqrt(max(TPR, 0) * max(TNR, 0) * max(PPV, 0) * max(NPV, 0)) - np.sqrt(
        max(FNR, 0) * max(FPR, 0) * max(FOR, 0) * max(FDR, 0)
    )
    TS = TP / (TP + FN + FP) if (TP + FN + FP) else 0.0
    LRp = TPR / FPR if FPR else np.inf
    LRm = FNR / TNR if TNR else np.inf
    DOR = LRp / LRm if (LRm and LRm != 0 and np.isfinite(LRp)) else np.inf

    metrics_out = {
        "Metric": [
            "True Positive Rate (Sensitivity, Recall)",
            "False Negative Rate",
            "True Negative Rate (Specificity)",
            "False Positive Rate",
            "Positive Predictive Value (Precision)",
            "Negative Predictive Value",
            "False Omission Rate",
            "False Discovery Rate",
            "Accuracy",
            "Balanced Accuracy",
            "F1 Score",
            "Matthews Correlation Coefficient",
            "Threat Score (CSI, Jaccard Index)",
            "Positive Likelihood Ratio",
            "Negative Likelihood Ratio",
            "Diagnostic Odds Ratio",
        ],
        "Value": [TPR, FNR, TNR, FPR, PPV, NPV, FOR, FDR, ACC, BA, F1, MCC, TS, LRp, LRm, DOR],
    }
    return pd.DataFrame(metrics_out)

def _safe_div(a, b):
    return (a / b) if b else 0.0

def ba_from_cm(cm):
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    P = TP + FN
    N = TN + FP
    tpr = _safe_div(TP, P)
    tnr = _safe_div(TN, N)
    return 0.5 * (tpr + tnr)

def f1_from_cm(cm):
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    prec = _safe_div(TP, TP + FP)
    rec = _safe_div(TP, TP + FN)
    return _safe_div(2 * prec * rec, prec + rec)


# ---------------------------
# Misc
# ---------------------------
def _norm_list(xs: List[str]) -> List[str]:
    return [str(x).strip().upper() for x in xs]

def _load_thresholds_from_json(path: str) -> Optional[List[float]]:
    try:
        obj = json.loads(Path(path).read_text())
    except Exception:
        return None
    # expected: results -> standard -> thresholds
    try:
        th = obj.get("results", {}).get("standard", {}).get("thresholds", None)
        if th is None:
            th = obj.get("results", {}).get("standard", {}).get("metrics_table", {}).get("thresholds", None)
        if th is None:
            return None
        out = [float(x) for x in th]
        return out
    except Exception:
        return None


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tag", required=True, help="Run tag; outputs under outputs/<TAG>/")
    ap.add_argument("--lp_txt", required=True, help="Path to longest_path_*.txt (one gene per line)")
    ap.add_argument("--topo_txt", required=True, help="Path to topo_order_*.txt (one gene per line)")
    ap.add_argument("--lp_rds", default=None, help="Optional longest_path_*.rds (to read R metadata)")

    # mats (used in 'coad_vs_dfci_mgi' and 'old_80_20')
    ap.add_argument("--coad_mat", default="study_data/COAD2.mat")
    ap.add_argument("--dfci_mat", default="study_data/DFCI2.mat")
    ap.add_argument("--mgi_mat", default="study_data/MGI2.mat")
    ap.add_argument("--coad_ds", default="COAD2")
    ap.add_argument("--dfci_ds", default="DFCI2")
    ap.add_argument("--mgi_ds", default="MGI2")
    ap.add_argument("--coad_blocks", default="2,1,3")
    ap.add_argument("--dfci_blocks", default="3,2,1")
    ap.add_argument("--mgi_blocks", default="3,2,1")

    # CSV (used in 'newcsv_80_20')
    ap.add_argument("--combined_csv", default="combined_union.csv", help="CSV rows=genes, cols=samples")

    # modes & context
    ap.add_argument(
        "--mode",
        choices=["coad_vs_dfci_mgi", "old_80_20", "newcsv_80_20"],
        default="coad_vs_dfci_mgi",
    )
    ap.add_argument(
        "--context",
        choices=["full", "lp_only"],
        default="full",
        help="'full' uses TOPO-prefix (all predecessors). 'lp_only' uses only earlier LP genes.",
    )

    # training
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=27)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="If a required model file is missing, error instead of training it.",
    )

    # model architecture
    ap.add_argument("--model_type", choices=["lstm", "dilated_cnn"], default="lstm")
    ap.add_argument("--lstm_units", type=int, default=5)
    ap.add_argument("--cnn_filters", type=int, default=32)
    ap.add_argument("--cnn_kernel_size", type=int, default=3)
    ap.add_argument("--cnn_dilation_rates", type=str, default="1,2,4")

    # early stopping
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    ap.add_argument("--validation_split", type=float, default=0.1)

    # determinism
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)

    # thresholding
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--tune_metric", choices=["ba", "f1"], default="ba")
    ap.add_argument(
        "--th_grid",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    ap.add_argument(
        "--thresholds_from",
        default=None,
        help="Path to a prior run_summary.json to reuse per-gene thresholds (recommended for real->null eval runs).",
    )

    # coverage checks
    ap.add_argument("--min_topo_coverage", type=float, default=0.95)
    ap.add_argument("--min_lp_coverage", type=float, default=0.95)
    ap.add_argument("--fail_on_missing_topo", action="store_true")
    ap.add_argument("--fail_on_missing_lp", action="store_true")
    ap.add_argument("--write_missing_lists", action="store_true")

    args = ap.parse_args()

    # seed everything (per run)
    SEED = int(args.seed)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    TAG = args.tag
    BASE_DIR = Path("outputs") / TAG
    MODELS_DIR = BASE_DIR / "models"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # orders
    lp_raw = _norm_list(read_longest_path_txt(args.lp_txt))
    topo_raw = _norm_list(read_longest_path_txt(args.topo_txt))
    print(f"[info] loaded {len(lp_raw)} LP genes, {len(topo_raw)} TOPO genes (normalized)")

    lp_meta = try_read_rds_meta(args.lp_rds) if args.lp_rds else None

    # load thresholds (optional, useful for real->null eval)
    thresholds_external = None
    if args.thresholds_from:
        thresholds_external = _load_thresholds_from_json(args.thresholds_from)
        if thresholds_external is None:
            print("[WARN] Could not read thresholds from --thresholds_from; will fall back to tuning or 0.5.")
        else:
            print(f"[info] Loaded {len(thresholds_external)} thresholds from {args.thresholds_from}")

    # -------------------------
    # DATA LOAD
    # -------------------------
    if args.mode in ("coad_vs_dfci_mgi", "old_80_20"):
        parse_blocks = lambda s: tuple(int(x) for x in s.split(","))
        COAD2, g1, _ = read_named_mat(args.coad_mat, args.coad_ds, parse_blocks(args.coad_blocks))
        DFCI2, g2, _ = read_named_mat(args.dfci_mat, args.dfci_ds, parse_blocks(args.dfci_blocks))
        MGI2, g3, _ = read_named_mat(args.mgi_mat, args.mgi_ds, parse_blocks(args.mgi_blocks))

        gene_names = _norm_list(g1)  # align to COAD gene list (your convention)
        name_to_idx = {g: i for i, g in enumerate(gene_names)}
        combined = np.concatenate([COAD2, DFCI2, MGI2], axis=1).astype(np.float32)
        nC, nD, nM = COAD2.shape[1], DFCI2.shape[1], MGI2.shape[1]

    elif args.mode == "newcsv_80_20":
        df = pd.read_csv(args.combined_csv, index_col=0)
        X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
        gene_names = _norm_list(df.index.tolist())
        name_to_idx = {g: i for i, g in enumerate(gene_names)}
        combined = X
        nC = nD = nM = 0

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # -------------------------
    # TOPO coverage + reorder
    # -------------------------
    genes_set = set(gene_names)
    topo_set = set(topo_raw)

    missing_topo = [g for g in topo_raw if g not in genes_set]
    present_topo = [g for g in topo_raw if g in genes_set]
    topo_cov = (len(present_topo) / len(topo_raw)) if topo_raw else 0.0

    print(f"[topo] present={len(present_topo)}/{len(topo_raw)} ({topo_cov:.2%}), missing={len(missing_topo)}")

    if args.write_missing_lists and missing_topo:
        (BASE_DIR / "missing_topo_genes.txt").write_text("\n".join(missing_topo))

    if topo_cov < args.min_topo_coverage:
        msg = f"TOPO coverage {topo_cov:.2%} < required {args.min_topo_coverage:.2%}. Missing {len(missing_topo)}."
        if args.fail_on_missing_topo:
            raise RuntimeError(msg)
        print("[WARN]", msg)

    if not present_topo:
        raise RuntimeError("No TOPO genes found in data after alignment.")

    name_to_topo_pos = {g: i for i, g in enumerate(present_topo)}
    topo_idx = np.array([name_to_idx[g] for g in present_topo], dtype=np.int64)
    combined_topo = combined[topo_idx, :]

    # -------------------------
    # LP coverage + TOPO-ordered LP subset
    # -------------------------
    lp_not_in_topo = [g for g in lp_raw if g not in topo_set]
    if lp_not_in_topo:
        msg = f"{len(lp_not_in_topo)} LP genes not in TOPO (first few: {lp_not_in_topo[:10]})"
        if args.write_missing_lists:
            (BASE_DIR / "lp_not_in_topo.txt").write_text("\n".join(lp_not_in_topo))
        print("[WARN]", msg)

    lp_present = [g for g in lp_raw if g in name_to_topo_pos]
    lp_cov = (len(lp_present) / len(lp_raw)) if lp_raw else 0.0
    print(f"[lp] present={len(lp_present)}/{len(lp_raw)} ({lp_cov:.2%}), missing={len(lp_raw) - len(lp_present)}")

    if args.write_missing_lists and len(lp_present) < len(lp_raw):
        missing_lp = [g for g in lp_raw if g not in name_to_topo_pos]
        (BASE_DIR / "missing_lp_genes.txt").write_text("\n".join(missing_lp))

    if lp_cov < args.min_lp_coverage:
        msg = f"LP coverage {lp_cov:.2%} < required {args.min_lp_coverage:.2%}."
        if args.fail_on_missing_lp:
            raise RuntimeError(msg)
        print("[WARN]", msg)

    if not lp_present:
        raise RuntimeError("No LP genes found in data (after TOPO alignment).")

    lp_set = set(lp_present)
    lp = [g for g in present_topo if g in lp_set]  # strictly TOPO-ordered subset
    lp_positions = [name_to_topo_pos[g] for g in lp]
    K = len(lp)

    # -------------------------
    # SPLIT
    # -------------------------
    if args.mode == "coad_vs_dfci_mgi":
        # Train on COAD, test on DFCI+MGI (no shuffle)
        coad_cols = np.arange(0, nC, dtype=np.int64)
        dfci_cols = np.arange(nC, nC + nD, dtype=np.int64)
        mgi_cols = np.arange(nC + nD, nC + nD + nM, dtype=np.int64)
        train_indices = coad_cols
        test_indices = np.concatenate([dfci_cols, mgi_cols], axis=0)

    elif args.mode in ("old_80_20", "newcsv_80_20"):
        rng = np.random.default_rng(SEED)
        total_samples = combined_topo.shape[1]
        col_perm = rng.permutation(total_samples)
        train_size = int(np.floor(args.train_frac * total_samples))
        train_indices = np.sort(col_perm[:train_size])
        test_indices = np.sort(col_perm[train_size:])

    else:
        raise ValueError(f"Unhandled mode: {args.mode}")

    # optional threshold-tuning val split taken from TRAIN
    if args.tune_threshold and thresholds_external is None:
        rng = np.random.default_rng(SEED)
        train_indices = np.array(train_indices, dtype=np.int64)
        n_train = len(train_indices)
        n_val = int(np.floor(args.val_frac * n_train))
        if n_val < 1 or n_train - n_val < 2:
            raise RuntimeError("Not enough training samples to create validation split for threshold tuning.")
        val_indices = np.sort(rng.choice(train_indices, size=n_val, replace=False))
        train_fit_indices = np.sort(np.array(sorted(set(train_indices) - set(val_indices)), dtype=np.int64))
    else:
        val_indices = None
        train_fit_indices = np.array(train_indices, dtype=np.int64)

    # persist split
    pd.to_pickle(
        {"train": np.array(train_indices), "train_fit": np.array(train_fit_indices), "val": val_indices, "test": np.array(test_indices)},
        BASE_DIR / "split_indices.pkl",
    )

    # -------------------------
    # TRAIN/EVAL per LP target
    # -------------------------
    ths = np.array([float(x) for x in args.th_grid.split(",")], dtype=np.float32)

    n_test = len(test_indices)
    accuracy_matrix = np.zeros((K, n_test), dtype=np.int8)
    yte_rows = []

    # pooled for main model
    pooled_pred = []
    pooled_prob = []
    pooled_true = []

    per_gene_thresholds = []

    cnn_dilation_rates = tuple(int(x) for x in args.cnn_dilation_rates.split(","))

    for k, gene in enumerate(lp):
        pos = name_to_topo_pos[gene]

        # targets
        y_tr_fit = combined_topo[pos, train_fit_indices].astype(np.float32)
        y_te = combined_topo[pos, test_indices].astype(np.float32)
        yte_rows.append(y_te.astype(np.int8))

        if thresholds_external is None and args.tune_threshold:
            y_val = combined_topo[pos, val_indices].astype(np.float32)
        else:
            y_val = None

        # choose context rows
        if args.context == "full":
            T_now = pos
            if T_now == 0:
                # no baseline model for zero predecessors
                per_gene_thresholds.append(0.5 if thresholds_external is None else float(thresholds_external[k]) if k < len(thresholds_external) else 0.5)
                continue
            X_tr_2d = combined_topo[:pos, :][:, train_fit_indices]
            X_te_2d = combined_topo[:pos, :][:, test_indices]
            X_val_2d = combined_topo[:pos, :][:, val_indices] if (y_val is not None) else None
        else:
            T_now = k
            if T_now == 0:
                per_gene_thresholds.append(0.5 if thresholds_external is None else float(thresholds_external[k]) if k < len(thresholds_external) else 0.5)
                continue
            rows_sel = lp_positions[:k]
            X_tr_2d = combined_topo[rows_sel, :][:, train_fit_indices]
            X_te_2d = combined_topo[rows_sel, :][:, test_indices]
            X_val_2d = combined_topo[rows_sel, :][:, val_indices] if (y_val is not None) else None

        x_tr = array3_from_2d(X_tr_2d)
        x_te = array3_from_2d(X_te_2d)
        x_val = array3_from_2d(X_val_2d) if (X_val_2d is not None) else None

        model_path = MODELS_DIR / f"model_t_{k+1}.keras"

        # load or train
        if (not args.retrain) and model_path.exists():
            model = keras.models.load_model(model_path)
        else:
            if args.eval_only:
                raise RuntimeError(f"--eval_only set, but missing model file: {model_path}")
            model = build_model(
                T_now,
                model_type=args.model_type,
                lstm_units=args.lstm_units,
                cnn_filters=args.cnn_filters,
                cnn_kernel_size=args.cnn_kernel_size,
                cnn_dilation_rates=cnn_dilation_rates,
                seed=SEED + (k + 1),
            )

            fit_callbacks = []
            if args.early_stopping:
                fit_callbacks.append(
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=args.early_stopping_patience,
                        min_delta=args.early_stopping_min_delta,
                        restore_best_weights=True,
                        verbose=1,
                    )
                )

            model.fit(
                x_tr,
                y_tr_fit,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1,
                shuffle=False,
                validation_split=args.validation_split if args.early_stopping else 0.0,
                callbacks=fit_callbacks if fit_callbacks else None,
            )
            model.save(model_path, include_optimizer=True)

        # predict
        preds = model.predict(x_te, verbose=0).reshape(-1)

        # threshold selection
        if thresholds_external is not None:
            th = float(thresholds_external[k]) if k < len(thresholds_external) else 0.5
        elif args.tune_threshold:
            preds_val = model.predict(x_val, verbose=0).reshape(-1)
            best_th, best_score = 0.5, -1.0
            for cand in ths:
                cls = (preds_val > cand).astype(np.int8)
                cmv = standard_confusion_from_preds(cls, y_val.astype(np.int8))
                score = ba_from_cm(cmv) if args.tune_metric == "ba" else f1_from_cm(cmv)
                if score > best_score:
                    best_score = score
                    best_th = float(cand)
            th = best_th
        else:
            th = 0.5

        per_gene_thresholds.append(th)

        pred_cls = (preds > th).astype(np.int8)

        # accuracy-matrix (your convention)
        acc_vec = (pred_cls == y_te.astype(np.int8)).astype(np.int8)
        accuracy_matrix[k, :] = acc_vec

        # pooled (main)
        pooled_pred.append(pred_cls)
        pooled_prob.append(preds.astype(np.float32))
        pooled_true.append(y_te.astype(np.int8))

        # cleanup
        del model, x_tr, x_te, preds, pred_cls, acc_vec

    # Test labels by LP row
    TestSetLP = np.vstack(yte_rows) if len(yte_rows) == K else np.array(yte_rows, dtype=np.int8)

    # save accuracy matrix
    np.save(BASE_DIR / "accuracy_matrix.npy", accuracy_matrix)

    # custom metrics (accuracy-matrix convention)
    cm_custom = confusion_from_accuracy_matrix(accuracy_matrix, TestSetLP)
    metrics_custom = metrics_table_from_confusion(cm_custom)
    pd.DataFrame(
        cm_custom,
        index=["Actual_Positive", "Actual_Negative"],
        columns=["Pred_Positive", "Pred_Negative"],
    ).to_csv(BASE_DIR / "confusion_matrix.csv")
    metrics_custom.to_csv(BASE_DIR / "metrics_table.csv", index=False)

    # standard pooled metrics
    auc_score = None
    if pooled_true:
        pooled_pred_all = np.concatenate(pooled_pred, axis=0)
        pooled_prob_all = np.concatenate(pooled_prob, axis=0)
        pooled_true_all = np.concatenate(pooled_true, axis=0)

        cm_std = standard_confusion_from_preds(pooled_pred_all, pooled_true_all)
        metrics_std = metrics_table_from_confusion(cm_std)

        if len(np.unique(pooled_true_all)) > 1:
            auc_score = float(roc_auc_score(pooled_true_all, pooled_prob_all))
            metrics_std = pd.concat(
                [metrics_std, pd.DataFrame({"Metric": ["AUC (Area Under ROC Curve)"], "Value": [auc_score]})],
                ignore_index=True,
            )
            print(f"[info] AUC: {auc_score:.6f}")
        else:
            print("[warn] AUC not computed: only one class present in pooled test labels")

        pd.DataFrame(
            cm_std,
            index=["Actual_Positive", "Actual_Negative"],
            columns=["Pred_Positive", "Pred_Negative"],
        ).to_csv(BASE_DIR / "standard_confusion_matrix.csv")
        metrics_std.to_csv(BASE_DIR / "standard_metrics_table.csv", index=False)
    else:
        cm_std = np.array([[0, 0], [0, 0]], dtype=np.int64)
        metrics_std = pd.DataFrame({"Metric": [], "Value": []})

    # -----------------------------
    # Run summary (TXT + JSON)
    # -----------------------------
    lines = [
        f"Run Summary for TAG: {TAG}",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== Graph Meta (from RDS) ===" if lp_meta else "=== Graph Meta ===",
    ]
    if lp_meta:
        for k_, v_ in lp_meta.items():
            lines.append(f"{k_}: {v_}")
    else:
        lines.append("(not available — LP sourced from TXT or attributes not accessible)")

    lines += [
        "",
        "=== Orders & Coverage ===",
        f"TOPO: {len(present_topo)}/{len(topo_raw)} ({(len(present_topo)/len(topo_raw)):.2%}) present",
        f"LP:   {len(lp_present)}/{len(lp_raw)} ({(len(lp_present)/len(lp_raw)):.2%}) present (after TOPO)",
        "",
        f"=== Model Evaluation ===",
        f"Mode: {args.mode}",
        f"Context: {args.context}",
        f"Model: {args.model_type}",
        f"Num LP targets: {K}",
        f"Num training samples: {len(train_indices)}",
        f"Num test samples: {len(test_indices)}",
        f"Seed: {SEED}",
        f"Thresholding: {'external' if thresholds_external is not None else ('tuned' if args.tune_threshold else 'fixed(0.5)')}",
    ]
    (BASE_DIR / "run_summary.txt").write_text("\n".join(lines))

    def matrix_to_obj(mat: np.ndarray):
        return {"dim": list(mat.shape), "data_byrow": [list(r) for r in mat.tolist()]}

    model_config = {
        "model_type": args.model_type,
        "num_lp_targets": int(K),
        "num_train_samples": int(len(train_indices)),
        "num_test_samples": int(len(test_indices)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "context": args.context,
        "train_frac": args.train_frac,
        "seed": SEED,
        "thresholding": {
            "tune_threshold": bool(args.tune_threshold),
            "tune_metric": args.tune_metric,
            "val_frac": args.val_frac,
            "thresholds_from": args.thresholds_from,
            "thresholds": per_gene_thresholds,
        },
        "eval_only": bool(args.eval_only),
    }
    if args.model_type == "lstm":
        model_config["lstm_units"] = args.lstm_units
    else:
        model_config["cnn_filters"] = args.cnn_filters
        model_config["cnn_kernel_size"] = args.cnn_kernel_size
        model_config["cnn_dilation_rates"] = args.cnn_dilation_rates

    json_payload = {
        "tag": TAG,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph_meta": lp_meta,
        "orders": {
            "topo_total": len(topo_raw),
            "topo_present": len(present_topo),
            "lp_total": len(lp_raw),
            "lp_present": len(lp_present),
        },
        "model": model_config,
        "results": {
            "custom": {
                "confusion_matrix": matrix_to_obj(cm_custom),
                "metrics_table": {
                    "columns": list(metrics_custom.columns),
                    "rows": [dict(zip(metrics_custom.columns, row)) for row in metrics_custom.values],
                },
            },
            "standard": {
                "confusion_matrix": matrix_to_obj(cm_std),
                "metrics_table": {
                    "columns": list(metrics_std.columns),
                    "rows": [dict(zip(metrics_std.columns, row)) for row in metrics_std.values],
                },
                "auc": auc_score,
                "thresholds": per_gene_thresholds,
            },
        },
    }
    (BASE_DIR / "run_summary.json").write_text(json.dumps(json_payload, indent=2))

    print(f"[done] outputs written to: {BASE_DIR}")


if __name__ == "__main__":
    main()