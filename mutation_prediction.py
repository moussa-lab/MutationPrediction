import os, json, time, argparse, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

import tensorflow as tf
import keras
from keras import layers, models, initializers, losses, metrics, optimizers

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def read_longest_path_txt(path_txt: str) -> list[str]:
    with open(path_txt, 'r') as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    return genes

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
            if 'meta' in r_obj and isinstance(r_obj['meta'], dict):
                return r_obj['meta']
            if 'params' in r_obj and isinstance(r_obj['params'], dict):
                return r_obj['params']
            return {k: v for k, v in r_obj.items() if not hasattr(v, "__len__") or isinstance(v, (str, int, float))}
        
        meta = None
        if hasattr(r_obj, 'attrs') and isinstance(r_obj.attrs, dict) and len(r_obj.attrs):
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
    
def _to_str_list(x) -> List[str]:
    """
    Robustly convert MATLAB char arrays / cellstr / object arrays into a list[str].
    - MATLAB char matrix: shape (n, maxlen), dtype kind 'U' or 'S' -> join per row
    - cell array of strings: dtype object -> flatten and str()
    """
    arr = np.asarray(x)
    if arr.dtype.kind in ("U", "S"):
        if arr.ndim == 2:
            # each row is characters -> join row into a string
            return [''.join(row).strip() for row in arr]
        else:
            return arr.astype(str).ravel().tolist()
    if arr.dtype == object:
        out = []
        for el in arr.ravel(order="K"):
            if isinstance(el, np.ndarray) and el.dtype.kind in ("U","S"):
                # char vector
                out.append(''.join(el).strip())
            else:
                out.append(str(el))
        return out
    # fallback
    return [str(z) for z in arr.ravel(order="K")]

def read_named_mat(mat_path: str, ds_name: str,
                   data_genes_samples: Tuple[int,int,int]=(1,2,3)
                   ) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Returns (data, gene_names, sample_names) with rows=genes, cols=samples.
    Indexes the MATLAB struct by ds._fieldnames (stable), not __dict__.values() (unstable).
    Also handles MATLAB char arrays / cellstr robustly.
    """
    md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if ds_name not in md:
        raise ValueError(f"Dataset '{ds_name}' not found in {mat_path}")
    ds = md[ds_name]

    # Use explicit field order from the MATLAB struct
    fields = list(getattr(ds, "_fieldnames", []) or list(ds.__dict__.keys()))
    if len(fields) < 3:
        raise ValueError(f"Unexpected struct for '{ds_name}' in {mat_path}: fields={fields}")

    def get_by_1based(idx: int):
        name = fields[idx-1]
        return name, getattr(ds, name)

    (f_data, raw_data)     = get_by_1based(data_genes_samples[0])
    (f_genes, raw_genes)   = get_by_1based(data_genes_samples[1])
    (f_samples, raw_samps) = get_by_1based(data_genes_samples[2])

    # Coerce data to float32 2D
    try:
        data = np.array(raw_data, dtype=np.float32)
    except Exception as e:
        raise TypeError(
            f"Field '{f_data}' (index {data_genes_samples[0]}) is not numeric; "
            f"type={type(raw_data)}, example={repr(raw_data)[:80]}"
        ) from e
    if data.ndim != 2:
        data = np.atleast_2d(data)

    # Convert names robustly
    genes   = _to_str_list(raw_genes)
    samples = _to_str_list(raw_samps)

    # Fix orientation if needed
    if not (data.shape[0] == len(genes) and data.shape[1] == len(samples)):
        if data.shape[1] == len(genes) and data.shape[0] == len(samples):
            data = data.T
        else:
            raise ValueError(
                f"Shape mismatch in '{ds_name}': data {data.shape}, "
                f"genes {len(genes)} ({f_genes}), samples {len(samples)} ({f_samples}). "
                f"Fields order={fields}"
            )

    return data, genes, samples

def array3_from_2d(seq2d: np.ndarray) -> np.ndarray:
    return np.expand_dims(seq2d.T.astype(np.float32, copy=False), axis=-1)

def build_model(n_timesteps, seed=123):
    xavier = initializers.GlorotUniform(seed=seed)
    ortho  = initializers.Orthogonal(seed=seed)
    zeros  = initializers.Zeros()

    inputs = layers.Input(shape=(n_timesteps, 1))
    x = layers.LSTM(
        units=5, return_sequences=False,
        kernel_initializer=xavier,
        recurrent_initializer=ortho,
        bias_initializer=zeros
    )(inputs)
    outputs = layers.Dense(1, activation="sigmoid",
                           kernel_initializer=xavier,
                           bias_initializer=zeros)(x)

    model = models.Model(inputs, outputs)
    try:
        loss = losses.BinaryFocalCrossentropy()
    except Exception:
        loss = losses.BinaryCrossentropy()
    model.compile(optimizer="adam", loss=loss,
                  metrics=["accuracy", metrics.Recall(name="recall"), metrics.Precision(name="precision")])
    return model

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def confusion_from_accuracy_matrix(acc_mat: np.ndarray, test_set: np.ndarray) -> np.ndarray:
    """
    acc_mat[r, s] = 1 if the model's class prediction at step t=r+1 was correct for sample s.
    test_set[r, s] is the actual class at that step.
    Returns [[TP, FN],
             [FP, TN]] (this is what your metrics() expects).
    """
    T, S = acc_mat.shape
    TP = TN = FP = FN = 0
    for r in range(1, T):  # r=1..T-1, because there's no t=1 prediction
        for s in range(S):
            correctness = int(acc_mat[r, s])  # 1=correct, 0=incorrect
            actual      = int(test_set[r, s]) # same row r
            if correctness == 1 and actual == 1:
                TP += 1
            elif correctness == 1 and actual == 0:
                TN += 1
            elif correctness == 0 and actual == 0:
                FP += 1
            elif correctness == 0 and actual == 1:
                FN += 1
    return np.array([[TP, FN], [FP, TN]], dtype=np.int64)

def metrics_table_from_confusion(cm: np.ndarray) -> pd.DataFrame:
    TP, FN = cm[0,0], cm[0,1]
    FP, TN = cm[1,0], cm[1,1]
    P = TP + FN; N = TN + FP; total = (P + N) or 1
    TPR = TP / P if P else 0.0
    FNR = FN / P if P else 0.0
    TNR = TN / N if N else 0.0
    FPR = FP / N if N else 0.0
    PPV = TP / (TP + FP) if (TP+FP) else 0.0
    NPV = TN / (TN + FN) if (TN+FN) else 0.0
    FOR = FN / (FN + TN) if (FN+TN) else 0.0
    FDR = FP / (TP + FP) if (TP+FP) else 0.0
    ACC = (TP + TN) / total
    BA  = (TPR + TNR) / 2.0
    F1  = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) else 0.0
    MCC = np.sqrt(max(TPR,0)*max(TNR,0)*max(PPV,0)*max(NPV,0)) - np.sqrt(max(FNR,0)*max(FPR,0)*max(FOR,0)*max(FDR,0))
    TS  = TP / (TP + FN + FP) if (TP + FN + FP) else 0.0
    LRp = TPR / FPR if FPR else np.inf
    LRm = FNR / TNR if TNR else np.inf
    DOR = LRp / LRm if (LRm and LRm != 0 and np.isfinite(LRp)) else np.inf

    metrics = {
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
    return pd.DataFrame(metrics)

def standard_confusion_from_preds(all_pred_cls: np.ndarray, all_true_cls: np.ndarray) -> np.ndarray:
    pred = all_pred_cls.astype(int).ravel()
    true = all_true_cls.astype(int).ravel()
    TP = int(((pred == 1) & (true == 1)).sum())
    TN = int(((pred == 0) & (true == 0)).sum())
    FP = int(((pred == 1) & (true == 0)).sum())
    FN = int(((pred == 0) & (true == 1)).sum())
    return np.array([[TP, FN], [FP, TN]], dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Run tag; outputs under outputs/<TAG>/")
    ap.add_argument("--lp_txt", required=True, help="Path to longest_path_genes_<TAG>.txt (one gene per line)")
    ap.add_argument("--lp_rds", default=None, help="Optional path to longest_path_genes_<TAG>.rds (to read R metadata)")
    # dataset paths/blocks
    ap.add_argument("--coad_mat", default="study_data/COAD2.mat")
    ap.add_argument("--dfci_mat", default="study_data/DFCI2.mat")
    ap.add_argument("--mgi_mat",  default="study_data/MGI2.mat")
    ap.add_argument("--coad_ds", default="COAD2")
    ap.add_argument("--dfci_ds", default="DFCI2")
    ap.add_argument("--mgi_ds",  default="MGI2")
    ap.add_argument("--coad_blocks", default="2,1,3")
    ap.add_argument("--dfci_blocks", default="3,2,1")
    ap.add_argument("--mgi_blocks",  default="3,2,1")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=27)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--shuffle_genes", action="store_true", help="Shuffle gene order for baseline comparison")
    args = ap.parse_args()

    TAG = args.tag
    BASE_DIR = Path("outputs") / TAG
    MODELS_DIR = BASE_DIR / "models"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load LP (genes) from TXT
    lp = read_longest_path_txt(args.lp_txt)
    print(f"[info] loaded {len(lp)} genes from TXT")

    # Try to read RDS meta (optional)
    lp_meta = try_read_rds_meta(args.lp_rds) if args.lp_rds else None

    # Read mats
    parse_blocks = lambda s: tuple(int(x) for x in s.split(","))
    COAD2, g1, _ = read_named_mat(args.coad_mat, args.coad_ds, parse_blocks(args.coad_blocks))
    DFCI2, g2, _ = read_named_mat(args.dfci_mat, args.dfci_ds, parse_blocks(args.dfci_blocks))
    MGI2,  g3, _ = read_named_mat(args.mgi_mat,  args.mgi_ds,  parse_blocks(args.mgi_blocks))

    # Combine samples
    combined = np.concatenate([COAD2, DFCI2, MGI2], axis=1).astype(np.float32)
    gene_names = g1
    name_to_idx = {g:i for i,g in enumerate(gene_names)}

    path_genes_in_data = [g for g in lp if g in name_to_idx]
    if len(path_genes_in_data) < 2:
        raise RuntimeError("Not enough genes from longest path in data")
    row_idx = np.array([name_to_idx[g] for g in path_genes_in_data], dtype=np.int64)
    ordered_combined = combined[row_idx, :]

    if args.shuffle_genes:
        rng = np.random.default_rng(SEED)
        row_idx = rng.permutation(row_idx)
        ordered_combined = combined[row_idx, :]
        print("[info] Gene order shuffled for baseline run.")

    # Shuffle columns reproducibly & split
    rng = np.random.default_rng(SEED)
    col_perm = rng.permutation(ordered_combined.shape[1])
    combined_data = ordered_combined[:, col_perm]
    total_samples = combined_data.shape[1]
    train_size = int(np.floor(args.train_frac * total_samples))
    train_indices = rng.choice(total_samples, size=train_size, replace=False)
    test_indices  = np.array(sorted(set(range(total_samples)) - set(train_indices)))
    TrainingSet = combined_data[:, train_indices]
    TestSet     = combined_data[:, test_indices]
    pd.to_pickle({"train": train_indices, "test": test_indices}, BASE_DIR / "split_indices.pkl")

    # Train per-t models + collect predictions
    num_mutations = TrainingSet.shape[0]
    accuracy_matrix = np.zeros((num_mutations, TestSet.shape[1]), dtype=np.int8)
    # For standard metrics, we also pool predicted and actual classes across all (t>=2, s)
    pooled_pred = []
    pooled_true = []

    for t in range(2, num_mutations+1):
        T_now = t - 1
        x_tr = array3_from_2d(TrainingSet[:T_now, :])
        y_tr = TrainingSet[T_now, :].astype(np.float32)

        model_path = MODELS_DIR / f"model_t_{t}.keras"
        if (not args.retrain) and model_path.exists():
            model = keras.models.load_model(model_path)
        else:
            model = build_model(T_now, seed=SEED + t)
            model.fit(
                x_tr, y_tr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1,
                shuffle=False,
            )
            model.save(model_path, include_optimizer=True)

        x_te = array3_from_2d(TestSet[:T_now, :])
        y_te = TestSet[T_now, :].astype(np.float32)
        preds = model.predict(x_te, verbose=0).reshape(-1)
        pred_cls = (preds > 0.5).astype(np.int8)
        # accuracy-matrix convention
        acc_vec = (pred_cls == y_te.astype(np.int8)).astype(np.int8)
        accuracy_matrix[t-1, :] = acc_vec

        pooled_pred.append(pred_cls)
        pooled_true.append(y_te.astype(np.int8))

        del model, x_tr, y_tr, x_te, y_te, preds, pred_cls, acc_vec

    # Save accuracy matrix
    np.save(BASE_DIR / "accuracy_matrix.npy", accuracy_matrix)

    # ----- Your original-style confusion/metrics (based on accuracy_matrix) -----
    cm_custom = confusion_from_accuracy_matrix(accuracy_matrix, TestSet)
    metrics_custom = metrics_table_from_confusion(cm_custom)
    pd.DataFrame(cm_custom, index=["Actual_Positive", "Actual_Negative"],
                 columns=["Pred_Positive", "Pred_Negative"]).to_csv(BASE_DIR / "confusion_matrix.csv")
    metrics_custom.to_csv(BASE_DIR / "metrics_table.csv", index=False)

    # ----- Standard confusion/metrics (pred vs actual) -----
    pooled_pred = np.concatenate(pooled_pred, axis=0)
    pooled_true = np.concatenate(pooled_true, axis=0)
    cm_std = standard_confusion_from_preds(pooled_pred, pooled_true)
    metrics_std = metrics_table_from_confusion(cm_std)
    pd.DataFrame(cm_std, index=["Actual_Positive", "Actual_Negative"],
                 columns=["Pred_Positive", "Pred_Negative"]).to_csv(BASE_DIR / "standard_confusion_matrix.csv")
    metrics_std.to_csv(BASE_DIR / "standard_metrics_table.csv", index=False)

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
        for k, v in lp_meta.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append("(not available — LP sourced from TXT or attributes not accessible)")

    lines += [
        "",
        "=== LSTM Evaluation ===",
        f"Num mutations (longest path): {TrainingSet.shape[0]}",
        f"Num training samples: {TrainingSet.shape[1]}",
        f"Num test samples: {TestSet.shape[1]}",
        "",
        "Custom Confusion Matrix (rows: Actual [+,-], cols: Pred [+,-]):",
        str(cm_custom),
        "",
        "Custom Metrics Table:",
        metrics_custom.to_string(index=False),
        "",
        "Standard Confusion Matrix (rows: Actual [+,-], cols: Pred [+,-]):",
        str(cm_std),
        "",
        "Standard Metrics Table:",
        metrics_std.to_string(index=False),
        ""
    ]
    (BASE_DIR / "run_summary.txt").write_text("\n".join(lines))

    def matrix_to_obj(mat: np.ndarray):
        return {"dim": list(mat.shape), "data_byrow": [list(r) for r in mat.tolist()]}

    json_payload = {
        "tag": TAG,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "graph_meta": lp_meta,
        "lstm": {
            "num_mutations": int(TrainingSet.shape[0]),
            "num_train_samples": int(TrainingSet.shape[1]),
            "num_test_samples": int(TestSet.shape[1]),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "results": {
            "custom": {
                "confusion_matrix": matrix_to_obj(cm_custom),
                "metrics_table": {
                    "columns": list(metrics_custom.columns),
                    "rows": [dict(zip(metrics_custom.columns, row)) for row in metrics_custom.values]
                }
            },
            "standard": {
                "confusion_matrix": matrix_to_obj(cm_std),
                "metrics_table": {
                    "columns": list(metrics_std.columns),
                    "rows": [dict(zip(metrics_std.columns, row)) for row in metrics_std.values]
                }
            }
        }
    }
    (BASE_DIR / "run_summary.json").write_text(json.dumps(json_payload, indent=2))

    print(f"[done] outputs written to: {BASE_DIR}")

if __name__ == "__main__":
    main()