"""
Aggregate all experiment outputs in the results/ folder into easy-to-read CSVs.

It scans for:
- *_acc_results.pkl            -> list[float] of accuracies per run
- *_pred_results.pkl           -> list of (y_true, y_pred) per run
- *transition_matrices*.pkl    -> list[np.ndarray CxC] per run

It produces:
- results/metrics_per_run.csv              (one row per experiment per run)
- results/metrics_summary.csv              (aggregated per experiment)
- results/per_class_metrics_summary.csv    (aggregated per-class metrics for pred files)
- results/confusion_matrices_mean.csv      (mean normalized confusion matrix per experiment)
- results/transition_matrices_summary.csv  (mean/std per entry if found)

This script uses only the standard library plus numpy and scikit-learn (already used elsewhere).
"""

from __future__ import annotations

import os
import re
import glob
import csv
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def ensure_results_dir(path: str) -> None:
    if not os.path.exists(path):
        print(f"[INFO] Results folder not found: {path}")
    else:
        print(f"[INFO] Scanning results in: {path}")


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def strip_suffixes(name: str) -> str:
    # remove known suffixes
    for suf in (
        "_acc_results",
        "_pred_results",
        "_transition_matrices",
        "_transition_matrix",
        "_T_hat",
    ):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def parse_label_from_filename(p: str) -> Dict[str, str]:
    base = os.path.basename(p)
    name, _ = os.path.splitext(base)
    core = strip_suffixes(name)
    # Heuristic: model_dataset or model-dataset
    tokens = re.split(r"[_-]", core)
    model = tokens[0] if tokens else core
    dataset = None
    if len(tokens) >= 2:
        dataset = tokens[1]
        # Some datasets include dots or mixed case; keep as-is but normalize a bit
        dataset = dataset.replace("mnist", "MNIST").replace("fashion", "Fashion")
        dataset = dataset.replace("FashionMNIST", "FashionMNIST")
    label = core
    return {"experiment": label, "model": model, "dataset": dataset or "unknown"}


def ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    if n <= 1:
        return (float("nan"), float("nan"))
    half = 1.96 * (std / (n ** 0.5))
    return (mean - half, mean + half)


def summarize_acc_list(accs: List[float]) -> Dict[str, Any]:
    arr = np.array(accs, dtype=float)
    mean = float(np.mean(arr)) if arr.size else float("nan")
    std = float(np.std(arr)) if arr.size else float("nan")
    med = float(np.median(arr)) if arr.size else float("nan")
    minv = float(np.min(arr)) if arr.size else float("nan")
    maxv = float(np.max(arr)) if arr.size else float("nan")
    low, high = ci95(mean, std, len(arr)) if arr.size else (float("nan"), float("nan"))
    best_idx = int(np.argmax(arr)) if arr.size else -1
    best_val = float(arr[best_idx]) if arr.size else float("nan")
    return {
        "runs": int(arr.size),
        "mean_acc": mean,
        "std_acc": std,
        "median_acc": med,
        "min_acc": minv,
        "max_acc": maxv,
        "ci95_low": low,
        "ci95_high": high,
        "best_run_idx": best_idx,
        "best_run_acc": best_val,
    }


def safe_to_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return [x]


def aggregate_from_pred_runs(
    runs: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[int, Dict[str, float]], np.ndarray]:
    """Compute per-run metrics, summary metrics, per-class summary, and mean normalized confusion matrix.

    Returns:
        per_run_rows: list of dicts with acc, macro_f1, weighted_f1 per run
        summary: dict with aggregated metrics
        per_class_summary: dict[class_id] -> aggregated precision/recall/f1 means/stds
        mean_conf_mat: averaged row-normalized confusion matrix across runs
    """
    per_run_rows: List[Dict[str, Any]] = []
    accs: List[float] = []
    macro_f1s: List[float] = []
    weighted_f1s: List[float] = []
    confs: List[np.ndarray] = []
    all_class_metrics: Dict[int, Dict[str, List[float]]] = {}

    num_classes = None
    for i, pair in enumerate(runs):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            y_true, y_pred = pair
        elif isinstance(pair, dict) and "y_true" in pair and "y_pred" in pair:
            y_true, y_pred = pair["y_true"], pair["y_pred"]
        else:
            # Unknown format; skip
            continue

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0:
            continue
        if num_classes is None:
            num_classes = int(max(y_true.max(), y_pred.max()) + 1)

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        accs.append(acc)
        macro_f1s.append(macro_f1)
        weighted_f1s.append(weighted_f1)
        per_run_rows.append({
            "run": i,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        # row-normalize by true counts to be comparable across runs
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        confs.append(cm_norm)

        prec, rec, f1c, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), zero_division=0
        )
        for c in range(len(prec)):
            d = all_class_metrics.setdefault(c, {"precision": [], "recall": [], "f1": []})
            d["precision"].append(float(prec[c]))
            d["recall"].append(float(rec[c]))
            d["f1"].append(float(f1c[c]))

    # Summary
    summary = summarize_acc_list(accs)
    # add f1 summaries if present
    if macro_f1s:
        mf = np.array(macro_f1s)
        wf = np.array(weighted_f1s)
        summary.update({
            "mean_macro_f1": float(np.mean(mf)),
            "std_macro_f1": float(np.std(mf)),
            "mean_weighted_f1": float(np.mean(wf)),
            "std_weighted_f1": float(np.std(wf)),
        })
    # Per-class summary
    per_class_summary: Dict[int, Dict[str, float]] = {}
    for c, d in all_class_metrics.items():
        per_class_summary[c] = {
            "precision_mean": float(np.mean(d["precision"])) if d["precision"] else float("nan"),
            "precision_std": float(np.std(d["precision"])) if d["precision"] else float("nan"),
            "recall_mean": float(np.mean(d["recall"])) if d["recall"] else float("nan"),
            "recall_std": float(np.std(d["recall"])) if d["recall"] else float("nan"),
            "f1_mean": float(np.mean(d["f1"])) if d["f1"] else float("nan"),
            "f1_std": float(np.std(d["f1"])) if d["f1"] else float("nan"),
        }

    mean_conf = np.mean(confs, axis=0) if confs else None
    return per_run_rows, summary, per_class_summary, mean_conf


def summarize_transition_matrices(mats: List[np.ndarray]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not mats:
        return rows
    # infer C from first matrix
    C = mats[0].shape[0]
    stack = np.stack(mats, axis=0)  # R x C x C
    mean_mat = np.mean(stack, axis=0)
    std_mat = np.std(stack, axis=0)
    for i in range(C):
        for j in range(C):
            rows.append({
                "i": i,
                "j": j,
                "mean": float(mean_mat[i, j]),
                "std": float(std_mat[i, j]),
            })
    return rows


def main() -> None:
    ensure_results_dir(RESULTS_DIR)

    acc_files = glob.glob(os.path.join(RESULTS_DIR, "*_acc_results.pkl"))
    pred_files = glob.glob(os.path.join(RESULTS_DIR, "*_pred_results.pkl"))
    tmat_files = glob.glob(os.path.join(RESULTS_DIR, "*transition_matrices*.pkl"))

    print(f"[INFO] Found {len(acc_files)} acc files, {len(pred_files)} pred files, {len(tmat_files)} transition-matrix files.")

    per_run_out: List[Dict[str, Any]] = []
    summary_out: Dict[str, Dict[str, Any]] = {}
    per_class_out: List[Dict[str, Any]] = []
    conf_rows: List[Dict[str, Any]] = []

    # Process accuracy-only files
    for path in acc_files:
        label = parse_label_from_filename(path)
        accs = safe_to_list(load_pickle(path))
        accs = [float(a) for a in accs]
        # per-run rows
        for i, a in enumerate(accs):
            per_run_out.append({
                **label,
                "run": i,
                "accuracy": a,
                "macro_f1": float("nan"),
                "weighted_f1": float("nan"),
                "source": os.path.basename(path),
            })
        # summary
        summary_out[label["experiment"]] = {**label, **summarize_acc_list(accs), "source": os.path.basename(path)}

    # Process prediction files
    for path in pred_files:
        label = parse_label_from_filename(path)
        runs = safe_to_list(load_pickle(path))
        per_run_rows, summary, per_class_summary, mean_conf = aggregate_from_pred_runs(runs)
        # append per-run rows with labels
        for row in per_run_rows:
            per_run_out.append({**label, **row, "source": os.path.basename(path)})
        # merge/override summary if also present from acc file
        key = label["experiment"]
        merged = {**label, **summary, "source": os.path.basename(path)}
        if key in summary_out:
            # Prefer pred-derived accuracy stats if runs match; else keep richer fields
            prev = summary_out[key]
            prev.update(merged)
            summary_out[key] = prev
        else:
            summary_out[key] = merged

        # per-class
        for c, stats in per_class_summary.items():
            per_class_out.append({**label, "class": c, **stats, "source": os.path.basename(path)})

        # confusion matrix mean -> flatten rows
        if mean_conf is not None:
            C = mean_conf.shape[0]
            for i in range(C):
                for j in range(C):
                    conf_rows.append({**label, "i": i, "j": j, "value": float(mean_conf[i, j]), "source": os.path.basename(path)})

    # Process transition matrices if any
    tmat_summary_rows: List[Dict[str, Any]] = []
    for path in tmat_files:
        label = parse_label_from_filename(path)
        mats = safe_to_list(load_pickle(path))
        # Convert to numpy, handling PyTorch CUDA tensors
        converted_mats = []
        for m in mats:
            if m is None:
                continue
            # Check if it's a torch tensor and move to CPU first
            if hasattr(m, 'cpu'):
                m = m.cpu()
            if hasattr(m, 'numpy'):
                m = m.numpy()
            else:
                m = np.asarray(m)
            converted_mats.append(m)
        rows = summarize_transition_matrices(converted_mats)
        for r in rows:
            tmat_summary_rows.append({**label, **r, "source": os.path.basename(path)})

    # Write CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_run_csv = os.path.join(RESULTS_DIR, "metrics_per_run.csv")
    if per_run_out:
        fieldnames = sorted({k for row in per_run_out for k in row.keys()})
        with open(per_run_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_run_out)
        print(f"[OK] Wrote {per_run_csv} ({len(per_run_out)} rows)")
    else:
        print("[WARN] No per-run metrics found.")

    summary_csv = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    if summary_out:
        rows = list(summary_out.values())
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] Wrote {summary_csv} ({len(rows)} rows)")
    else:
        print("[WARN] No summary metrics found.")

    per_class_csv = os.path.join(RESULTS_DIR, "per_class_metrics_summary.csv")
    if per_class_out:
        fieldnames = sorted({k for row in per_class_out for k in row.keys()})
        with open(per_class_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_class_out)
        print(f"[OK] Wrote {per_class_csv} ({len(per_class_out)} rows)")
    else:
        print("[INFO] No per-class metrics to write (no pred files found?).")

    conf_csv = os.path.join(RESULTS_DIR, "confusion_matrices_mean.csv")
    if conf_rows:
        fieldnames = sorted({k for row in conf_rows for k in row.keys()})
        with open(conf_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(conf_rows)
        print(f"[OK] Wrote {conf_csv} ({len(conf_rows)} rows)")
    else:
        print("[INFO] No confusion matrix rows to write.")

    tmat_csv = os.path.join(RESULTS_DIR, "transition_matrices_summary.csv")
    if tmat_summary_rows:
        fieldnames = sorted({k for row in tmat_summary_rows for k in row.keys()})
        with open(tmat_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(tmat_summary_rows)
        print(f"[OK] Wrote {tmat_csv} ({len(tmat_summary_rows)} rows)")
    else:
        print("[INFO] No transition matrix summaries to write.")


if __name__ == "__main__":
    main()



