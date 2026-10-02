#!/usr/bin/env python3
"""Train and benchmark a LightGBM credit-card fraud classifier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def find_dataset(explicit_path: Path | None) -> Path:
    """Resolve a CSV path or locate Kaggle's extracted creditcard.csv."""
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    candidates = (
        Path.cwd() / "creditcard.csv",
        Path.cwd() / "ml-benchmark" / "creditcard.csv",
        Path.home() / "ml-benchmark" / "creditcard.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find creditcard.csv. Download and unzip the Kaggle dataset to "
        "~/ml-benchmark, or pass --data /path/to/creditcard.csv."
    )


def timed_predict(
    model: lgb.LGBMClassifier, rows: pd.DataFrame, repeats: int
) -> dict[str, float | int]:
    """Measure average one-row latency and throughput for a 1000-row batch."""
    if len(rows) < 1000:
        raise ValueError("At least 1000 test rows are required to measure 1000-row throughput.")

    one_row = rows.iloc[[0]]
    model.predict_proba(one_row)  # warm up the prediction path

    start = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(one_row)
    one_row_mean_ms = (time.perf_counter() - start) * 1000 / repeats

    batch = rows.iloc[:1000]
    batch_repeats = max(1, min(100, repeats))
    start = time.perf_counter()
    for _ in range(batch_repeats):
        model.predict_proba(batch)
    batch_seconds = time.perf_counter() - start
    rows_per_second = (len(batch) * batch_repeats) / batch_seconds

    return {
        "inference_latency_1_row_ms": float(one_row_mean_ms),
        "inference_throughput_rows_per_second": float(rows_per_second),
        "throughput_batch_rows": len(batch),
        "throughput_repeats": batch_repeats,
    }


def run_benchmark(data_path: Path, output_path: Path, inference_repeats: int) -> dict[str, Any]:
    load_start = time.perf_counter()
    data = pd.read_csv(data_path)
    data_load_seconds = time.perf_counter() - load_start

    if "Class" not in data.columns:
        raise ValueError("Expected the Kaggle dataset to contain a 'Class' target column.")
    if data.empty:
        raise ValueError("The dataset is empty.")

    features = data.drop(columns=["Class"])
    target = data["Class"].astype("int8")
    if target.nunique() != 2:
        raise ValueError("Expected a binary 'Class' target with both fraud and non-fraud rows.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=negatives / positives,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    training_start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    training_seconds = time.perf_counter() - training_start

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    metrics: dict[str, Any] = {
        "data_file": str(data_path),
        "rows": len(data),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "split": "60/20/20",
        "seed": 42,
        "data_load_seconds": float(data_load_seconds),
        "training_seconds": float(training_seconds),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }
    metrics.update(timed_predict(model, X_test, inference_repeats))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to creditcard.csv (defaults to common Kaggle download locations).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="JSON output file (default: benchmark_result.json).",
    )
    parser.add_argument(
        "--inference-repeats",
        type=int,
        default=200,
        help="Number of one-row predictions used to average latency (default: 200).",
    )
    args = parser.parse_args()
    if args.inference_repeats < 1:
        parser.error("--inference-repeats must be at least 1")

    try:
        data_path = find_dataset(args.data)
        output_path = args.output.expanduser().resolve()
        metrics = run_benchmark(data_path, output_path, args.inference_repeats)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("LightGBM Credit Card Fraud Benchmark")
    print(f"Dataset: {metrics['data_file']} ({metrics['rows']:,} rows)")
    print(
        "Split: 60/20/20 "
        f"(train={metrics['train_rows']:,}, "
        f"validation={metrics['validation_rows']:,}, "
        f"test={metrics['test_rows']:,})"
    )
    print(f"Data load: {metrics['data_load_seconds']:.3f} s")
    print(f"Training: {metrics['training_seconds']:.3f} s")
    print(f"Best iteration: {metrics['best_iteration']}")
    print(f"AUC-ROC: {metrics['auc_roc']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"F1-Score: {metrics['f1_score']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"Inference latency (1 row, mean): {metrics['inference_latency_1_row_ms']:.4f} ms")
    print(
        "Inference throughput "
        f"({metrics['throughput_batch_rows']} rows): "
        f"{metrics['inference_throughput_rows_per_second']:.2f} rows/s"
    )
    print(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
