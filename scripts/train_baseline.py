#!/usr/bin/env python3
"""
AegisLLM - Tier 1 & Tier 2 Baseline Training & Evaluation.
Software Engineering for Deep Learning - ETSIINF (UPM)
"""

import argparse
import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aegis_dl.metrics import (
    compute_binary_malicious_score,
    compute_comprehensive_metrics,
    find_operating_thresholds,
)


def build_tier1_pipeline() -> Pipeline:
    """
    Constructs Tier-1 Baseline:
    Dual Word (1-3) + Char Subword (3-5) TF-IDF + Calibrated LinearSVC.
    """
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=25000,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=25000,
        sublinear_tf=True,
    )
    union = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer),
    ])
    base_svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto")
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5)

    pipeline = Pipeline([
        ("features", union),
        ("classifier", calibrated_svc),
    ])
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Tier-1 Baseline")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--ood-dir", type=str, default="data/ood")
    parser.add_argument("--model-out", type=str, default="models/baseline_tfidf_linearsvc.joblib")
    parser.add_argument("--report-out", type=str, default="reports/baseline_tier1_metrics.json")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(repo_root, args.data_dir, "train.parquet")
    val_path = os.path.join(repo_root, args.data_dir, "val.parquet")
    test_path = os.path.join(repo_root, args.data_dir, "test.parquet")
    xstest_path = os.path.join(repo_root, args.ood_dir, "xstest.parquet")

    print(f"[*] Loading datasets from {args.data_dir}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    pipeline = build_tier1_pipeline()
    print(f"[*] Training Tier-1 TF-IDF + Calibrated LinearSVC on {len(train_df):,} samples...")
    t0 = time.perf_counter()
    pipeline.fit(train_df["text"], train_df["label"])
    train_duration = time.perf_counter() - t0
    print(f"[+] Model trained in {train_duration:.2f} seconds.")

    # Save artifact
    os.makedirs(os.path.dirname(os.path.join(repo_root, args.model_out)), exist_ok=True)
    joblib.dump(pipeline, os.path.join(repo_root, args.model_out))
    print(f"[+] Saved model artifact to {args.model_out}")

    # Validation calibration
    val_probs = pipeline.predict_proba(val_df["text"])
    val_true_binary = (val_df["label"].values > 0).astype(int)
    val_s_mal = compute_binary_malicious_score(val_probs)
    tau_95, tau_1pct = find_operating_thresholds(val_true_binary, val_s_mal)
    print(f"[+] Calibrated thresholds on Validation: tau_95={tau_95:.4f}, tau_1pct={tau_1pct:.4f}")

    # Latency benchmark
    latencies = []
    sample_queries = test_df["text"].head(50).tolist()
    for q in sample_queries:
        t_start = time.perf_counter()
        _ = pipeline.predict_proba([q])
        latencies.append((time.perf_counter() - t_start) * 1000.0)
    p99_latency_ms = float(np.percentile(latencies, 99))
    p50_latency_ms = float(np.percentile(latencies, 50))
    print(f"[+] Latency benchmark: P50={p50_latency_ms:.3f} ms, P99={p99_latency_ms:.3f} ms")

    # Evaluation on Test set
    test_probs = pipeline.predict_proba(test_df["text"])
    test_metrics = compute_comprehensive_metrics(
        test_df["label"].values,
        test_probs,
        tau_95=tau_95,
        tau_1pct=tau_1pct,
    )
    test_metrics["p99_latency_ms"] = p99_latency_ms
    test_metrics["p50_latency_ms"] = p50_latency_ms

    # Isolated DevOps Hard Negatives FPR
    devops_mask = (test_df["sub_category"] == "hard_negative_devops").values
    if np.sum(devops_mask) > 0:
        devops_s_mal = test_probs[devops_mask, 1] + test_probs[devops_mask, 2]
        devops_fpr = float(np.mean(devops_s_mal >= tau_95))
        test_metrics["devops_hard_neg_fpr"] = devops_fpr
        print(f"[!] DevOps Hard-Negative FPR: {devops_fpr * 100:.2f}%")

    # Evaluation on OOD XSTest
    if os.path.exists(xstest_path):
        xstest_df = pd.read_parquet(xstest_path)
        xstest_probs = pipeline.predict_proba(xstest_df["text"])
        safe_mask = (xstest_df["label"] == 0).values
        if np.sum(safe_mask) > 0:
            safe_s_mal = xstest_probs[safe_mask, 1] + xstest_probs[safe_mask, 2]
            xstest_fpr = float(np.mean(safe_s_mal >= tau_95))
            test_metrics["xstest_safe_fpr"] = xstest_fpr
            print(f"[!] XSTest Safe FPR (Over-refusal): {xstest_fpr * 100:.2f}%")

    # Save metrics report
    os.makedirs(os.path.dirname(os.path.join(repo_root, args.report_out)), exist_ok=True)
    with open(os.path.join(repo_root, args.report_out), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"[+] Metrics exported to {args.report_out}:")
    print(f"    Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"    ROC-AUC:         {test_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:          {test_metrics['pr_auc']:.4f}")
    print(f"    FPR @ 95% TPR:   {test_metrics['fpr_at_95_tpr'] * 100:.2f}%")
    print(f"    TPR @ 1% FPR:    {test_metrics['tpr_at_1pct_fpr'] * 100:.2f}%")
    print("[OK] Baseline evaluation completed.")


if __name__ == "__main__":
    main()
