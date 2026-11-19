#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aegis_dl.metrics import (
    compute_binary_malicious_score,
    compute_comprehensive_metrics,
    find_operating_thresholds,
)


def build_tier1_pipeline() -> Pipeline:
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
    union = FeatureUnion(
        [
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ]
    )
    base_svc = LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual="auto")
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5)

    return Pipeline(
        [
            ("features", union),
            ("classifier", calibrated_svc),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and Evaluate Tier-1 Baseline")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--ood-dir", type=str, default="data/ood")
    parser.add_argument("--model-out", type=str, default="models/baseline_tfidf_linearsvc.joblib")
    parser.add_argument("--report-out", type=str, default="reports/baseline_tier1_metrics.json")
    parser.add_argument("--predictions-dir", type=str, default="reports/predictions")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / args.data_dir
    ood_dir = repo_root / args.ood_dir
    model_out = repo_root / args.model_out
    report_out = repo_root / args.report_out
    preds_dir = repo_root / args.predictions_dir

    print(f"[*] Loading datasets from {data_dir}...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    pipeline = build_tier1_pipeline()
    print(f"[*] Training Tier-1 TF-IDF + Calibrated LinearSVC on {len(train_df):,} samples...")
    t0 = time.perf_counter()
    pipeline.fit(train_df["text"], train_df["label"])
    train_duration = time.perf_counter() - t0
    print(f"[+] Model trained in {train_duration:.2f} seconds.")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    val_probs = pipeline.predict_proba(val_df["text"])
    val_true_binary = (val_df["label"].values > 0).astype(int)
    val_s_mal = compute_binary_malicious_score(val_probs)
    tau_95, tau_1pct = find_operating_thresholds(val_true_binary, val_s_mal)
    print(f"[+] Calibrated thresholds on Validation: tau_95={tau_95:.4f}, tau_1pct={tau_1pct:.4f}")

    sample_queries = test_df["text"].sample(n=min(500, len(test_df)), random_state=42).tolist()
    for q in sample_queries[:20]:
        _ = pipeline.predict_proba([q])

    latencies = []
    for q in sample_queries:
        t_start = time.perf_counter()
        _ = pipeline.predict_proba([q])
        latencies.append((time.perf_counter() - t_start) * 1000.0)
    p99_latency_ms = float(np.percentile(latencies, 99))
    p50_latency_ms = float(np.percentile(latencies, 50))
    print(f"[+] Single-request latency (500 prompts): P50={p50_latency_ms:.3f} ms, P99={p99_latency_ms:.3f} ms")

    test_probs = pipeline.predict_proba(test_df["text"])
    test_metrics = compute_comprehensive_metrics(
        test_df["label"].values,
        test_probs,
        tau_95=tau_95,
        tau_1pct=tau_1pct,
    )
    test_metrics["p99_latency_ms"] = p99_latency_ms
    test_metrics["p50_latency_ms"] = p50_latency_ms
    test_metrics["train_samples"] = int(len(train_df))
    test_metrics["test_samples"] = int(len(test_df))

    devops_mask = (test_df["sub_category"] == "hard_negative_devops").values
    if np.sum(devops_mask) > 0:
        devops_s_mal = compute_binary_malicious_score(test_probs[devops_mask])
        devops_fpr_95 = float(np.mean(devops_s_mal >= tau_95))
        devops_fpr_argmax = float(np.mean(np.argmax(test_probs[devops_mask], axis=-1) != 0))
        test_metrics["devops_hard_neg_count"] = int(np.sum(devops_mask))
        test_metrics["devops_hard_neg_fpr"] = devops_fpr_95
        test_metrics["devops_hard_neg_fpr_argmax"] = devops_fpr_argmax
        print(
            f"[!] DevOps Hard-Negative FPR (n={int(np.sum(devops_mask))}): {devops_fpr_95 * 100:.2f}% (@tau_95) | {devops_fpr_argmax * 100:.2f}% (argmax)"
        )

    xstest_path = ood_dir / "xstest.parquet"
    xstest_probs = None
    if xstest_path.exists():
        xstest_df = pd.read_parquet(xstest_path)
        xstest_probs = pipeline.predict_proba(xstest_df["text"])
        xstest_s_mal = compute_binary_malicious_score(xstest_probs)
        safe_mask = (xstest_df["label"].values == 0)
        unsafe_mask = (xstest_df["label"].values > 0)
        if np.sum(safe_mask) > 0:
            xstest_safe_fpr = float(np.mean(xstest_s_mal[safe_mask] >= tau_95))
            test_metrics["xstest_safe_fpr"] = xstest_safe_fpr
            print(f"[!] XSTest Safe FPR (Over-refusal, n={int(np.sum(safe_mask))}): {xstest_safe_fpr * 100:.2f}%")
        if np.sum(unsafe_mask) > 0:
            xstest_unsafe_tpr = float(np.mean(xstest_s_mal[unsafe_mask] >= tau_95))
            test_metrics["xstest_unsafe_tpr"] = xstest_unsafe_tpr

    notinject_path = ood_dir / "notinject.parquet"
    notinject_probs = None
    if notinject_path.exists():
        notinject_df = pd.read_parquet(notinject_path)
        notinject_probs = pipeline.predict_proba(notinject_df["text"])
        notinject_s_mal = compute_binary_malicious_score(notinject_probs)
        notinject_fpr = float(np.mean(notinject_s_mal >= tau_95))
        test_metrics["notinject_fpr"] = notinject_fpr
        print(f"[!] NotInject Trigger-Word Benign FPR (n={len(notinject_df)}): {notinject_fpr * 100:.2f}%")

    jbb_path = ood_dir / "jbb.parquet"
    jbb_probs = None
    if jbb_path.exists():
        jbb_df = pd.read_parquet(jbb_path)
        jbb_probs = pipeline.predict_proba(jbb_df["text"])
        jbb_s_mal = compute_binary_malicious_score(jbb_probs)
        harm_mask = (jbb_df["label"].values > 0)
        ben_mask = (jbb_df["label"].values == 0)
        if np.sum(harm_mask) > 0:
            jbb_tpr = float(np.mean(jbb_s_mal[harm_mask] >= tau_95))
            test_metrics["jbb_harmful_tpr"] = jbb_tpr
            print(f"[!] JailbreakBench Harmful TPR (n={int(np.sum(harm_mask))}): {jbb_tpr * 100:.2f}%")
        if np.sum(ben_mask) > 0:
            test_metrics["jbb_benign_fpr"] = float(np.mean(jbb_s_mal[ben_mask] >= tau_95))

    preds_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "y_true": test_df["label"].values,
        "probs": test_probs,
        "tau_95": np.array(tau_95, dtype=np.float64),
        "tau_1pct": np.array(tau_1pct, dtype=np.float64),
    }
    if xstest_probs is not None:
        npz_payload["xstest_probs"] = xstest_probs
    if notinject_probs is not None:
        npz_payload["notinject_probs"] = notinject_probs
    if jbb_probs is not None:
        npz_payload["jbb_probs"] = jbb_probs

    np.savez_compressed(preds_dir / "baseline_tier1_test.npz", **npz_payload)
    with open(preds_dir / "baseline_tier1_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": "Tier 1: TF-IDF + Calibrated LinearSVC",
                "model_key": "baseline_tier1",
                "params": "0.15M",
                "p99_latency_ms": p99_latency_ms,
                "p50_latency_ms": p50_latency_ms,
                "tau_95": tau_95,
                "tau_1pct": tau_1pct,
                "metrics": test_metrics,
            },
            f,
            indent=2,
        )

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"[+] Metrics exported to {report_out} and predictions saved to {preds_dir / 'baseline_tier1_test.npz'}:")
    print(f"    Macro F1:        {test_metrics['macro_f1']:.4f}")
    print(f"    ROC-AUC:         {test_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC:          {test_metrics['pr_auc']:.4f}")
    print(f"    FPR @ 95% TPR:   {test_metrics['fpr_at_95_tpr'] * 100:.2f}%")
    print(f"    TPR @ 1% FPR:    {test_metrics['tpr_at_1pct_fpr'] * 100:.2f}%")


if __name__ == "__main__":
    main()
