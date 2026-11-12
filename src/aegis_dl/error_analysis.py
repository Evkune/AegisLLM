"""
Error analysis and failure mode taxonomy for AegisLLM.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def extract_top_failures(
    df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    k: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts Top-K highest-confidence False Positives and False Negatives.
    """
    preds = np.argmax(probs, axis=-1)
    s_mal = 1.0 - probs[:, 0]
    confidences = np.max(probs, axis=-1)

    analysis_df = df.copy()
    analysis_df["y_true"] = y_true
    analysis_df["y_pred"] = preds
    analysis_df["s_mal"] = s_mal
    analysis_df["confidence"] = confidences

    # False Positives: Ground truth is Benign (0), predicted as Malicious (1 or 2)
    fp_mask = (y_true == 0) & (preds != 0)
    fp_df = (
        analysis_df[fp_mask]
        .sort_values(by="s_mal", ascending=False)
        .head(k)
        .copy()
    )

    # False Negatives: Ground truth is Malicious (1 or 2), predicted as Benign (0)
    fn_mask = (y_true != 0) & (preds == 0)
    fn_df = (
        analysis_df[fn_mask]
        .sort_values(by="confidence", ascending=False)
        .head(k)
        .copy()
    )

    return fp_df, fn_df


def analyze_length_correlation(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    token_lengths: np.ndarray,
) -> Dict:
    """
    Partitions samples into length quartiles and computes:
    - Error rate, FPR, and FNR per bin
    - Point-biserial correlation between length and classification error
    """
    errors = (y_true != y_pred).astype(int)
    r_pb, p_val = stats.pointbiserialr(errors, token_lengths)

    bins = [0, 32, 128, 256, 512, float("inf")]
    bin_labels = ["1-32", "33-128", "129-256", "257-512", ">512"]
    assigned_bins = pd.cut(token_lengths, bins=bins, labels=bin_labels)

    breakdown = []
    for label in bin_labels:
        mask = (assigned_bins == label)
        count = int(np.sum(mask))
        if count == 0:
            continue

        bin_errors = errors[mask]
        bin_y_true = y_true[mask]
        bin_y_pred = y_pred[mask]

        bin_neg = (bin_y_true == 0)
        bin_pos = (bin_y_true != 0)

        fpr = float(np.mean(bin_y_pred[bin_neg] != 0)) if np.sum(bin_neg) > 0 else 0.0
        fnr = float(np.mean(bin_y_pred[bin_pos] == 0)) if np.sum(bin_pos) > 0 else 0.0

        breakdown.append({
            "bin": label,
            "sample_count": count,
            "error_rate": float(np.mean(bin_errors)),
            "fpr": fpr,
            "fnr": fnr,
        })

    return {
        "point_biserial_correlation": float(r_pb),
        "correlation_p_value": float(p_val),
        "bins_breakdown": breakdown,
    }


def analyze_subcategory_performance(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Computes accuracy, precision, and recall per fine-grained sub_category.
    """
    df_eval = df.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred
    df_eval["is_correct"] = (y_true == y_pred).astype(int)

    summary = (
        df_eval.groupby("sub_category")
        .agg(
            sample_count=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    return summary
