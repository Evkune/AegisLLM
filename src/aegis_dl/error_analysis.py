from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def extract_top_failures(
    df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    k: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preds = np.argmax(probs, axis=-1)
    s_mal = 1.0 - probs[:, 0]
    confidences = np.max(probs, axis=-1)

    analysis_df = df.copy()
    analysis_df["y_true"] = y_true
    analysis_df["y_pred"] = preds
    analysis_df["s_mal"] = s_mal
    analysis_df["confidence"] = confidences

    fp_mask = (y_true == 0) & (preds != 0)
    fp_df = (
        analysis_df[fp_mask]
        .sort_values(by="s_mal", ascending=False)
        .head(k)
        .copy()
    )

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
    errors = (y_true != y_pred).astype(int)
    if len(np.unique(errors)) < 2 or len(np.unique(token_lengths)) < 2:
        r_pb, p_val = 0.0, 1.0
    else:
        r_pb, p_val = stats.pointbiserialr(errors, token_lengths)

    bins = [0, 32, 128, 256, 512, float("inf")]
    bin_labels = ["1-32", "33-128", "129-256", "257-512", ">512"]
    assigned_bins = pd.cut(token_lengths, bins=bins, labels=bin_labels)

    breakdown: List[Dict] = []
    for label in bin_labels:
        mask = assigned_bins == label
        count = int(np.sum(mask))
        if count == 0:
            continue

        bin_errors = errors[mask]
        bin_y_true = y_true[mask]
        bin_y_pred = y_pred[mask]

        bin_neg = bin_y_true == 0
        bin_pos = bin_y_true != 0

        fpr = float(np.mean(bin_y_pred[bin_neg] != 0)) if np.sum(bin_neg) > 0 else 0.0
        fnr = float(np.mean(bin_y_pred[bin_pos] == 0)) if np.sum(bin_pos) > 0 else 0.0

        breakdown.append(
            {
                "bin": label,
                "sample_count": count,
                "error_rate": float(np.mean(bin_errors)),
                "fpr": fpr,
                "fnr": fnr,
            }
        )

    return {
        "point_biserial_correlation": float(r_pb),
        "correlation_p_value": float(p_val),
        "bins_breakdown": breakdown,
    }


def analyze_subcategory_performance(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true)
    y_pred_in = np.asarray(y_pred)

    if probs is not None:
        probs_arr = np.asarray(probs)
        s_mal = 1.0 - probs_arr[:, 0] if probs_arr.ndim > 1 else probs_arr
        y_pred_arr = np.argmax(probs_arr, axis=-1) if y_pred_in.ndim > 1 else y_pred_in
    elif y_pred_in.ndim > 1:
        s_mal = 1.0 - y_pred_in[:, 0]
        y_pred_arr = np.argmax(y_pred_in, axis=-1)
    elif "s_mal" in df.columns:
        s_mal = np.asarray(df["s_mal"].values, dtype=np.float64)
        y_pred_arr = y_pred_in
    else:
        y_pred_arr = y_pred_in
        s_mal = (y_pred_arr != 0).astype(np.float64)

    df_eval = df.copy()
    df_eval["y_true"] = y_true_arr
    df_eval["y_pred"] = y_pred_arr
    df_eval["s_mal"] = s_mal

    rows = []
    for sub_cat, grp in df_eval.groupby("sub_category", sort=True):
        yt = grp["y_true"].values
        yp = grp["y_pred"].values
        sm = grp["s_mal"].values

        neg_mask = yt == 0
        pos_mask = yt != 0

        fpr = float(np.mean(yp[neg_mask] != 0)) if np.sum(neg_mask) > 0 else 0.0
        fnr = float(np.mean(yp[pos_mask] == 0)) if np.sum(pos_mask) > 0 else 0.0

        rows.append(
            {
                "sub_category": sub_cat,
                "sample_count": int(len(grp)),
                "accuracy": float(np.mean(yt == yp)),
                "fpr": fpr,
                "fnr": fnr,
                "mean_s_mal": float(np.mean(sm)),
            }
        )

    return pd.DataFrame(rows)
