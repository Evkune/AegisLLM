"""
Empirical evaluation metrics, calibration tests, and statistical tests for AegisLLM.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_malicious_score(probs: np.ndarray) -> np.ndarray:
    """
    Computes binary malicious probability:
    s_mal(x) = P(PROMPT_INJECTION) + P(JAILBREAK) = 1.0 - P(BENIGN)
    """
    if probs.ndim == 1:
        return probs
    # Class 0: BENIGN, Class 1: PROMPT_INJECTION, Class 2: JAILBREAK
    return 1.0 - probs[:, 0]


def find_operating_thresholds(
    y_true_binary: np.ndarray,
    malicious_scores: np.ndarray,
) -> Tuple[float, float]:
    """
    Finds operating thresholds:
    1. tau_95: threshold achieving >= 95% TPR with minimal FPR
    2. tau_1pct: threshold achieving <= 1% FPR with maximal TPR
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, malicious_scores)

    # 1. tau_95 (TPR >= 0.95)
    tpr_idx = np.where(tpr >= 0.95)[0]
    tau_95 = float(thresholds[tpr_idx[0]]) if len(tpr_idx) > 0 else 0.50

    # 2. tau_1pct (FPR <= 0.01)
    fpr_idx = np.where(fpr <= 0.01)[0]
    tau_1pct = float(thresholds[fpr_idx[-1]]) if len(fpr_idx) > 0 else 0.50

    return tau_95, tau_1pct


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width bins."""
    preds = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)
    accuracies = preds == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    tau_95: Optional[float] = None,
    tau_1pct: Optional[float] = None,
) -> Dict[str, float]:
    """
    Computes all standard and security-specific metrics.
    """
    y_pred = np.argmax(probs, axis=-1)
    y_true_binary = (y_true > 0).astype(int)
    s_mal = compute_binary_malicious_score(probs)

    # Threshold-independent metrics
    roc_auc = float(roc_auc_score(y_true_binary, s_mal))
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, s_mal)
    pr_auc = float(auc(recall_curve, precision_curve))

    # Standard multi-class
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    # Per-class metrics (strictly enforce 3 classes: 0, 1, 2)
    p_per_class = precision_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    r_per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)

    # Operational Security Metrics
    if tau_95 is None or tau_1pct is None:
        computed_tau_95, computed_tau_1pct = find_operating_thresholds(y_true_binary, s_mal)
        tau_95 = tau_95 if tau_95 is not None else computed_tau_95
        tau_1pct = tau_1pct if tau_1pct is not None else computed_tau_1pct

    preds_at_tau95 = (s_mal >= tau_95).astype(int)
    preds_at_tau1pct = (s_mal >= tau_1pct).astype(int)

    # FPR @ 95% TPR
    neg_mask = (y_true_binary == 0)
    pos_mask = (y_true_binary == 1)

    fpr_at_tau95 = float(np.mean(preds_at_tau95[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
    tpr_at_tau95 = float(np.mean(preds_at_tau95[pos_mask])) if np.sum(pos_mask) > 0 else 0.0

    fpr_at_tau1pct = float(np.mean(preds_at_tau1pct[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
    tpr_at_tau1pct = float(np.mean(preds_at_tau1pct[pos_mask])) if np.sum(pos_mask) > 0 else 0.0

    ece = compute_expected_calibration_error(probs, y_true)
    brier = float(brier_score_loss(y_true_binary, s_mal))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_benign": float(p_per_class[0]),
        "recall_benign": float(r_per_class[0]),
        "f1_benign": float(f1_per_class[0]),
        "precision_injection": float(p_per_class[1]),
        "recall_injection": float(r_per_class[1]),
        "f1_injection": float(f1_per_class[1]),
        "precision_jailbreak": float(p_per_class[2]),
        "recall_jailbreak": float(r_per_class[2]),
        "f1_jailbreak": float(f1_per_class[2]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tau_95": float(tau_95),
        "tau_1pct": float(tau_1pct),
        "fpr_at_95_tpr": fpr_at_tau95,
        "tpr_at_95_tpr": tpr_at_tau95,
        "fpr_at_1pct_fpr": fpr_at_tau1pct,
        "tpr_at_1pct_fpr": tpr_at_tau1pct,
        "ece": ece,
        "brier_score": brier,
    }


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Computes McNemar's test comparing model A vs model B.
    H0: both models have equal error rates.
    chi2 = (|b - c| - 1)^2 / (b + c) with Edwards' continuity correction.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B incorrect
    b = int(np.sum(correct_a & ~correct_b))
    # c: A incorrect, B correct
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return 0.0, 1.0

    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_value = float(1.0 - stats.chi2.cdf(stat, df=1))
    return float(stat), p_value


def compute_paired_bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes point estimate and empirical bootstrap confidence interval.
    Returns: (point_estimate, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    point_val = metric_fn(y_true, probs)

    boot_scores = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, n_samples)
        score = metric_fn(y_true[indices], probs[indices])
        boot_scores.append(score)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_scores, 100.0 * alpha))
    upper = float(np.percentile(boot_scores, 100.0 * (1.0 - alpha)))
    return float(point_val), lower, upper
