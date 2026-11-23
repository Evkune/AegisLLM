#!/usr/bin/env python3
"""
Generate publication-grade 300 DPI figures (PNG + PDF) for AegisLLM Milestone 2
strictly from empirical prediction artifacts (`reports/predictions/*.npz`),
`reports/evaluation_summary.json`, `reports/length_correlation.json`, and
`reports/latency_length_curve.json`. Zero hardcoded metric values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from aegis_dl.metrics import compute_binary_malicious_score

REPO_ROOT = Path(__file__).resolve().parent.parent


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "grid.alpha": 0.28,
        }
    )


def _save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.savefig(fig_dir / f"{stem}.png")
    fig.savefig(fig_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_roc_pr_curves(preds_dir: Path, fig_dir: Path) -> None:
    series = [
        ("baseline_tier1", "Tier-1 TF-IDF + LinearSVC (0.15M)", "--", "#64748b"),
        ("teacher_deberta_v3_base", "Teacher: DeBERTa-v3-Base FP32 (86M)", "-.", "#7c3aed"),
        ("deberta_v3_small_soup_fp32", "Student: DeBERTa-v3-Small FP32 (44M)", "-", "#0f766e"),
        ("deberta_v3_small_int8_dynamic", "Prod: DeBERTa-v3-Small ONNX INT8 (44M)", ":", "#dc2626"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    for key, label, style, color in series:
        npz_path = preds_dir / f"{key}_test.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        y_bin = (npz["y_true"] > 0).astype(int)
        scores = compute_binary_malicious_score(npz["probs"])
        fpr_c, tpr_c, _ = roc_curve(y_bin, scores)
        prec_c, rec_c, _ = precision_recall_curve(y_bin, scores)
        axes[0].plot(fpr_c, tpr_c, linestyle=style, color=color, linewidth=2.2, label=label)
        axes[1].plot(rec_c, prec_c, linestyle=style, color=color, linewidth=2.2, label=label)

    axes[0].axvline(0.01, color="#b91c1c", linestyle="--", linewidth=1.2, alpha=0.7, label="1% FPR SLA Boundary")
    axes[0].set_xlim(0.0, 0.15)
    axes[0].set_ylim(0.80, 1.005)
    axes[0].set_xlabel("False Positive Rate (Benign Blocked)")
    axes[0].set_ylabel("True Positive Rate (Attack Recall)")
    axes[0].set_title("Zoomed ROC Curve (Low-FPR Security Regime)")
    axes[0].legend(loc="lower right", fontsize=8.5)

    axes[1].set_xlim(0.80, 1.005)
    axes[1].set_ylim(0.80, 1.005)
    axes[1].set_xlabel("Recall (Attack Detection)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (In-Domain Test, N=4,438)")
    axes[1].legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    _save_fig(fig, fig_dir, "fig01_roc_pr_curves")


def plot_reliability_diagram(preds_dir: Path, fig_dir: Path) -> None:
    npz = np.load(preds_dir / "deberta_v3_small_int8_dynamic_test.npz")
    meta = json.loads((preds_dir / "deberta_v3_small_int8_dynamic_meta.json").read_text(encoding="utf-8"))
    y_bin = (npz["y_true"] > 0).astype(int)
    s_cal = compute_binary_malicious_score(npz["probs"])
    raw_probs = _softmax_np(npz["logits"].astype(np.float64))
    s_uncal = compute_binary_malicious_score(raw_probs)

    ece_cal = float(meta["metrics"]["ece"])
    temp_val = float(meta["temperature"])

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect Calibration")
    for label, scores, color, marker in [
        ("Before Temp Scaling (T = 1.00)", s_uncal, "#ea580c", "s"),
        (f"After Recalibration (T = {temp_val:.2f}, ECE = {ece_cal:.4f})", s_cal, "#0f766e", "o"),
    ]:
        p_true, p_pred = calibration_curve(y_bin, np.clip(scores, 0.0, 1.0), n_bins=10)
        ax.plot(p_pred, p_true, marker=marker, markersize=6, linewidth=2.2, color=color, label=label)

    ax.set_xlabel("Mean Predicted Malicious Probability")
    ax.set_ylabel("Empirical Fraction of Malicious Prompts")
    ax.set_title("Reliability Diagram — Production ONNX INT8 Dynamic")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _save_fig(fig, fig_dir, "fig02_reliability_calibration")


def plot_confusion_matrix(preds_dir: Path, fig_dir: Path) -> None:
    npz = np.load(preds_dir / "deberta_v3_small_int8_dynamic_test.npz")
    y_true = npz["y_true"]
    y_pred = np.argmax(npz["probs"], axis=-1)
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    cm_cnt = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    annot = np.empty_like(cm_norm, dtype=object)
    for r in range(3):
        for c in range(3):
            annot[r, c] = f"{cm_norm[r, c] * 100:.1f}%\n(n={cm_cnt[r, c]})"

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=["BENIGN", "INJECTION", "JAILBREAK"],
        yticklabels=["BENIGN", "INJECTION", "JAILBREAK"],
        cbar_kws={"label": "Row-Normalized Recall"},
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix — ONNX INT8 Dynamic (N=4,438)")
    ax.set_ylabel("Ground Truth Label")
    ax.set_xlabel("Predicted Class")
    plt.tight_layout()
    _save_fig(fig, fig_dir, "fig03_confusion_matrices")


def plot_latency_and_error_vs_length(reports_dir: Path, fig_dir: Path) -> None:
    len_corr = json.loads((reports_dir / "length_correlation.json").read_text(encoding="utf-8"))
    bins_df = pd.DataFrame(len_corr["bins_breakdown"])
    curve_path = reports_dir / "latency_length_curve.json"

    if curve_path.exists():
        curve_data = json.loads(curve_path.read_text(encoding="utf-8"))
        curve_df = pd.DataFrame(curve_data["length_curve"])
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))

        x_idx = np.arange(len(curve_df))
        axes[0].plot(x_idx, curve_df["p50_ms"], marker="o", linewidth=2.2, color="#0f766e", label="E2E P50 Latency")
        axes[0].plot(x_idx, curve_df["p99_ms"], marker="s", linewidth=2.2, color="#dc2626", label="E2E P99 Latency")
        axes[0].axhline(15.0, color="#1d4ed8", linestyle="--", linewidth=1.3, alpha=0.75, label="15 ms Interactive SLA")
        axes[0].axvline(3.5, color="#6b7280", linestyle=":", linewidth=1.2, alpha=0.8)
        axes[0].text(3.6, float(curve_df["p99_ms"].max()) * 0.25, "Head+Tail+Hot-Spot\nBounded Windowing", fontsize=8.5, color="#374151")
        axes[0].set_xticks(x_idx)
        axes[0].set_xticklabels([str(v) for v in curve_df["target_tokens"]])
        axes[0].set_xlabel("Input Prompt Length (Tokens)")
        axes[0].set_ylabel("Gateway Inspection Latency (ms)")
        axes[0].set_title("Live Proxy Latency vs. Prompt Length (32 -> 4,096 tok)")
        axes[0].legend(loc="upper left", fontsize=8.5)
        ax_err = axes[1]
    else:
        fig, ax_err = plt.subplots(figsize=(7.0, 4.6))

    x_pos = np.arange(len(bins_df))
    width = 0.27
    ax_err.bar(x_pos - width, bins_df["error_rate"] * 100.0, width=width, color="#334155", label="Overall Error (%)")
    ax_err.bar(x_pos, bins_df["fpr"] * 100.0, width=width, color="#ea580c", label="False Positive Rate (%)")
    ax_err.bar(x_pos + width, bins_df["fnr"] * 100.0, width=width, color="#2563eb", label="False Negative Rate (%)")
    ax_err.set_xticks(x_pos)
    ax_err.set_xticklabels(bins_df["bin"])
    ax_err.set_xlabel("Token Length Bin (DeBERTa-v3 SentencePiece)")
    ax_err.set_ylabel("Rate (%)")
    ax_err.set_title(
        f"Error Rate vs. Sequence Length (r_pb = {len_corr['point_biserial_correlation']:.4f}, p = {len_corr['correlation_p_value']:.2f})"
    )
    ax_err.legend(fontsize=8.5)

    plt.tight_layout()
    _save_fig(fig, fig_dir, "fig04_latency_and_error_by_length")


def plot_ablation_comparison(reports_dir: Path, fig_dir: Path) -> None:
    summary = json.loads((reports_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    by_key = {m["model_key"]: m for m in summary.get("models", [])}
    agg_by_key = {a["group_key"]: a for a in summary.get("multi_seed_aggregates", [])}

    ordered_specs = [
        ("baseline_tier1", "Tier-1\nTF-IDF", "#94a3b8", False),
        ("teacher_deberta_v3_base", "Teacher\n86M", "#7c3aed", False),
        ("deberta_v3_small_ce_seed42", "Loss:\nCE", "#60a5fa", False),
        ("deberta_v3_small_focal_seed42", "Loss:\nFocal", "#3b82f6", False),
        ("deberta_v3_small_cs_afl_no_distill_seed42", "CS-AFL\nNo Distill", "#f59e0b", False),
        ("deberta_v3_small_cs_afl", "CS-AFL\n(3-seed)", "#10b981", True),
        ("deberta_v3_small_soup_fp32", "Soup\nFP32", "#0f766e", False),
        ("deberta_v3_small_int8_dynamic", "INT8\nDynamic", "#dc2626", False),
        ("deberta_v3_small_int8_static_qdq", "INT8\nStatic QDQ", "#fb7185", False),
    ]

    labels: list[str] = []
    colors: list[str] = []
    f1s: list[float] = []
    f1_err_low: list[float] = []
    f1_err_high: list[float] = []
    fprs: list[float] = []
    fpr_errs: list[float] = []

    for key, lbl, col, is_agg in ordered_specs:
        if is_agg and key in agg_by_key:
            agg = agg_by_key[key]
            labels.append(lbl)
            colors.append(col)
            f1s.append(float(agg["macro_f1_mean"]))
            std_f1 = float(agg["macro_f1_std"])
            f1_err_low.append(std_f1)
            f1_err_high.append(std_f1)
            fprs.append(float(agg["fpr_at_95_tpr_mean"]) * 100.0)
            fpr_errs.append(float(agg["fpr_at_95_tpr_std"]) * 100.0)
        elif key in by_key:
            row = by_key[key]
            labels.append(lbl)
            colors.append(col)
            val = float(row["macro_f1"])
            f1s.append(val)
            f1_err_low.append(max(0.0, val - float(row["macro_f1_ci_low"])))
            f1_err_high.append(max(0.0, float(row["macro_f1_ci_high"]) - val))
            fprs.append(float(row["fpr_at_95_tpr"]) * 100.0)
            fpr_errs.append(0.0)

    if not labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    x = np.arange(len(labels))

    bars1 = axes[0].bar(
        x, f1s, color=colors, yerr=[f1_err_low, f1_err_high], capsize=3.5, edgecolor="#1e293b", linewidth=0.6
    )
    axes[0].set_ylim(0.86, 0.972)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("Macro F1-Score (95% CI; ±1σ for 3-seed)")
    axes[0].set_title("Ablation Study: Loss, Distillation, Multi-Seed & Quantization")
    for b, val, eh in zip(bars1, f1s, f1_err_high, strict=True):
        axes[0].text(
            b.get_x() + b.get_width() / 2.0,
            val + eh + 0.0022,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    bars2 = axes[1].bar(
        x, fprs, color=colors, yerr=fpr_errs, capsize=3.0, edgecolor="#1e293b", linewidth=0.6
    )
    axes[1].axhline(1.0, color="#b91c1c", linestyle="--", linewidth=1.2, alpha=0.75, label="1.0% FPR SLA Target")
    axes[1].set_ylim(0.0, max(7.7, max(fprs) * 1.12))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylabel("FPR @ 95% TPR (%) — Lower is Better")
    axes[1].set_title("Security Operating Point: FPR @ 95% TPR Across Ablations")
    axes[1].legend(loc="upper right", fontsize=8.5)
    for b, val, ef in zip(bars2, fprs, fpr_errs, strict=True):
        axes[1].text(
            b.get_x() + b.get_width() / 2.0,
            val + ef + 0.12,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    plt.tight_layout()
    _save_fig(fig, fig_dir, "fig05_ablation_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication 300 DPI figures for AegisLLM.")
    parser.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    args = parser.parse_args()

    _setup_style()
    preds_dir = args.reports_dir / "predictions"
    fig_dir = args.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_pr_curves(preds_dir, fig_dir)
    plot_reliability_diagram(preds_dir, fig_dir)
    plot_confusion_matrix(preds_dir, fig_dir)
    plot_latency_and_error_vs_length(args.reports_dir, fig_dir)
    plot_ablation_comparison(args.reports_dir, fig_dir)
    print(f"[+] Generated 5 publication figures (PNG + PDF, 300 DPI) in {fig_dir}")


if __name__ == "__main__":
    main()
