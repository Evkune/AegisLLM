#!/usr/bin/env python3
"""Two-way factorial DOE statistical analysis (ANOVA, assumption diagnostics, 300 DPI plots)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from aegis_dl.metrics import compute_binary_malicious_score, compute_comprehensive_metrics

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent

DOE_LOSSES: tuple[str, ...] = ("ce", "cs_afl")
DOE_DISTILL_LEVELS: tuple[str, ...] = ("off", "on")
DOE_SEEDS: tuple[int, ...] = (42, 1337, 2026, 777, 999)

CSV_COLUMNS: list[str] = [
    "run_id",
    "loss",
    "distillation",
    "seed",
    "macro_f1",
    "roc_auc",
    "fpr_at_95_tpr",
    "notinject_fpr",
    "ece",
]

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "fpr_at_95_tpr": {
        "label": "In-Domain FPR @ 95% TPR",
        "short": "FPR @ 95% TPR",
        "direction": "minimize",
        "as_percent": True,
        "decimals": 4,
    },
    "notinject_fpr": {
        "label": "OOD NotInject FPR (at tau_95)",
        "short": "NotInject FPR",
        "direction": "minimize",
        "as_percent": True,
        "decimals": 4,
    },
    "macro_f1": {
        "label": "In-Domain 3-Class Macro-F1",
        "short": "Macro-F1",
        "direction": "maximize",
        "as_percent": False,
        "decimals": 4,
    },
    "ece": {
        "label": "Expected Calibration Error (ECE)",
        "short": "ECE",
        "direction": "minimize",
        "as_percent": False,
        "decimals": 4,
    },
    "roc_auc": {
        "label": "Binary Malicious ROC-AUC",
        "short": "ROC-AUC",
        "direction": "maximize",
        "as_percent": False,
        "decimals": 5,
    },
}

DEFAULT_RESPONSES: list[str] = ["fpr_at_95_tpr", "notinject_fpr", "macro_f1"]

REPORT_START_MARKER = "<!-- DOE_RESULTS_START -->"
REPORT_END_MARKER = "<!-- DOE_RESULTS_END -->"


def expected_run_key(loss: str, distillation: str, seed: int) -> str:
    tag = f"{loss}_no_distill" if distillation == "off" else loss
    return f"deberta_v3_small_{tag}_seed{seed}"


def load_doe_raw_dataframe(
    preds_dir: Path,
    test_parquet: Path,
    notinject_parquet: Path,
    losses: tuple[str, ...] = DOE_LOSSES,
    distill_levels: tuple[str, ...] = DOE_DISTILL_LEVELS,
    seeds: tuple[int, ...] = DOE_SEEDS,
) -> tuple[pd.DataFrame, list[str]]:
    test_df = pd.read_parquet(test_parquet) if test_parquet.exists() else None
    has_notinject = notinject_parquet.exists()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for loss in losses:
        for distill in distill_levels:
            for seed in seeds:
                run_key = expected_run_key(loss, distill, seed)
                npz_path = preds_dir / f"{run_key}_test.npz"
                meta_path = preds_dir / f"{run_key}_meta.json"
                if not (npz_path.exists() and meta_path.exists()):
                    missing.append(run_key)
                    continue

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                npz = np.load(npz_path)

                y_true = npz["y_true"] if "y_true" in npz else (
                    test_df["label"].values if test_df is not None else None
                )
                if y_true is None:
                    raise ValueError(f"Cannot resolve ground-truth labels for {run_key}")

                probs = npz["probs"]
                tau_95 = float(npz["tau_95"]) if "tau_95" in npz else meta.get("tau_95")
                tau_1pct = float(npz["tau_1pct"]) if "tau_1pct" in npz else meta.get("tau_1pct")
                metrics = compute_comprehensive_metrics(
                    y_true=y_true,
                    probs=probs,
                    tau_95=tau_95,
                    tau_1pct=tau_1pct,
                )

                notinject_fpr = float("nan")
                if "notinject_probs" in npz:
                    ni_scores = compute_binary_malicious_score(npz["notinject_probs"])
                    notinject_fpr = float(np.mean(ni_scores >= metrics["tau_95"]))
                elif has_notinject and "notinject_fpr" in meta.get("metrics", {}):
                    notinject_fpr = float(meta["metrics"]["notinject_fpr"])

                rows.append(
                    {
                        "run_id": run_key,
                        "loss": loss,
                        "distillation": distill,
                        "seed": int(seed),
                        "macro_f1": float(metrics["macro_f1"]),
                        "roc_auc": float(metrics["roc_auc"]),
                        "fpr_at_95_tpr": float(metrics["fpr_at_95_tpr"]),
                        "notinject_fpr": notinject_fpr,
                        "ece": float(metrics["ece"]),
                    }
                )

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    return df, missing


def _summarize_series(vals: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(vals, dtype=float)
    n = int(len(arr))
    mean = float(np.mean(arr)) if n > 0 else float("nan")
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr)) if n > 0 else float("nan")
    q75, q25 = np.percentile(arr, [75, 25]) if n > 0 else (float("nan"), float("nan"))
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "median": median,
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)) if n > 0 else float("nan"),
        "max": float(np.max(arr)) if n > 0 else float("nan"),
        "ci95_low": mean - t_crit * se,
        "ci95_high": mean + t_crit * se,
    }


def compute_descriptives(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    by_loss = {
        lvl: _summarize_series(df.loc[df["loss"] == lvl, metric].to_numpy())
        for lvl in DOE_LOSSES
        if (df["loss"] == lvl).any()
    }
    by_distill = {
        lvl: _summarize_series(df.loc[df["distillation"] == lvl, metric].to_numpy())
        for lvl in DOE_DISTILL_LEVELS
        if (df["distillation"] == lvl).any()
    }
    by_cell: dict[str, dict[str, float | int]] = {}
    for loss in DOE_LOSSES:
        for distill in DOE_DISTILL_LEVELS:
            sub = df.loc[(df["loss"] == loss) & (df["distillation"] == distill), metric].to_numpy()
            if len(sub) > 0:
                by_cell[f"{loss}__{distill}"] = _summarize_series(sub)
    return {
        "overall": _summarize_series(df[metric].to_numpy()),
        "by_loss": by_loss,
        "by_distillation": by_distill,
        "by_cell": by_cell,
    }


def compute_two_way_anova(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> dict[str, Any]:
    formula = f"{metric} ~ C(loss, Sum) * C(distillation, Sum)"
    model = ols(formula, data=df).fit()
    anova_t3 = sm.stats.anova_lm(model, typ=3)
    anova_t2 = sm.stats.anova_lm(model, typ=2)

    term_map = [
        ("C(loss, Sum)", "Loss"),
        ("C(distillation, Sum)", "Distillation"),
        ("C(loss, Sum):C(distillation, Sum)", "Loss:Distillation"),
        ("Residual", "Residual"),
    ]
    ss_total = float(sum(anova_t3.loc[raw, "sum_sq"] for raw, _ in term_map))
    ss_resid = float(anova_t3.loc["Residual", "sum_sq"])
    df_resid = int(round(float(anova_t3.loc["Residual", "df"])))
    ms_resid = ss_resid / df_resid if df_resid > 0 else float("nan")

    table_rows: list[dict[str, Any]] = []
    for raw_key, display_name in term_map:
        ss = float(anova_t3.loc[raw_key, "sum_sq"])
        df_val = int(round(float(anova_t3.loc[raw_key, "df"])))
        ms = ss / df_val if df_val > 0 else float("nan")
        if raw_key == "Residual":
            f_val = None
            p_val = None
            eta_sq = ss / ss_total if ss_total > 0 else 0.0
            partial_eta_sq = None
            omega_sq = None
        else:
            f_val = float(anova_t3.loc[raw_key, "F"])
            p_val = float(anova_t3.loc[raw_key, "PR(>F)"])
            eta_sq = ss / ss_total if ss_total > 0 else 0.0
            partial_eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
            omega_sq = (
                max(0.0, (ss - df_val * ms_resid) / (ss_total + ms_resid))
                if (ss_total + ms_resid) > 0
                else 0.0
            )
        table_rows.append(
            {
                "source": display_name,
                "ss": ss,
                "df": df_val,
                "ms": ms,
                "f": f_val,
                "p_value": p_val,
                "significant": (p_val is not None and p_val < alpha),
                "eta_sq": eta_sq,
                "partial_eta_sq": partial_eta_sq,
                "omega_sq": omega_sq,
            }
        )

    table_rows.append(
        {
            "source": "Total",
            "ss": ss_total,
            "df": int(len(df) - 1),
            "ms": ss_total / max(1, len(df) - 1),
            "f": None,
            "p_value": None,
            "significant": False,
            "eta_sq": 1.0,
            "partial_eta_sq": None,
            "omega_sq": None,
        }
    )

    type2_max_diff = max(
        abs(float(anova_t3.loc[raw, "sum_sq"]) - float(anova_t2.loc[raw, "sum_sq"]))
        for raw, _ in term_map
    )

    residuals = model.resid.to_numpy(dtype=float)
    fitted = model.fittedvalues.to_numpy(dtype=float)
    shapiro_w, shapiro_p = stats.shapiro(residuals)

    groups = [
        df.loc[(df["loss"] == loss) & (df["distillation"] == distill), metric].to_numpy(dtype=float)
        for loss in DOE_LOSSES
        for distill in DOE_DISTILL_LEVELS
    ]
    lev_mean_f, lev_mean_p = stats.levene(*groups, center="mean")
    lev_med_f, lev_med_p = stats.levene(*groups, center="median")
    kw_stat, kw_p = stats.kruskal(*groups)

    cell_labels = (df["loss"] + "+" + df["distillation"]).to_numpy()
    tukey = pairwise_tukeyhsd(endog=df[metric].to_numpy(dtype=float), groups=cell_labels, alpha=alpha)
    tukey_rows: list[dict[str, Any]] = []
    for row in tukey._results_table.data[1:]:
        tukey_rows.append(
            {
                "group1": str(row[0]),
                "group2": str(row[1]),
                "meandiff": float(row[2]),
                "p_adj": float(row[3]),
                "lower": float(row[4]),
                "upper": float(row[5]),
                "reject": bool(row[6]),
            }
        )

    return {
        "formula": formula,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "type2_type3_max_ss_diff": float(type2_max_diff),
        "anova_table": table_rows,
        "residuals": residuals.tolist(),
        "fitted": fitted.tolist(),
        "assumptions": {
            "shapiro_wilk": {
                "W": float(shapiro_w),
                "p_value": float(shapiro_p),
                "passed": bool(shapiro_p >= alpha),
            },
            "levene_mean": {
                "F": float(lev_mean_f),
                "df1": len(groups) - 1,
                "df2": int(len(df) - len(groups)),
                "p_value": float(lev_mean_p),
                "passed": bool(lev_mean_p >= alpha),
            },
            "levene_median_brown_forsythe": {
                "F": float(lev_med_f),
                "df1": len(groups) - 1,
                "df2": int(len(df) - len(groups)),
                "p_value": float(lev_med_p),
                "passed": bool(lev_med_p >= alpha),
            },
            "kruskal_wallis_sensitivity": {
                "H": float(kw_stat),
                "df": len(groups) - 1,
                "p_value": float(kw_p),
            },
        },
        "tukey_hsd": tukey_rows,
    }


def _configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_factor_boxplot(
    df: pd.DataFrame,
    metric: str,
    factor: str,
    levels: tuple[str, ...],
    out_path: Path,
) -> None:
    spec = METRIC_SPECS[metric]
    plot_df = df.copy()
    scale = 100.0 if spec["as_percent"] else 1.0
    plot_df["val_plot"] = plot_df[metric] * scale
    ylabel = f"{spec['label']} (%)" if spec["as_percent"] else spec["label"]
    factor_title = "Loss Function (Factor A)" if factor == "loss" else "Teacher Distillation (Factor B)"

    palette = ["#4C72B0", "#DD8452"] if factor == "loss" else ["#8172B3", "#55A868"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    sns.boxplot(
        data=plot_df,
        x=factor,
        y="val_plot",
        order=list(levels),
        hue=factor,
        hue_order=list(levels),
        palette=palette,
        width=0.48,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#1F1F1F",
            "markersize": 6.5,
        },
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x=factor,
        y="val_plot",
        order=list(levels),
        color="#222222",
        alpha=0.75,
        size=5.5,
        jitter=0.08,
        ax=ax,
    )

    for idx, lvl in enumerate(levels):
        sub = plot_df.loc[plot_df[factor] == lvl, "val_plot"].to_numpy()
        if len(sub) > 0:
            mean_v = float(np.mean(sub))
            sd_v = float(np.std(sub, ddof=1)) if len(sub) > 1 else 0.0
            unit = "%" if spec["as_percent"] else ""
            fmt = f"{mean_v:.2f}{unit} ± {sd_v:.2f}{unit}" if spec["as_percent"] else f"{mean_v:.4f} ± {sd_v:.4f}"
            ax.annotate(
                f"μ = {fmt}\n(n={len(sub)})",
                xy=(idx, float(np.max(sub))),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8F9FA", "edgecolor": "#CCCCCC"},
            )

    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.18 * span)
    ax.set_xlabel(factor_title)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{spec['short']} by {factor_title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_qq_residuals(
    residuals: np.ndarray,
    metric: str,
    shapiro_w: float,
    shapiro_p: float,
    levene_f: float,
    levene_p: float,
    out_path: Path,
) -> None:
    spec = METRIC_SPECS[metric]
    res = np.asarray(residuals, dtype=float)
    std_res = (res - np.mean(res)) / (np.std(res, ddof=1) if np.std(res, ddof=1) > 0 else 1.0)
    (osm, osr), (slope, intercept, _) = stats.probplot(std_res, dist="norm")

    n = len(std_res)
    se = (slope / stats.norm.pdf(osm)) * np.sqrt((stats.norm.cdf(osm) * (1 - stats.norm.cdf(osm))) / n)
    z_crit = stats.norm.ppf(0.975)
    fit_line = slope * osm + intercept

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.fill_between(osm, fit_line - z_crit * se, fit_line + z_crit * se, color="#4C72B0", alpha=0.14, label="95% Normal Band")
    ax.plot(osm, fit_line, color="#C44E52", linewidth=1.8, linestyle="--", label="Reference Normal Line")
    ax.scatter(osm, osr, color="#1F4E79", edgecolor="white", s=48, zorder=3, label="Standardized Residuals")

    ax.text(
        0.04,
        0.95,
        f"Shapiro-Wilk: W = {shapiro_w:.4f}, p = {shapiro_p:.4f}\n"
        f"Levene (mean): F(3,16) = {levene_f:.3f}, p = {levene_p:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.92},
    )
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Standardized ANOVA Residuals")
    ax.set_title(f"Normal Probability (Q-Q) Plot of Residuals — {spec['short']}")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_interaction(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    spec = METRIC_SPECS[metric]
    scale = 100.0 if spec["as_percent"] else 1.0
    ylabel = f"{spec['label']} (%)" if spec["as_percent"] else spec["label"]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x_positions = np.arange(len(DOE_LOSSES), dtype=float)
    styles = {
        "off": {"color": "#8172B3", "marker": "o", "linestyle": "--", "label": "Distillation = OFF"},
        "on": {"color": "#55A868", "marker": "s", "linestyle": "-", "label": "Distillation = ON"},
    }

    for offset, distill in zip((-0.04, 0.04), DOE_DISTILL_LEVELS, strict=True):
        means: list[float] = []
        ci_half: list[float] = []
        for loss in DOE_LOSSES:
            vals = df.loc[(df["loss"] == loss) & (df["distillation"] == distill), metric].to_numpy() * scale
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            half = float(stats.t.ppf(0.975, df=len(vals) - 1)) * (sd_v / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            means.append(mean_v)
            ci_half.append(half)

        st = styles[distill]
        ax.errorbar(
            x_positions + offset,
            means,
            yerr=ci_half,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=2.0,
            markersize=7.5,
            capsize=5,
            label=st["label"],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Loss = {lvl}" for lvl in DOE_LOSSES])
    ax.set_xlim(-0.35, 1.35)
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.16 * span)
    ax.set_xlabel("Loss Function (Factor A)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Interaction Plot (Cell Means ± 95% CI) — {spec['short']}")
    ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _fmt_metric(metric: str, val: float) -> str:
    spec = METRIC_SPECS[metric]
    if math.isnan(val):
        return "N/A"
    if spec["as_percent"]:
        return f"{val * 100:.2f}%"
    return f"{val:.{spec['decimals']}f}"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


def recommend_best_combination(
    summary_by_metric: dict[str, dict[str, Any]],
    responses: list[str],
) -> dict[str, Any]:
    cell_keys = [f"{loss}__{distill}" for loss in DOE_LOSSES for distill in DOE_DISTILL_LEVELS]
    cell_scores: dict[str, dict[str, Any]] = {}

    for ck in cell_keys:
        loss, distill = ck.split("__", 1)
        cell_scores[ck] = {"loss": loss, "distillation": distill, "metrics": {}, "ranks": {}}

    for metric in responses:
        desc = summary_by_metric[metric]["descriptives"]["by_cell"]
        direction = METRIC_SPECS[metric]["direction"]
        ordered_cells = sorted(
            cell_keys,
            key=lambda c: desc[c]["mean"],
            reverse=(direction == "maximize"),
        )
        for rank_idx, ck in enumerate(ordered_cells, start=1):
            cell_scores[ck]["metrics"][metric] = {
                "mean": desc[ck]["mean"],
                "sd": desc[ck]["sd"],
            }
            cell_scores[ck]["ranks"][metric] = rank_idx

    weights = {"fpr_at_95_tpr": 0.45, "notinject_fpr": 0.35, "macro_f1": 0.20}
    for ck in cell_keys:
        total_w = 0.0
        weighted_rank = 0.0
        for m in responses:
            w = weights.get(m, 1.0 / max(1, len(responses)))
            weighted_rank += w * cell_scores[ck]["ranks"][m]
            total_w += w
        cell_scores[ck]["weighted_rank"] = weighted_rank / max(total_w, 1e-9)

    best_key = min(
        cell_keys,
        key=lambda c: (
            cell_scores[c]["weighted_rank"],
            cell_scores[c]["ranks"].get("fpr_at_95_tpr", 99),
        ),
    )
    best = cell_scores[best_key]
    return {
        "recommended_cell": best_key,
        "recommended_loss": best["loss"],
        "recommended_distillation": best["distillation"],
        "ranking_weights": {m: weights.get(m, 1.0 / len(responses)) for m in responses},
        "cells": cell_scores,
    }


def _interpret_anova(metric: str, res: dict[str, Any], alpha: float) -> str:
    spec = METRIC_SPECS[metric]
    rows = {r["source"]: r for r in res["anova"]["anova_table"]}
    desc = res["descriptives"]
    a = rows["Loss"]
    b = rows["Distillation"]
    ab = rows["Loss:Distillation"]
    shapiro = res["anova"]["assumptions"]["shapiro_wilk"]
    levene = res["anova"]["assumptions"]["levene_mean"]

    parts: list[str] = []
    if shapiro["passed"] and levene["passed"]:
        parts.append(
            f"Both ANOVA parametric assumptions hold at $\\alpha = {alpha:.2f}$ "
            f"(Shapiro-Wilk $W = {shapiro['W']:.4f}, p = {_fmt_p(shapiro['p_value'])}$; "
            f"Levene $F(3, 16) = {levene['F']:.3f}, p = {_fmt_p(levene['p_value'])}$)."
        )
    else:
        kw = res["anova"]["assumptions"]["kruskal_wallis_sensitivity"]
        parts.append(
            f"Assumption diagnostic caution: Shapiro-Wilk $p = {_fmt_p(shapiro['p_value'])}$, "
            f"Levene $p = {_fmt_p(levene['p_value'])}$; non-parametric Kruskal-Wallis sensitivity check across the 4 cells yields "
            f"$H(3) = {kw['H']:.3f}, p = {_fmt_p(kw['p_value'])}$."
        )

    if ab["significant"]:
        parts.append(
            f"A statistically significant **Loss × Distillation interaction** is observed "
            f"($F(1, 16) = {ab['f']:.3f}, p = {_fmt_p(ab['p_value'])}, \\eta^2 = {ab['eta_sq']:.3f}, "
            f"\\eta_p^2 = {ab['partial_eta_sq']:.3f}$), so main effects must be interpreted jointly via cell simple effects."
        )
    else:
        parts.append(
            f"No statistically significant **Loss × Distillation interaction** is detected "
            f"($F(1, 16) = {ab['f']:.3f}, p = {_fmt_p(ab['p_value'])}, \\eta^2 = {ab['eta_sq']:.3f}$), "
            f"allowing direct interpretation of the main effects."
        )

    ce_m = _fmt_metric(metric, desc["by_loss"]["ce"]["mean"])
    cs_m = _fmt_metric(metric, desc["by_loss"]["cs_afl"]["mean"])
    off_m = _fmt_metric(metric, desc["by_distillation"]["off"]["mean"])
    on_m = _fmt_metric(metric, desc["by_distillation"]["on"]["mean"])

    if a["significant"]:
        parts.append(
            f"**Factor A (Loss)** has a significant main effect ($F(1, 16) = {a['f']:.3f}, "
            f"p = {_fmt_p(a['p_value'])}, \\eta^2 = {a['eta_sq']:.3f}$): `cs_afl` averages {cs_m} vs `ce` at {ce_m}."
        )
    else:
        parts.append(
            f"**Factor A (Loss)** does not reach statistical significance at $\\alpha = {alpha:.2f}$ "
            f"($F(1, 16) = {a['f']:.3f}, p = {_fmt_p(a['p_value'])}, \\eta^2 = {a['eta_sq']:.3f}$; "
            f"`cs_afl` = {cs_m} vs `ce` = {ce_m})."
        )

    if b["significant"]:
        parts.append(
            f"**Factor B (Distillation)** has a significant main effect ($F(1, 16) = {b['f']:.3f}, "
            f"p = {_fmt_p(b['p_value'])}, \\eta^2 = {b['eta_sq']:.3f}$): `on` averages {on_m} vs `off` at {off_m}."
        )
    else:
        parts.append(
            f"**Factor B (Distillation)** does not reach statistical significance at $\\alpha = {alpha:.2f}$ "
            f"($F(1, 16) = {b['f']:.3f}, p = {_fmt_p(b['p_value'])}, \\eta^2 = {b['eta_sq']:.3f}$; "
            f"`on` = {on_m} vs `off` = {off_m})."
        )

    best_cell_key = (
        min(desc["by_cell"], key=lambda c: desc["by_cell"][c]["mean"])
        if spec["direction"] == "minimize"
        else max(desc["by_cell"], key=lambda c: desc["by_cell"][c]["mean"])
    )
    bl, bd = best_cell_key.split("__", 1)
    bc = desc["by_cell"][best_cell_key]
    parts.append(
        f"Best cell mean for **{spec['short']}** ({spec['direction']}): "
        f"`(loss={bl}, distillation={bd})` with mean {_fmt_metric(metric, bc['mean'])} (SD = {_fmt_metric(metric, bc['sd'])})."
    )
    return " ".join(parts)


def render_section3_markdown(
    df: pd.DataFrame,
    summary_by_metric: dict[str, dict[str, Any]],
    recommendation: dict[str, Any],
    responses: list[str],
    fig_rel_prefix: str,
    alpha: float,
) -> str:
    lines: list[str] = []

    lines.append("### 3.1 Raw Experimental Data ($N = 20$ Runs)")
    lines.append("")
    lines.append(
        "Complete tidy dataset exported to `reports/doe/doe_raw_data.csv` (directly importable in JASP, Jamovi, and SPSS). "
        "All operating points (`tau_95`) and temperatures (`T`) were calibrated strictly on Validation (`N = 4,439`) prior to blind Test (`N = 4,438`) and OOD `NotInject` (`N = 339`) evaluation."
    )
    lines.append("")
    lines.append("| # | Run ID | Loss (A) | Distillation (B) | Seed | FPR @ 95% TPR | NotInject FPR | Macro-F1 | ROC-AUC | ECE |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    df_sorted = df.sort_values(["loss", "distillation", "seed"]).reset_index(drop=True)
    for idx, row in df_sorted.iterrows():
        lines.append(
            f"| {idx + 1} | `{row['run_id']}` | `{row['loss']}` | `{row['distillation']}` | `{int(row['seed'])}` | "
            f"{row['fpr_at_95_tpr'] * 100:.2f}% ({row['fpr_at_95_tpr']:.5f}) | "
            f"{row['notinject_fpr'] * 100:.2f}% ({row['notinject_fpr']:.5f}) | "
            f"{row['macro_f1']:.4f} | {row['roc_auc']:.5f} | {row['ece']:.4f} |"
        )
    lines.append("")

    lines.append("### 3.2 Descriptive Statistics and Factor Boxplots")
    lines.append("")
    for metric in responses:
        spec = METRIC_SPECS[metric]
        desc = summary_by_metric[metric]["descriptives"]
        lines.append(f"#### {spec['label']} (`{metric}`)")
        lines.append("")
        lines.append("| Grouping | Level / Cell | $n$ | Mean | SD | Median | IQR | Min | Max |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for lvl in DOE_LOSSES:
            s = desc["by_loss"][lvl]
            lines.append(
                f"| Factor A (Loss) | `{lvl}` | {s['n']} | {_fmt_metric(metric, s['mean'])} | "
                f"{_fmt_metric(metric, s['sd'])} | {_fmt_metric(metric, s['median'])} | "
                f"{_fmt_metric(metric, s['iqr'])} | {_fmt_metric(metric, s['min'])} | {_fmt_metric(metric, s['max'])} |"
            )
        for lvl in DOE_DISTILL_LEVELS:
            s = desc["by_distillation"][lvl]
            lines.append(
                f"| Factor B (Distillation) | `{lvl}` | {s['n']} | {_fmt_metric(metric, s['mean'])} | "
                f"{_fmt_metric(metric, s['sd'])} | {_fmt_metric(metric, s['median'])} | "
                f"{_fmt_metric(metric, s['iqr'])} | {_fmt_metric(metric, s['min'])} | {_fmt_metric(metric, s['max'])} |"
            )
        for loss in DOE_LOSSES:
            for distill in DOE_DISTILL_LEVELS:
                s = desc["by_cell"][f"{loss}__{distill}"]
                lines.append(
                    f"| Cell (A × B) | `({loss}, {distill})` | {s['n']} | {_fmt_metric(metric, s['mean'])} | "
                    f"{_fmt_metric(metric, s['sd'])} | {_fmt_metric(metric, s['median'])} | "
                    f"{_fmt_metric(metric, s['iqr'])} | {_fmt_metric(metric, s['min'])} | {_fmt_metric(metric, s['max'])} |"
                )
        lines.append("")
        lines.append(f"![Boxplot {spec['short']} by Loss]({fig_rel_prefix}/boxplot_{metric}_by_loss.png)")
        lines.append("")
        lines.append(f"![Boxplot {spec['short']} by Distillation]({fig_rel_prefix}/boxplot_{metric}_by_distillation.png)")
        lines.append("")

    lines.append("### 3.3 Model Validation (Residual Normality Q-Q Plots & Levene's Test)")
    lines.append("")
    lines.append("| Response Variable | Shapiro-Wilk $W$ | Shapiro-Wilk $p$ | Normality ($\\alpha=0.05$) | Levene $F(3,16)$ (mean) | Levene $p$ | Homoscedasticity ($\\alpha=0.05$) | Brown-Forsythe $p$ |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for metric in responses:
        spec = METRIC_SPECS[metric]
        a = summary_by_metric[metric]["anova"]["assumptions"]
        sw = a["shapiro_wilk"]
        lm = a["levene_mean"]
        bf = a["levene_median_brown_forsythe"]
        lines.append(
            f"| **{spec['short']}** (`{metric}`) | {sw['W']:.4f} | {_fmt_p(sw['p_value'])} | "
            f"{'PASS' if sw['passed'] else 'VIOLATION'} | {lm['F']:.3f} | {_fmt_p(lm['p_value'])} | "
            f"{'PASS' if lm['passed'] else 'VIOLATION'} | {_fmt_p(bf['p_value'])} |"
        )
    lines.append("")
    for metric in responses:
        spec = METRIC_SPECS[metric]
        lines.append(f"![Normal Probability Q-Q Plot — {spec['short']}]({fig_rel_prefix}/qqplot_{metric}_residuals.png)")
        lines.append("")

    lines.append("### 3.4 Two-Way ANOVA Tables, Interaction Plots, and Interpretation")
    lines.append("")
    for metric in responses:
        spec = METRIC_SPECS[metric]
        anova = summary_by_metric[metric]["anova"]
        lines.append(f"#### Two-Way ANOVA — {spec['label']} (`{metric}`)")
        lines.append("")
        lines.append(
            f"Model fit: $R^2 = {anova['r_squared']:.4f}$, Adjusted $R^2 = {anova['adj_r_squared']:.4f}$ "
            f"(balanced design verification: $\\max |SS_{{\\text{{II}}}} - SS_{{\\text{{III}}}}| = {anova['type2_type3_max_ss_diff']:.2e}$)."
        )
        lines.append("")
        lines.append("| Source | $SS$ | $df$ | $MS$ | $F$ | $p$-value | $\\eta^2$ | $\\eta_p^2$ |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in anova["anova_table"]:
            f_str = f"{r['f']:.4f}" if r["f"] is not None else "—"
            p_str = _fmt_p(r["p_value"])
            if r["significant"]:
                p_str = f"**{p_str}***"
            pe_str = f"{r['partial_eta_sq']:.4f}" if r["partial_eta_sq"] is not None else "—"
            lines.append(
                f"| {r['source']} | {r['ss']:.6e} | {r['df']} | {r['ms']:.6e} | {f_str} | {p_str} | {r['eta_sq']:.4f} | {pe_str} |"
            )
        lines.append("")
        lines.append(f"![Interaction Plot — {spec['short']}]({fig_rel_prefix}/interaction_{metric}.png)")
        lines.append("")
        lines.append(f"**Interpretation ({spec['short']}):** {_interpret_anova(metric, summary_by_metric[metric], alpha)}")
        lines.append("")

    lines.append("### 3.5 Recommendation of the Best Treatment Combination")
    lines.append("")
    rec_loss = recommendation["recommended_loss"]
    rec_dist = recommendation["recommended_distillation"]
    lines.append(
        f"Synthesizing the primary inline-security objective (`fpr_at_95_tpr`, weight 45%), "
        f"OOD benign trigger-word resistance (`notinject_fpr`, weight 35%), and 3-class utility (`macro_f1`, weight 20%), "
        f"the recommended treatment combination is **`loss = {rec_loss}` + `distillation = {rec_dist}`**."
    )
    lines.append("")
    lines.append("| Treatment `(Loss, Distillation)` | FPR @ 95% TPR (Mean ± SD) | NotInject FPR (Mean ± SD) | Macro-F1 (Mean ± SD) | Weighted Rank |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    ranked_cells = sorted(recommendation["cells"].items(), key=lambda kv: kv[1]["weighted_rank"])
    for _, c in ranked_cells:
        tag = f"`({c['loss']}, {c['distillation']})`"
        if c["loss"] == rec_loss and c["distillation"] == rec_dist:
            tag = f"**{tag} (Recommended)**"
        m_fpr = c["metrics"].get("fpr_at_95_tpr")
        m_ni = c["metrics"].get("notinject_fpr")
        m_f1 = c["metrics"].get("macro_f1")
        s_fpr = f"{m_fpr['mean']*100:.2f}% ± {m_fpr['sd']*100:.2f}%" if m_fpr else "—"
        s_ni = f"{m_ni['mean']*100:.2f}% ± {m_ni['sd']*100:.2f}%" if m_ni else "—"
        s_f1 = f"{m_f1['mean']:.4f} ± {m_f1['sd']:.4f}" if m_f1 else "—"
        lines.append(f"| {tag} | {s_fpr} | {s_ni} | {s_f1} | {c['weighted_rank']:.2f} |")
    lines.append("")

    return "\n".join(lines)


def inject_into_report(report_path: Path, body_md: str) -> None:
    content = report_path.read_text(encoding="utf-8")
    if REPORT_START_MARKER not in content or REPORT_END_MARKER not in content:
        raise ValueError(
            f"Markers {REPORT_START_MARKER} / {REPORT_END_MARKER} not found in {report_path}"
        )
    before, rest = content.split(REPORT_START_MARKER, 1)
    _, after = rest.split(REPORT_END_MARKER, 1)
    updated = f"{before}{REPORT_START_MARKER}\n\n{body_md}\n{REPORT_END_MARKER}{after}"
    report_path.write_text(updated, encoding="utf-8")


def run_analysis(
    preds_dir: Path,
    out_dir: Path,
    test_parquet: Path,
    notinject_parquet: Path,
    responses: list[str],
    alpha: float = 0.05,
    allow_partial: bool = False,
    update_report: Path | None = None,
) -> dict[str, Any]:
    df, missing = load_doe_raw_dataframe(
        preds_dir=preds_dir,
        test_parquet=test_parquet,
        notinject_parquet=notinject_parquet,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "doe_raw_data.csv"
    df.sort_values(["loss", "distillation", "seed"]).to_csv(csv_path, index=False)

    if missing and not allow_partial:
        raise RuntimeError(
            f"Incomplete DOE matrix: {len(df)}/20 runs found, {len(missing)} missing: {missing}. "
            "Pass --allow-partial to export partial CSV/descriptives without balanced ANOVA."
        )

    if missing:
        partial_payload = {
            "complete": False,
            "found_runs": len(df),
            "expected_runs": len(DOE_LOSSES) * len(DOE_DISTILL_LEVELS) * len(DOE_SEEDS),
            "missing_runs": missing,
            "csv_path": str(csv_path),
            "descriptives": {m: compute_descriptives(df, m) for m in responses if len(df) > 0},
        }
        (out_dir / "doe_anova_summary.json").write_text(
            json.dumps(partial_payload, indent=2), encoding="utf-8"
        )
        return partial_payload

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    summary_by_metric: dict[str, dict[str, Any]] = {}
    for metric in responses:
        desc = compute_descriptives(df, metric)
        anova = compute_two_way_anova(df, metric, alpha=alpha)

        plot_factor_boxplot(df, metric, "loss", DOE_LOSSES, fig_dir / f"boxplot_{metric}_by_loss.png")
        plot_factor_boxplot(
            df, metric, "distillation", DOE_DISTILL_LEVELS, fig_dir / f"boxplot_{metric}_by_distillation.png"
        )
        plot_qq_residuals(
            residuals=np.array(anova["residuals"], dtype=float),
            metric=metric,
            shapiro_w=anova["assumptions"]["shapiro_wilk"]["W"],
            shapiro_p=anova["assumptions"]["shapiro_wilk"]["p_value"],
            levene_f=anova["assumptions"]["levene_mean"]["F"],
            levene_p=anova["assumptions"]["levene_mean"]["p_value"],
            out_path=fig_dir / f"qqplot_{metric}_residuals.png",
        )
        plot_interaction(df, metric, fig_dir / f"interaction_{metric}.png")

        summary_by_metric[metric] = {
            "meta": METRIC_SPECS[metric],
            "descriptives": desc,
            "anova": anova,
            "figures": {
                "boxplot_loss": str(fig_dir / f"boxplot_{metric}_by_loss.png"),
                "boxplot_distillation": str(fig_dir / f"boxplot_{metric}_by_distillation.png"),
                "qqplot_residuals": str(fig_dir / f"qqplot_{metric}_residuals.png"),
                "interaction": str(fig_dir / f"interaction_{metric}.png"),
            },
        }

    recommendation = recommend_best_combination(summary_by_metric, responses)
    payload: dict[str, Any] = {
        "complete": True,
        "design": "2x2 Full Factorial CRFD (5 replicates per cell, N=20)",
        "alpha": alpha,
        "responses": responses,
        "csv_path": str(csv_path),
        "results": summary_by_metric,
        "recommendation": recommendation,
    }
    (out_dir / "doe_anova_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    local_md = render_section3_markdown(
        df=df,
        summary_by_metric=summary_by_metric,
        recommendation=recommendation,
        responses=responses,
        fig_rel_prefix="figures",
        alpha=alpha,
    )
    (out_dir / "doe_analysis_report.md").write_text(
        "# AegisLLM — Milestone 2 DOE Statistical Analysis Summary\n\n" + local_md,
        encoding="utf-8",
    )

    if update_report is not None:
        rel_fig = "../reports/doe/figures"
        report_md = render_section3_markdown(
            df=df,
            summary_by_metric=summary_by_metric,
            recommendation=recommendation,
            responses=responses,
            fig_rel_prefix=rel_fig,
            alpha=alpha,
        )
        inject_into_report(update_report, report_md)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AegisLLM 2x2x5 factorial DOE via Two-Way ANOVA.")
    parser.add_argument("--preds-dir", type=Path, default=REPO_ROOT / "reports/predictions")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports/doe")
    parser.add_argument("--test-parquet", type=Path, default=REPO_ROOT / "data/processed/test.parquet")
    parser.add_argument("--notinject-parquet", type=Path, default=REPO_ROOT / "data/ood/notinject.parquet")
    parser.add_argument(
        "--responses",
        nargs="+",
        choices=list(METRIC_SPECS.keys()),
        default=DEFAULT_RESPONSES,
        help="Response variables to analyze (default: fpr_at_95_tpr notinject_fpr macro_f1).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--allow-partial", action="store_true", help="Export partial CSV without balanced ANOVA.")
    parser.add_argument(
        "--update-report",
        type=Path,
        default=None,
        help="Optional Markdown report path containing DOE_RESULTS_START/END markers to populate.",
    )
    args = parser.parse_args()

    preds_dir = args.preds_dir if args.preds_dir.is_absolute() else (REPO_ROOT / args.preds_dir)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir)
    test_parquet = args.test_parquet if args.test_parquet.is_absolute() else (REPO_ROOT / args.test_parquet)
    notinject_parquet = (
        args.notinject_parquet if args.notinject_parquet.is_absolute() else (REPO_ROOT / args.notinject_parquet)
    )
    update_report = (
        (args.update_report if args.update_report.is_absolute() else (REPO_ROOT / args.update_report))
        if args.update_report is not None
        else None
    )

    result = run_analysis(
        preds_dir=preds_dir,
        out_dir=out_dir,
        test_parquet=test_parquet,
        notinject_parquet=notinject_parquet,
        responses=args.responses,
        alpha=args.alpha,
        allow_partial=args.allow_partial,
        update_report=update_report,
    )
    if result["complete"]:
        rec = result["recommendation"]
        print(
            f"[DOE ANOVA] Complete N=20 analysis written to {out_dir} | "
            f"Recommended: (loss={rec['recommended_loss']}, distillation={rec['recommended_distillation']})"
        )
    else:
        print(
            f"[DOE ANOVA] Partial export ({result['found_runs']}/{result['expected_runs']} runs) "
            f"written to {result['csv_path']}."
        )


if __name__ == "__main__":
    main()
