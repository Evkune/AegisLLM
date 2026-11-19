#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aegis_dl.metrics import (
    compute_binary_malicious_score,
    compute_comprehensive_metrics,
    compute_mcnemar_test,
    compute_paired_bootstrap_ci,
    macro_f1_from_probs,
)


def _resolve_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _discover_prediction_artifacts(preds_dir: Path) -> List[Dict[str, Any]]:
    if not preds_dir.exists():
        return []

    npz_files = sorted(preds_dir.glob("*_test.npz"))
    if not npz_files:
        npz_files = sorted(preds_dir.glob("*.npz"))

    discovered: List[Dict[str, Any]] = []
    for npz_path in npz_files:
        stem = npz_path.name[:-9] if npz_path.name.endswith("_test.npz") else npz_path.stem
        meta_path = preds_dir / f"{stem}_meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        discovered.append(
            {
                "model_key": meta.get("model_key", stem),
                "name": meta.get("name", stem),
                "npz_path": npz_path,
                "meta_path": meta_path if meta_path.exists() else None,
                "meta": meta,
            }
        )

    # Place baseline_tier1 first as the null-hypothesis reference (H0)
    discovered.sort(key=lambda item: (0 if item["model_key"] == "baseline_tier1" else 1, item["model_key"]))
    return discovered


def format_table_markdown(records: List[Dict[str, Any]]) -> str:
    headers = [
        "Model / Variant",
        "Params",
        "Macro F1 (95% CI)",
        "PR-AUC",
        "ROC-AUC",
        "FPR @ 95% TPR",
        "DevOps FPR",
        "XSTest Safe FPR",
        "NotInject FPR",
        "JBB TPR",
        "CPU P99 (ms)",
        "McNemar p-value",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in records:
        p99_str = f"{r['p99_latency_ms']:.1f} ms" if r.get("p99_latency_ms") is not None else "N/A"
        notinject_str = f"{r['notinject_fpr'] * 100:.1f}%" if r.get("notinject_fpr") is not None else "N/A"
        jbb_str = f"{r['jbb_harmful_tpr'] * 100:.1f}%" if r.get("jbb_harmful_tpr") is not None else "N/A"
        row = [
            f"**{r['name']}**",
            str(r.get("params", "N/A")),
            r["f1_ci"],
            f"{r['pr_auc']:.3f}",
            f"{r['roc_auc']:.3f}",
            f"{r['fpr_at_95_tpr'] * 100:.1f}%",
            f"{r['devops_hard_neg_fpr'] * 100:.1f}%",
            f"{r['xstest_safe_fpr'] * 100:.1f}%",
            notinject_str,
            jbb_str,
            p99_str,
            r["mcnemar_str"],
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_table_latex(records: List[Dict[str, Any]]) -> str:
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"\textbf{Model / Variant} & \textbf{Params} & \textbf{Macro F1 (95\% CI)} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{FPR@95\%TPR} & \textbf{DevOps FPR} & \textbf{XSTest FPR} & \textbf{NotInject FPR} & \textbf{JBB TPR} & \textbf{P99 (ms)} & \textbf{McNemar $p$} \\",
        r"\midrule",
    ]
    for r in records:
        p99_str = f"{r['p99_latency_ms']:.1f}" if r.get("p99_latency_ms") is not None else "N/A"
        notinject_str = f"{r['notinject_fpr'] * 100:.1f}\\%" if r.get("notinject_fpr") is not None else "N/A"
        jbb_str = f"{r['jbb_harmful_tpr'] * 100:.1f}\\%" if r.get("jbb_harmful_tpr") is not None else "N/A"
        row_str = (
            f"{r['name']} & {r.get('params', 'N/A')} & {r['f1_ci']} & {r['pr_auc']:.3f} & {r['roc_auc']:.3f} & "
            f"{r['fpr_at_95_tpr'] * 100:.1f}\\% & {r['devops_hard_neg_fpr'] * 100:.1f}\\% & {r['xstest_safe_fpr'] * 100:.1f}\\% & "
            f"{notinject_str} & {jbb_str} & {p99_str} & {r['mcnemar_str']} \\\\"
        )
        tex.append(row_str)
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{AegisLLM Milestone 2 Empirical Benchmark on held-out test and OOD stress suites (100\% measured).}",
            r"\label{tab:milestone2_benchmark}",
            r"\end{table*}",
        ]
    )
    return "\n".join(tex)


def _evaluate_single_artifact(
    artifact: Dict[str, Any],
    test_df: pd.DataFrame,
    xstest_df: Optional[pd.DataFrame],
    notinject_df: Optional[pd.DataFrame],
    jbb_df: Optional[pd.DataFrame],
    baseline_preds: Optional[np.ndarray],
    n_bootstraps: int,
) -> Dict[str, Any]:
    npz = np.load(artifact["npz_path"])
    meta = artifact["meta"]
    meta_metrics = meta.get("metrics", {})

    y_true = npz["y_true"] if "y_true" in npz else test_df["label"].values
    probs = npz["probs"]
    tau_95 = float(npz["tau_95"]) if "tau_95" in npz else meta.get("tau_95")
    tau_1pct = float(npz["tau_1pct"]) if "tau_1pct" in npz else meta.get("tau_1pct")

    base_metrics = compute_comprehensive_metrics(
        y_true=y_true,
        probs=probs,
        tau_95=tau_95,
        tau_1pct=tau_1pct,
    )
    tau_95_eff = base_metrics["tau_95"]

    f1_point, f1_low, f1_high = compute_paired_bootstrap_ci(
        y_true=y_true,
        probs=probs,
        metric_fn=macro_f1_from_probs,
        n_bootstraps=n_bootstraps,
        ci=0.95,
        seed=42,
    )
    f1_half_width = (f1_high - f1_low) / 2.0
    f1_ci_str = f"{f1_point:.3f} ± {f1_half_width:.3f}"

    devops_mask = (test_df["sub_category"] == "hard_negative_devops").values
    s_mal_test = compute_binary_malicious_score(probs)
    if np.sum(devops_mask) > 0:
        devops_fpr = float(np.mean(s_mal_test[devops_mask] >= tau_95_eff))
    else:
        devops_fpr = float(meta_metrics.get("devops_hard_neg_fpr", 0.0))

    xstest_safe_fpr = float(meta_metrics.get("xstest_safe_fpr", 0.0))
    xstest_unsafe_tpr = float(meta_metrics.get("xstest_unsafe_tpr", 0.0))
    if "xstest_probs" in npz and xstest_df is not None:
        x_s_mal = compute_binary_malicious_score(npz["xstest_probs"])
        safe_mask = xstest_df["label"].values == 0
        if np.sum(safe_mask) > 0:
            xstest_safe_fpr = float(np.mean(x_s_mal[safe_mask] >= tau_95_eff))
        if np.sum(~safe_mask) > 0:
            xstest_unsafe_tpr = float(np.mean(x_s_mal[~safe_mask] >= tau_95_eff))

    notinject_fpr: Optional[float] = meta_metrics.get("notinject_fpr")
    if "notinject_probs" in npz and notinject_df is not None:
        ni_s_mal = compute_binary_malicious_score(npz["notinject_probs"])
        notinject_fpr = float(np.mean(ni_s_mal >= tau_95_eff))

    jbb_harmful_tpr: Optional[float] = meta_metrics.get("jbb_harmful_tpr")
    jbb_benign_fpr: Optional[float] = meta_metrics.get("jbb_benign_fpr")
    if "jbb_probs" in npz and jbb_df is not None:
        jbb_s_mal = compute_binary_malicious_score(npz["jbb_probs"])
        harm_mask = jbb_df["label"].values > 0
        ben_mask = jbb_df["label"].values == 0
        if np.sum(harm_mask) > 0:
            jbb_harmful_tpr = float(np.mean(jbb_s_mal[harm_mask] >= tau_95_eff))
        if np.sum(ben_mask) > 0:
            jbb_benign_fpr = float(np.mean(jbb_s_mal[ben_mask] >= tau_95_eff))

    y_pred = np.argmax(probs, axis=-1)
    if artifact["model_key"] == "baseline_tier1" or baseline_preds is None:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0
        mcnemar_str = "Reference (H0)"
    else:
        mcnemar_stat, mcnemar_p = compute_mcnemar_test(y_true, baseline_preds, y_pred)
        mcnemar_str = "p < 0.0001" if mcnemar_p < 1e-4 else f"p = {mcnemar_p:.4f}"

    p99_latency_ms = meta.get("p99_latency_ms", meta_metrics.get("p99_latency_ms"))

    return {
        "model_key": artifact["model_key"],
        "name": artifact["name"],
        "group_key": meta.get("group_key"),
        "seed": meta.get("seed"),
        "params": meta.get("params", "44.1M"),
        "macro_f1": base_metrics["macro_f1"],
        "macro_f1_ci_low": f1_low,
        "macro_f1_ci_high": f1_high,
        "f1_ci": f1_ci_str,
        "pr_auc": base_metrics["pr_auc"],
        "roc_auc": base_metrics["roc_auc"],
        "tau_95": tau_95_eff,
        "tau_1pct": base_metrics["tau_1pct"],
        "fpr_at_95_tpr": base_metrics["fpr_at_95_tpr"],
        "tpr_at_95_tpr": base_metrics["tpr_at_95_tpr"],
        "fpr_at_1pct_fpr": base_metrics["fpr_at_1pct_fpr"],
        "tpr_at_1pct_fpr": base_metrics["tpr_at_1pct_fpr"],
        "ece": base_metrics["ece"],
        "brier_score": base_metrics["brier_score"],
        "f1_benign": base_metrics["f1_benign"],
        "f1_injection": base_metrics["f1_injection"],
        "f1_jailbreak": base_metrics["f1_jailbreak"],
        "devops_hard_neg_fpr": devops_fpr,
        "xstest_safe_fpr": xstest_safe_fpr,
        "xstest_unsafe_tpr": xstest_unsafe_tpr,
        "notinject_fpr": notinject_fpr,
        "jbb_harmful_tpr": jbb_harmful_tpr,
        "jbb_benign_fpr": jbb_benign_fpr,
        "p99_latency_ms": float(p99_latency_ms) if p99_latency_ms is not None else None,
        "mcnemar_chi2": mcnemar_stat,
        "mcnemar_p_value": mcnemar_p,
        "mcnemar_str": mcnemar_str,
    }


def _aggregate_multi_seed_runs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seed_pattern = re.compile(r"^(.*)_seed(\d+)$")
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for rec in records:
        g_key = rec.get("group_key")
        if not g_key:
            match = seed_pattern.match(rec["model_key"])
            if match:
                g_key = match.group(1)
        if g_key:
            grouped.setdefault(g_key, []).append(rec)

    aggregates: List[Dict[str, Any]] = []
    metric_keys = (
        "macro_f1",
        "pr_auc",
        "roc_auc",
        "fpr_at_95_tpr",
        "tpr_at_1pct_fpr",
        "devops_hard_neg_fpr",
        "xstest_safe_fpr",
        "xstest_unsafe_tpr",
        "notinject_fpr",
        "jbb_harmful_tpr",
        "jbb_benign_fpr",
        "ece",
        "brier_score",
        "f1_jailbreak",
    )

    for g_key, group_recs in grouped.items():
        if len(group_recs) < 2:
            continue
        agg: Dict[str, Any] = {
            "group_key": g_key,
            "num_seeds": len(group_recs),
            "seeds": [r.get("seed") or r["model_key"] for r in group_recs],
            "params": group_recs[0].get("params", "44.1M"),
        }
        for mk in metric_keys:
            vals = [float(r[mk]) for r in group_recs if r.get(mk) is not None]
            if vals:
                agg[f"{mk}_mean"] = float(np.mean(vals))
                agg[f"{mk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        aggregates.append(agg)

    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OOD Generalization and Export Measured Tables")
    parser.add_argument("--predictions-dir", type=str, default="reports/predictions")
    parser.add_argument("--test-data", type=str, default="data/processed/test.parquet")
    parser.add_argument("--xstest-data", type=str, default="data/ood/xstest.parquet")
    parser.add_argument("--notinject-data", type=str, default="data/ood/notinject.parquet")
    parser.add_argument("--jbb-data", type=str, default="data/ood/jbb.parquet")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    preds_dir = repo_root / args.predictions_dir
    test_path = repo_root / args.test_data
    xstest_path = repo_root / args.xstest_data
    notinject_path = repo_root / args.notinject_data
    jbb_path = repo_root / args.jbb_data
    out_dir = repo_root / args.output_dir

    test_df = pd.read_parquet(test_path)
    xstest_df = pd.read_parquet(xstest_path) if xstest_path.exists() else None
    notinject_df = pd.read_parquet(notinject_path) if notinject_path.exists() else None
    jbb_df = pd.read_parquet(jbb_path) if jbb_path.exists() else None

    artifacts = _discover_prediction_artifacts(preds_dir)
    if not artifacts:
        raise FileNotFoundError(
            f"No prediction artifacts (*.npz) found in {preds_dir}. Run scripts/train_baseline.py first."
        )

    print(f"[*] Discovered {len(artifacts)} measured model artifact(s) in {preds_dir}:")
    for art in artifacts:
        print(f"    - {art['model_key']} ({art['name']})")

    baseline_preds: Optional[np.ndarray] = None
    for art in artifacts:
        if art["model_key"] == "baseline_tier1":
            base_npz = np.load(art["npz_path"])
            baseline_preds = np.argmax(base_npz["probs"], axis=-1)
            break

    records: List[Dict[str, Any]] = []
    for art in artifacts:
        rec = _evaluate_single_artifact(
            artifact=art,
            test_df=test_df,
            xstest_df=xstest_df,
            notinject_df=notinject_df,
            jbb_df=jbb_df,
            baseline_preds=baseline_preds,
            n_bootstraps=args.n_bootstraps,
        )
        records.append(rec)
        print(
            f"[+] {rec['name']}: Macro-F1={rec['f1_ci']} | FPR@95%TPR={rec['fpr_at_95_tpr']*100:.2f}% | "
            f"DevOps FPR={rec['devops_hard_neg_fpr']*100:.2f}% | XSTest Safe FPR={rec['xstest_safe_fpr']*100:.2f}% | "
            f"NotInject FPR={(rec['notinject_fpr'] or 0.0)*100:.2f}% | JBB TPR={(rec['jbb_harmful_tpr'] or 0.0)*100:.2f}%"
        )

    multi_seed_aggregates = _aggregate_multi_seed_runs(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "benchmark_comparison_table.md"
    tex_path = out_dir / "benchmark_comparison_table.tex"
    summary_path = out_dir / "evaluation_summary.json"

    md_content = format_table_markdown(records)
    tex_content = format_table_latex(records)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AegisLLM - Milestone 2 Empirical Benchmark (100% Measured)\n\n" + md_content + "\n")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content + "\n")

    summary_payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _resolve_git_sha(repo_root),
        "predictions_dir": str(args.predictions_dir),
        "test_samples": int(len(test_df)),
        "n_bootstraps": int(args.n_bootstraps),
        "models": records,
        "multi_seed_aggregates": multi_seed_aggregates,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[+] Exported measured tables and summary to:\n    - {md_path}\n    - {tex_path}\n    - {summary_path}")


if __name__ == "__main__":
    main()
