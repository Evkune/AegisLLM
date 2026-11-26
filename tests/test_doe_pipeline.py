from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


doe_anova = _load_script("analyze_doe_anova", "analyze_doe_anova.py")
doe_batch = _load_script("run_doe_batch", "run_doe_batch.py")


def _make_synthetic_run(
    preds_dir: Path,
    loss: str,
    distill: str,
    seed: int,
    rng: np.random.Generator,
    n_test: int = 300,
    n_ni: int = 100,
) -> None:
    run_key = doe_anova.expected_run_key(loss, distill, seed)
    y_true = np.array([0] * 160 + [1] * 100 + [2] * 40, dtype=np.int64)

    # Inject controlled treatment signal so ANOVA detects non-trivial effects
    benign_sep = 2.2 + (0.6 if loss == "cs_afl" else 0.0) + (0.4 if distill == "on" else 0.0)
    logits = np.zeros((n_test, 3), dtype=np.float32)
    for idx, y in enumerate(y_true):
        logits[idx] = rng.normal(0.0, 0.8, size=3)
        logits[idx, y] += benign_sep

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)

    ni_bias = -1.2 - (0.7 if loss == "cs_afl" else 0.0) - (0.5 if distill == "on" else 0.0)
    ni_logits = rng.normal(0.0, 0.9, size=(n_ni, 3))
    ni_logits[:, 0] -= ni_bias
    ni_shift = ni_logits - np.max(ni_logits, axis=1, keepdims=True)
    ni_probs = np.exp(ni_shift) / np.sum(np.exp(ni_shift), axis=1, keepdims=True)

    np.savez_compressed(
        preds_dir / f"{run_key}_test.npz",
        y_true=y_true,
        probs=probs,
        logits=logits,
        tau_95=np.array(0.55),
        tau_1pct=np.array(0.85),
        notinject_probs=ni_probs,
    )
    (preds_dir / f"{run_key}_meta.json").write_text(
        json.dumps(
            {
                "name": run_key,
                "model_key": run_key,
                "loss_variant": loss,
                "distillation_enabled": distill == "on",
                "seed": seed,
                "tau_95": 0.55,
                "tau_1pct": 0.85,
            }
        ),
        encoding="utf-8",
    )


def test_doe_batch_selection_and_skipping(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    matrix = doe_batch.build_doe_matrix()
    assert len(matrix) == 20

    # Mark 5 runs complete
    for spec in matrix[:5]:
        (preds_dir / f"{spec.run_key}_test.npz").write_bytes(b"npz")
        (preds_dir / f"{spec.run_key}_meta.json").write_text("{}", encoding="utf-8")

    ordered, pending = doe_batch.select_scheduled_runs(
        matrix=matrix,
        preds_dir=preds_dir,
        force_rerun=False,
        randomize_order=True,
        order_seed=2026,
        filter_loss=None,
        filter_distill=None,
        filter_seeds=None,
        batch_index=None,
        batch_size=None,
        max_runs=None,
    )
    assert len(ordered) == 20
    assert len(pending) == 15

    _, batch0 = doe_batch.select_scheduled_runs(
        matrix=matrix,
        preds_dir=preds_dir,
        force_rerun=False,
        randomize_order=True,
        order_seed=2026,
        filter_loss=None,
        filter_distill=None,
        filter_seeds=None,
        batch_index=0,
        batch_size=5,
        max_runs=None,
    )
    assert len(batch0) == 5
    assert batch0 == pending[:5]


def test_full_20_run_anova_pipeline_and_report_injection(tmp_path: Path) -> None:
    preds_dir = tmp_path / "predictions"
    preds_dir.mkdir()
    rng = np.random.default_rng(12345)

    for loss in doe_anova.DOE_LOSSES:
        for distill in doe_anova.DOE_DISTILL_LEVELS:
            for seed in doe_anova.DOE_SEEDS:
                _make_synthetic_run(preds_dir, loss, distill, seed, rng)

    report_copy = tmp_path / "milestone2_doe_report.md"
    report_copy.write_text(
        (REPO_ROOT / "docs/milestone2_doe_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    out_dir = tmp_path / "doe_out"
    result = doe_anova.run_analysis(
        preds_dir=preds_dir,
        out_dir=out_dir,
        test_parquet=tmp_path / "unused_test.parquet",
        notinject_parquet=tmp_path / "unused_ni.parquet",
        responses=doe_anova.DEFAULT_RESPONSES,
        alpha=0.05,
        allow_partial=False,
        update_report=report_copy,
    )

    assert result["complete"] is True
    csv_df = pd.read_csv(out_dir / "doe_raw_data.csv")
    assert list(csv_df.columns) == doe_anova.CSV_COLUMNS
    assert len(csv_df) == 20

    for metric in doe_anova.DEFAULT_RESPONSES:
        anova = result["results"][metric]["anova"]
        assert anova["type2_type3_max_ss_diff"] < 1e-10
        dfs = [row["df"] for row in anova["anova_table"]]
        assert dfs == [1, 1, 1, 16, 19]
        eta_sum = sum(row["eta_sq"] for row in anova["anova_table"][:4])
        assert abs(eta_sum - 1.0) < 1e-9

        for fig_key in ("boxplot_loss", "boxplot_distillation", "qqplot_residuals", "interaction"):
            fig_path = Path(result["results"][metric]["figures"][fig_key])
            assert fig_path.exists() and fig_path.stat().st_size > 5_000

    updated_md = report_copy.read_text(encoding="utf-8")
    assert "## 1.- Experiment Planning and Definition" in updated_md
    assert "## 2.- Experimental Design" in updated_md
    assert "### 3.1 Raw Experimental Data" in updated_md
    assert "### 3.4 Two-Way ANOVA Tables" in updated_md
    assert "### 3.6 Statistical Tool Options & Verification Screenshots" in updated_md
    assert "[PLACEHOLDER — SCREENSHOT 1" in updated_md
