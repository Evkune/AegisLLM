#!/usr/bin/env python3
"""Resilient Colab/GPU batch runner for the AegisLLM 2x2x5 factorial DOE."""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

DOE_LOSSES: tuple[str, ...] = ("ce", "cs_afl")
DOE_DISTILL_LEVELS: tuple[str, ...] = ("off", "on")
DOE_SEEDS: tuple[int, ...] = (42, 1337, 2026, 777, 999)


@dataclass(frozen=True)
class DOERunSpec:
    loss: str
    distillation: str
    seed: int

    @property
    def no_distill(self) -> bool:
        return self.distillation == "off"

    @property
    def variant_tag(self) -> str:
        return f"{self.loss}_no_distill" if self.no_distill else self.loss

    @property
    def run_key(self) -> str:
        return f"deberta_v3_small_{self.variant_tag}_seed{self.seed}"


def build_doe_matrix(
    losses: tuple[str, ...] = DOE_LOSSES,
    distill_levels: tuple[str, ...] = DOE_DISTILL_LEVELS,
    seeds: tuple[int, ...] = DOE_SEEDS,
) -> list[DOERunSpec]:
    return [
        DOERunSpec(loss=loss, distillation=distill, seed=seed)
        for loss in losses
        for distill in distill_levels
        for seed in seeds
    ]


def is_run_complete(preds_dir: Path, spec: DOERunSpec) -> bool:
    return (preds_dir / f"{spec.run_key}_test.npz").exists() and (
        preds_dir / f"{spec.run_key}_meta.json"
    ).exists()


class KernelHeartbeat:
    """Periodic stdout pulse to prevent Colab kernel idle timeouts during long GPU runs."""

    def __init__(self, interval_sec: int = 240) -> None:
        self.interval_sec = max(30, int(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ts = time.time()
        self._current_run = "initializing"

    def set_run(self, run_key: str) -> None:
        self._current_run = run_key

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            elapsed_min = (time.time() - self._start_ts) / 60.0
            gpu_info = ""
            try:
                import torch

                if torch.cuda.is_available():
                    alloc_gb = torch.cuda.memory_allocated() / (1024**3)
                    res_gb = torch.cuda.memory_reserved() / (1024**3)
                    gpu_info = f" | vram={alloc_gb:.2f}/{res_gb:.2f}GB"
            except (ImportError, RuntimeError):
                pass
            ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            print(
                f"[HEARTBEAT {ts}] active_run={self._current_run} elapsed={elapsed_min:.1f}m{gpu_info}",
                flush=True,
            )


def _in_colab() -> bool:
    return "google.colab" in sys.modules


def _persist_run_artifacts(
    preds_dir: Path,
    run_key: str,
    drive_backup_dir: Path | None,
    colab_download: bool,
    bundle_dir: Path,
) -> None:
    npz_path = preds_dir / f"{run_key}_test.npz"
    meta_path = preds_dir / f"{run_key}_meta.json"
    if not (npz_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Expected artifacts missing for {run_key} in {preds_dir}")

    if drive_backup_dir is not None:
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz_path, drive_backup_dir / npz_path.name)
        shutil.copy2(meta_path, drive_backup_dir / meta_path.name)
        print(f"[DOE] Synced {run_key} to Drive: {drive_backup_dir}", flush=True)

    if colab_download:
        try:
            from google.colab import files  # type: ignore[import-not-found]

            bundle_dir.mkdir(parents=True, exist_ok=True)
            zip_path = bundle_dir / f"{run_key}_artifacts.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(meta_path, arcname=meta_path.name)
                zf.write(npz_path, arcname=npz_path.name)
            files.download(str(zip_path))
            print(f"[DOE] Triggered Colab download: {zip_path.name}", flush=True)
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"[DOE][WARN] Colab download skipped ({exc})", flush=True)


def _load_progress(progress_path: Path) -> dict[str, Any]:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"updated_at_utc": None, "runs": {}}


def _save_progress(progress_path: Path, payload: dict[str, Any], drive_backup_dir: Path | None) -> None:
    payload["updated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if drive_backup_dir is not None:
        drive_backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(progress_path, drive_backup_dir / progress_path.name)


def select_scheduled_runs(
    matrix: list[DOERunSpec],
    preds_dir: Path,
    force_rerun: bool,
    randomize_order: bool,
    order_seed: int,
    filter_loss: str | None,
    filter_distill: str | None,
    filter_seeds: list[int] | None,
    batch_index: int | None,
    batch_size: int | None,
    max_runs: int | None,
) -> tuple[list[DOERunSpec], list[DOERunSpec]]:
    ordered = list(matrix)
    if randomize_order:
        rng = random.Random(order_seed)
        rng.shuffle(ordered)

    pending: list[DOERunSpec] = []
    for spec in ordered:
        if filter_loss and spec.loss != filter_loss:
            continue
        if filter_distill and spec.distillation != filter_distill:
            continue
        if filter_seeds and spec.seed not in filter_seeds:
            continue
        if not force_rerun and is_run_complete(preds_dir, spec):
            continue
        pending.append(spec)

    if batch_index is not None and batch_size is not None:
        start = batch_index * batch_size
        selected = pending[start : start + batch_size]
    else:
        selected = pending

    if max_runs is not None:
        selected = selected[:max_runs]

    return ordered, selected


def _print_audit(
    matrix: list[DOERunSpec],
    selected: list[DOERunSpec],
    preds_dir: Path,
    epochs: int,
    enable_fgm: bool,
) -> None:
    done = [s for s in matrix if is_run_complete(preds_dir, s)]
    missing = [s for s in matrix if not is_run_complete(preds_dir, s)]
    est_min_per_run = 27.5 if (epochs == 3 and enable_fgm) else (9.5 if (epochs == 2 and not enable_fgm) else 15.0)

    print(f"DOE Matrix: 2x2x5 = {len(matrix)} runs | preds_dir={preds_dir}")
    print(f"Protocol: epochs={epochs}, fgm={'ON' if enable_fgm else 'OFF'} (~{est_min_per_run:.1f} min/run on T4)")
    print(f"Status: {len(done)}/{len(matrix)} DONE, {len(missing)} MISSING, {len(selected)} SCHEDULED THIS BATCH\n")

    print(f"{'Cell (Loss, Distill)':<22} {'Done':<8} {'Missing Seeds'}")
    print("-" * 64)
    for loss in DOE_LOSSES:
        for distill in DOE_DISTILL_LEVELS:
            cell_specs = [s for s in matrix if s.loss == loss and s.distillation == distill]
            cell_done = [s.seed for s in cell_specs if is_run_complete(preds_dir, s)]
            cell_miss = [s.seed for s in cell_specs if not is_run_complete(preds_dir, s)]
            print(f"({loss:<6}, {distill:<3})         {len(cell_done)}/5      {cell_miss}")

    if selected:
        total_h = (len(selected) * est_min_per_run) / 60.0
        print(f"\nScheduled batch ({len(selected)} runs, est. {total_h:.2f}h on T4):")
        for idx, spec in enumerate(selected, start=1):
            print(f"  {idx:2d}. {spec.run_key}  [loss={spec.loss}, distill={spec.distillation}, seed={spec.seed}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute missing AegisLLM 2x2x5 DOE runs in resilient batches.")
    parser.add_argument("--preds-dir", type=Path, default=REPO_ROOT / "reports/predictions")
    parser.add_argument("--doe-dir", type=Path, default=REPO_ROOT / "reports/doe")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per student run (default: 3).")
    parser.add_argument("--disable-fgm", action="store_true", help="Disable FGM adversarial pass (fast DOE mode).")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of runs in this batch.")
    parser.add_argument("--batch-index", type=int, default=None, help="Zero-based batch index (requires --batch-size).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size when slicing pending runs.")
    parser.add_argument("--filter-loss", choices=DOE_LOSSES, default=None)
    parser.add_argument("--filter-distill", choices=DOE_DISTILL_LEVELS, default=None)
    parser.add_argument("--filter-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--force-rerun", action="store_true", help="Re-execute runs even if artifacts already exist.")
    parser.add_argument("--order-seed", type=int, default=2026, help="PRNG seed for randomized run order.")
    parser.add_argument("--no-randomize", action="store_true", help="Run in canonical order instead of randomized.")
    parser.add_argument("--heartbeat-sec", type=int, default=240, help="Colab keep-alive pulse interval in seconds.")
    parser.add_argument("--drive-backup-dir", type=Path, default=None, help="Optional Google Drive backup path.")
    parser.add_argument("--colab-download", action="store_true", help="Trigger google.colab.files.download after each run.")
    parser.add_argument("--no-colab-download", action="store_true", help="Disable automatic Colab downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Audit matrix and print batch schedule without training.")
    args = parser.parse_args()

    preds_dir = args.preds_dir if args.preds_dir.is_absolute() else (REPO_ROOT / args.preds_dir)
    doe_dir = args.doe_dir if args.doe_dir.is_absolute() else (REPO_ROOT / args.doe_dir)
    enable_fgm = not args.disable_fgm
    randomize = not args.no_randomize

    matrix = build_doe_matrix()
    ordered, selected = select_scheduled_runs(
        matrix=matrix,
        preds_dir=preds_dir,
        force_rerun=args.force_rerun,
        randomize_order=randomize,
        order_seed=args.order_seed,
        filter_loss=args.filter_loss,
        filter_distill=args.filter_distill,
        filter_seeds=args.filter_seeds,
        batch_index=args.batch_index,
        batch_size=args.batch_size,
        max_runs=args.max_runs,
    )

    _print_audit(matrix, selected, preds_dir, epochs=args.epochs, enable_fgm=enable_fgm)
    if args.dry_run or not selected:
        return

    doe_dir.mkdir(parents=True, exist_ok=True)
    order_manifest = {
        "order_seed": args.order_seed,
        "randomized": randomize,
        "epochs": args.epochs,
        "fgm_enabled": enable_fgm,
        "preds_dir": str(preds_dir),
        "full_order": [asdict(s) | {"run_key": s.run_key} for s in ordered],
    }
    (doe_dir / "run_order.json").write_text(json.dumps(order_manifest, indent=2), encoding="utf-8")

    import torch
    import yaml
    from colab_train_pipeline import load_all_datasets, run_student_training
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load((REPO_ROOT / "configs/milestone2_deberta_v3_small.yaml").read_text(encoding="utf-8"))
    max_seq_len = int(cfg["model"].get("max_sequence_length", 256))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"], use_fast=True)
    data = load_all_datasets(tokenizer, max_seq_len)

    teacher_logits_path = REPO_ROOT / "models/teacher_train_logits.pt"
    teacher_logits: torch.Tensor | None = None
    if any(not s.no_distill for s in selected):
        if not teacher_logits_path.exists():
            raise FileNotFoundError(
                f"Teacher logits not found at {teacher_logits_path}. Run Phase 1 teacher caching first."
            )
        teacher_logits = torch.load(teacher_logits_path, map_location="cpu", weights_only=True)
        print(f"[DOE] Loaded cached teacher logits: shape={tuple(teacher_logits.shape)}", flush=True)

    use_colab_download = (args.colab_download or _in_colab()) and not args.no_colab_download
    progress_path = doe_dir / "doe_batch_progress.json"
    progress = _load_progress(progress_path)

    heartbeat = KernelHeartbeat(interval_sec=args.heartbeat_sec)
    heartbeat.start()
    try:
        for idx, spec in enumerate(selected, start=1):
            heartbeat.set_run(f"{spec.run_key} ({idx}/{len(selected)})")
            t0 = time.time()
            print(
                f"\n[DOE] >>> Starting {idx}/{len(selected)}: {spec.run_key} "
                f"(loss={spec.loss}, distill={spec.distillation}, seed={spec.seed}, gpu={gpu_name})",
                flush=True,
            )
            run_student_training(
                cfg=cfg,
                data=data,
                tokenizer=tokenizer,
                teacher_logits=teacher_logits,
                loss_variant=spec.loss,
                run_seed=spec.seed,
                keep_ckpts=False,
                epochs=args.epochs,
                device=device,
                no_distill=spec.no_distill,
                enable_fgm=enable_fgm,
                preds_dir=preds_dir,
            )
            elapsed_sec = round(time.time() - t0, 2)
            _persist_run_artifacts(
                preds_dir=preds_dir,
                run_key=spec.run_key,
                drive_backup_dir=args.drive_backup_dir,
                colab_download=use_colab_download,
                bundle_dir=doe_dir / "downloads",
            )
            progress["runs"][spec.run_key] = {
                "loss": spec.loss,
                "distillation": spec.distillation,
                "seed": spec.seed,
                "epochs": args.epochs,
                "fgm_enabled": enable_fgm,
                "gpu": gpu_name,
                "elapsed_sec": elapsed_sec,
                "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            _save_progress(progress_path, progress, args.drive_backup_dir)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[DOE] <<< Finished {spec.run_key} in {elapsed_sec / 60:.1f} min", flush=True)
    finally:
        heartbeat.stop()

    print("\n[DOE] Batch complete.", flush=True)


if __name__ == "__main__":
    main()
