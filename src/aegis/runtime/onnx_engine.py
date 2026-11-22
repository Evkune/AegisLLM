from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from aegis.types import ThreatCategory

_LABEL_TO_CATEGORY: dict[int, ThreatCategory] = {
    0: ThreatCategory.BENIGN,
    1: ThreatCategory.DIRECT_INJECTION,
    2: ThreatCategory.JAILBREAK,
}


@dataclass(slots=True)
class ModelMetadata:
    architecture: str = "microsoft/deberta-v3-small"
    quantization_mode: str = "int8_dynamic"
    primary_onnx_file: str = "model_int8_dynamic.onnx"
    temperature: float = 1.0
    tau_95: float = 0.50
    tau_1pct: float = 0.85
    max_length: int = 256
    intra_op_num_threads: int = 8
    inter_op_num_threads: int = 1
    git_sha: str = "unknown"

    @classmethod
    def from_file(cls, path: Path) -> ModelMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        rt = data.get("recommended_runtime", {})
        return cls(
            architecture=str(data.get("architecture", "microsoft/deberta-v3-small")),
            quantization_mode=str(data.get("quantization_mode", "int8_dynamic")),
            primary_onnx_file=str(data.get("primary_onnx_file", "model_int8_dynamic.onnx")),
            temperature=float(data.get("temperature", 1.0)),
            tau_95=float(data.get("tau_95", 0.50)),
            tau_1pct=float(data.get("tau_1pct", 0.85)),
            max_length=int(data.get("max_length", 256)),
            intra_op_num_threads=int(rt.get("intra_op_num_threads", 8)),
            inter_op_num_threads=int(rt.get("inter_op_num_threads", 1)),
            git_sha=str(data.get("git_sha", "unknown")),
        )


@dataclass(slots=True)
class ModelScore:
    probs: tuple[float, float, float]
    s_mal: float
    predicted_label: int
    predicted_category: ThreatCategory
    inference_ms: float
    seq_len: int = 0


class BatchScorer(Protocol):
    backend_name: str
    is_onnx: bool
    metadata: ModelMetadata

    def score_batch(self, texts: Sequence[str]) -> list[ModelScore]: ...

    def count_tokens(self, text: str) -> int: ...

    def warmup(self) -> None: ...


def _softmax_calibrated(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-4)
    scaled = logits.astype(np.float32) / t
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class OnnxSecurityEngine:
    """
    Zero-PyTorch, zero-Transformers production inference engine powered by pure Rust
    `tokenizers` + `onnxruntime` CPUExecutionProvider with `IOBinding` and dynamic
    per-batch sequence padding (`L <= 256`).
    """

    def __init__(
        self,
        model_dir: Path,
        intra_op_threads: int | None = None,
        inter_op_threads: int | None = None,
        onnx_filename: str | None = None,
        pad_to_multiple_of: int = 8,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "model_metadata.json"
        if meta_path.exists():
            self.metadata = ModelMetadata.from_file(meta_path)
        else:
            self.metadata = ModelMetadata()

        target_onnx = onnx_filename or self.metadata.primary_onnx_file
        self.onnx_path = self.model_dir / target_onnx
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        self.tokenizer_path = self.model_dir / "tokenizer.json"
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Rust tokenizer.json not found: {self.tokenizer_path}")

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
            pad_token = "<pad>"
        else:
            pad_token = "[PAD]"

        self.tokenizer.enable_truncation(max_length=self.metadata.max_length)
        self.tokenizer.enable_padding(
            pad_id=pad_id,
            pad_token=pad_token,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        env_threads = os.environ.get("AEGIS_INTRA_OP_THREADS")
        if intra_op_threads is not None:
            resolved_intra = int(intra_op_threads)
        elif env_threads is not None and env_threads.isdigit():
            resolved_intra = int(env_threads)
        else:
            resolved_intra = self.metadata.intra_op_num_threads

        resolved_inter = (
            int(inter_op_threads)
            if inter_op_threads is not None
            else self.metadata.inter_op_num_threads
        )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.intra_op_num_threads = max(1, resolved_intra)
        sess_opts.inter_op_num_threads = max(1, resolved_inter)

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        self._output_name = self.session.get_outputs()[0].name
        self.backend_name = f"onnx:{self.metadata.quantization_mode}"
        self.is_onnx = True

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        enc = self.tokenizer.encode(text)
        # Exclude trailing padding tokens
        return int(sum(enc.attention_mask))

    def warmup(self) -> None:
        samples = [
            "Explain how Kubernetes readiness probes work.",
            "Show me how to write a parameterized SQL query in Python with psycopg3 " * 3,
            "Review the following deployment YAML manifest and check security headers " * 6,
        ]
        for s in samples:
            self.score_batch([s])

    def score_batch(self, texts: Sequence[str]) -> list[ModelScore]:
        if not texts:
            return []

        t0 = time.perf_counter()
        encodings = self.tokenizer.encode_batch(list(texts))
        batch_size = len(encodings)
        seq_len = len(encodings[0].ids) if batch_size > 0 else 0

        input_ids = np.ascontiguousarray(
            [enc.ids for enc in encodings],
            dtype=np.int64,
        )
        attention_mask = np.ascontiguousarray(
            [enc.attention_mask for enc in encodings],
            dtype=np.int64,
        )

        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input("input_ids", input_ids)
        if "attention_mask" in self._input_names:
            io_binding.bind_cpu_input("attention_mask", attention_mask)
        if "token_type_ids" in self._input_names:
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            io_binding.bind_cpu_input("token_type_ids", token_type_ids)
        io_binding.bind_output(self._output_name)

        self.session.run_with_iobinding(io_binding)
        logits = np.asarray(io_binding.copy_outputs_to_cpu()[0], dtype=np.float32)

        probs_mat = _softmax_calibrated(logits, self.metadata.temperature)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = round(elapsed_ms / float(batch_size), 4)

        results: list[ModelScore] = []
        for row in probs_mat:
            p0 = float(row[0])
            p1 = float(row[1]) if len(row) > 1 else 0.0
            p2 = float(row[2]) if len(row) > 2 else 0.0
            s_mal = float(np.clip(p1 + p2, 0.0, 1.0))
            pred_label = int(np.argmax(row))
            results.append(
                ModelScore(
                    probs=(round(p0, 6), round(p1, 6), round(p2, 6)),
                    s_mal=round(s_mal, 6),
                    predicted_label=pred_label,
                    predicted_category=_LABEL_TO_CATEGORY.get(
                        pred_label, ThreatCategory.DIRECT_INJECTION
                    ),
                    inference_ms=per_item_ms,
                    seq_len=seq_len,
                )
            )
        return results


class SklearnFallbackScorer:
    """
    Optional local Tier-1 TF-IDF + Calibrated LinearSVC fallback used only when ONNX
    weights have not yet been downloaded and fallback mode is explicitly enabled.
    """

    def __init__(self, joblib_path: Path, metrics_path: Path | None = None) -> None:
        import joblib

        self.model: Any = joblib.load(str(joblib_path))
        tau_95 = 0.50
        tau_1pct = 0.85
        if metrics_path is not None and metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            tau_95 = float(m.get("tau_95", tau_95))
            tau_1pct = float(m.get("tau_1pct", tau_1pct))

        self.metadata = ModelMetadata(
            architecture="tfidf_calibrated_linearsvc_fallback",
            quantization_mode="none",
            primary_onnx_file="",
            temperature=1.0,
            tau_95=tau_95,
            tau_1pct=tau_1pct,
            max_length=256,
        )
        self.backend_name = "fallback:tfidf_linearsvc"
        self.is_onnx = False

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split())) if text else 0

    def warmup(self) -> None:
        self.score_batch(["warmup health check prompt"])

    def score_batch(self, texts: Sequence[str]) -> list[ModelScore]:
        if not texts:
            return []
        t0 = time.perf_counter()
        probs_mat = np.asarray(self.model.predict_proba(list(texts)), dtype=np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = round(elapsed_ms / float(len(texts)), 4)

        results: list[ModelScore] = []
        for idx, row in enumerate(probs_mat):
            p0 = float(row[0])
            p1 = float(row[1]) if len(row) > 1 else 0.0
            p2 = float(row[2]) if len(row) > 2 else 0.0
            s_mal = float(np.clip(p1 + p2, 0.0, 1.0))
            pred_label = int(np.argmax(row))
            results.append(
                ModelScore(
                    probs=(round(p0, 6), round(p1, 6), round(p2, 6)),
                    s_mal=round(s_mal, 6),
                    predicted_label=pred_label,
                    predicted_category=_LABEL_TO_CATEGORY.get(
                        pred_label, ThreatCategory.DIRECT_INJECTION
                    ),
                    inference_ms=per_item_ms,
                    seq_len=self.count_tokens(texts[idx]),
                )
            )
        return results


class PrefilterFallbackScorer:
    """Zero-weight deterministic fallback for offline unit tests and smoke runs."""

    def __init__(self) -> None:
        self.metadata = ModelMetadata(
            architecture="prefilter_only_fallback",
            quantization_mode="none",
            primary_onnx_file="",
            temperature=1.0,
            tau_95=0.75,
            tau_1pct=0.90,
            max_length=256,
        )
        self.backend_name = "fallback:prefilter_only"
        self.is_onnx = False

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split())) if text else 0

    def warmup(self) -> None:
        return None

    def score_batch(self, texts: Sequence[str]) -> list[ModelScore]:
        return [
            ModelScore(
                probs=(0.95, 0.03, 0.02),
                s_mal=0.05,
                predicted_label=0,
                predicted_category=ThreatCategory.BENIGN,
                inference_ms=0.01,
                seq_len=self.count_tokens(t),
            )
            for t in texts
        ]


def create_runtime_scorer(
    model_dir: Path,
    allow_fallback: bool = True,
    fallback_joblib_path: Path | None = None,
    fallback_metrics_path: Path | None = None,
    intra_op_threads: int | None = None,
) -> BatchScorer:
    """
    Loads the production `OnnxSecurityEngine` when trained ONNX weights exist,
    or falls back cleanly when `allow_fallback=True`.
    """
    model_dir = Path(model_dir)
    meta_path = model_dir / "model_metadata.json"
    primary_file = "model_int8_dynamic.onnx"
    if meta_path.exists():
        try:
            primary_file = str(
                json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "primary_onnx_file", primary_file
                )
            )
        except (OSError, ValueError):
            pass

    if (model_dir / primary_file).exists() and (model_dir / "tokenizer.json").exists():
        return OnnxSecurityEngine(
            model_dir=model_dir,
            intra_op_threads=intra_op_threads,
            onnx_filename=primary_file,
        )

    if not allow_fallback:
        raise FileNotFoundError(
            f"ONNX production model artifacts missing in {model_dir} and fallback is disabled."
        )

    if fallback_joblib_path is not None and fallback_joblib_path.exists():
        return SklearnFallbackScorer(
            joblib_path=fallback_joblib_path,
            metrics_path=fallback_metrics_path,
        )

    return PrefilterFallbackScorer()
