import inspect
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from aegis_dl.metrics import compute_quantization_drift

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_MODEL_DIR = REPO_ROOT / "models" / "deberta_v3_small_aegis_v1"


class MiniEncoderSequenceClassifier(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 64, num_labels: int = 3) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layernorm_emb = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layernorm_attn = nn.LayerNorm(hidden_size)
        self.ffn_1 = nn.Linear(hidden_size, hidden_size * 2)
        self.act = nn.GELU()
        self.ffn_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layernorm_ffn = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.scale = float(hidden_size) ** -0.5

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.layernorm_emb(self.word_embeddings(input_ids))
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = (1.0 - attention_mask.float()).unsqueeze(1) * -1e4
        attn = torch.softmax(scores + mask, dim=-1)
        context = self.out_proj(torch.matmul(attn, v))
        h = self.layernorm_attn(x + context)

        ffn_out = self.ffn_2(self.act(self.ffn_1(h)))
        h = self.layernorm_ffn(h + ffn_out)

        mask_f = attention_mask.float().unsqueeze(-1)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


def test_encoder_onnx_fp32_and_int8_parity_and_drift(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = MiniEncoderSequenceClassifier().float().eval().cpu()

    trace_ids = torch.randint(1, 128, (2, 16), dtype=torch.long)
    trace_mask = torch.ones((2, 16), dtype=torch.long)

    fp32_raw_path = tmp_path / "model_fp32_raw.onnx"
    export_kwargs = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        "opset_version": 18,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            (trace_ids, trace_mask),
            str(fp32_raw_path),
            **export_kwargs,
        )

    fp32_opt_path = tmp_path / "model_fp32_opt.onnx"
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_opts.optimized_model_filepath = str(fp32_opt_path)
    _ = ort.InferenceSession(str(fp32_raw_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    proto = onnx.load(str(fp32_opt_path), load_external_data=False)
    exclude_patterns = ("embeddings", "classifier", "layernorm")
    excluded_nodes = sorted(
        {
            node.name
            for node in proto.graph.node
            if node.name and any(pat in node.name.lower() for pat in exclude_patterns)
        }
    )

    int8_dyn_path = tmp_path / "model_int8_dynamic.onnx"
    quantize_dynamic(
        model_input=str(fp32_opt_path),
        model_output=str(int8_dyn_path),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excluded_nodes,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )

    sess_fp32 = ort.InferenceSession(str(fp32_opt_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_dyn_path), providers=["CPUExecutionProvider"])

    # Evaluate across different batch sizes and sequence lengths to verify dynamic axes
    rng = torch.Generator().manual_seed(1337)
    eval_ids = torch.randint(1, 128, (24, 32), generator=rng, dtype=torch.long)
    eval_mask = torch.ones((24, 32), dtype=torch.long)
    eval_mask[:, 24:] = 0

    with torch.no_grad():
        pt_logits = model(eval_ids, eval_mask).numpy()

    feed = {
        "input_ids": eval_ids.numpy().astype(np.int64),
        "attention_mask": eval_mask.numpy().astype(np.int64),
    }
    ort_fp32_logits = sess_fp32.run(["logits"], feed)[0]
    ort_int8_logits = sess_int8.run(["logits"], feed)[0]

    np.testing.assert_allclose(pt_logits, ort_fp32_logits, rtol=1e-4, atol=1e-4)

    y_true = np.argmax(pt_logits, axis=-1)
    y_true[:8] = 0
    y_true[8:16] = 1
    y_true[16:] = 2

    drift = compute_quantization_drift(
        fp32_logits=ort_fp32_logits,
        int8_logits=ort_int8_logits,
        y_true=y_true,
        tau_95=0.5,
    )

    assert drift["max_abs_diff"] < 0.25
    assert drift["kl_div_mean"] < 0.02
    assert drift["prediction_agreement"] >= 0.95
    assert drift["delta_fpr_at_95_tpr"] <= 0.05


def test_production_deberta_v3_small_int8_artifact_integration() -> None:
    int8_candidates = sorted(PROD_MODEL_DIR.glob("model_int8*.onnx")) if PROD_MODEL_DIR.exists() else []
    if not int8_candidates:
        pytest.skip(
            f"Trained DeBERTa-v3-Small INT8 ONNX artifacts not yet repatriated in {PROD_MODEL_DIR}."
        )

    primary_int8_path = PROD_MODEL_DIR / "model_int8_dynamic.onnx"
    if not primary_int8_path.exists():
        primary_int8_path = int8_candidates[0]

    meta_path = PROD_MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta.get("quantization_mode") == "int8_dynamic"
        assert float(meta.get("temperature", 0.0)) > 0.0
        assert 0.0 < float(meta.get("tau_95", 0.5)) < 1.0

    sess_int8 = ort.InferenceSession(str(primary_int8_path), providers=["CPUExecutionProvider"])
    dummy_ids = np.ones((2, 32), dtype=np.int64)
    dummy_mask = np.ones((2, 32), dtype=np.int64)
    out = sess_int8.run(["logits"], {"input_ids": dummy_ids, "attention_mask": dummy_mask})[0]
    assert out.shape == (2, 3)
    assert np.all(np.isfinite(out))
