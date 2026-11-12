"""
Unit test asserting numerical equivalence between PyTorch and ONNX Runtime.
Software Engineering for Deep Learning - ETSIINF (UPM)
"""

import os
import pytest
import numpy as np


def test_onnx_numerical_parity_gate():
    """
    Asserts that exported ONNX model produces logits matching PyTorch within tolerance.
    """
    try:
        import torch
        import torch.nn as nn
        import onnxruntime as ort
    except ImportError:
        pytest.skip("PyTorch or ONNXRuntime not installed in local environment.")

    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 3)

        def forward(self, x):
            return self.linear(x)

    model = TinyClassifier().eval()
    dummy_input = torch.randn(2, 16, dtype=torch.float32)

    with torch.no_grad():
        pt_logits = model(dummy_input).numpy()

    onnx_path = "/tmp/test_parity_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=18,
    )

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logits"], {"input": dummy_input.numpy()})[0]

    np.testing.assert_allclose(pt_logits, ort_logits, rtol=1e-3, atol=1e-4)
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
