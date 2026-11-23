import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from aegis_dl.distillation import (
    FGMAdversarialPerturber,
    build_llrd_optimizer_params,
    greedy_model_soup,
    sort_checkpoints_numerically,
)
from aegis_dl.error_analysis import (
    analyze_subcategory_performance,
    extract_top_failures,
)
from aegis_dl.losses import (
    CostSensitiveAsymmetricFocalLoss,
    FocalLoss,
    TemperatureScaler,
    build_loss_fn,
    compute_class_weights_from_counts,
)
from aegis_dl.metrics import (
    compute_quantization_drift,
    macro_f1_from_probs,
    tpr_at_1pct_fpr_from_probs,
)


def test_cs_afl_smooth_fp_penalty_differentiable():
    loss_fn = CostSensitiveAsymmetricFocalLoss(
        class_weights=(1.0, 1.35, 3.0),
        gammas=(2.5, 1.5, 2.0),
        beta_fp=2.25,
        label_smoothing=0.03,
    )
    targets = torch.tensor([0, 0], dtype=torch.long)

    logits_low_fp = torch.tensor([[2.0, -1.0, -1.0], [2.0, -1.0, -1.0]], requires_grad=True)
    logits_high_fp = torch.tensor([[-1.5, 2.5, 0.5], [-1.5, 2.5, 0.5]], requires_grad=True)

    loss_low = loss_fn(logits_low_fp, targets)
    loss_high = loss_fn(logits_high_fp, targets)
    loss_low.backward()
    loss_high.backward()

    assert loss_high.item() > loss_low.item()
    assert logits_low_fp.grad is not None and torch.all(torch.isfinite(logits_low_fp.grad))
    assert logits_high_fp.grad is not None and torch.all(torch.isfinite(logits_high_fp.grad))

    # Verify smoothness across the decision boundary (no jump discontinuity at logit_0 == logit_1)
    eps = 1e-4
    b_minus = torch.tensor([[1.0 + eps, 1.0 - eps, -2.0]], dtype=torch.float32)
    b_plus = torch.tensor([[1.0 - eps, 1.0 + eps, -2.0]], dtype=torch.float32)
    diff = abs(loss_fn(b_plus, torch.tensor([0])).item() - loss_fn(b_minus, torch.tensor([0])).item())
    assert diff < 1e-2, f"Discontinuity detected across argmax boundary: delta={diff}"


def test_class_imbalance_weights_for_6pct_jailbreak():
    # Empirical train distribution from data/manifest.json: 11358 benign, 8268 injection, 1252 jailbreak (6.0%)
    weights = compute_class_weights_from_counts([11358, 8268, 1252], smoothing_power=0.5)
    assert len(weights) == 3
    assert math.isclose(weights[0], 1.0, rel_tol=1e-4)
    assert 1.1 < weights[1] < 1.5
    assert 2.5 <= weights[2] <= 3.5

    loss_fn = build_loss_fn("cs_afl", cfg={}, class_counts=[11358, 8268, 1252])
    assert isinstance(loss_fn, CostSensitiveAsymmetricFocalLoss)
    assert float(loss_fn.class_weights[2].item()) >= 2.5


def test_build_loss_fn_ablation_variants():
    cfg = {
        "training": {
            "loss": {
                "class_weights": [1.0, 1.35, 3.0],
                "gammas": [2.5, 1.5, 2.0],
                "beta_fp": 2.25,
                "label_smoothing": 0.03,
            }
        }
    }
    logits = torch.randn(8, 3, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)

    for variant, expected_cls in [
        ("ce", nn.Module),
        ("focal", FocalLoss),
        ("cs_afl", CostSensitiveAsymmetricFocalLoss),
    ]:
        fn = build_loss_fn(variant, cfg)
        assert isinstance(fn, expected_cls)
        out = fn(logits, targets)
        assert out.ndim == 0
        assert torch.isfinite(out)

    with pytest.raises(ValueError):
        build_loss_fn("unknown_variant", cfg)


def test_temperature_scaler_bf16_safety():
    rng = torch.Generator().manual_seed(42)
    raw_logits = torch.randn(120, 3, generator=rng) * 3.5
    labels = torch.argmax(raw_logits, dim=-1)
    # Flip 15% of labels so model is overconfident and requires T > 1
    labels[:18] = (labels[:18] + 1) % 3

    bf16_logits = raw_logits.to(torch.bfloat16)
    scaler = TemperatureScaler(init_temperature=1.0)
    learned_t = scaler.fit(bf16_logits, labels, max_iter=50)

    assert math.isfinite(learned_t)
    assert learned_t > 1.0
    scaled_bf16 = scaler.scale_logits(bf16_logits)
    assert scaled_bf16.dtype == torch.bfloat16


def test_fgm_perturber_bf16_and_grad_accumulation():
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(32, 16)
            self.classifier = nn.Linear(16, 3)

        def forward(self, input_ids):
            x = self.word_embeddings(input_ids).mean(dim=1)
            return self.classifier(x)

    model = DummyBackbone().to(torch.bfloat16)
    perturber = FGMAdversarialPerturber(model, emb_names=["word_embeddings"], epsilon=0.5)

    orig_weight = model.word_embeddings.weight.detach().clone()
    x = torch.randint(0, 32, (4, 8))
    y = torch.tensor([0, 1, 2, 0])

    loss = nn.CrossEntropyLoss()(model(x).float(), y)
    loss.backward()
    perturber.backup_grad()

    perturber.attack()
    assert model.word_embeddings.weight.dtype == torch.bfloat16
    assert not torch.allclose(model.word_embeddings.weight, orig_weight)

    perturber.restore()
    assert torch.allclose(model.word_embeddings.weight, orig_weight)
    perturber.restore_grad()
    assert model.word_embeddings.weight.grad is not None


def test_build_llrd_optimizer_params():
    class MockDeberta(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(20, 8)
            self.encoder_layer_0 = nn.Linear(8, 8)
            self.encoder_layer_5 = nn.Linear(8, 8)
            self.classifier = nn.Linear(8, 3)

    # Create module matching HF DeBERTa parameter naming convention
    model = nn.Module()
    model.deberta = nn.Module()
    model.deberta.embeddings = nn.Embedding(20, 8)
    model.deberta.encoder = nn.Module()
    model.deberta.encoder.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])
    model.deberta.encoder.LayerNorm = nn.LayerNorm(8)
    model.classifier = nn.Linear(8, 3)

    groups = build_llrd_optimizer_params(
        model,
        backbone_lr=2.5e-5,
        head_lr=1.0e-4,
        llrd_decay=0.90,
        weight_decay=0.01,
    )

    total_params = sum(len(g["params"]) for g in groups)
    expected_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_params == expected_params

    lrs = sorted({round(g["lr"], 10) for g in groups})
    assert max(lrs) == round(1.0e-4, 10)
    assert min(lrs) == round(2.5e-5 * (0.90 ** 6), 10)

    # Ensure bias and LayerNorm have 0.0 weight decay
    for g in groups:
        if g["weight_decay"] == 0.0:
            assert len(g["params"]) > 0


def test_greedy_model_soup_numerical_sort_and_monotonic_improvement(tmp_path: Path):
    paths = [
        str(tmp_path / "checkpoint-1050"),
        str(tmp_path / "checkpoint-175"),
        str(tmp_path / "checkpoint-350"),
    ]
    sorted_paths = sort_checkpoints_numerically(paths)
    assert [Path(p).name for p in sorted_paths] == [
        "checkpoint-175",
        "checkpoint-350",
        "checkpoint-1050",
    ]

    target_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    def make_model():
        m = nn.Linear(2, 2, bias=False)
        return m

    # Checkpoint 175: slightly above target; Checkpoint 350: slightly below target (averaging improves!)
    # Checkpoint 1050: far outlier (averaging hurts and must be rejected by greedy soup)
    sd_175 = {"weight": target_weight + torch.tensor([[0.2, -0.2], [0.2, -0.2]])}
    sd_350 = {"weight": target_weight + torch.tensor([[-0.2, 0.2], [-0.2, 0.2]])}
    sd_1050 = {"weight": target_weight + torch.tensor([[5.0, 5.0], [5.0, 5.0]])}

    for p, sd in zip(sorted_paths, [sd_175, sd_350, sd_1050], strict=True):
        d = Path(p)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(sd, d / "state_dict.pt")

    def eval_fn(model: nn.Module) -> float:
        return float(torch.mean((model.weight - target_weight) ** 2).item())

    soup_sd, retained, best_loss = greedy_model_soup(
        checkpoint_paths=paths,
        eval_fn=eval_fn,
        model_factory=make_model,
        higher_is_better=False,
    )

    assert len(retained) == 2
    assert str(tmp_path / "checkpoint-1050") not in retained
    assert best_loss < 1e-6
    assert torch.allclose(soup_sd["weight"], target_weight, atol=1e-5)


def test_error_analysis_and_quantization_drift_metrics():
    df = pd.DataFrame(
        {
            "text": ["cmd 1", "cmd 2", "attack 1", "attack 2"],
            "sub_category": [
                "hard_negative_devops",
                "hard_negative_devops",
                "direct_delimiter_override",
                "jailbreak_persona",
            ],
        }
    )
    y_true = np.array([0, 0, 1, 2])
    probs = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.20, 0.70, 0.10],
            [0.05, 0.90, 0.05],
            [0.65, 0.15, 0.20],
        ]
    )
    y_pred = np.argmax(probs, axis=-1)

    fp_df, fn_df = extract_top_failures(df, y_true, probs, k=5)
    assert len(fp_df) == 1
    assert len(fn_df) == 1

    sub_df = analyze_subcategory_performance(df, y_true, y_pred, probs=probs)
    expected_cols = {"sub_category", "sample_count", "accuracy", "fpr", "fnr", "mean_s_mal"}
    assert expected_cols.issubset(set(sub_df.columns))

    fp32_logits = np.array([[2.0, -1.0, -1.0], [-1.0, 2.5, -0.5], [-1.0, -0.5, 2.2], [1.8, -0.8, -0.9]])
    int8_logits = fp32_logits + np.array([[0.01, -0.01, 0.0], [-0.01, 0.02, -0.01], [0.0, 0.01, -0.01], [-0.01, 0.01, 0.0]])
    drift = compute_quantization_drift(fp32_logits, int8_logits, y_true=np.array([0, 1, 2, 0]), tau_95=0.5)

    for key in (
        "max_abs_diff",
        "mean_abs_diff",
        "kl_div_mean",
        "prediction_agreement",
        "delta_fpr_at_95_tpr",
        "delta_pr_auc",
    ):
        assert key in drift
    assert drift["prediction_agreement"] == 1.0
    assert drift["max_abs_diff"] < 0.05

    f1_val = macro_f1_from_probs(y_true, probs)
    tpr_val = tpr_at_1pct_fpr_from_probs(y_true, probs)
    assert 0.0 <= f1_val <= 1.0
    assert 0.0 <= tpr_val <= 1.0
