from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EMPIRICAL_CLASS_COUNTS: tuple[int, int, int] = (11358, 8268, 1252)


def compute_class_weights_from_counts(
    class_counts: Sequence[int] | Mapping[int, int],
    smoothing_power: float = 0.5,
    max_weight: float = 4.0,
) -> tuple[float, ...]:
    if isinstance(class_counts, Mapping):
        ordered = [int(class_counts[idx]) for idx in sorted(class_counts.keys())]
    else:
        ordered = [int(c) for c in class_counts]

    if not ordered or any(c <= 0 for c in ordered):
        return (1.0, 1.35, 3.0)

    ref_count = float(ordered[0])
    weights = []
    for count in ordered:
        ratio = ref_count / max(float(count), 1.0)
        w = min(max_weight, max(1.0, ratio ** smoothing_power))
        weights.append(round(float(w), 4))
    return tuple(weights)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weights: Sequence[float] = (1.0, 1.35, 3.0),
        label_smoothing: float = 0.03,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_fp32 = logits.float()
        targets_long = targets.long()
        weights = self.class_weights.to(device=logits_fp32.device, dtype=torch.float32)
        return F.cross_entropy(
            logits_fp32,
            targets_long,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        class_weights: Sequence[float] = (1.0, 1.35, 3.0),
        gammas: Sequence[float] = (2.0, 1.5, 2.0),
        label_smoothing: float = 0.03,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_fp32 = logits.float()
        targets_long = targets.long()
        gammas = self.gammas.to(device=logits_fp32.device, dtype=torch.float32)
        class_weights = self.class_weights.to(device=logits_fp32.device, dtype=torch.float32)

        probs = F.softmax(logits_fp32, dim=-1)
        log_probs = F.log_softmax(logits_fp32, dim=-1)

        p_t = probs.gather(dim=-1, index=targets_long.unsqueeze(-1)).squeeze(-1)
        gamma_t = gammas[targets_long]
        weight_t = class_weights[targets_long]

        clamped_p = p_t.clamp(min=1e-6, max=1.0 - 1e-6)
        focal_factor = torch.pow(1.0 - clamped_p, gamma_t)

        nll_loss = -log_probs.gather(dim=-1, index=targets_long.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        ce_smoothed = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss

        loss = weight_t * focal_factor * ce_smoothed
        return loss.mean()


class CostSensitiveAsymmetricFocalLoss(nn.Module):
    def __init__(
        self,
        class_weights: Sequence[float] = (1.0, 1.35, 3.0),
        gammas: Sequence[float] = (2.5, 1.5, 2.0),
        beta_fp: float = 2.25,
        label_smoothing: float = 0.03,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32))
        self.beta_fp = float(beta_fp)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_fp32 = logits.float()
        targets_long = targets.long()
        gammas = self.gammas.to(device=logits_fp32.device, dtype=torch.float32)
        class_weights = self.class_weights.to(device=logits_fp32.device, dtype=torch.float32)

        probs = F.softmax(logits_fp32, dim=-1)
        log_probs = F.log_softmax(logits_fp32, dim=-1)

        p_t = probs.gather(dim=-1, index=targets_long.unsqueeze(-1)).squeeze(-1)
        gamma_t = gammas[targets_long]
        weight_t = class_weights[targets_long]

        clamped_p = p_t.clamp(min=1e-6, max=1.0 - 1e-6)
        focal_factor = torch.pow(1.0 - clamped_p, gamma_t)

        nll_loss = -log_probs.gather(dim=-1, index=targets_long.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        ce_smoothed = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss

        fp_penalty = torch.where(
            targets_long == 0,
            1.0 + (self.beta_fp - 1.0) * (1.0 - clamped_p),
            torch.ones_like(clamped_p),
        )

        loss = weight_t * fp_penalty * focal_factor * ce_smoothed
        return loss.mean()


def build_loss_fn(
    variant: Literal["ce", "focal", "cs_afl"],
    cfg: dict[str, Any] | None = None,
    class_counts: Sequence[int] | Mapping[int, int] | None = None,
) -> nn.Module:
    loss_cfg = ((cfg or {}).get("training") or {}).get("loss") or (cfg or {})

    if class_counts is not None:
        class_weights: Sequence[float] = compute_class_weights_from_counts(class_counts)
    elif "class_weights" in loss_cfg:
        raw_weights = [float(w) for w in loss_cfg["class_weights"]]
        if len(raw_weights) == 3 and raw_weights[2] < 2.0:
            emp_weights = compute_class_weights_from_counts(DEFAULT_EMPIRICAL_CLASS_COUNTS)
            raw_weights[2] = emp_weights[2]
        class_weights = tuple(raw_weights)
    else:
        class_weights = compute_class_weights_from_counts(DEFAULT_EMPIRICAL_CLASS_COUNTS)

    gammas = tuple(float(g) for g in loss_cfg.get("gammas", (2.5, 1.5, 2.0)))
    beta_fp = float(loss_cfg.get("beta_fp", 2.25))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.03))

    if variant == "ce":
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    if variant == "focal":
        return FocalLoss(
            class_weights=class_weights,
            gammas=gammas,
            label_smoothing=label_smoothing,
        )
    if variant == "cs_afl":
        return CostSensitiveAsymmetricFocalLoss(
            class_weights=class_weights,
            gammas=gammas,
            beta_fp=beta_fp,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unsupported loss variant '{variant}'. Expected one of ('ce', 'focal', 'cs_afl').")


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.to(device=logits.device, dtype=logits.dtype).clamp(min=1e-3)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        self.cpu().float()
        nll_criterion = nn.CrossEntropyLoss()
        logits_fp32 = val_logits.detach().float().cpu()
        labels_long = val_labels.detach().long().cpu()

        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.05,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def eval_step() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits_fp32 / self.temperature.clamp(min=1e-3, max=100.0)
            loss = nll_criterion(scaled, labels_long)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3, max=100.0)
        return float(self.temperature.item())
