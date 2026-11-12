"""
Cost-Sensitive Asymmetric Focal Loss & Post-Hoc Temperature Scaling.
Designed for high-precision in-line AI security gateways.
"""

from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CostSensitiveAsymmetricFocalLoss(nn.Module):
    """
    Cost-Sensitive Asymmetric Focal Loss (CS-AFL).

    Concentrates benign gradient updates on DevOps hard negatives (gamma_0 = 2.5),
    applies asymmetric false-positive penalty (beta_fp = 2.25) when benign inputs
    are falsely predicted as malicious, and incorporates label smoothing (eps = 0.03).
    """

    def __init__(
        self,
        class_weights: Sequence[float] = (1.0, 1.35, 1.65),
        gammas: Sequence[float] = (2.5, 1.5, 1.5),
        beta_fp: float = 2.25,
        label_smoothing: float = 0.03,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.register_buffer("gammas", torch.tensor(gammas, dtype=torch.float32))
        self.beta_fp = float(beta_fp)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gammas = self.gammas.to(logits.device)
        class_weights = self.class_weights.to(logits.device)

        num_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Probabilities and hyperparams for the true classes
        p_t = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        gamma_t = gammas[targets]
        weight_t = class_weights[targets]

        # Focal factor: (1 - p_t)^gamma_t
        clamped_p = p_t.clamp(min=1e-6, max=1.0 - 1e-6)
        focal_factor = torch.pow(1.0 - clamped_p, gamma_t)

        # Label-smoothed negative log-likelihood
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        ce_smoothed = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss

        # Asymmetric False Positive penalty: benign (y == 0) predicted as malicious (pred != 0)
        preds = logits.argmax(dim=-1)
        is_false_positive = (targets == 0) & (preds != 0)
        fp_penalty = torch.where(is_false_positive, self.beta_fp, 1.0)

        loss = weight_t * fp_penalty * focal_factor * ce_smoothed
        return loss.mean()


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for calibrated multi-class probabilities.
    Learns scalar parameter T > 0 by minimizing NLL on validation logits using L-BFGS.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        nll_criterion = nn.CrossEntropyLoss()
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.scale_logits(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)
        return float(self.temperature.item())
