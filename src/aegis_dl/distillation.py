"""
High-Compute Training Extensions for AegisLLM.
- Fast Gradient Method (FGM) Adversarial Perturbations
- Teacher-Student Knowledge Distillation with CS-AFL
- Uniform Model Soup (Weight Averaging)
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGMAdversarialPerturber:
    """
    Fast Gradient Method (FGM) for adversarial embedding training.
    Perturbs word embeddings in the gradient direction during backprop
    to regularize decision boundaries and suppress shortcut learning.
    """

    def __init__(
        self,
        model: nn.Module,
        emb_names: Optional[List[str]] = None,
        epsilon: float = 0.5,
    ) -> None:
        self.model = model
        self.emb_names = emb_names or ["word_embeddings"]
        self.epsilon = float(epsilon)
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        """Injects normalized adversarial perturbation into embedding weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(emb in name for emb in self.emb_names) and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self) -> None:
        """Restores original unperturbed embedding weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class HybridDistillationCSAFL(nn.Module):
    """
    Hybrid Distillation Loss:
    Blends Cost-Sensitive Asymmetric Focal Loss (CS-AFL) on ground-truth labels
    with Kullback-Leibler divergence on Teacher soft logits (DeBERTa-v3-large).
    """

    def __init__(
        self,
        cs_afl_loss: nn.Module,
        alpha_distill: float = 0.4,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.cs_afl = cs_afl_loss
        self.alpha = float(alpha_distill)
        self.temperature = float(temperature)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hard_loss = self.cs_afl(student_logits, targets)
        if teacher_logits is None or self.alpha <= 0.0:
            return hard_loss

        # Soft distillation loss with temperature scaling tau_d
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = (self.temperature ** 2) * self.kl_div(student_log_probs, teacher_probs)

        return (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss


def average_checkpoints_model_soup(checkpoint_state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Uniform Model Soup (Wortsman et al., ICML 2022).
    Averages parameters of the K best checkpoints to boost OOD generalization
    with zero inference runtime overhead.
    """
    if not checkpoint_state_dicts:
        raise ValueError("Cannot compute model soup over an empty list of checkpoints.")

    soup_state: Dict[str, torch.Tensor] = {}
    k = len(checkpoint_state_dicts)
    first_dict = checkpoint_state_dicts[0]

    for key in first_dict.keys():
        if first_dict[key].is_floating_point():
            soup_state[key] = sum(sd[key].float() for sd in checkpoint_state_dicts) / k
            soup_state[key] = soup_state[key].to(first_dict[key].dtype)
        else:
            soup_state[key] = first_dict[key].clone()
    return soup_state
