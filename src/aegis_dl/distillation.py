from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGMAdversarialPerturber:
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
        self.grad_backup: Dict[str, torch.Tensor] = {}

    def _iter_target_embeddings(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(emb in name for emb in self.emb_names):
                yield name, param

    def backup_grad(self) -> None:
        self.grad_backup.clear()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.detach().clone()

    def restore_grad(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.grad_backup:
                    if param.grad is None:
                        param.grad = self.grad_backup[name].clone()
                    else:
                        param.grad.copy_(self.grad_backup[name])
        self.grad_backup.clear()

    def attack(self) -> None:
        self.backup.clear()
        with torch.no_grad():
            for name, param in self._iter_target_embeddings():
                if param.grad is None:
                    continue
                self.backup[name] = param.detach().clone()
                grad_fp32 = param.grad.detach().float()
                norm = torch.linalg.vector_norm(grad_fp32)
                if torch.isfinite(norm) and norm.item() > 1e-12:
                    r_at = (self.epsilon * grad_fp32 / (norm + 1e-12)).to(dtype=param.dtype)
                    param.add_(r_at)

    def restore(self) -> None:
        with torch.no_grad():
            for name, param in self._iter_target_embeddings():
                if name in self.backup:
                    param.copy_(self.backup[name])
        self.backup.clear()


class HybridDistillationCSAFL(nn.Module):
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

        student_fp32 = student_logits.float()
        teacher_fp32 = teacher_logits.to(device=student_logits.device, dtype=torch.float32)

        student_log_probs = F.log_softmax(student_fp32 / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_fp32 / self.temperature, dim=-1)
        soft_loss = (self.temperature ** 2) * self.kl_div(student_log_probs, teacher_probs)

        return (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss


def build_llrd_optimizer_params(
    model: nn.Module,
    backbone_lr: float = 2.5e-5,
    head_lr: float = 1.0e-4,
    llrd_decay: float = 0.90,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    layer_pattern = re.compile(r"(?:encoder\.layer|layers)\.(\d+)\.")
    no_decay_tokens = ("bias", "layernorm", "layer_norm", "norm")
    head_tokens = ("classifier", "pooler", "score", "head")
    embedding_tokens = ("embeddings", "rel_embeddings")

    max_layer_idx = -1
    for name, _ in model.named_parameters():
        match = layer_pattern.search(name)
        if match:
            max_layer_idx = max(max_layer_idx, int(match.group(1)))
    num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else 6

    grouped: Dict[Tuple[float, float], List[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lower_name = name.lower()
        wd = 0.0 if any(tok in lower_name for tok in no_decay_tokens) else float(weight_decay)

        match = layer_pattern.search(name)
        if any(tok in lower_name for tok in head_tokens):
            lr = float(head_lr)
        elif match:
            layer_idx = int(match.group(1))
            depth_from_top = max(0, num_layers - 1 - layer_idx)
            lr = float(backbone_lr * (llrd_decay ** depth_from_top))
        elif any(tok in lower_name for tok in embedding_tokens):
            lr = float(backbone_lr * (llrd_decay ** num_layers))
        else:
            lr = float(backbone_lr)

        key = (round(lr, 12), wd)
        grouped.setdefault(key, []).append(param)

    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in grouped.items()
        if params
    ]


def sort_checkpoints_numerically(checkpoint_paths: Sequence[str]) -> List[str]:
    def _extract_step(path_str: str) -> Tuple[int, str]:
        name = Path(path_str).name
        matches = re.findall(r"(\d+)", name)
        step = int(matches[-1]) if matches else -1
        return (step, path_str)

    return sorted(checkpoint_paths, key=_extract_step)


def average_checkpoints_model_soup(
    checkpoint_state_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not checkpoint_state_dicts:
        raise ValueError("Cannot compute model soup over an empty list of checkpoints.")

    soup_state: Dict[str, torch.Tensor] = {}
    k = len(checkpoint_state_dicts)
    first_dict = checkpoint_state_dicts[0]

    for key, ref_tensor in first_dict.items():
        if ref_tensor.is_floating_point():
            acc = torch.zeros_like(ref_tensor, dtype=torch.float32, device="cpu")
            for sd in checkpoint_state_dicts:
                acc.add_(sd[key].detach().cpu().float())
            soup_state[key] = (acc / float(k)).to(dtype=ref_tensor.dtype)
        else:
            soup_state[key] = ref_tensor.detach().cpu().clone()
    return soup_state


def _load_checkpoint_state_dict(
    ckpt_path: str,
    model_factory: Optional[Callable[[], nn.Module]] = None,
) -> Dict[str, torch.Tensor]:
    p = Path(ckpt_path)
    if p.is_file():
        payload = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            return payload["model_state_dict"]
        return payload

    if p.is_dir():
        for fname in ("state_dict.pt", "pytorch_model.bin", "best_model_state.pt"):
            candidate = p / fname
            if candidate.exists():
                payload = torch.load(candidate, map_location="cpu", weights_only=True)
                if isinstance(payload, dict) and "model_state_dict" in payload:
                    return payload["model_state_dict"]
                return payload

        safetensors_file = p / "model.safetensors"
        if safetensors_file.exists():
            from safetensors.torch import load_file

            return load_file(str(safetensors_file), device="cpu")

        from transformers import AutoModelForSequenceClassification

        hf_model = AutoModelForSequenceClassification.from_pretrained(str(p))
        return {k: v.detach().cpu() for k, v in hf_model.state_dict().items()}

    raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")


def greedy_model_soup(
    checkpoint_paths: List[str],
    eval_fn: Callable[[nn.Module], float],
    model_factory: Callable[[], nn.Module],
    higher_is_better: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List[str], float]:
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty.")

    ordered_paths = sort_checkpoints_numerically(checkpoint_paths)
    loaded_sds: Dict[str, Dict[str, torch.Tensor]] = {}
    individual_scores: List[Tuple[float, str]] = []

    eval_model = model_factory()
    for path in ordered_paths:
        sd = _load_checkpoint_state_dict(path, model_factory)
        loaded_sds[path] = sd
        eval_model.load_state_dict(sd)
        eval_model.eval()
        score = float(eval_fn(eval_model))
        individual_scores.append((score, path))

    individual_scores.sort(key=lambda item: item[0], reverse=higher_is_better)

    best_score, best_first_path = individual_scores[0]
    retained_paths: List[str] = [best_first_path]
    retained_sds: List[Dict[str, torch.Tensor]] = [loaded_sds[best_first_path]]

    for _, candidate_path in individual_scores[1:]:
        trial_sds = retained_sds + [loaded_sds[candidate_path]]
        trial_soup = average_checkpoints_model_soup(trial_sds)
        eval_model.load_state_dict(trial_soup)
        eval_model.eval()
        trial_score = float(eval_fn(eval_model))

        improved = (trial_score > best_score) if higher_is_better else (trial_score < best_score)
        if improved:
            best_score = trial_score
            retained_paths.append(candidate_path)
            retained_sds.append(loaded_sds[candidate_path])

    final_soup = average_checkpoints_model_soup(retained_sds)
    return final_soup, retained_paths, float(best_score)
