from aegis_dl.normalization import extract_structural_skeleton, normalize_text

try:
    from aegis_dl.distillation import (
        FGMAdversarialPerturber,
        HybridDistillationCSAFL,
        average_checkpoints_model_soup,
        build_llrd_optimizer_params,
        greedy_model_soup,
        sort_checkpoints_numerically,
    )
    from aegis_dl.losses import (
        CostSensitiveAsymmetricFocalLoss,
        FocalLoss,
        TemperatureScaler,
        WeightedCrossEntropyLoss,
        build_loss_fn,
        compute_class_weights_from_counts,
    )
except ImportError:
    CostSensitiveAsymmetricFocalLoss = None
    FocalLoss = None
    WeightedCrossEntropyLoss = None
    TemperatureScaler = None
    build_loss_fn = None
    compute_class_weights_from_counts = None
    FGMAdversarialPerturber = None
    HybridDistillationCSAFL = None
    average_checkpoints_model_soup = None
    build_llrd_optimizer_params = None
    greedy_model_soup = None
    sort_checkpoints_numerically = None

__version__ = "0.2.0"
__all__ = [
    "normalize_text",
    "extract_structural_skeleton",
    "CostSensitiveAsymmetricFocalLoss",
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "TemperatureScaler",
    "build_loss_fn",
    "compute_class_weights_from_counts",
    "FGMAdversarialPerturber",
    "HybridDistillationCSAFL",
    "average_checkpoints_model_soup",
    "build_llrd_optimizer_params",
    "greedy_model_soup",
    "sort_checkpoints_numerically",
]
