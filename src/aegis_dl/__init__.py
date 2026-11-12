"""
AegisLLM Deep Learning Module (Milestone 2)
Software Engineering for Deep Learning - ETSIINF (UPM)
"""

from aegis_dl.normalization import normalize_text, extract_structural_skeleton

try:
    from aegis_dl.losses import CostSensitiveAsymmetricFocalLoss, TemperatureScaler
    from aegis_dl.distillation import (
        FGMAdversarialPerturber,
        HybridDistillationCSAFL,
        average_checkpoints_model_soup,
    )
except ImportError:
    CostSensitiveAsymmetricFocalLoss = None
    TemperatureScaler = None
    FGMAdversarialPerturber = None
    HybridDistillationCSAFL = None
    average_checkpoints_model_soup = None

__version__ = "0.2.0"
__all__ = [
    "normalize_text",
    "extract_structural_skeleton",
    "CostSensitiveAsymmetricFocalLoss",
    "TemperatureScaler",
    "FGMAdversarialPerturber",
    "HybridDistillationCSAFL",
    "average_checkpoints_model_soup",
]
