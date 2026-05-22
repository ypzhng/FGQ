"""
Quantization module for VGGT model - FlatQuant integration.
"""
from .config import FlatQuantConfig
from .apply_flatquant import (
    apply_flatquant_to_vggt,
    calibrate_flatquant_vggt,
    reparameterize_vggt,
    load_calibration_images,
    FlatQuantVGGTAttention,
    FlatQuantVGGTMLP,
)

__all__ = [
    "FlatQuantConfig",
    "apply_flatquant_to_vggt",
    "calibrate_flatquant_vggt",
    "reparameterize_vggt",
    "load_calibration_images",
    "FlatQuantVGGTAttention",
    "FlatQuantVGGTMLP",
]
