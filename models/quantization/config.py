"""
FlatQuant configuration for VGGT model.
"""
import argparse
from dataclasses import dataclass, field


@dataclass
class FlatQuantConfig:
    """Configuration for FlatQuant VGGT quantization."""
    # Quantization bits
    w_bits: int = 4
    a_bits: int = 4
    w_asym: bool = False
    a_asym: bool = False
    a_groupsize: int = -1

    # FlatQuant calibration
    epochs: int = 15
    flat_lr: float = 5e-3
    cali_bsz: int = 2
    nsamples: int = 64
    num_frames: int = 4
    img_size: int = 518

    # FlatQuant transforms
    cali_trans: bool = True
    add_diag: bool = True
    lwc: bool = True
    lac: bool = True
    direct_inv: bool = False
    separate_vtrans: bool = False

    # Diagonal initialization
    diag_init: str = "sq_style"
    diag_alpha: float = 0.5

    # Training
    warmup: bool = True
    deactive_amp: bool = False

    # KV-cache (unused for VGGT but needed for FlatQuant API compat)
    q_bits: int = 16
    k_bits: int = 16
    v_bits: int = 16
    q_asym: bool = False
    k_asym: bool = False
    v_asym: bool = False

    # Calibration data
    cali_dataset: str = "co3d"
    cali_data_dir: str = "data/co3dv2/data"
    cali_annotation_dir: str = "data/co3dv2/annotations"
    cali_split: str = "train"
    cali_seed: int = 42
    min_num_images: int = 50

    # Fisher-weighted calibration
    use_fisher: bool = False          # Enable Fisher-weighted reconstruction loss
    fisher_path: str = ""             # Path to precomputed fisher.pt
    # Save/load
    save_dir: str = "/data1/vggt/quantized_model"
    save_matrix: bool = True
    resume: bool = False
    matrix_path: str = ""

    def to_args(self):
        """Convert to argparse.Namespace for FlatQuant API compatibility."""
        args = argparse.Namespace()
        for k, v in self.__dict__.items():
            setattr(args, k, v)
        return args
