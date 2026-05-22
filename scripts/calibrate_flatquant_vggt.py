#!/usr/bin/env python
"""
FlatQuant calibration script for VGGT model.

Usage:
    conda activate flatquant_vggt
    export PYTHONPATH="/data/projects/zyp/FlatQuant:$PYTHONPATH"
    cd /data/projects/zyp/recons_eval

    python scripts/calibrate_flatquant_vggt.py \
        --model_path /data1/vggt/model \
        --save_dir /data1/vggt/quantized_model \
        --w_bits 4 --a_bits 4 \
        --cali_dataset co3d --cali_data_dir data/co3dv2/data \
        --nsamples 64 --num_frames 4 \
        --epochs 15 --flat_lr 5e-3 \
        --cali_trans --add_diag --lwc --lac --warmup
"""
import os
import sys
import argparse
import datetime
import json
import logging
import tempfile
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from models.vggt.models.vggt import VGGT
from models.quantization.config import FlatQuantConfig
from models.quantization.apply_flatquant import (
    apply_flatquant_to_vggt,
    calibrate_flatquant_vggt,
    reparameterize_vggt,
    load_calibration_images,
    load_fisher_weights,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("flatquant-vggt-calibration")


def parse_args():
    parser = argparse.ArgumentParser(description="FlatQuant calibration for VGGT")

    # Model paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VGGT model checkpoint directory or file")
    parser.add_argument("--save_dir", type=str, default="/data1/vggt/quantized_model",
                        help="Directory to save quantized model")
    parser.add_argument("--output_name", "--output-name", type=str, default=None,
                        help="Checkpoint filename inside --save_dir. Defaults to model_flatquant_w{w_bits}a{a_bits}[_fisher].pt")
    parser.add_argument("--min_save_free_gb", "--min-save-free-gb", type=float, default=8.0,
                        help="Minimum free GiB required in --save_dir before calibration starts. Set to 0 to disable.")
    parser.add_argument("--skip_save", "--skip-save", action="store_true",
                        help="Run calibration and optional profiling without writing the quantized checkpoint")
    parser.add_argument("--profile_path", "--profile-path", type=str, default=None,
                        help="Optional JSON path for wall time and CUDA memory stats")

    # Quantization bits
    parser.add_argument("--w_bits", type=int, default=4, help="Weight quantization bits")
    parser.add_argument("--a_bits", type=int, default=4, help="Activation quantization bits")
    parser.add_argument("--w_asym", action="store_true", help="Asymmetric weight quantization")
    parser.add_argument("--a_asym", action="store_true", help="Asymmetric activation quantization")
    parser.add_argument("--a_groupsize", type=int, default=-1, help="Activation groupsize")

    # FlatQuant transforms
    parser.add_argument("--cali_trans", action="store_true", help="Use calibrated transforms")
    parser.add_argument("--add_diag", action="store_true", help="Add diagonal scaling")
    parser.add_argument("--lwc", action="store_true", help="Learnable weight clipping")
    parser.add_argument("--lac", action="store_true", help="Learnable activation clipping")
    parser.add_argument("--direct_inv", action="store_true", help="Use direct inverse transforms")
    parser.add_argument("--separate_vtrans", action="store_true", help="Separate V transform")

    # Calibration
    parser.add_argument("--epochs", type=int, default=15, help="Calibration epochs per block")
    parser.add_argument("--flat_lr", type=float, default=5e-3, help="Learning rate for transforms")
    parser.add_argument("--cali_bsz", type=int, default=2, help="Calibration batch size")
    parser.add_argument("--nsamples", type=int, default=64, help="Number of calibration samples")
    parser.add_argument("--num_frames", type=int, default=4, help="Frames per calibration sample")
    parser.add_argument("--img_size", type=int, default=518, help="Image size")

    # Initialization
    parser.add_argument("--diag_init", type=str, default="sq_style", help="Diagonal init method")
    parser.add_argument("--diag_alpha", type=float, default=0.5, help="Diagonal init alpha")
    parser.add_argument("--warmup", action="store_true", help="Use warmup")
    parser.add_argument("--deactive_amp", action="store_true", help="Deactivate AMP")

    # Calibration data
    parser.add_argument("--cali_dataset", type=str, default="co3d", help="Calibration dataset")
    parser.add_argument("--cali_data_dir", type=str, default="data/co3dv2/data",
                        help="Calibration data directory")
    parser.add_argument("--cali_annotation_dir", type=str, default="data/co3dv2/annotations",
                        help="Deprecated for VGGT calibration; random raw CO3D sampling does not use annotations")
    parser.add_argument("--cali_split", type=str, default="train", choices=["train", "test"],
                        help="Deprecated for VGGT calibration; kept for CLI compatibility")
    parser.add_argument("--cali_seed", type=int, default=42, help="Calibration sampling seed")
    parser.add_argument("--min_num_images", type=int, default=50,
                        help="Deprecated for VGGT calibration; random raw sampling only requires --num_frames")

    # Fisher-weighted calibration
    parser.add_argument("--use_fisher", action="store_true", help="Use Fisher-weighted reconstruction loss")
    parser.add_argument("--fisher_path", type=str, default="outputs/fisher/fisher.pt",
                        help="Path to precomputed fisher.pt")

    return parser.parse_args()


def _format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _resolve_output_name(args):
    fisher_suffix = "_fisher" if args.use_fisher else ""
    if args.output_name is None:
        return f"model_flatquant_w{args.w_bits}a{args.a_bits}{fisher_suffix}.pt"
    if os.path.basename(args.output_name) != args.output_name:
        raise ValueError("--output_name must be a filename inside --save_dir, not a path")
    return args.output_name


def _preflight_save_path(save_dir, output_name, min_free_gb):
    save_dir = os.path.abspath(save_dir)
    save_path = os.path.join(save_dir, output_name)

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create --save_dir {save_dir}: {exc}") from exc

    if not os.path.isdir(save_dir):
        raise RuntimeError(f"--save_dir is not a directory: {save_dir}")

    if os.path.exists(save_path) and not os.access(save_path, os.W_OK):
        raise RuntimeError(f"Cannot overwrite existing checkpoint because it is not writable: {save_path}")

    try:
        stat = os.statvfs(save_dir)
    except OSError as exc:
        raise RuntimeError(f"Cannot inspect free space for --save_dir {save_dir}: {exc}") from exc

    free_bytes = stat.f_bavail * stat.f_frsize
    min_free_bytes = int(max(min_free_gb, 0.0) * 1024 ** 3)
    if min_free_bytes > 0 and free_bytes < min_free_bytes:
        raise RuntimeError(
            f"Not enough free space in --save_dir {save_dir}: "
            f"{_format_bytes(free_bytes)} available, at least {_format_bytes(min_free_bytes)} required. "
            "Use --save_dir on a writable filesystem with more space, or lower --min_save_free_gb."
        )

    test_path = None
    try:
        fd, test_path = tempfile.mkstemp(prefix=".flatquant_write_test_", dir=save_dir)
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RuntimeError(
            f"Cannot write to --save_dir {save_dir}: {exc}. "
            "Check mount options, permissions, and free space."
        ) from exc
    finally:
        if test_path is not None:
            try:
                os.unlink(test_path)
            except OSError:
                pass

    return save_path


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def _cuda_memory_stats():
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()
    device_index = torch.cuda.current_device()
    stats = {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "current_allocated_bytes": torch.cuda.memory_allocated(device_index),
        "current_reserved_bytes": torch.cuda.memory_reserved(device_index),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
    }
    for key in list(stats):
        if key.endswith("_bytes"):
            stats[key.replace("_bytes", "_gib")] = stats[key] / (1024 ** 3)
    return stats


def _write_profile(path, payload):
    if path is None:
        return
    profile_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    total_start = time.perf_counter()
    profile = {
        "script": os.path.relpath(__file__, root),
        "argv": sys.argv[1:],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": _now_iso(),
        "phases": {},
    }

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    output_name = _resolve_output_name(args)
    if args.skip_save:
        save_path = os.path.join(os.path.abspath(args.save_dir), output_name)
        logger.info("Skipping checkpoint save because --skip_save was set")
    else:
        try:
            save_path = _preflight_save_path(args.save_dir, output_name, args.min_save_free_gb)
        except RuntimeError as exc:
            logger.error(str(exc))
            raise SystemExit(2) from exc
        logger.info(f"Checkpoint will be saved to {save_path}")

    # ---- 1. Load VGGT model ----
    phase_start = time.perf_counter()
    logger.info(f"Loading VGGT model from {args.model_path}")
    model = VGGT()

    # Load weights
    model_file = args.model_path
    if os.path.isdir(model_file):
        # Try common filenames
        for fname in ["model.safetensors", "model.pt", "pytorch_model.bin"]:
            candidate = os.path.join(model_file, fname)
            if os.path.exists(candidate):
                model_file = candidate
                break

    if model_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(model_file))
    else:
        model.load_state_dict(torch.load(model_file, map_location="cpu"))

    model.eval()
    logger.info("Model loaded successfully")
    profile["phases"]["load_model_seconds"] = time.perf_counter() - phase_start

    # ---- 2. Create FlatQuant config ----
    config = FlatQuantConfig(
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        w_asym=args.w_asym,
        a_asym=args.a_asym,
        a_groupsize=args.a_groupsize,
        epochs=args.epochs,
        flat_lr=args.flat_lr,
        cali_bsz=args.cali_bsz,
        nsamples=args.nsamples,
        num_frames=args.num_frames,
        img_size=args.img_size,
        cali_trans=args.cali_trans,
        add_diag=args.add_diag,
        lwc=args.lwc,
        lac=args.lac,
        direct_inv=args.direct_inv,
        separate_vtrans=args.separate_vtrans,
        diag_init=args.diag_init,
        diag_alpha=args.diag_alpha,
        warmup=args.warmup,
        deactive_amp=args.deactive_amp,
        cali_dataset=args.cali_dataset,
        cali_data_dir=args.cali_data_dir,
        cali_annotation_dir=args.cali_annotation_dir,
        cali_split=args.cali_split,
        cali_seed=args.cali_seed,
        min_num_images=args.min_num_images,
        save_dir=args.save_dir,
        use_fisher=args.use_fisher,
        fisher_path=args.fisher_path,
    )

    fq_args = config.to_args()

    # ---- 3. Apply FlatQuant structure ----
    phase_start = time.perf_counter()
    logger.info("Applying FlatQuant wrappers to model...")
    model = apply_flatquant_to_vggt(fq_args, model)
    profile["phases"]["apply_flatquant_seconds"] = time.perf_counter() - phase_start

    # ---- 4. Load calibration data ----
    phase_start = time.perf_counter()
    logger.info("Loading calibration data...")
    calibration_data = load_calibration_images(config)
    profile["phases"]["load_calibration_data_seconds"] = time.perf_counter() - phase_start

    # ---- 5. Run calibration ----
    logger.info("Starting FlatQuant calibration...")
    fisher_weights = None
    if config.use_fisher:
        fisher_weights = load_fisher_weights(config.fisher_path)
        logger.info(f"Loaded Fisher weights from {config.fisher_path}, shape: {fisher_weights.shape}")
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    phase_start = time.perf_counter()
    profile["calibration_started_at"] = _now_iso()
    calibrate_flatquant_vggt(fq_args, model, calibration_data, device, logger, fisher_weights=fisher_weights)
    if device == "cuda":
        torch.cuda.synchronize()
    profile["calibration_finished_at"] = _now_iso()
    profile["phases"]["calibration_seconds"] = time.perf_counter() - phase_start
    profile["calibration_cuda_memory"] = _cuda_memory_stats()
    logger.info(f"Calibration wall time: {profile['phases']['calibration_seconds']:.2f} seconds")
    if profile["calibration_cuda_memory"] is not None:
        mem = profile["calibration_cuda_memory"]
        logger.info(
            "Calibration CUDA peak memory: "
            f"allocated={mem['peak_allocated_gib']:.2f} GiB, "
            f"reserved={mem['peak_reserved_gib']:.2f} GiB"
        )

    # ---- 6. Reparameterize ----
    phase_start = time.perf_counter()
    logger.info("Reparameterizing model...")
    model = model.to(device)
    reparameterize_vggt(model)
    if device == "cuda":
        torch.cuda.synchronize()
    profile["phases"]["reparameterize_seconds"] = time.perf_counter() - phase_start

    # ---- 7. Save ----
    save_start = time.perf_counter()
    # Save model state dict and config
    save_dict = {
        "model": model.state_dict(),
        "config": {
            "model_path": args.model_path,
            "output_name": output_name,
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "w_asym": args.w_asym,
            "a_asym": args.a_asym,
            "a_groupsize": args.a_groupsize,
            "cali_trans": args.cali_trans,
            "add_diag": args.add_diag,
            "lwc": args.lwc,
            "lac": args.lac,
            "direct_inv": args.direct_inv,
            "separate_vtrans": args.separate_vtrans,
            "epochs": args.epochs,
            "flat_lr": args.flat_lr,
            "cali_bsz": args.cali_bsz,
            "nsamples": args.nsamples,
            "num_frames": args.num_frames,
            "img_size": args.img_size,
            "cali_sampling": "random_raw",
            "diag_init": args.diag_init,
            "diag_alpha": args.diag_alpha,
            "warmup": args.warmup,
            "deactive_amp": args.deactive_amp,
            "cali_dataset": args.cali_dataset,
            "cali_data_dir": args.cali_data_dir,
            "cali_annotation_dir": args.cali_annotation_dir,
            "cali_split": args.cali_split,
            "cali_seed": args.cali_seed,
            "min_num_images": args.min_num_images,
            "use_fisher": args.use_fisher,
            "fisher_path": args.fisher_path if args.use_fisher else None,
        },
    }
    profile["config"] = save_dict["config"]
    profile["checkpoint_path"] = None if args.skip_save else save_path
    if args.skip_save:
        logger.info("Quantized model save skipped")
    else:
        torch.save(save_dict, save_path)
        logger.info(f"Quantized model saved to {save_path}")
    profile["phases"]["save_seconds"] = time.perf_counter() - save_start
    profile["finished_at"] = _now_iso()
    profile["total_seconds"] = time.perf_counter() - total_start
    profile["final_cuda_memory"] = _cuda_memory_stats()
    _write_profile(args.profile_path, profile)
    if args.profile_path is not None:
        logger.info(f"Profile summary saved to {args.profile_path}")


if __name__ == "__main__":
    main()
