#!/usr/bin/env python
"""
Compute per-block and per-channel quantization loss for VGGT task sensitivity analysis.

Measures the actual output degradation caused by W4A4 quantization of each aggregator
block individually, broken down by task (camera, depth, points) and channel. This
provides a direct empirical measurement of quantization sensitivity, complementing the
Fisher-based proxy in compute_fisher.py.

Usage:
    conda activate flatquant_vggt
    cd /data/projects/zyp/recons_eval
    python scripts/compute_quant_loss.py \
        --model_path /data1/vggt/model \
        --nsamples 8 --num_frames 4 --img_size 518 \
        --save_dir outputs/quant_loss

    # Quick test on first 4 blocks
    python scripts/compute_quant_loss.py \
        --model_path /data1/vggt/model --nsamples 2 --max_blocks 4

    # Re-plot from saved data
    python scripts/compute_quant_loss.py --plot_only --save_dir outputs/quant_loss
"""
import os
import sys
import random
import argparse
import logging

import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from models.vggt.models.vggt import VGGT
from models.quantization.apply_flatquant import (
    apply_flatquant_to_vggt, reparameterize_vggt, load_calibration_images,
)
from models.quantization.config import FlatQuantConfig
from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.quant_utils import WeightQuantizer, set_act_quantizer_state

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger("quant_loss")

TASK_NAMES = ["camera", "depth", "points"]
TASK_KEYS = ["pose_enc", "depth", "world_points"]
NUM_BLOCKS = 48
EMBED_DIM = 1024

# Shrink each heatmap's vmin/vmax range to this fraction of the data span so
# differences in the bulk of the data become more visible (extremes get clipped).
RANGE_FACTOR = 0.6

CALI_DATASET_DEFAULTS = {
    "co3d": "data/co3dv2/data",
    "kitti": "data/kitti/depth_selection/val_selection_cropped/image_gathered",
}


# ============================================================================
# Calibration data loading (reused from compute_fisher.py / plot_sensitivity_validation.py)
# ============================================================================

def load_calibration_images_kitti(data_dir, nsamples, num_frames, img_size):
    """Load calibration images from KITTI image_gathered directory."""
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    seq_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    seq_dirs = [d for d in seq_dirs if len(os.listdir(d)) >= num_frames]
    if not seq_dirs:
        raise RuntimeError(f"No KITTI sequences with >= {num_frames} frames in {data_dir}")

    calibration_data = []
    rng = random.Random(42)
    for _ in range(nsamples):
        seq_dir = rng.choice(seq_dirs)
        all_imgs = sorted([
            os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
            if f.lower().endswith((".png", ".jpg"))
        ])
        selected = sorted(rng.sample(all_imgs, num_frames))
        frames = [transform(Image.open(p).convert("RGB")) for p in selected]
        calibration_data.append(torch.stack(frames))

    return calibration_data


# ============================================================================
# Model loading with FlatQuant wrappers
# ============================================================================

def load_model_flatquant(model_path, device, w_bits=4, a_bits=4):
    """Load VGGT model, apply FlatQuant wrappers, reparameterize, move to device.

    After this, each block's linear layers are FlatQuantizedLinear with _eval_mode=True
    and FP (unquantized) weights. Activation quantizers are disabled.
    """
    model = VGGT()

    model_file = model_path
    if os.path.isdir(model_file):
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

    config = FlatQuantConfig(w_bits=w_bits, a_bits=a_bits)
    fq_args = config.to_args()

    apply_flatquant_to_vggt(fq_args, model)
    reparameterize_vggt(model)

    # Delete random diag_scale buffers (same fix as in fastmodel.py)
    for name, module in model.named_modules():
        if 'ln_trans' in name and hasattr(module, 'diag_scale'):
            delattr(module, 'diag_scale')

    # Convert transform matrices and aggregator LayerNorms to FP16
    for name, module in model.named_modules():
        if hasattr(module, 'matrix_left'):
            module.matrix_left.data = module.matrix_left.data.to(torch.float16)
        if hasattr(module, 'matrix_right'):
            module.matrix_right.data = module.matrix_right.data.to(torch.float16)
        if hasattr(module, 'matrix_left_inv'):
            module.matrix_left_inv.data = module.matrix_left_inv.data.to(torch.float16)
        if hasattr(module, 'matrix_right_inv'):
            module.matrix_right_inv.data = module.matrix_right_inv.data.to(torch.float16)
        if hasattr(module, 'matrix') and isinstance(module.matrix, torch.nn.Parameter):
            module.matrix.data = module.matrix.data.to(torch.float16)
        if hasattr(module, 'matrix_inv_t'):
            module.matrix_inv_t.data = module.matrix_inv_t.data.to(torch.float16)
        if isinstance(module, torch.nn.LayerNorm):
            target_dtype = torch.float16 if 'aggregator' in name else torch.float32
            if module.weight is not None:
                module.weight.data = module.weight.data.to(target_dtype)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(target_dtype)

    # Disable all activation quantizers
    set_act_quantizer_state(model, enable=False)

    model.eval()
    model.to(device)
    logger.info(f"Loaded FlatQuant-wrapped VGGT model (W{w_bits}A{a_bits}) from {model_file}")
    return model


# ============================================================================
# Helpers
# ============================================================================

def make_block_labels():
    """Generate labels: F0, G0, F1, G1, ..., F23, G23."""
    labels = []
    for i in range(24):
        labels.append(f"F{i}")
        labels.append(f"G{i}")
    return labels


def get_block(model, block_idx):
    """Map linear block index (0..47) to frame_blocks[i] or global_blocks[i]."""
    pair_idx = block_idx // 2
    if block_idx % 2 == 0:
        return model.aggregator.frame_blocks[pair_idx]
    else:
        return model.aggregator.global_blocks[pair_idx]


def get_flatquant_linears(block):
    """Collect all FlatQuantizedLinear modules in a block.

    Returns: [q_proj, k_proj, v_proj, o_proj, fc1, fc2]
    """
    return [
        block.attn.q_proj,
        block.attn.k_proj,
        block.attn.v_proj,
        block.attn.o_proj,
        block.mlp.fc1,
        block.mlp.fc2,
    ]


# ============================================================================
# FP reference computation
# ============================================================================

@torch.no_grad()
def compute_fp_references(model, calibration_data, device):
    """Compute FP reference predictions and block outputs.

    Returns:
        fp_preds: dict mapping task_key -> list of per-sample prediction tensors (on CPU)
        fp_block_outs: dict mapping block_idx -> list of per-sample block output tensors (on CPU)
    """
    fp_preds = {k: [] for k in TASK_KEYS}
    fp_block_outs = {i: [] for i in range(NUM_BLOCKS)}

    # Register hooks to capture block outputs
    hook_outputs = {}

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            hook_outputs[block_idx] = output.detach().float()
        return hook_fn

    hooks = []
    for i in range(24):
        hooks.append(
            model.aggregator.frame_blocks[i].register_forward_hook(make_hook(2 * i))
        )
        hooks.append(
            model.aggregator.global_blocks[i].register_forward_hook(make_hook(2 * i + 1))
        )

    for sample_idx, sample in enumerate(calibration_data):
        logger.info(f"FP reference: sample {sample_idx + 1}/{len(calibration_data)}")
        hook_outputs.clear()
        images = sample.unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)

        for key in TASK_KEYS:
            fp_preds[key].append(predictions[key].float().cpu())

        for block_idx in range(NUM_BLOCKS):
            if block_idx in hook_outputs:
                fp_block_outs[block_idx].append(hook_outputs[block_idx].cpu())

        del predictions, images
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    return fp_preds, fp_block_outs


# ============================================================================
# Per-block quantization
# ============================================================================

@torch.no_grad()
def quantize_block(block, w_bits=4):
    """Apply RTN W4 weight quantization and enable activation quantizer on a block.

    Returns saved_weights dict for restore.
    """
    linears = get_flatquant_linears(block)

    # Save original FP weights
    saved_weights = {}
    for mod in linears:
        saved_weights[id(mod)] = mod.linear.weight.data.clone()

    # Apply RTN weight quantization
    quantizer = WeightQuantizer()
    quantizer.configure(w_bits, perchannel=True, sym=True, mse=False)
    for mod in linears:
        W = mod.linear.weight.data
        w_dtype = W.dtype
        quantizer.find_params(W)
        mod.linear.weight.data = quantizer.quantize(W).to(w_dtype)

    # Enable activation quantizer
    for mod in linears:
        mod.act_quantizer.enable = True

    return saved_weights


@torch.no_grad()
def restore_block(block, saved_weights):
    """Restore FP weights and disable activation quantizer."""
    for mod in get_flatquant_linears(block):
        mod.linear.weight.data = saved_weights[id(mod)]
        mod.act_quantizer.enable = False


def compute_quant_loss_for_block(
    model, block_idx, calibration_data, fp_preds, fp_block_outs, device, w_bits=4,
):
    """Quantize one block, run full model, compute per-task loss and per-task per-channel loss.

    Per-task per-channel loss is computed via first-order Taylor attribution:
        channel_loss[k, c] = mean_samples( grad_k[c] * delta[c] )
    where grad_k = d(task_k_loss.sum())/d(block_output) and delta = quant_out - fp_out.

    Returns:
        task_mse: [3] tensor -- per-task MSE (camera, depth, points)
        channel_loss: [3, 1024] tensor -- per-task per-channel attributed loss at block output
    """
    block = get_block(model, block_idx)

    with torch.no_grad():
        saved_weights = quantize_block(block, w_bits=w_bits)

    task_mse_acc = torch.zeros(3, dtype=torch.float64)
    channel_loss_acc = torch.zeros(3, EMBED_DIM, dtype=torch.float64)

    # Hook to capture this block's output with gradient tracking
    captured_output = {}

    def capture_hook(module, input, output):
        output.retain_grad()
        captured_output["out"] = output

    handle = block.register_forward_hook(capture_hook)

    nsamples = len(calibration_data)
    for sample_idx in range(nsamples):
        images = calibration_data[sample_idx].unsqueeze(0).to(device)
        captured_output.clear()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)

        # Per-task MSE (no grad needed)
        with torch.no_grad():
            for task_idx, key in enumerate(TASK_KEYS):
                q_val = predictions[key].float().cpu()
                fp_val = fp_preds[key][sample_idx]
                mse = (q_val - fp_val).pow(2).mean().item()
                task_mse_acc[task_idx] += mse

        # Per-task per-channel attributed loss via gradients
        if "out" in captured_output:
            q_out = captured_output["out"]  # has grad, on GPU
            with torch.no_grad():
                fp_out = fp_block_outs[block_idx][sample_idx].to(q_out.device).to(q_out.dtype)
                delta = (q_out.detach() - fp_out)  # channel perturbation

            for task_idx, key in enumerate(TASK_KEYS):
                # Zero any existing grads
                if q_out.grad is not None:
                    q_out.grad.zero_()

                task_loss = predictions[key].sum()
                task_loss.backward(retain_graph=(task_idx < len(TASK_KEYS) - 1))

                if q_out.grad is not None:
                    grad = q_out.grad.float()
                    # Attribution: grad * delta, averaged over all dims except channel
                    contrib = (grad * delta.float())
                    with torch.no_grad():
                        channel_loss_acc[task_idx] += contrib.mean(
                            dim=tuple(range(contrib.ndim - 1))
                        ).cpu().double()

        del predictions, images
        torch.cuda.empty_cache()

    handle.remove()
    with torch.no_grad():
        restore_block(block, saved_weights)

    task_mse = (task_mse_acc / nsamples).float()
    channel_loss = (channel_loss_acc / nsamples).float()

    return task_mse, channel_loss


def compute_quant_loss(model, calibration_data, device, block_range=None, w_bits=4):
    """Main computation loop over specified blocks.

    Args:
        block_range: tuple (start, end) or None for all blocks

    Returns:
        block_task_loss: [3, 48] tensor (per-task MSE per block)
        channel_loss: [3, 48, 1024] tensor (per-task per-channel attributed loss)
    """
    # Phase 1: FP reference
    logger.info("Computing FP reference predictions...")
    fp_preds, fp_block_outs = compute_fp_references(model, calibration_data, device)

    # Phase 2: Per-block quantization
    block_task_loss = torch.zeros(3, NUM_BLOCKS)
    channel_loss = torch.zeros(3, NUM_BLOCKS, EMBED_DIM)

    # Determine which blocks to compute
    if block_range is None:
        block_indices = range(NUM_BLOCKS)
    else:
        block_start, block_end = block_range
        block_indices = range(block_start, block_end)

    block_labels = make_block_labels()
    logger.info(f"Computing quantization loss for blocks {list(block_indices)[0]}-{list(block_indices)[-1]} ({len(block_indices)} blocks)")

    for block_idx in block_indices:
        label = block_labels[block_idx]
        logger.info(f"Quantizing block {block_idx} ({label})...")

        task_mse, ch_loss = compute_quant_loss_for_block(
            model, block_idx, calibration_data, fp_preds, fp_block_outs, device, w_bits=w_bits,
        )

        block_task_loss[:, block_idx] = task_mse
        channel_loss[:, block_idx] = ch_loss

        logger.info(
            f"  {label}: camera={task_mse[0]:.6e}, depth={task_mse[1]:.6e}, "
            f"points={task_mse[2]:.6e}"
        )

    return block_task_loss, channel_loss


# ============================================================================
# Visualization
# ============================================================================

def _safe_log10(arr, eps=1e-30):
    """log10(|arr|), masking entries that are exactly zero so they render as NaN.

    Channel-attribution values can be negative (first-order Taylor: grad*delta);
    uncomputed blocks are exactly zero and should be displayed as missing rather
    than as the colorbar minimum.
    """
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    nonzero = arr != 0
    out[nonzero] = np.log10(np.abs(arr[nonzero]) + eps)
    return out


def _shrunk_range(data, factor=RANGE_FACTOR):
    """Return (vmin, vmax) covering `factor` of the finite data range, centered.

    Shrinking the range increases contrast: values near the extremes are clipped
    to the colormap endpoints while differences in the bulk become more visible.
    """
    finite = data[np.isfinite(data)] if hasattr(data, "__iter__") else data
    if not hasattr(finite, "size") or finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    mid = (lo + hi) / 2
    half = (hi - lo) / 2 * factor
    return mid - half, mid + half


def plot_quant_loss_heatmap(block_task_loss, block_labels, save_path):
    """Plot 1: Quantization loss heatmap -- 3 rows (tasks) x 48 cols (blocks)."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 6), sharex=True)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    for task_idx, (ax, task_name) in enumerate(zip(axes, TASK_NAMES)):
        data = block_task_loss[task_idx].numpy()
        log_data = _safe_log10(data[None, :])
        vmin, vmax = _shrunk_range(log_data)
        im = ax.imshow(
            log_data,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin, vmax=vmax,
        )
        ax.set_ylabel(task_name, fontsize=12)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="log10(quant loss)")

    axes[-1].set_xticks(range(NUM_BLOCKS))
    axes[-1].set_xticklabels(block_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel("Block")
    fig.suptitle("W4A4 Quantization Loss per Task and Block", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_quant_loss_bar_chart(block_task_loss, block_labels, save_path):
    """Plot 2: Grouped bar chart -- one bar per task per block."""
    fig, ax = plt.subplots(figsize=(20, 6))

    x = np.arange(NUM_BLOCKS)
    width = 0.25

    for task_idx, task_name in enumerate(TASK_NAMES):
        data = block_task_loss[task_idx].numpy()
        ax.bar(x + task_idx * width, data, width, label=task_name)

    ax.set_yscale("log")
    ax.set_xlabel("Block")
    ax.set_ylabel("Quantization Loss (MSE, log scale)")
    ax.set_title("Per-Block W4A4 Quantization Loss by Task")
    ax.set_xticks(x + width)
    ax.set_xticklabels(block_labels, rotation=90, fontsize=6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_channel_loss_heatmap(channel_loss, block_labels, save_path):
    """Plot 3: Per-channel quantization loss heatmaps [48 x 1024], one per task."""
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    for task_idx, task_name in enumerate(TASK_NAMES):
        fig = plt.figure(figsize=(8, 4.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.03], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        data = _safe_log10(channel_loss[task_idx].numpy())
        vmin, vmax = _shrunk_range(data)
        im = ax.imshow(
            data, aspect="auto", cmap=cmap, interpolation="nearest",
            vmin=vmin, vmax=vmax,
        )
        ax.set_ylabel("Block", fontsize=14)
        ax.set_xlabel("Channel", fontsize=14)

        labels = [block_labels[i] if i % 2 == 0 else "" for i in range(NUM_BLOCKS)]
        ax.set_yticks(range(NUM_BLOCKS))
        ax.set_yticklabels(labels, fontsize=6)

        tick_pos = list(range(0, EMBED_DIM, 128))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_pos, fontsize=10)

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$\log_{10}$(attributed loss)", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        fig.suptitle(f"Per-Channel Quantization Loss — {task_name.capitalize()}", fontsize=15)
        task_save_path = save_path.replace(".pdf", f"_{task_name}.pdf")
        fig.savefig(task_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {task_save_path}")


def plot_channel_loss_distributions(channel_loss, block_labels, save_path):
    """Plot 4: Histograms of per-channel quant loss for selected blocks, per task."""
    selected = [0, 1, 22, 23, 44, 45, 46, 47]

    fig, axes = plt.subplots(
        len(selected), 3, figsize=(15, 3 * len(selected))
    )

    for row, block_idx in enumerate(selected):
        for col, task_name in enumerate(TASK_NAMES):
            ax = axes[row, col]
            data = channel_loss[col, block_idx].numpy()
            data_nz = np.abs(data[data != 0])
            if len(data_nz) > 0:
                ax.hist(np.log10(data_nz), bins=50, alpha=0.7)
            ax.set_title(
                f"{block_labels[block_idx]} -- {task_name}", fontsize=9
            )
            if row == len(selected) - 1:
                ax.set_xlabel("log10(attributed loss)")
            if col == 0:
                ax.set_ylabel("Count")

    fig.suptitle("Per-Channel Quantization Loss Distributions (by Task)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def select_typical_block(block_task_loss, block_labels):
    """Select the block whose total task loss is closest to the median (log scale)."""
    total = block_task_loss.sum(dim=0)  # [48]
    log_total = torch.log10(total + 1e-30)
    # Only consider blocks that have been computed (nonzero)
    nonzero_mask = total > 0
    if nonzero_mask.sum() == 0:
        return 0
    median_val = log_total[nonzero_mask].median()
    idx = (log_total[nonzero_mask] - median_val).abs().argmin().item()
    # Map back to original index
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
    idx = nonzero_indices[idx].item()
    logger.info(
        f"Selected typical block: {block_labels[idx]} (index {idx}, "
        f"log10(total)={log_total[idx]:.2f}, median={median_val:.2f})"
    )
    return idx


def _compute_common_span(block_task_loss, channel_loss, block_idx):
    """Compute max data span across all task rows in both figures."""
    max_span = 0.0
    for task_idx in range(3):
        # Per-block: task loss
        block_data = _safe_log10(block_task_loss[task_idx].numpy())
        finite = np.isfinite(block_data)
        if finite.any():
            max_span = max(max_span, float(block_data[finite].max() - block_data[finite].min()))
        # Per-channel: channel loss for the selected block (per-task)
        ch_data = _safe_log10(channel_loss[task_idx, block_idx].numpy())
        finite = np.isfinite(ch_data)
        if finite.any():
            max_span = max(max_span, float(ch_data[finite].max() - ch_data[finite].min()))
    return max_span


def plot_block_sensitivity_subfig(block_task_loss, block_labels, save_path, span):
    """Per-block quantization loss heatmap, compact for subfigure use.

    3 rows (tasks) x 48 columns (blocks), per-row colorbars with same span.
    """
    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1, 0.03], hspace=0.7, wspace=0.05,
    )
    axes = []

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    for task_idx, task_name in enumerate(TASK_NAMES):
        ax = fig.add_subplot(gs[task_idx, 0])
        cax = fig.add_subplot(gs[task_idx, 1])
        axes.append(ax)

        data = _safe_log10(block_task_loss[task_idx].numpy()[None, :])
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            mid = (finite.min() + finite.max()) / 2
        else:
            mid = 0.0
        half = span * RANGE_FACTOR / 2
        norm = matplotlib.colors.Normalize(
            vmin=mid - half, vmax=mid + half,
        )
        im = ax.imshow(
            data, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest",
        )
        ax.set_ylabel(task_name.capitalize(), fontsize=14, labelpad=10)
        ax.set_yticks([])
        if task_idx < 2:
            ax.set_xticks([])

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)

    # Show frame-block labels (even indices) for readability
    axes[-1].set_xticks(range(NUM_BLOCKS))
    labels = [block_labels[i] if i % 2 == 0 else "" for i in range(NUM_BLOCKS)]
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=9)
    axes[-1].set_xlabel("Block", fontsize=14)

    fig.suptitle("Per-Block Quantization Loss by Task", fontsize=15)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_channel_sensitivity_subfig(
    channel_loss, block_labels, block_idx, save_path, span,
):
    """Per-channel quantization loss for one block, per task, compact for subfigure.

    3 rows (tasks) x 1024 columns (channels), per-row colorbars with same span.
    """
    block_name = block_labels[block_idx]

    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1, 0.03], hspace=0.7, wspace=0.05,
    )
    axes = []

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    for task_idx, task_name in enumerate(TASK_NAMES):
        ax = fig.add_subplot(gs[task_idx, 0])
        cax = fig.add_subplot(gs[task_idx, 1])
        axes.append(ax)

        data = _safe_log10(channel_loss[task_idx, block_idx].numpy()[None, :])
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            mid = (finite.min() + finite.max()) / 2
        else:
            mid = 0.0
        half = span * RANGE_FACTOR / 2
        norm = matplotlib.colors.Normalize(
            vmin=mid - half, vmax=mid + half,
        )
        im = ax.imshow(
            data, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest",
        )
        ax.set_ylabel(task_name.capitalize(), fontsize=14, labelpad=10)
        ax.set_yticks([])
        if task_idx < 2:
            ax.set_xticks([])

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)

    tick_pos = list(range(0, EMBED_DIM, 128))
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(tick_pos, fontsize=10)
    axes[-1].set_xlabel("Channel", fontsize=14)

    fig.suptitle(
        f"Per-Channel Quantization Loss (Block {block_name})",
        fontsize=15,
    )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_block_channel_heatmap(channel_loss, block_labels, save_dir, span):
    """2D block x channel-group heatmaps.

    Generates one PDF per task.  X-axis: 26 blocks (F11-G23), Y-axis: 26 channel groups.
    Same colorbar span as other subfigures.
    """
    block_start, block_end = 22, 48
    n_blocks = block_end - block_start  # 26
    selected_labels = block_labels[block_start:block_end]

    # Bin 1024 channels into n_blocks groups to make a square heatmap
    n_groups = n_blocks
    channel_indices = np.array_split(np.arange(EMBED_DIM), n_groups)

    for task_idx, task_name in enumerate(TASK_NAMES):
        fig = plt.figure(figsize=(5, 4.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        # Build 2D matrix: [n_groups, n_blocks]
        # Use |loss| before averaging to avoid sign cancellation in Taylor attribution
        data_2d = np.zeros((n_groups, n_blocks))
        for g, ch_idx in enumerate(channel_indices):
            block_ch_loss = channel_loss[task_idx, block_start:block_end, :].abs()  # [26, 1024]
            data_2d[g] = block_ch_loss[:, ch_idx].mean(dim=1).numpy()

        log_data = _safe_log10(data_2d)
        finite = log_data[np.isfinite(log_data)]
        if finite.size > 0:
            mid = (finite.min() + finite.max()) / 2
        else:
            mid = 0.0
        half = span * RANGE_FACTOR / 2
        norm = matplotlib.colors.Normalize(
            vmin=mid - half, vmax=mid + half,
        )

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(
            log_data, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest", origin="lower",
        )

        # X-axis: show frame-block labels (even positions)
        ax.set_xticks(range(n_blocks))
        xlabels = [
            selected_labels[i] if i % 2 == 0 else ""
            for i in range(n_blocks)
        ]
        ax.set_xticklabels(xlabels, rotation=90, fontsize=9)
        ax.set_xlabel("Block", fontsize=14)

        # Y-axis: channel group index
        ax.set_yticks(range(0, n_groups, 5))
        ax.set_yticklabels(range(0, n_groups, 5), fontsize=10)
        ax.set_ylabel("Channel Group", fontsize=14)

        ax.set_title(
            f"Block x Channel Loss — {task_name.capitalize()}",
            fontsize=14,
        )

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$\log_{10}$(attributed loss)", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        out_path = os.path.join(save_dir, f"block_channel_loss_{task_name}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-block and per-channel quantization loss for VGGT"
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to VGGT model checkpoint directory or file")
    parser.add_argument("--save_dir", type=str, default="outputs/quant_loss",
                        help="Directory for output files")
    parser.add_argument("--nsamples", type=int, default=8,
                        help="Number of calibration samples")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Frames per sample")
    parser.add_argument("--img_size", type=int, default=518,
                        help="Image resize dimension")
    parser.add_argument("--w_bits", type=int, default=4,
                        help="Weight quantization bits")
    parser.add_argument("--a_bits", type=int, default=4,
                        help="Activation quantization bits")
    parser.add_argument("--max_blocks", type=int, default=NUM_BLOCKS,
                        help="Max blocks to quantize (for quick testing)")
    parser.add_argument("--block_range", type=str, default="22-48",
                        help="Block range to compute (e.g., '22-48' for F11-G23, or 'all' for all blocks)")
    parser.add_argument("--cali_dataset", type=str, default="co3d",
                        choices=["co3d", "kitti"],
                        help="Calibration dataset (co3d or kitti)")
    parser.add_argument("--cali_data_dir", type=str, default=None,
                        help="Calibration data directory (auto-set per dataset if omitted)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip computation; load existing quant_loss.pt and regenerate plots")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if not args.plot_only and args.model_path is None:
        raise ValueError("--model_path is required when not using --plot_only")

    if args.cali_data_dir is None:
        args.cali_data_dir = CALI_DATASET_DEFAULTS[args.cali_dataset]

    os.makedirs(args.save_dir, exist_ok=True)
    block_labels = make_block_labels()

    if args.plot_only:
        data_path = os.path.join(args.save_dir, "quant_loss.pt")
        data = torch.load(data_path, map_location="cpu", weights_only=True)
        block_task_loss = data["block_task_loss"]
        channel_loss = data["channel_loss"]
        block_labels = data.get("block_labels", block_labels)
        logger.info(f"Loaded quantization loss data from {data_path}")
    else:
        # Load model
        logger.info(f"Loading VGGT model with FlatQuant (W{args.w_bits}A{args.a_bits}) from {args.model_path}")
        model = load_model_flatquant(args.model_path, device, w_bits=args.w_bits, a_bits=args.a_bits)

        # Load calibration data
        logger.info(f"Loading calibration data ({args.cali_dataset})...")
        if args.cali_dataset == "kitti":
            calibration_data = load_calibration_images_kitti(
                args.cali_data_dir, args.nsamples, args.num_frames, args.img_size,
            )
        else:
            config = FlatQuantConfig(
                nsamples=args.nsamples,
                num_frames=args.num_frames,
                img_size=args.img_size,
                cali_data_dir=args.cali_data_dir,
            )
            calibration_data = load_calibration_images(config)
        logger.info(f"Loaded {len(calibration_data)} samples")

        # Parse block range
        if args.block_range.lower() == "all":
            block_range = None
            logger.info("Computing all 48 blocks")
        else:
            start, end = map(int, args.block_range.split("-"))
            block_range = (start, end)
            logger.info(f"Computing blocks {start}-{end-1} ({end-start} blocks)")

        # Compute quantization loss
        logger.info("Computing quantization loss...")
        block_task_loss, channel_loss = compute_quant_loss(
            model, calibration_data, device,
            block_range=block_range, w_bits=args.w_bits,
        )

        # Save
        save_path = os.path.join(args.save_dir, "quant_loss.pt")
        torch.save(
            {
                "block_task_loss": block_task_loss,
                "channel_loss": channel_loss,
                "task_names": TASK_NAMES,
                "block_labels": block_labels,
                "nsamples": len(calibration_data),
                "w_bits": args.w_bits,
                "a_bits": args.a_bits,
            },
            save_path,
        )
        logger.info(f"Saved {save_path}")

        # Cleanup
        del model, calibration_data
        torch.cuda.empty_cache()

    # --- Generate plots ---
    logger.info("Generating visualizations...")
    n_computed = (block_task_loss.sum(dim=0) > 0).sum().item()
    logger.info(f"Blocks with data: {n_computed}/{NUM_BLOCKS}")

    plot_quant_loss_heatmap(
        block_task_loss, block_labels,
        os.path.join(args.save_dir, "quant_loss_heatmap.pdf"),
    )
    plot_quant_loss_bar_chart(
        block_task_loss, block_labels,
        os.path.join(args.save_dir, "quant_loss_bar_chart.pdf"),
    )
    plot_channel_loss_heatmap(
        channel_loss, block_labels,
        os.path.join(args.save_dir, "channel_loss_heatmap.pdf"),
    )
    plot_channel_loss_distributions(
        channel_loss, block_labels,
        os.path.join(args.save_dir, "channel_loss_distributions.pdf"),
    )

    # --- Subfigure heatmaps ---
    block_idx = select_typical_block(block_task_loss, block_labels)
    span = _compute_common_span(block_task_loss, channel_loss, block_idx)
    logger.info(f"Common colorbar span: {span:.2f} log10 units")

    plot_block_sensitivity_subfig(
        block_task_loss, block_labels,
        os.path.join(args.save_dir, "block_sensitivity.pdf"),
        span,
    )
    block_name = block_labels[block_idx]
    plot_channel_sensitivity_subfig(
        channel_loss, block_labels, block_idx,
        os.path.join(args.save_dir, f"channel_sensitivity_{block_name}.pdf"),
        span,
    )
    plot_block_channel_heatmap(
        channel_loss, block_labels, args.save_dir, span,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
