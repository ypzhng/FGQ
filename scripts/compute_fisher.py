#!/usr/bin/env python
"""
Compute diagonal Fisher information for VGGT task sensitivity analysis.

Measures how sensitive each of the 48 aggregator blocks is to perturbations,
broken down by task (camera, depth, points) and channel. Also collects activation
statistics and estimates Fisher-weighted quantization error across bitwidth configs.

Usage:
    conda activate flatquant_vggt
    cd /data/projects/zyp/recons_eval
    python scripts/compute_fisher.py \
        --model_path /data1/vggt/model \
        --nsamples 32 --num_frames 4 --img_size 518 \
        --save_dir outputs/fisher
"""
import os
import sys
import random
import argparse
import logging

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from models.vggt.models.vggt import VGGT
from models.quantization.apply_flatquant import load_calibration_images
from models.quantization.config import FlatQuantConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger("fisher")

TASK_NAMES = ["camera", "depth", "points"]
NUM_BLOCKS = 48
EMBED_DIM = 1024

CALI_DATASET_DEFAULTS = {
    "co3d": "data/co3dv2/data",
    "kitti": "data/kitti/depth_selection/val_selection_cropped/image_gathered",
}


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Fisher information for VGGT aggregator blocks"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VGGT model checkpoint directory or file")
    parser.add_argument("--save_dir", type=str, default="outputs/fisher",
                        help="Directory for output files")
    parser.add_argument("--nsamples", type=int, default=32,
                        help="Number of calibration samples")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Frames per sample")
    parser.add_argument("--img_size", type=int, default=518,
                        help="Image resize dimension")
    parser.add_argument("--cali_dataset", type=str, default="co3d",
                        choices=["co3d", "kitti"],
                        help="Calibration dataset (co3d or kitti)")
    parser.add_argument("--cali_data_dir", type=str, default=None,
                        help="Calibration data directory (auto-set per dataset if omitted)")
    parser.add_argument("--cali_annotation_dir", type=str, default="data/co3dv2/annotations",
                        help="Deprecated for VGGT Fisher; random raw CO3D sampling does not use annotations")
    parser.add_argument("--cali_split", type=str, default="train", choices=["train", "test"],
                        help="Deprecated for VGGT Fisher; kept for CLI compatibility")
    parser.add_argument("--cali_seed", type=int, default=42,
                        help="Calibration sampling seed")
    parser.add_argument("--min_num_images", type=int, default=50,
                        help="Deprecated for VGGT Fisher; random raw sampling only requires --num_frames")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip computation; load existing fisher.pt and regenerate plots")
    return parser.parse_args()


def load_model(model_path, device):
    """Load FP VGGT model."""
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

    model.eval()
    model.to(device)
    return model


def make_block_labels():
    """Generate labels: F0, G0, F1, G1, ..., F23, G23."""
    labels = []
    for i in range(24):
        labels.append(f"F{i}")
        labels.append(f"G{i}")
    return labels


def compute_fisher(model, calibration_data, device):
    """
    Compute diagonal Fisher information and activation statistics.

    For each task k, block l, channel c:
        fisher[k][l][c] = (1/N) * sum_n sum_{b,s} (dL_k / dh_{l,b,s,c})^2

    Returns:
        fisher: [3, 48, 1024] tensor
        act_stats: dict with mean, variance, abs_max [48, 1024] and total_tokens [48]
    """
    aggregator = model.aggregator

    # Storage for hook outputs (block_idx -> output tensor)
    block_outputs = {}

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            output.retain_grad()
            block_outputs[block_idx] = output
        return hook_fn

    # Register hooks: frame_blocks[i] -> block 2*i, global_blocks[i] -> block 2*i+1
    hooks = []
    for i in range(24):
        hooks.append(
            aggregator.frame_blocks[i].register_forward_hook(make_hook(2 * i))
        )
        hooks.append(
            aggregator.global_blocks[i].register_forward_hook(make_hook(2 * i + 1))
        )

    # Accumulators
    fisher = torch.zeros(3, NUM_BLOCKS, EMBED_DIM, dtype=torch.float32)
    act_mean_acc = torch.zeros(NUM_BLOCKS, EMBED_DIM, dtype=torch.float64)
    act_var_acc = torch.zeros(NUM_BLOCKS, EMBED_DIM, dtype=torch.float64)
    act_abs_max = torch.zeros(NUM_BLOCKS, EMBED_DIM, dtype=torch.float32)
    total_tokens = torch.zeros(NUM_BLOCKS, dtype=torch.int64)

    nsamples = len(calibration_data)

    for sample_idx in range(nsamples):
        logger.info(f"Processing sample {sample_idx + 1}/{nsamples}")
        block_outputs.clear()

        images = calibration_data[sample_idx].unsqueeze(0).to(device)  # [1, S, 3, H, W]

        # Forward pass with gradients enabled, bfloat16 autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)

        # --- Collect activation statistics (before backward destroys anything) ---
        for block_idx in range(NUM_BLOCKS):
            if block_idx not in block_outputs:
                continue
            out = block_outputs[block_idx].detach().float()  # [tokens_dim0, tokens_dim1, C]
            n_tokens = out.shape[0] * out.shape[1]
            act_mean_acc[block_idx] += out.mean(dim=(0, 1)).cpu().double()
            act_var_acc[block_idx] += out.var(dim=(0, 1)).cpu().double()
            act_abs_max[block_idx] = torch.maximum(
                act_abs_max[block_idx], out.abs().amax(dim=(0, 1)).cpu()
            )
            total_tokens[block_idx] += n_tokens

        # --- Multi-backward: one per task ---
        task_outputs = [
            predictions["pose_enc"],       # [B, S, 9]
            predictions["depth"],          # [B, S, H, W, 1]
            predictions["world_points"],   # [B, S, H, W, 3]
        ]

        for task_idx, task_out in enumerate(task_outputs):
            is_last = task_idx == len(task_outputs) - 1

            # Zero block output grads before each backward
            for block_idx in range(NUM_BLOCKS):
                if block_idx in block_outputs and block_outputs[block_idx].grad is not None:
                    block_outputs[block_idx].grad.zero_()

            loss = task_out.sum()
            loss.backward(retain_graph=not is_last)

            # Collect squared gradients -> Fisher
            for block_idx in range(NUM_BLOCKS):
                if block_idx not in block_outputs:
                    continue
                output = block_outputs[block_idx]
                if output.grad is None:
                    continue
                grad = output.grad.float()  # [tokens..., C]
                # Sum squared grad over all dims except last (channel)
                grad_sq_sum = (grad ** 2).sum(
                    dim=tuple(range(grad.ndim - 1))
                )  # [C]
                fisher[task_idx, block_idx] += grad_sq_sum.cpu()

        # Cleanup
        del predictions, task_outputs, images, loss
        block_outputs.clear()
        torch.cuda.empty_cache()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average over samples
    fisher /= nsamples
    act_mean_acc /= nsamples
    act_var_acc /= nsamples

    act_stats = {
        "mean": act_mean_acc.float(),
        "variance": act_var_acc.float(),
        "abs_max": act_abs_max,
        "total_tokens": total_tokens,
    }

    return fisher, act_stats


def compute_quantization_impact(fisher, act_stats):
    """
    Estimate Fisher-weighted quantization error for different bitwidth configs.

    Uses uniform quantization noise model:
        step_size = 2 * abs_max / 2^bits
        quant_error_var = step_size^2 / 12

    Returns:
        impact: dict mapping config_name -> [3] tensor (per-task weighted error)
    """
    abs_max = act_stats["abs_max"]  # [48, 1024]

    configs = {
        "W4A4": (4, 4),
        "W4A8": (4, 8),
        "W8A8": (8, 8),
        "W8A16": (8, 16),
    }

    impact = {}
    for name, (w_bits, a_bits) in configs.items():
        # Activation quantization error (uniform noise model)
        a_step = 2 * abs_max / (2 ** a_bits)
        a_quant_error_var = a_step ** 2 / 12  # [48, 1024]

        # Fisher-weighted error per task
        task_impact = torch.zeros(3)
        for task_idx in range(3):
            task_impact[task_idx] = (fisher[task_idx] * a_quant_error_var).sum()

        impact[name] = task_impact

    return impact


# ============================================================================
# Visualization
# ============================================================================

def plot_fisher_heatmap(fisher, block_labels, save_path):
    """Plot 1: Fisher heatmap -- 3 rows (tasks) x 48 cols (blocks)."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 6), sharex=True)

    for task_idx, (ax, task_name) in enumerate(zip(axes, TASK_NAMES)):
        # Mean Fisher over channels per block
        block_fisher = fisher[task_idx].mean(dim=1).numpy()  # [48]
        im = ax.imshow(
            np.log10(block_fisher[None, :] + 1e-30),
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_ylabel(task_name, fontsize=12)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="log10(mean Fisher)")

    axes[-1].set_xticks(range(NUM_BLOCKS))
    axes[-1].set_xticklabels(block_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel("Block")
    fig.suptitle("Fisher Information per Task and Block", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_fisher_bar_chart(fisher, block_labels, save_path):
    """Plot 2: Grouped bar chart -- one bar per task per block."""
    fig, ax = plt.subplots(figsize=(20, 6))

    x = np.arange(NUM_BLOCKS)
    width = 0.25

    for task_idx, task_name in enumerate(TASK_NAMES):
        block_fisher = fisher[task_idx].mean(dim=1).numpy()  # [48]
        ax.bar(x + task_idx * width, block_fisher, width, label=task_name)

    ax.set_yscale("log")
    ax.set_xlabel("Block")
    ax.set_ylabel("Mean Fisher (log scale)")
    ax.set_title("Per-Block Fisher Information by Task")
    ax.set_xticks(x + width)
    ax.set_xticklabels(block_labels, rotation=90, fontsize=6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_quantization_impact(impact, save_path):
    """Plot 3: Grouped bars -- 3 tasks x 4 bitwidth configs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    config_names = list(impact.keys())
    x = np.arange(len(config_names))
    width = 0.25

    for task_idx, task_name in enumerate(TASK_NAMES):
        values = [impact[cfg][task_idx].item() for cfg in config_names]
        ax.bar(x + task_idx * width, values, width, label=task_name)

    ax.set_yscale("log")
    ax.set_xlabel("Quantization Config")
    ax.set_ylabel("Fisher-Weighted Quant Error (log scale)")
    ax.set_title("Estimated Quantization Impact by Task and Bitwidth")
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_channel_distributions(fisher, block_labels, save_path):
    """Plot 4: Histograms of per-channel Fisher for selected blocks."""
    # Early, middle, late blocks from both frame and global
    selected = [0, 1, 22, 23, 44, 45, 46, 47]

    fig, axes = plt.subplots(
        len(selected), 3, figsize=(15, 3 * len(selected))
    )

    for row, block_idx in enumerate(selected):
        for col, task_name in enumerate(TASK_NAMES):
            ax = axes[row, col]
            data = fisher[col, block_idx].numpy()
            data_pos = data[data > 0]
            if len(data_pos) > 0:
                ax.hist(np.log10(data_pos), bins=50, alpha=0.7)
            ax.set_title(
                f"{block_labels[block_idx]} -- {task_name}", fontsize=9
            )
            if row == len(selected) - 1:
                ax.set_xlabel("log10(Fisher)")
            if col == 0:
                ax.set_ylabel("Count")

    fig.suptitle("Per-Channel Fisher Distributions", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def select_typical_block(fisher, block_labels):
    """Select the block whose total Fisher is closest to the median (log scale)."""
    total = fisher.sum(dim=0).sum(dim=1)  # [48]
    log_total = torch.log10(total + 1e-30)
    median_val = log_total.median()
    idx = (log_total - median_val).abs().argmin().item()
    logger.info(
        f"Selected typical block: {block_labels[idx]} (index {idx}, "
        f"log10(total)={log_total[idx]:.2f}, median={median_val:.2f})"
    )
    return idx


def _compute_common_span(fisher, block_idx):
    """Compute max data span across all task rows in both figures.

    Each task row gets its own colorbar center, but they all share the same
    span (number of log10 units) so color differences are comparable.
    """
    max_span = 0.0
    for task_idx in range(3):
        # Per-block: mean Fisher over channels
        block_data = np.log10(fisher[task_idx].mean(dim=1).numpy() + 1e-30)
        max_span = max(max_span, float(block_data.max() - block_data.min()))
        # Per-channel: Fisher for the selected block
        ch_data = np.log10(fisher[task_idx, block_idx].numpy() + 1e-30)
        max_span = max(max_span, float(ch_data.max() - ch_data.min()))
    return max_span


def plot_block_sensitivity_subfig(fisher, block_labels, save_path, span):
    """Per-block task sensitivity heatmap, compact for subfigure use.

    3 rows (tasks) x 48 columns (blocks), per-row colorbars with same span.
    """
    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1, 0.03], hspace=0.7, wspace=0.05,
    )
    axes = []

    for task_idx, task_name in enumerate(TASK_NAMES):
        ax = fig.add_subplot(gs[task_idx, 0])
        cax = fig.add_subplot(gs[task_idx, 1])
        axes.append(ax)

        data = np.log10(fisher[task_idx].mean(dim=1).numpy()[None, :] + 1e-30)
        mid = (data.min() + data.max()) / 2
        norm = matplotlib.colors.Normalize(
            vmin=mid - span / 2, vmax=mid + span / 2,
        )
        im = ax.imshow(
            data, aspect="auto", cmap="viridis", norm=norm,
            interpolation="nearest",
        )
        ax.set_ylabel(task_name.capitalize(), fontsize=14, labelpad=10)
        ax.set_yticks([])
        if task_idx < 2:
            ax.set_xticks([])

        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)

    # Show frame-block labels (even indices) for readability; ticks at all 48
    axes[-1].set_xticks(range(NUM_BLOCKS))
    labels = [block_labels[i] if i % 2 == 0 else "" for i in range(NUM_BLOCKS)]
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=9)
    axes[-1].set_xlabel("Block", fontsize=14)

    fig.suptitle("Per-Block Task Sensitivity", fontsize=15)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_channel_sensitivity_subfig(
    fisher, block_labels, block_idx, save_path, span,
):
    """Per-channel task sensitivity heatmap for one block, compact for subfigure.

    3 rows (tasks) x 1024 columns (channels), per-row colorbars with same span.
    """
    block_name = block_labels[block_idx]

    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1, 0.03], hspace=0.7, wspace=0.05,
    )
    axes = []

    for task_idx, task_name in enumerate(TASK_NAMES):
        ax = fig.add_subplot(gs[task_idx, 0])
        cax = fig.add_subplot(gs[task_idx, 1])
        axes.append(ax)

        data = np.log10(fisher[task_idx, block_idx].numpy()[None, :] + 1e-30)
        mid = (data.min() + data.max()) / 2
        norm = matplotlib.colors.Normalize(
            vmin=mid - span / 2, vmax=mid + span / 2,
        )
        im = ax.imshow(
            data, aspect="auto", cmap="viridis", norm=norm,
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
        f"Per-Channel Task Sensitivity (Block {block_name})",
        fontsize=15,
    )
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_block_channel_heatmap(fisher, block_labels, save_dir, span):
    """2D block × channel-group heatmaps for the second half of the network.

    Generates one PDF per task.  X-axis: 26 blocks (F11–G23), Y-axis: 26
    channel groups (~39 channels each).  Same colorbar span as other subfigures.
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
        data_2d = np.zeros((n_groups, n_blocks))
        for g, ch_idx in enumerate(channel_indices):
            block_fisher = fisher[task_idx, block_start:block_end, :]  # [26, 1024]
            data_2d[g] = block_fisher[:, ch_idx].mean(dim=1).numpy()

        log_data = np.log10(data_2d + 1e-30)
        mid = (log_data.min() + log_data.max()) / 2
        norm = matplotlib.colors.Normalize(
            vmin=mid - span / 2, vmax=mid + span / 2,
        )

        im = ax.imshow(
            log_data, aspect="auto", cmap="viridis", norm=norm,
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
            f"Block × Channel Sensitivity — {task_name.capitalize()}",
            fontsize=14,
        )

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$\log_{10}$(Fisher)", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        out_path = os.path.join(save_dir, f"block_channel_{task_name}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out_path}")

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Auto-set cali_data_dir if not specified
    if args.cali_data_dir is None:
        args.cali_data_dir = CALI_DATASET_DEFAULTS[args.cali_dataset]

    os.makedirs(args.save_dir, exist_ok=True)

    if args.plot_only:
        # Load existing Fisher data -- skip model/calibration
        fisher_path = os.path.join(args.save_dir, "fisher.pt")
        data = torch.load(fisher_path, map_location="cpu", weights_only=True)
        fisher = data["fisher"]
        block_labels = data["block_labels"]
        logger.info(f"Loaded Fisher data from {fisher_path}")
    else:
        # Load model
        logger.info(f"Loading VGGT model from {args.model_path}")
        model = load_model(args.model_path, device)
        logger.info("Model loaded")

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
                cali_dataset=args.cali_dataset,
                cali_data_dir=args.cali_data_dir,
                cali_annotation_dir=args.cali_annotation_dir,
                cali_split=args.cali_split,
                cali_seed=args.cali_seed,
                min_num_images=args.min_num_images,
            )
            calibration_data = load_calibration_images(config)
        logger.info(f"Loaded {len(calibration_data)} samples")

        # Compute Fisher and activation stats
        logger.info("Computing Fisher information...")
        fisher, act_stats = compute_fisher(model, calibration_data, device)

        # Verify
        logger.info(f"Fisher shape: {fisher.shape}")
        logger.info(f"Fisher range: [{fisher.min():.6e}, {fisher.max():.6e}]")
        for task_idx, name in enumerate(TASK_NAMES):
            nonzero = (fisher[task_idx] > 0).sum().item()
            total = fisher[task_idx].numel()
            logger.info(
                f"  {name}: {nonzero}/{total} nonzero channels "
                f"({100 * nonzero / total:.1f}%)"
            )

        # Save tensors
        block_labels = make_block_labels()

        fisher_path = os.path.join(args.save_dir, "fisher.pt")
        torch.save(
            {
                "fisher": fisher,
                "task_names": TASK_NAMES,
                "block_labels": block_labels,
                "nsamples": len(calibration_data),
                "num_frames": args.num_frames,
                "img_size": args.img_size,
                "cali_sampling": "random_raw",
                "cali_dataset": args.cali_dataset,
                "cali_data_dir": args.cali_data_dir,
                "cali_annotation_dir": args.cali_annotation_dir if args.cali_dataset == "co3d" else None,
                "cali_split": args.cali_split if args.cali_dataset == "co3d" else None,
                "cali_seed": args.cali_seed,
                "min_num_images": args.min_num_images if args.cali_dataset == "co3d" else None,
            },
            fisher_path,
        )
        logger.info(f"Saved {fisher_path}")

        stats_path = os.path.join(args.save_dir, "act_stats.pt")
        torch.save(act_stats, stats_path)
        logger.info(f"Saved {stats_path}")

        # Compute quantization impact
        impact = compute_quantization_impact(fisher, act_stats)
        for name, vals in impact.items():
            logger.info(
                f"Quant impact {name}: "
                f"camera={vals[0]:.4e}, depth={vals[1]:.4e}, points={vals[2]:.4e}"
            )

        # Generate original visualizations
        logger.info("Generating visualizations...")
        plot_fisher_heatmap(
            fisher, block_labels,
            os.path.join(args.save_dir, "fisher_heatmap.pdf"),
        )
        plot_fisher_bar_chart(
            fisher, block_labels,
            os.path.join(args.save_dir, "fisher_bar_chart.pdf"),
        )
        plot_quantization_impact(
            impact,
            os.path.join(args.save_dir, "quantization_impact.pdf"),
        )
        plot_channel_distributions(
            fisher, block_labels,
            os.path.join(args.save_dir, "channel_distributions.pdf"),
        )

    # --- Subfigure heatmaps (always generated) ---
    block_idx = select_typical_block(fisher, block_labels)
    span = _compute_common_span(fisher, block_idx)
    logger.info(f"Common colorbar span: {span:.2f} log10 units")

    plot_block_sensitivity_subfig(
        fisher, block_labels,
        os.path.join(args.save_dir, "block_sensitivity.pdf"),
        span,
    )
    block_name = block_labels[block_idx]
    plot_channel_sensitivity_subfig(
        fisher, block_labels, block_idx,
        os.path.join(args.save_dir, f"channel_sensitivity_{block_name}.pdf"),
        span,
    )
    plot_block_channel_heatmap(
        fisher, block_labels, args.save_dir, span,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
