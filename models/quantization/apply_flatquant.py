"""
FlatQuant integration for VGGT model.

Provides FlatQuant wrappers for VGGT's Attention and MLP modules,
calibration loop, and reparameterization functions.
"""
import os
import gc
import time
import math
import functools
import logging
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.trans_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix
from flatquant.trans_utils import InvSingleTransMatrix, InvDecomposeTransMatrix
from flatquant.function_utils import (
    get_init_scale,
    get_decompose_dim,
    set_require_grad_all,
    get_n_set_parameters_byname,
    get_paras_dict_by_name,
)
from flatquant.quant_utils import ActivationQuantizer, set_quantizer_state

logger = logging.getLogger(__name__)


# ============================================================================
# Helper: Split fused QKV linear
# ============================================================================

def split_qkv_linear(fused_qkv: nn.Linear, idx: int, dim: int) -> nn.Linear:
    """
    Extract the idx-th projection (Q=0, K=1, V=2) from a fused QKV linear.

    Args:
        fused_qkv: nn.Linear with weight [3*dim, dim] and optional bias [3*dim]
        idx: 0 for Q, 1 for K, 2 for V
        dim: hidden dimension (1024 for VGGT)

    Returns:
        nn.Linear with weight [dim, dim] and optional bias [dim]
    """
    has_bias = fused_qkv.bias is not None
    linear = nn.Linear(dim, dim, bias=has_bias, device=fused_qkv.weight.device, dtype=fused_qkv.weight.dtype)
    linear.weight.data = fused_qkv.weight.data[idx * dim:(idx + 1) * dim].clone()
    if has_bias:
        linear.bias.data = fused_qkv.bias.data[idx * dim:(idx + 1) * dim].clone()
    return linear


# ============================================================================
# FlatQuantVGGTAttention
# ============================================================================

class FlatQuantVGGTAttention(nn.Module):
    """
    FlatQuant wrapper for VGGT's Attention module.

    Key differences from FlatQuantLlamaAttention:
    - Splits fused QKV into separate Q, K, V projections
    - No KV-cache transforms (VGGT is non-autoregressive)
    - Uses self.rope (single) for 2D RoPE
    - Preserves QK-norm (per-head LayerNorm)
    - Uses F.scaled_dot_product_attention (no causal mask)
    """

    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.scale = module.scale
        self.fused_attn = module.fused_attn

        dim = module.qkv.in_features  # 1024

        # Split fused QKV into separate projections and wrap with FlatQuantizedLinear
        self.q_proj = FlatQuantizedLinear(args, split_qkv_linear(module.qkv, 0, dim))
        self.k_proj = FlatQuantizedLinear(args, split_qkv_linear(module.qkv, 1, dim))
        self.v_proj = FlatQuantizedLinear(args, split_qkv_linear(module.qkv, 2, dim))
        self.o_proj = FlatQuantizedLinear(args, module.proj)

        # Keep QK norms as-is
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm

        # Keep RoPE (single rope, recons_eval convention)
        self.rope = module.rope

        # Dropout
        self.attn_drop = module.attn_drop
        self.proj_drop = module.proj_drop

        # Add FlatQuant transforms
        self.add_fq_trans()

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            device = self.q_proj.linear.weight.device
            self.ln_smax = torch.ones_like(
                self.q_proj.linear.weight.abs().max(dim=0)[0]
            ).to(device) * 1e-5

    def add_fq_trans(self):
        if self.args.direct_inv:
            SingleTransMatrix = InvSingleTransMatrix
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            SingleTransMatrix = SVDSingleTransMatrix
            DecomposeTransMatrix = SVDDecomposeTransMatrix

        if self.args.w_bits < 16 or self.args.a_bits < 16:
            dim = self.q_proj.linear.weight.shape[1]
            ln_dim_left, ln_dim_right = get_decompose_dim(dim)
            self.ln_trans = DecomposeTransMatrix(
                ln_dim_left, ln_dim_right, add_diag=self.args.add_diag
            )
            # O-proj transform: across heads
            self.o_trans = SingleTransMatrix(self.num_heads)
            # V transform: per head_dim
            self.vcache_trans = SingleTransMatrix(self.head_dim)
        else:
            self.ln_trans = None
            self.o_trans = None
            self.vcache_trans = None

    def _trans_forward_after_ln(self, hidden_states):
        """Quantized forward: apply learned transforms before projection."""
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)

        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(
                hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans
            )
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        """Original FP forward: no transforms, collect stats for diag init."""
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            if self.ln_smax.device != hidden_states.device:
                self.ln_smax = self.ln_smax.to(hidden_states.device)
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1])
                .abs()
                .max(0)[0]
                .clone()
                .detach(),
            )
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    def forward(self, x, pos=None):
        B, N, C = x.shape

        if self._ori_mode:
            q, k, v = self._ori_forward_after_ln(x)
        else:
            q, k, v = self._trans_forward_after_ln(x)

        # Reshape to multi-head: [B, N, dim] -> [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # QK normalization
        q, k = self.q_norm(q), self.k_norm(k)

        # 2D RoPE
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # V-cache transform for non-ori mode
        if not self._ori_mode and self.vcache_trans is not None and self.args.separate_vtrans:
            v = self.vcache_trans(v)

        # Attention
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # Reshape back: [B, num_heads, N, head_dim] -> [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection
        if self._ori_mode:
            x = self.o_proj._ori_forward(x)
        else:
            if self.o_trans is None and self.vcache_trans is not None:
                init_shape = x.shape
                x = x.reshape(-1, self.num_heads, self.head_dim)
                x = torch.matmul(
                    x, self.vcache_trans.get_matrix(inv_t=True).T.to(x)
                ).reshape(init_shape)
                x = self.o_proj(x)
            elif self.o_trans is not None:
                init_shape = x.shape
                x = x.reshape(-1, self.num_heads, self.head_dim)
                x = torch.matmul(
                    self.o_trans.get_matrix().T.to(x), x
                ).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
                    attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
                    x = self.o_proj(x, qa_trans=[attn_o_og_it, attn_v_og_it])
                else:
                    x = self.o_proj(x)
            else:
                x = self.o_proj(x)

        x = self.proj_drop(x)
        return x

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()

        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            self.v_proj.reparameterize(qa_trans=self.ln_trans)
        else:
            self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)

        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])

        # Convert transform matrices to FP16 to match linear layer weights
        if self.ln_trans is not None and hasattr(self.ln_trans, 'matrix_left'):
            self.ln_trans.matrix_left.data = self.ln_trans.matrix_left.data.to(torch.float16)
            self.ln_trans.matrix_right.data = self.ln_trans.matrix_right.data.to(torch.float16)
            if hasattr(self.ln_trans, 'matrix_left_inv'):
                self.ln_trans.matrix_left_inv.data = self.ln_trans.matrix_left_inv.data.to(torch.float16)
                self.ln_trans.matrix_right_inv.data = self.ln_trans.matrix_right_inv.data.to(torch.float16)

        if self.vcache_trans is not None and hasattr(self.vcache_trans, 'matrix'):
            self.vcache_trans.matrix.data = self.vcache_trans.matrix.data.to(torch.float16)
            if hasattr(self.vcache_trans, 'matrix_inv_t'):
                self.vcache_trans.matrix_inv_t.data = self.vcache_trans.matrix_inv_t.data.to(torch.float16)

        if self.o_trans is not None and hasattr(self.o_trans, 'matrix'):
            self.o_trans.matrix.data = self.o_trans.matrix.data.to(torch.float16)
            if hasattr(self.o_trans, 'matrix_inv_t'):
                self.o_trans.matrix_inv_t.data = self.o_trans.matrix_inv_t.data.to(torch.float16)

        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = (
            torch.cat(
                [
                    self.q_proj.linear.weight,
                    self.k_proj.linear.weight,
                    self.v_proj.linear.weight,
                ],
                dim=0,
            )
            .abs()
            .max(dim=0)[0]
        )
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(
                qkvw_smax, self.ln_smax, alpha
            )
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


# ============================================================================
# FlatQuantVGGTMLP
# ============================================================================

class FlatQuantVGGTMLP(nn.Module):
    """
    FlatQuant wrapper for VGGT's Mlp module.

    Key differences from FlatQuantLlamaMLP:
    - Two-layer MLP (fc1→act→fc2) instead of three-layer gated (gate+up+down)
    - GELU activation instead of SiLU
    - All layers have bias
    """

    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.fc1 = FlatQuantizedLinear(args, module.fc1)
        self.act = module.act
        self.fc2 = FlatQuantizedLinear(args, module.fc2)
        # Handle drop attribute (may be named 'drop' or have dropout)
        self.drop = getattr(module, 'drop', nn.Identity())

        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            device = self.fc1.linear.weight.device
            self.fc1_smax = (
                torch.ones_like(self.fc1.linear.weight.abs().max(dim=0)[0]).to(device)
                * 1e-5
            )
            self.fc2_smax = (
                torch.ones_like(self.fc2.linear.weight.abs().max(dim=0)[0]).to(device)
                * 1e-5
            )

    def add_fq_trans(self):
        if self.args.direct_inv:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix

        if self.args.w_bits < 16 or self.args.a_bits < 16:
            fc1_dim_left, fc1_dim_right = get_decompose_dim(
                self.fc1.linear.weight.shape[1]
            )
            self.fc1_trans = DecomposeTransMatrix(
                fc1_dim_left, fc1_dim_right, add_diag=self.args.add_diag
            )
            fc2_dim_left, fc2_dim_right = get_decompose_dim(
                self.fc2.linear.weight.shape[1]
            )
            self.fc2_trans = DecomposeTransMatrix(
                fc2_dim_left, fc2_dim_right, add_diag=self.args.add_diag
            )
        else:
            self.fc1_trans = None
            self.fc2_trans = None

    def _trans_forward(self, x):
        if self.fc1_trans is not None:
            x_ts = self.fc1_trans(x)
        else:
            x_ts = x
        x = self.fc1(x_ts, qa_trans=self.fc1_trans)
        x = self.act(x)
        x = self.drop(x)

        if self.fc2_trans is not None:
            x_ts2 = self.fc2_trans(x)
        else:
            x_ts2 = x
        x = self.fc2(x_ts2, qa_trans=self.fc2_trans)
        x = self.drop(x)
        return x

    def _ori_forward(self, x):
        if self.diag_init == "sq_style":
            if self.fc1_smax.device != x.device:
                self.fc1_smax = self.fc1_smax.to(x.device)
            self.fc1_smax = torch.maximum(
                self.fc1_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        x = self.fc1._ori_forward(x)
        x = self.act(x)
        x = self.drop(x)
        if self.diag_init == "sq_style":
            if self.fc2_smax.device != x.device:
                self.fc2_smax = self.fc2_smax.to(x.device)
            self.fc2_smax = torch.maximum(
                self.fc2_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        x = self.fc2._ori_forward(x)
        x = self.drop(x)
        return x

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self):
        if self.fc1_trans is not None:
            self.fc1_trans.to_eval_mode()
            self.fc2_trans.to_eval_mode()

        self.fc1.reparameterize(qa_trans=self.fc1_trans)
        self.fc2.reparameterize(qa_trans=self.fc2_trans)

        if self.fc1_trans is not None:
            self.fc1_trans.use_diag = False

        # Convert transform matrices to FP16 to match linear layer weights
        if self.fc1_trans is not None and hasattr(self.fc1_trans, 'matrix_left'):
            self.fc1_trans.matrix_left.data = self.fc1_trans.matrix_left.data.to(torch.float16)
            self.fc1_trans.matrix_right.data = self.fc1_trans.matrix_right.data.to(torch.float16)
            if hasattr(self.fc1_trans, 'matrix_left_inv'):
                self.fc1_trans.matrix_left_inv.data = self.fc1_trans.matrix_left_inv.data.to(torch.float16)
                self.fc1_trans.matrix_right_inv.data = self.fc1_trans.matrix_right_inv.data.to(torch.float16)

        if self.fc2_trans is not None and hasattr(self.fc2_trans, 'matrix_left'):
            self.fc2_trans.matrix_left.data = self.fc2_trans.matrix_left.data.to(torch.float16)
            self.fc2_trans.matrix_right.data = self.fc2_trans.matrix_right.data.to(torch.float16)
            if hasattr(self.fc2_trans, 'matrix_left_inv'):
                self.fc2_trans.matrix_left_inv.data = self.fc2_trans.matrix_left_inv.data.to(torch.float16)
                self.fc2_trans.matrix_right_inv.data = self.fc2_trans.matrix_right_inv.data.to(torch.float16)
            # Also convert diag_scale if present
            if hasattr(self.fc2_trans, 'diag_scale'):
                self.fc2_trans.diag_scale.data = self.fc2_trans.diag_scale.data.to(torch.float16)

        # NOTE: Unlike LLaMA's gated MLP (down(silu(gate)*up)), VGGT uses a simple
        # two-layer MLP (fc2(gelu(fc1(x)))). In LLaMA, down_trans.diag_scale can be
        # absorbed into up_proj because the element-wise multiplication commutes with
        # diagonal scaling. In VGGT, fc2_trans.diag_scale CANNOT be absorbed into fc1
        # because GELU is nonlinear: gelu(diag*x) != diag*gelu(x). The diagonal must
        # be applied at inference time, after GELU, so we keep fc2_trans.use_diag=True.

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "fc1_smax") and hasattr(self, "fc2_smax")
        fc1w_smax = self.fc1.linear.weight.abs().max(dim=0)[0]
        fc2w_smax = self.fc2.linear.weight.abs().max(dim=0)[0]
        if self.fc1_trans is not None:
            self.fc1_trans.diag_scale.data = get_init_scale(
                fc1w_smax, self.fc1_smax, alpha
            )
        if self.fc2_trans is not None:
            self.fc2_trans.diag_scale.data = get_init_scale(
                fc2w_smax, self.fc2_smax, alpha
            )
        del self.fc1_smax, self.fc2_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.fc1_trans is not None:
            self.fc1_trans.to_eval_mode()
            self.fc2_trans.to_eval_mode()


# ============================================================================
# Apply FlatQuant to VGGT
# ============================================================================

def apply_flatquant_to_vggt(args, model):
    """
    Replace VGGT's Attention and MLP modules with FlatQuant versions.

    Args:
        args: FlatQuant args (from FlatQuantConfig.to_args())
        model: VGGT model instance

    Returns:
        Modified model with FlatQuant wrappers
    """
    aggregator = model.aggregator

    num_blocks = len(aggregator.frame_blocks)
    logger.info(f"Applying FlatQuant to VGGT: {num_blocks} frame blocks + {num_blocks} global blocks")

    for i in range(num_blocks):
        # Frame blocks
        aggregator.frame_blocks[i].attn = FlatQuantVGGTAttention(
            args, aggregator.frame_blocks[i].attn
        )
        aggregator.frame_blocks[i].mlp = FlatQuantVGGTMLP(
            args, aggregator.frame_blocks[i].mlp
        )
        # Global blocks
        aggregator.global_blocks[i].attn = FlatQuantVGGTAttention(
            args, aggregator.global_blocks[i].attn
        )
        aggregator.global_blocks[i].mlp = FlatQuantVGGTMLP(
            args, aggregator.global_blocks[i].mlp
        )

    logger.info("FlatQuant applied to all aggregator blocks")
    return model


# ============================================================================
# RTN Weight Quantization (post-reparameterization)
# ============================================================================

@torch.no_grad()
def apply_rtn_weight_quantization(model, w_bits, w_asym=False):
    """
    Apply Round-To-Nearest (RTN) fake weight quantization to all FlatQuantizedLinear
    modules in the aggregator. This must be called AFTER reparameterization.

    In the original FlatQuant (main.py), this is done via gptq_utils.rtn_fwrd() after
    reparameterize_model(). Without this step, _eval_forward uses FP weights and w_bits
    has no effect on evaluation results.

    Args:
        model: VGGT model with FlatQuant wrappers (after reparameterization)
        w_bits: Weight quantization bits (e.g., 4 or 8)
        w_asym: Whether to use asymmetric weight quantization
    """
    from flatquant.quant_utils import WeightQuantizer

    if w_bits >= 16:
        logger.info("w_bits >= 16, skipping weight quantization")
        return

    n_quantized = 0
    total_mse = 0.0
    for name, module in model.named_modules():
        if isinstance(module, FlatQuantizedLinear):
            W = module.linear.weight.data
            w_dtype = W.dtype

            quantizer = WeightQuantizer()
            quantizer.configure(w_bits, perchannel=True, sym=not w_asym, mse=False)
            quantizer.find_params(W)
            W_quant = quantizer.quantize(W).to(w_dtype)
            mse = (W.float() - W_quant.float()).pow(2).mean().item()
            total_mse += mse
            module.linear.weight.data = W_quant
            n_quantized += 1

    avg_mse = total_mse / max(n_quantized, 1)
    logger.info(
        f"RTN weight quantization: w_bits={w_bits}, {n_quantized} layers, "
        f"avg weight MSE={avg_mse:.2e}"
    )


# ============================================================================
# Reparameterization
# ============================================================================

def reparameterize_ln_with_bias(ln, trans):
    """
    Fuse FlatQuant diagonal scaling into LayerNorm.

    Unlike FlatQuant's original reparameterize_ln (for RMSNorm, weight-only),
    VGGT uses LayerNorm with both weight AND bias, so we scale both.

    Math: LayerNorm(x) = (x - mu) / sigma * gamma + beta
    After scaling: (x - mu) / sigma * (gamma*s) + (beta*s) = s * LayerNorm(x)
    """
    if trans is None or not trans.add_diag:
        return
    ori_dtype = ln.weight.dtype
    ln.weight.data = (
        ln.weight.data.to(torch.float64) * trans.diag_scale.to(torch.float64)
    ).to(ori_dtype)
    if ln.bias is not None:
        ln.bias.data = (
            ln.bias.data.to(torch.float64) * trans.diag_scale.to(torch.float64)
        ).to(ori_dtype)
    trans.use_diag = False


def reparameterize_vggt(model):
    """
    After calibration, absorb all learned transforms into model weights.
    """
    aggregator = model.aggregator

    for block_list_name in ("frame_blocks", "global_blocks"):
        block_list = getattr(aggregator, block_list_name)
        for idx in range(len(block_list)):
            block = block_list[idx]

            # Reparameterize attention
            block.attn.reparameterize()

            # Reparameterize MLP
            block.mlp.reparameterize()

            # Fuse diagonal scaling into LayerNorm
            if hasattr(block.attn, "ln_trans"):
                reparameterize_ln_with_bias(block.norm1, block.attn.ln_trans)
            if hasattr(block.mlp, "fc1_trans"):
                reparameterize_ln_with_bias(block.norm2, block.mlp.fc1_trans)

            # Convert LayerNorm parameters to FP16 to match linear layer weights
            # (After dtype fix in flat_linear.py, all weights are FP16)
            block.norm1.weight.data = block.norm1.weight.data.to(torch.float16)
            block.norm1.bias.data = block.norm1.bias.data.to(torch.float16)
            block.norm2.weight.data = block.norm2.weight.data.to(torch.float16)
            block.norm2.bias.data = block.norm2.bias.data.to(torch.float16)

            # Convert QK LayerNorm to FP16
            if hasattr(block.attn, 'q_norm') and not isinstance(block.attn.q_norm, nn.Identity):
                block.attn.q_norm.weight.data = block.attn.q_norm.weight.data.to(torch.float16)
                block.attn.q_norm.bias.data = block.attn.q_norm.bias.data.to(torch.float16)
                block.attn.k_norm.weight.data = block.attn.k_norm.weight.data.to(torch.float16)
                block.attn.k_norm.bias.data = block.attn.k_norm.bias.data.to(torch.float16)

    # NOTE: Heads' LayerNorms are intentionally kept in FP32 here.
    # Under outer autocast(bf16) at inference, the aggregator's residual chain outputs FP32,
    # and vggt.py wraps the heads in autocast(enabled=False). Converting head LNs to FP16
    # would cause "expected scalar type Float but found Half" in fake-quant mode.
    # Mark model as using fake-quant mode (for dtype preservation in .to() calls)
    model._is_flatquant_fake = True

    logger.info("Reparameterization complete")


# ============================================================================
# Calibration
# ============================================================================

def load_calibration_images(config):
    """
    Load calibration images.

    Args:
        config: FlatQuantConfig

    Returns:
        List of image tensors, each [S, 3, H, W] in [0, 1]
    """
    import glob
    import random
    from PIL import Image
    from torchvision import transforms

    data_dir = config.cali_data_dir
    nsamples = config.nsamples
    num_frames = config.num_frames
    img_size = config.img_size

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Randomly sample directly from the raw calibration image directory.
    # This intentionally does not use CO3D annotation files or quality filters.
    categories = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not categories:
        raise RuntimeError(f"No categories found in {data_dir}")

    calibration_data = []
    rng = random.Random(getattr(config, "cali_seed", 42))

    attempts = 0
    max_attempts = nsamples * 10

    while len(calibration_data) < nsamples and attempts < max_attempts:
        attempts += 1
        # Pick a random category
        category = rng.choice(categories)
        cat_dir = os.path.join(data_dir, category)

        # Pick a random sequence
        sequences = [
            d for d in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, d))
        ]
        if not sequences:
            continue

        seq = rng.choice(sequences)
        seq_dir = os.path.join(cat_dir, seq, "images")
        if not os.path.isdir(seq_dir):
            # Try without 'images' subdirectory
            seq_dir = os.path.join(cat_dir, seq)

        # Find image files
        image_files = sorted(
            glob.glob(os.path.join(seq_dir, "*.jpg"))
            + glob.glob(os.path.join(seq_dir, "*.png"))
        )

        if len(image_files) < num_frames:
            continue

        # Sample frames
        selected = rng.sample(image_files, num_frames)
        frames = []
        valid = True
        for img_path in selected:
            try:
                img = Image.open(img_path).convert("RGB")
                frames.append(transform(img))
            except Exception:
                valid = False
                break

        if valid and len(frames) == num_frames:
            calibration_data.append(torch.stack(frames))

    if len(calibration_data) < nsamples:
        logger.warning(
            f"Only loaded {len(calibration_data)}/{nsamples} calibration samples"
        )

    logger.info(
        "Loaded %d random calibration samples from %s with %d frames each",
        len(calibration_data),
        data_dir,
        num_frames,
    )
    return calibration_data


def _set_block_mode(block, ori_mode):
    """Set the _ori_mode flag for attn and mlp in a block."""
    block.attn._ori_mode = ori_mode
    block.mlp._ori_mode = ori_mode


def load_fisher_weights(fisher_path):
    """
    Load and normalize precomputed Fisher information for per-channel loss weighting.

    Args:
        fisher_path: Path to fisher.pt containing 'fisher' key with shape [3, 48, 1024]

    Returns:
        Tensor of shape [48, 1024] with per-block, per-channel weights (mean=1 per block)
    """
    data = torch.load(fisher_path, map_location='cpu', weights_only=True)
    fisher = data['fisher']  # [3, 48, 1024]

    # Per-task normalization: divide each task's Fisher by its global mean
    for k in range(fisher.shape[0]):
        f_bar = fisher[k].mean()
        fisher[k] = fisher[k] / (f_bar + 1e-30)

    # Sum across tasks (equal weights)
    w = fisher.sum(dim=0)  # [48, 1024]

    # Per-block rescaling to mean 1
    for l in range(w.shape[0]):
        w[l] = w[l] / (w[l].mean() + 1e-30)

    # Flooring at 1% of per-block mean (which is 1.0)
    w = torch.clamp(w, min=0.01)

    return w


def calibrate_flatquant_vggt(args, model, calibration_data, dev, cali_logger=None, fisher_weights=None):
    """
    FlatQuant calibration loop for VGGT.

    Processes blocks in alternating order (frame[0] → global[0] → frame[1] → ...),
    calibrating one block at a time with MSE loss between FP and quantized outputs.

    Args:
        args: FlatQuant args namespace
        model: VGGT model with FlatQuant wrappers applied
        calibration_data: List of image tensors [S, 3, H, W]
        dev: Target device (e.g., 'cuda')
        cali_logger: Optional logger
        fisher_weights: Optional [48, 1024] tensor of per-channel Fisher weights
    """
    if cali_logger is None:
        cali_logger = logger

    model.eval()

    # Disable gradient for all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # AMP setup
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.bfloat16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

    aggregator = model.aggregator
    num_blocks = len(aggregator.frame_blocks)

    # ---- Step 1: Compute patch embeddings from calibration images ----
    cali_logger.info("Computing patch embeddings for calibration data...")
    all_tokens = []
    all_pos = []

    aggregator.patch_embed = aggregator.patch_embed.to(dev)
    # Move normalization buffers
    if hasattr(aggregator, '_resnet_mean'):
        aggregator._resnet_mean = aggregator._resnet_mean.to(dev)
        aggregator._resnet_std = aggregator._resnet_std.to(dev)

    with torch.no_grad():
        for sample_idx, images in enumerate(calibration_data):
            # images: [S, 3, H, W] in [0, 1]
            images = images.unsqueeze(0).to(dev)  # [1, S, 3, H, W]
            B, S, C_in, H, W = images.shape

            # Normalize
            images_norm = (images - aggregator._resnet_mean) / aggregator._resnet_std
            images_flat = images_norm.view(B * S, C_in, H, W)

            with traincast():
                patch_tokens = aggregator.patch_embed(images_flat)
                if isinstance(patch_tokens, dict):
                    patch_tokens = patch_tokens["x_norm_patchtokens"]

            _, P, C = patch_tokens.shape

            # Add special tokens
            from models.vggt.models.aggregator import slice_expand_and_flatten
            camera_token = slice_expand_and_flatten(aggregator.camera_token.to(dev), B, S)
            register_token = slice_expand_and_flatten(aggregator.register_token.to(dev), B, S)
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

            # Compute positions
            pos = None
            if aggregator.rope is not None:
                pos = aggregator.position_getter(
                    B * S, H // aggregator.patch_size, W // aggregator.patch_size, device=dev
                )
                if aggregator.patch_start_idx > 0:
                    pos = pos + 1
                    pos_special = torch.zeros(B * S, aggregator.patch_start_idx, 2).to(dev).to(pos.dtype)
                    pos = torch.cat([pos_special, pos], dim=1)

            all_tokens.append(tokens.to(dtype).cpu())
            if pos is not None:
                all_pos.append(pos.cpu())

    # Move patch embed back to CPU
    aggregator.patch_embed = aggregator.patch_embed.cpu()
    torch.cuda.empty_cache()

    # Get shapes for reshaping
    # tokens shape: [B*S, P_total, C]  (always stored in frame format as canonical)
    sample_tokens = all_tokens[0]
    P_total = sample_tokens.shape[1]
    C = sample_tokens.shape[2]

    # Store per-sample S values for reshaping
    # Each sample: all_tokens[i] has shape [S_i, P_total, C] where S_i = B*S
    sample_S_values = [t.shape[0] for t in all_tokens]

    nsamples = len(all_tokens)
    cali_logger.info(f"Collected {nsamples} calibration samples, tokens shape per sample: {sample_tokens.shape}")

    def reshape_for_block(tokens, pos, btype, S_val):
        """Reshape tokens and pos for frame or global block."""
        B_s = 1
        if btype == "frame":
            # Frame blocks expect [B*S, P, C] = [S_val, P_total, C]
            tokens = tokens.view(S_val, P_total, C)
            if pos is not None:
                pos = pos.view(S_val, P_total, 2)
        else:
            # Global blocks expect [B, S*P, C] = [1, S_val*P_total, C]
            tokens = tokens.view(B_s, S_val * P_total, C)
            if pos is not None:
                pos = pos.view(B_s, S_val * P_total, 2)
        return tokens, pos

    def reshape_output_to_canonical(out, btype, S_val):
        """Reshape block output back to canonical [S_val, P_total, C] format."""
        if btype == "frame":
            # Output is already [S_val, P_total, C]
            return out.view(S_val, P_total, C)
        else:
            # Output is [1, S_val*P_total, C] -> [S_val, P_total, C]
            return out.view(S_val, P_total, C)

    # ---- Step 2: Layer-by-layer calibration ----
    mse_criterion = nn.MSELoss()
    all_blocks = []
    block_types = []  # 'frame' or 'global'

    for i in range(num_blocks):
        all_blocks.append(aggregator.frame_blocks[i])
        block_types.append("frame")
        all_blocks.append(aggregator.global_blocks[i])
        block_types.append("global")

    # fp_inps always stored in canonical frame format: [S_val, P_total, C]
    fp_inps = all_tokens

    for block_idx, (block, btype) in enumerate(zip(all_blocks, block_types)):
        block_name = f"{btype}_block_{block_idx // 2}"
        cali_logger.info(f"=== Calibrating {block_name} ({block_idx + 1}/{len(all_blocks)}) ===")

        # Move block to GPU
        block = block.to(dev)

        # Determine position tensors
        block_pos = all_pos if all_pos else [None] * nsamples

        # ---- Collect FP outputs in FP32 for accurate targets ----
        # (Original FlatQuant uses layer.float() for FP targets)
        _set_block_mode(block, ori_mode=True)
        fp_outs = []

        with torch.no_grad():
            block.float()  # FP32 for accurate targets
            for s_idx in range(nsamples):
                tokens = fp_inps[s_idx].to(dev).float()
                pos = block_pos[s_idx].to(dev) if block_pos[s_idx] is not None else None
                S_val = sample_S_values[s_idx]

                tokens, pos = reshape_for_block(tokens, pos, btype, S_val)

                out = block(tokens, pos=pos)

                # Store output in canonical format
                out_canonical = reshape_output_to_canonical(out, btype, S_val)
                fp_outs.append(out_canonical.to(dtype).cpu())

        # ---- Initialize diagonal scaling ----
        if args.add_diag and args.diag_init == "sq_style":
            block.attn.init_diag_scale(alpha=args.diag_alpha)
            block.mlp.init_diag_scale(alpha=args.diag_alpha)

        # ---- Switch to quantized mode ----
        _set_block_mode(block, ori_mode=False)
        block = block.to(dev)  # Ensure on correct device after float()

        # Set up trainable parameters (matching original FlatQuant param groups)
        set_require_grad_all(block, False)
        trained_params = []
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(block, ["trans.linear", "trans.matrix"]), "lr": args.flat_lr})
        if args.add_diag:
            trained_params.append({"params": get_n_set_parameters_byname(block, ["trans.diag_scale"]), "lr": args.flat_lr})
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(block, ["clip_factor_w"]), "lr": args.flat_lr * 10})
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(block, ["clip_factor_a"]), "lr": args.flat_lr * 10})

        optimizer = torch.optim.AdamW(trained_params)

        n_steps_per_epoch = nsamples // args.cali_bsz
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * n_steps_per_epoch,
            eta_min=args.flat_lr * 1e-3
        )
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=16
            )
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [scheduler_warmup, scheduler_main]
            )
        else:
            scheduler = scheduler_main

        # ---- Training loop ----
        best_loss = float("inf")
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, nsamples, args.cali_bsz):
                index = batch_start
                batch_end = min(index + args.cali_bsz, nsamples)

                total_loss = 0.0
                for s_idx in range(index, batch_end):
                    tokens = fp_inps[s_idx].to(dev)
                    pos = block_pos[s_idx].to(dev) if block_pos[s_idx] is not None else None
                    fp_out_canonical = fp_outs[s_idx].to(dev)
                    S_val = sample_S_values[s_idx]

                    tokens, pos = reshape_for_block(tokens, pos, btype, S_val)
                    # fp_out needs same shape as block output for MSE
                    fp_out_reshaped, _ = reshape_for_block(fp_out_canonical, None, btype, S_val)

                    with traincast():
                        quant_out = block(tokens, pos=pos)
                        if fisher_weights is not None:
                            w_l = fisher_weights[block_idx].to(dev)  # [1024]
                            diff_sq = (quant_out - fp_out_reshaped) ** 2
                            loss = (w_l.view(1, 1, -1) * diff_sq).mean()
                        else:
                            loss = mse_criterion(quant_out, fp_out_reshaped)

                    total_loss += loss

                total_loss = total_loss / (batch_end - index)
                # Normalize loss to stabilize gradient magnitude (from original FlatQuant).
                # Without this, gradient magnitude decreases as MSE shrinks, making later
                # epochs ineffective. This explains why more epochs + lower lr was worse.
                epoch_loss += total_loss.item()
                total_loss = total_loss / total_loss.clone().detach()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            cur_lr = optimizer.param_groups[0]['lr']
            cali_logger.info(
                f"  {block_name} Epoch {epoch + 1}/{args.epochs} lr {cur_lr:.8f} Loss: {avg_loss:.6f} (best: {best_loss:.6f})"
            )

        # ---- Update fp_inps for next block ----
        _set_block_mode(block, ori_mode=True)
        # Use FP outputs (already in canonical format) as next block's input
        fp_inps = fp_outs

        # Move block back to CPU
        block = block.cpu()
        for param in block.parameters():
            param.requires_grad = False
        torch.cuda.empty_cache()

    cali_logger.info("Calibration complete for all blocks")
