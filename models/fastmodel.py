import os
from typing import Optional

import torch
import torch.nn as nn

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class VGGT(nn.Module):
    def __init__(self, pretrained_model_name_or_path: Optional[str] = None):
        super().__init__()

        if pretrained_model_name_or_path is None:
            raise ValueError("pretrained_model_name_or_path must be provided")

        from models.vggt.models.vggt import VGGT as VGGTModel

        self.model = VGGTModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images, query_points)


class VGGT_FlatQuant(nn.Module):
    """VGGT with FlatQuant-style fake quantization and optional Fisher guidance."""

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        quantized_model_path: Optional[str] = None,
        w_bits: int = 4,
        a_bits: int = 4,
        cali_trans: bool = True,
        add_diag: bool = True,
        lwc: bool = True,
        lac: bool = True,
        direct_inv: bool = False,
        separate_vtrans: bool = False,
        a_groupsize: int = -1,
    ):
        super().__init__()
        self._load_fakequant_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantized_model_path=quantized_model_path,
            w_bits=w_bits,
            a_bits=a_bits,
            cali_trans=cali_trans,
            add_diag=add_diag,
            lwc=lwc,
            lac=lac,
            direct_inv=direct_inv,
            separate_vtrans=separate_vtrans,
            a_groupsize=a_groupsize,
        )

    def _load_fakequant_model(
        self,
        pretrained_model_name_or_path: Optional[str],
        quantized_model_path: Optional[str],
        w_bits: int,
        a_bits: int,
        cali_trans: bool,
        add_diag: bool,
        lwc: bool,
        lac: bool,
        direct_inv: bool,
        separate_vtrans: bool,
        a_groupsize: int,
    ):
        from models.quantization.config import FlatQuantConfig
        from models.quantization.apply_flatquant import (
            apply_flatquant_to_vggt,
            apply_rtn_weight_quantization,
            reparameterize_vggt,
        )
        from models.vggt.models.vggt import VGGT as VGGTModel

        config = FlatQuantConfig(
            w_bits=w_bits,
            a_bits=a_bits,
            cali_trans=cali_trans,
            add_diag=add_diag,
            lwc=lwc,
            lac=lac,
            direct_inv=direct_inv,
            separate_vtrans=separate_vtrans,
            a_groupsize=a_groupsize,
        )
        fq_args = config.to_args()

        self.model = VGGTModel()

        if quantized_model_path is not None:
            model_file = quantized_model_path
            if os.path.isdir(model_file):
                candidate_names = [
                    f"model_flatquant_w{w_bits}a{a_bits}_fisher.pt",
                    f"model_flatquant_w{w_bits}a{a_bits}.pt",
                    "model_flatquant_fisher.pt",
                    "model_flatquant.pt",
                ]
                for candidate_name in candidate_names:
                    candidate = os.path.join(model_file, candidate_name)
                    if os.path.exists(candidate):
                        model_file = candidate
                        break

            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Quantized VGGT checkpoint not found at {model_file}")

            apply_flatquant_to_vggt(fq_args, self.model)
            reparameterize_vggt(self.model)

            checkpoint = torch.load(model_file, map_location="cpu")
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

            model_state = self.model.state_dict()
            filtered_state = {}
            skipped_unexpected = []
            for key, value in state_dict.items():
                if key in model_state and model_state[key].shape != value.shape:
                    if "weight_quantizer" not in key:
                        skipped_unexpected.append(key)
                    continue
                filtered_state[key] = value
            if skipped_unexpected:
                print(
                    "WARNING: skipped keys with unexpected shape mismatch: "
                    f"{skipped_unexpected[:5]}"
                )

            load_result = self.model.load_state_dict(filtered_state, strict=False)
            if load_result.unexpected_keys:
                print(
                    f"WARNING: {len(load_result.unexpected_keys)} unexpected keys in checkpoint: "
                    f"{load_result.unexpected_keys[:5]}"
                )
            if load_result.missing_keys:
                real_missing = [k for k in load_result.missing_keys if "weight_quantizer" not in k]
                if real_missing:
                    print(
                        f"WARNING: {len(real_missing)} missing keys in checkpoint: "
                        f"{real_missing[:5]}"
                    )

            if isinstance(checkpoint, dict) and "config" in checkpoint:
                ckpt_cfg = checkpoint["config"]
                ckpt_w = ckpt_cfg.get("w_bits", "?")
                ckpt_a = ckpt_cfg.get("a_bits", "?")
                if ckpt_w != w_bits or ckpt_a != a_bits:
                    print(
                        f"WARNING: checkpoint was calibrated for W{ckpt_w}A{ckpt_a} "
                        f"but loading as W{w_bits}A{a_bits}"
                    )

            apply_rtn_weight_quantization(self.model, w_bits=w_bits, w_asym=config.w_asym)
            self._cleanup_eval_dtypes()
            print(f"Loaded VGGT fake-quant checkpoint: W{w_bits}A{a_bits} from {model_file}")
        elif pretrained_model_name_or_path is not None:
            self.model = VGGTModel.from_pretrained(pretrained_model_name_or_path)
            apply_flatquant_to_vggt(fq_args, self.model)
        else:
            raise ValueError("Either quantized_model_path or pretrained_model_name_or_path must be provided")

    def _cleanup_eval_dtypes(self):
        for name, module in self.model.named_modules():
            if "ln_trans" in name and hasattr(module, "diag_scale"):
                delattr(module, "diag_scale")

            for attr in ("matrix_left", "matrix_right", "matrix_left_inv", "matrix_right_inv", "matrix_inv_t"):
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    if isinstance(value, torch.Tensor):
                        value.data = value.data.to(torch.float16)
            if hasattr(module, "matrix") and isinstance(module.matrix, torch.nn.Parameter):
                module.matrix.data = module.matrix.data.to(torch.float16)

            if isinstance(module, torch.nn.LayerNorm):
                target_dtype = torch.float16 if "aggregator" in name else torch.float32
                if module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(target_dtype)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images, query_points)
