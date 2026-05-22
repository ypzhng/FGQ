# Multi-view Reconstruction Evaluation

The released multi-view reconstruction evaluator targets VGGT and VGGT FGQ fake-quant checkpoints.

```bash
python mv_recon/eval.py \
  evaluation=mv_recon \
  eval_models=[vggt_flatquant_fisher] \
  eval_datasets=[7scenes-sparse,DTU,ETH3D]
```

The provided sequence maps under `datasets/seq-id-maps/` reproduce the default key-frame sampling. To change the sampling, edit `configs/data/mv_recon.yaml`, run `python mv_recon/sampling.py`, and point the config to the generated map.

The release does not include dataset preprocessing scripts. Prepare datasets with the expected directory structure before running evaluation.
