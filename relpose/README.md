# Relative Camera Pose Evaluation

The released relpose evaluators currently target VGGT and VGGT FGQ fake-quant checkpoints.

## Angular Metrics

```bash
python relpose/eval_angle.py \
  evaluation=relpose-angular \
  eval_models=[vggt_flatquant_fisher] \
  eval_datasets=[CO3Dv2]
```

The provided sequence maps under `datasets/seq-id-maps/` reproduce the default sampling. To change the sampling, edit `configs/data/relpose-angular.yaml`, run `python relpose/sampling.py`, and point the config to the generated map.

## Distance Metrics

```bash
python relpose/eval_dist.py \
  evaluation=relpose-distance \
  eval_models=[vggt_flatquant_fisher] \
  eval_datasets=[Re10K]
```

The release does not include dataset preprocessing scripts. Prepare datasets with the expected directory structure before running evaluation.
