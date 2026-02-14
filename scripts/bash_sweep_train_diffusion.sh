#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

# python scripts/sweep_layers.py --detach --gpus 0 --dec_range 5,5 --grid \
#   --encoder_name cls_graphormer --denoiser_name uni_o2_cat \
#   > logs/gpu0.log 2>&1 &

python scripts/sweep_layers.py --detach --gpus 1 --dec_range 5,5 --grid \
  --encoder_name cls_pearl --denoiser_name uni_o2_cat --pearl_fuse concat \
  > logs/gpu1.log 2>&1 &

python scripts/sweep_layers.py --detach --gpus 2 --dec_range 5,5 --grid \
  --encoder_name cls_pearl --denoiser_name uni_o2_cat --pearl_fuse add \
  > logs/gpu2.log 2>&1 &

# python scripts/sweep_layers.py --detach --gpus 3 --dec_range 5,5 --grid \
#   --encoder_name cls_graphormer_pearl --denoiser_name uni_o2_cat \
#   > logs/gpu3.log 2>&1 &

wait
echo "All jobs finished."



