#!/usr/bin/env bash
set -eo pipefail

RUN_DIR=${RUN_DIR:-/XYAIFS00/HDD_POOL/nsccgz_ywang/nsccgz_ywang_wzh/huangjh/encoder_condition_ladder}
cd "${RUN_DIR}"
source /app/bin/proxy.sh >/dev/null 2>&1 || true
source /XYAIFS00/HDD_POOL/nsccgz_ywang/nsccgz_ywang_wzh/anaconda3/etc/profile.d/conda.sh
conda activate hjhencoder

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SOURCE=${1:-none}
DATA_ROOT=${DATA_ROOT:-/tmp/hjhencoder/PCQM4M}
MAX_ITERS=${MAX_ITERS:-5000}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29541}

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" scripts/train_condition_ladder.py \
  --config configs/training_condition_ladder.yml \
  --condition_source "${SOURCE}" \
  --logdir logs_condition_ladder \
  --exp_name condition_ladder_a100_pilot \
  --max_iters "${MAX_ITERS}" \
  --set data.path="${DATA_ROOT}" \
  --set condition.require_valid_mask=false \
  --set train.batch_size=2048 \
  --set train.n_acc_batch=2 \
  --set train.num_workers=8 \
  --set train.persistent_workers=true \
  --set train.prefetch_factor=4 \
  --set train.pin_memory=true
