#!/usr/bin/env bash
set -eo pipefail

cd /root/graph_research/huangjh/encoder
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hjhencoder

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SOURCE=${1:-none}
GPUS=${GPUS:-0}
MAX_ITERS=${MAX_ITERS:-20}
DATA_ROOT=${DATA_ROOT:-/mnt2/luyifeng/diff4MoleculeRepresentation/data/PCQM4M}

NPROC=$(python -c "print(len('${GPUS}'.split(',')))")
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29531}
CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" scripts/train_condition_ladder.py \
  --config configs/training_condition_ladder.yml \
  --condition_source "${SOURCE}" \
  --logdir logs_condition_ladder_smoke \
  --exp_name condition_ladder_smoke \
  --max_iters "${MAX_ITERS}" \
  --set data.path="${DATA_ROOT}" \
  --set condition.require_valid_mask=false \
  --set train.batch_size=128 \
  --set train.n_acc_batch=1 \
  --set train.num_workers=0 \
  --set train.report_iter=1 \
  --set train.probe_freq=10 \
  --set train.val_freq=10 \
  --set train.validate_batches=2
