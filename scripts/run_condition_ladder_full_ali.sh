#!/usr/bin/env bash
set -eo pipefail

cd /root/graph_research/huangjh/encoder
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hjhencoder

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SOURCE=${1:?usage: bash scripts/run_condition_ladder_full_ali.sh none|property|grale_official [gpu_csv]}
GPUS=${2:-0,1,2,3}
DATA_ROOT=${DATA_ROOT:-/mnt2/luyifeng/diff4MoleculeRepresentation/data/PCQM4M}
NPROC=$(python -c "print(len('${GPUS}'.split(',')))")
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29551}

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" scripts/train_condition_ladder.py \
  --config configs/training_condition_ladder.yml \
  --condition_source "${SOURCE}" \
  --logdir logs_condition_ladder \
  --exp_name condition_ladder_v1 \
  --set data.path="${DATA_ROOT}" \
  --set train.batch_size=1024 \
  --set train.n_acc_batch=2 \
  --set train.num_workers=4 \
  --set train.persistent_workers=true \
  --set train.prefetch_factor=4 \
  --set train.pin_memory=true
