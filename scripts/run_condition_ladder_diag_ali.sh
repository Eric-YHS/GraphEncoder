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

SOURCE=${1:?usage: bash scripts/run_condition_ladder_diag_ali.sh none|property|grale_official [gpu_csv] [max_iters] [exp_suffix] [extra --set ...]}
GPUS=${2:-0,1,2,3}
MAX_ITERS=${3:-6000}
EXP_SUFFIX=${4:-diag}
if [ "$#" -ge 4 ]; then
  shift 4
else
  shift "$#"
fi

LOCAL_DATA_ROOT=/root/condition_ladder_data/PCQM4M
if [ -d "${LOCAL_DATA_ROOT}/pcqm4m-v2/processed" ]; then
  DATA_ROOT=${DATA_ROOT:-${LOCAL_DATA_ROOT}}
else
  DATA_ROOT=${DATA_ROOT:-/mnt2/luyifeng/diff4MoleculeRepresentation/data/PCQM4M}
fi

NPROC=$(python -c "print(len('${GPUS}'.split(',')))")
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29641}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-8192}
PER_GPU_BATCH=${PER_GPU_BATCH:-$((TARGET_GLOBAL_BATCH / NPROC))}
MIN_PER_GPU_BATCH=${MIN_PER_GPU_BATCH:-128}
AUTO_BATCH_ON_FAIL=${AUTO_BATCH_ON_FAIL:-true}

while true; do
  ACCUM=$((TARGET_GLOBAL_BATCH / (NPROC * PER_GPU_BATCH)))
  if [ "${ACCUM}" -lt 1 ] || [ $((NPROC * PER_GPU_BATCH * ACCUM)) -ne "${TARGET_GLOBAL_BATCH}" ]; then
    echo "[ERROR] TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH} must be divisible by NPROC*PER_GPU_BATCH=$((NPROC * PER_GPU_BATCH))" >&2
    exit 2
  fi
  echo "[INFO] diag source=${SOURCE} gpus=${GPUS} max_iters=${MAX_ITERS} per_gpu_batch=${PER_GPU_BATCH} accum=${ACCUM} extra=$*"
  set +e
  CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" scripts/train_condition_ladder.py \
    --config configs/training_condition_ladder.yml \
    --condition_source "${SOURCE}" \
    --logdir logs_condition_ladder_diag \
    --exp_name "condition_ladder_v1_${EXP_SUFFIX}" \
    --max_iters "${MAX_ITERS}" \
    --set data.path="${DATA_ROOT}" \
    --set train.batch_size="${PER_GPU_BATCH}" \
    --set train.n_acc_batch="${ACCUM}" \
    --set train.num_workers=4 \
    --set train.persistent_workers=true \
    --set train.prefetch_factor=4 \
    --set train.pin_memory=true \
    "$@"
  STATUS=$?
  set -e
  if [ "${STATUS}" -eq 0 ]; then
    exit 0
  fi
  if [ "${AUTO_BATCH_ON_FAIL}" != "true" ] || [ "${PER_GPU_BATCH}" -le "${MIN_PER_GPU_BATCH}" ]; then
    exit "${STATUS}"
  fi
  PER_GPU_BATCH=$((PER_GPU_BATCH / 2))
  echo "[WARN] diag exited with status=${STATUS}; retrying with per_gpu_batch=${PER_GPU_BATCH}" >&2
done
