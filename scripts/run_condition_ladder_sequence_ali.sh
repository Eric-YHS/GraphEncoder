#!/usr/bin/env bash
set -eo pipefail

cd /root/graph_research/huangjh/encoder

mkdir -p logs_condition_ladder_daemon

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
DATA_ROOT=${DATA_ROOT:-/mnt2/luyifeng/diff4MoleculeRepresentation/data/PCQM4M}
PER_GPU_BATCH=${PER_GPU_BATCH:-1024}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-8192}
AUTO_BATCH_ON_FAIL=${AUTO_BATCH_ON_FAIL:-true}
FIRST_SOURCE=${FIRST_SOURCE:-property}
SECOND_SOURCE=${SECOND_SOURCE:-grale_official}
MASTER_PORT_1=${MASTER_PORT_1:-29611}
MASTER_PORT_2=${MASTER_PORT_2:-29612}

export DATA_ROOT PER_GPU_BATCH TARGET_GLOBAL_BATCH AUTO_BATCH_ON_FAIL

echo "[INFO] sequential ali condition ladder: ${FIRST_SOURCE} -> ${SECOND_SOURCE}"
echo "[INFO] gpus=${GPUS} data_root=${DATA_ROOT} per_gpu_batch=${PER_GPU_BATCH} target_global=${TARGET_GLOBAL_BATCH}"

MASTER_PORT="${MASTER_PORT_1}" bash scripts/run_condition_ladder_full_ali.sh "${FIRST_SOURCE}" "${GPUS}"
MASTER_PORT="${MASTER_PORT_2}" bash scripts/run_condition_ladder_full_ali.sh "${SECOND_SOURCE}" "${GPUS}"
