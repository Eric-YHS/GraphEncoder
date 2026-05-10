#!/usr/bin/env bash
set -eo pipefail

cd /root/graph_research/huangjh/encoder
mkdir -p logs_condition_ladder_daemon

LEFT_SOURCE=${LEFT_SOURCE:-property}
RIGHT_SOURCE=${RIGHT_SOURCE:-grale_official}
LEFT_GPUS=${LEFT_GPUS:-0,1,2,3}
RIGHT_GPUS=${RIGHT_GPUS:-4,5,6,7}
LEFT_PORT=${LEFT_PORT:-29621}
RIGHT_PORT=${RIGHT_PORT:-29622}
DATA_ROOT=${DATA_ROOT:-/root/condition_ladder_data/PCQM4M}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-8192}
AUTO_BATCH_ON_FAIL=${AUTO_BATCH_ON_FAIL:-true}

export DATA_ROOT TARGET_GLOBAL_BATCH AUTO_BATCH_ON_FAIL
unset PER_GPU_BATCH

ts=$(date +%Y%m%d-%H%M%S)
left_log="logs_condition_ladder_daemon/dual_${LEFT_SOURCE}_${ts}.log"
right_log="logs_condition_ladder_daemon/dual_${RIGHT_SOURCE}_${ts}.log"

echo "[INFO] left ${LEFT_SOURCE} gpus=${LEFT_GPUS} log=${left_log}"
echo "[INFO] right ${RIGHT_SOURCE} gpus=${RIGHT_GPUS} log=${right_log}"

(MASTER_PORT="${LEFT_PORT}" bash scripts/run_condition_ladder_full_ali.sh "${LEFT_SOURCE}" "${LEFT_GPUS}" > "${left_log}" 2>&1) &
left_pid=$!
(MASTER_PORT="${RIGHT_PORT}" bash scripts/run_condition_ladder_full_ali.sh "${RIGHT_SOURCE}" "${RIGHT_GPUS}" > "${right_log}" 2>&1) &
right_pid=$!

set +e
wait "${left_pid}"
left_status=$?
wait "${right_pid}"
right_status=$?
set -e

if [ "${left_status}" -ne 0 ] || [ "${right_status}" -ne 0 ]; then
  echo "[ERROR] dual run failed left=${left_status} right=${right_status}" >&2
  exit 1
fi
