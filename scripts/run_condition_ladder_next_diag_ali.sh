#!/usr/bin/env bash
set -eo pipefail

cd /root/graph_research/huangjh/encoder
mkdir -p logs_condition_ladder_daemon

ts=$(date +%Y%m%d-%H%M%S)
pids=()

launch() {
  local name=$1
  local source=$2
  local gpus=$3
  local port=$4
  shift 4
  local log="logs_condition_ladder_daemon/diag_${name}_${ts}.log"
  echo "[INFO] launch ${name} source=${source} gpus=${gpus} log=${log}"
  (MASTER_PORT="${port}" bash scripts/run_condition_ladder_diag_ali.sh "${source}" "${gpus}" "${MAX_ITERS:-6000}" "${name}" "$@" > "${log}" 2>&1) &
  pids+=("$!")
}

wait_group() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  pids=()
  if [ "${status}" -ne 0 ]; then
    echo "[ERROR] one or more diagnostic jobs failed" >&2
    exit "${status}"
  fi
}

launch property_seed2026 property 0,1,2,3 29651 --set train.seed=2026
launch grale_seed2026 grale_official 4,5,6,7 29652 --set train.seed=2026
wait_group

launch grale_no_norm grale_official 0,1,2,3 29653 --set train.seed=2026 --set condition.normalize_grale=false
launch grale_big_adapter grale_official 4,5,6,7 29654 --set train.seed=2026 --set condition.adapter_hidden_dim=1024 --set condition.adapter_dropout=0.0
wait_group
