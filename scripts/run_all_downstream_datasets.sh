#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dataset-parallel sweep (config-serial):
# - For each CONFIG:
#     run ALL datasets with max parallelism (dataset-parallel)
# - Finish this CONFIG -> move to next CONFIG
#
# Supports --detach to run in background even if terminal closes.
#
# Usage:
#   chmod +x downstream_sweep_by_dataset.sh
#   ./downstream_sweep_by_dataset.sh
#   ./downstream_sweep_by_dataset.sh --detach
#
# Monitor:
#   tail -f ./logs_downstream_sweep/daemon.log
#
# Stop:
#   kill -TERM "$(cat ./logs_downstream_sweep/daemon.pid)"
# ============================================================

# -------------------------
# Detach wrapper
# -------------------------
DETACH=0
DAEMON_DIR="./logs_downstream_sweep"
DAEMON_LOG="${DAEMON_DIR}/daemon.log"
PIDFILE="${DAEMON_DIR}/daemon.pid"
REMAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=1
      shift
      ;;
    --daemon_dir)
      DAEMON_DIR="${2:?missing value for --daemon_dir}"
      DAEMON_LOG="${DAEMON_DIR}/daemon.log"
      PIDFILE="${DAEMON_DIR}/daemon.pid"
      shift 2
      ;;
    *)
      REMAIN_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${REMAIN_ARGS[@]}"

mkdir -p "${DAEMON_DIR}"

if [[ "${DETACH}" -eq 1 && "${DETACHED:-0}" -eq 0 ]]; then
  echo "[DETACH] launching in background..."
  echo "  daemon_dir: ${DAEMON_DIR}"
  echo "  daemon_log: ${DAEMON_LOG}"
  echo "  pidfile:    ${PIDFILE}"

  setsid nohup env DETACHED=1 bash "$0" "$@" >>"${DAEMON_LOG}" 2>&1 < /dev/null &
  echo $! > "${PIDFILE}"
  echo "[DETACH] ok. pid=$(cat "${PIDFILE}")"
  exit 0
fi

# Kill the whole process group (all background python jobs)
cleanup() {
  echo "[SIGNAL] received, terminating process group..."
  kill -- -$$ 2>/dev/null || true
  exit 1
}
trap cleanup INT TERM

# -------------------------
# User config (EDIT HERE)
# -------------------------
PREPARED_DIR="/mnt2/luyifeng/diff4MoleculeRepresentation/data/prepared"
SCRIPT="scripts/downstream_benchmark_port.py"

# GPUs to use (one dataset job gets one GPU; round-robin assignment)
GPUS=(1 2)            # e.g. (0 1 2 3)
MAX_PARALLEL=8        # max concurrent dataset jobs per config
                      # 建议 MAX_PARALLEL <= ${#GPUS[@]}，否则同一GPU会被多个任务抢

# Embedding performance knobs
EMBED_BS=256
NUM_WORKERS=0

# Results/logs root (统一放一起)
RUN_TS="$(date +"%Y_%m%d-%H%M%S")"
ROOT_DIR="./logs_downstream_sweep_runs/${RUN_TS}"
LOG_ROOT="${ROOT_DIR}/logs"
RESULT_ROOT="${ROOT_DIR}/results"
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"

# Skip some datasets if you want (按 prepared 文件名，不含 .json)
EXCLUDE_DATASETS=(

)


# Reduce warning spam (optional)
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# Limit CPU threads per process (optional, helps avoid CPU爆)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT=2
export PYTHONUNBUFFERED=1

# -------------------------
# CONFIGS: one line per config
# format: enlayer|delayer|encoder_name|denoiser_name|ckpt_date
# 你把下面示例改成你自己的 4 组即可
# -------------------------
CONFIGS=(
  "9|5|cls_graphormer|uni_o2_condition|20260204-163653"
  "9|5|cls_pearl|uni_o2_condition|20260204-175227"
  "9|5|cls_pearl|uni_o2_condition|20260204-180020"
  "9|5|cls_graphormer_pearl|uni_o2_condition|20260204-181909"
)

# -------------------------
# Helpers
# -------------------------
sanitize() {
  local s="$1"
  s="${s// /_}"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s//|/_}"
  s="${s//,/_}"
  echo "$s"
}

in_array() {
  local needle="$1"; shift
  local x
  for x in "$@"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}

merge_one_config_results() {
  local out_dir="$1"        # e.g. results/<cfg_tag>
  local model_name="$2"     # e.g. GraphGPS_Encoder_<cfg_tag>
  local en="$3"
  local de="$4"
  local encoder_name="$5"
  local denoiser_name="$6"
  local ckpt_date="$7"
  local cfg_tag="$8"

  local merged_csv="${out_dir}/ALL_${model_name}_results.csv"

  # 找到该 config 下每个数据集产出的 results.csv
  mapfile -t CSV_FILES < <(find "${out_dir}" -type f -name "${model_name}_results.csv" | sort)

  if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "[MERGE-WARN] No per-dataset result CSV found under: ${out_dir}"
    return 0
  fi

  local tmpfile
  tmpfile="$(mktemp)"

  local first=1
  for f in "${CSV_FILES[@]}"; do
    [[ -s "$f" ]] || { echo "[MERGE-WARN] Skip empty: $f"; continue; }
    if [[ $first -eq 1 ]]; then
      # header + extra columns
      head -n 1 "$f" | awk -v OFS=',' '{print $0,"enlayer","delayer","encoder_name","denoiser_name","ckpt_date","config_tag"}' >> "$tmpfile"
      tail -n +2 "$f" | awk -v OFS=',' -v en="$en" -v de="$de" -v e="$encoder_name" -v d="$denoiser_name" -v c="$ckpt_date" -v t="$cfg_tag" \
        '{print $0,en,de,e,d,c,t}' >> "$tmpfile"
      first=0
    else
      tail -n +2 "$f" | awk -v OFS=',' -v en="$en" -v de="$de" -v e="$encoder_name" -v d="$denoiser_name" -v c="$ckpt_date" -v t="$cfg_tag" \
        '{print $0,en,de,e,d,c,t}' >> "$tmpfile"
    fi
  done

  mv "$tmpfile" "${merged_csv}"
  echo "[MERGE-DONE] ${cfg_tag} -> ${merged_csv} | files=${#CSV_FILES[@]}"
}


# -------------------------
# 1) Collect datasets
# -------------------------
mapfile -t ALL_DS < <(ls -1 "${PREPARED_DIR}"/*.json 2>/dev/null | sort || true)
if [[ ${#ALL_DS[@]} -eq 0 ]]; then
  echo "[ERROR] No prepared datasets found under: ${PREPARED_DIR}"
  exit 1
fi

DATASETS=()
for p in "${ALL_DS[@]}"; do
  base="$(basename "$p" .json)"
  if in_array "$base" "${EXCLUDE_DATASETS[@]}"; then
    echo "[INFO] Exclude dataset: ${base}"
    continue
  fi
  DATASETS+=("$p")
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[ERROR] All datasets excluded; nothing to run."
  exit 1
fi

echo "[INFO] RUN_TS: ${RUN_TS}"
echo "[INFO] Found ${#DATASETS[@]} datasets under ${PREPARED_DIR}"
echo "[INFO] GPUs: ${GPUS[*]} | MAX_PARALLEL=${MAX_PARALLEL}"
echo "[INFO] Logs:    ${LOG_ROOT}"
echo "[INFO] Results: ${RESULT_ROOT}"

# Safety: avoid MAX_PARALLEL > number of GPUs by default
if [[ "${MAX_PARALLEL}" -gt "${#GPUS[@]}" ]]; then
  echo "[WARN] MAX_PARALLEL (${MAX_PARALLEL}) > #GPUs (${#GPUS[@]}). Multiple jobs may share the same GPU."
fi

# -------------------------
# 2) Run one dataset job (in background)
# -------------------------
run_one_dataset_job() {
  local cfg_tag="$1"
  local en="$2"
  local de="$3"
  local encoder_name="$4"
  local denoiser_name="$5"
  local ckpt_date="$6"
  local ds_path="$7"
  local gpu_id="$8"
  local out_dir="$9"
  local log_path="${10}"
  local model_name="${11}"

  local ds_base
  ds_base="$(basename "${ds_path}" .json)"

  mkdir -p "$(dirname "${log_path}")" "${out_dir}"

  # Skip if results already exist
  local expected_csv="${out_dir}/${ds_base}/${model_name}_results.csv"
  if [[ -s "${expected_csv}" ]]; then
    echo "[SKIP] ${cfg_tag} | ${ds_base} (exists: ${expected_csv})"
    return 0
  fi

  {
    echo "============================================================"
    echo "# START $(date)"
    echo "# CONFIG: ${cfg_tag}"
    echo "# DATASET: ${ds_base}"
    echo "# GPU: ${gpu_id}"
    echo "# OUT_DIR: ${out_dir}"
    echo "# EXPECTED_CSV: ${expected_csv}"
    echo "# CMD:"
    echo "CUDA_VISIBLE_DEVICES=${gpu_id} python ${SCRIPT} \\"
    echo "  --prepared_path ${ds_path} \\"
    echo "  --device cuda:0 \\"
    echo "  --out_dir ${out_dir} \\"
    echo "  --model_name ${model_name} \\"
    echo "  --embed_bs ${EMBED_BS} \\"
    echo "  --num_workers ${NUM_WORKERS} \\"
    echo "  --enlayer ${en} \\"
    echo "  --delayer ${de} \\"
    echo "  --encoder_name ${encoder_name} \\"
    echo "  --denoiser_name ${denoiser_name} \\"
    echo "  --ckpt_date ${ckpt_date} \\"
    echo "============================================================"
    echo
  } > "${log_path}"


  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python -u "${SCRIPT}" \
      --prepared_path "${ds_path}" \
      --device "cuda:0" \
      --out_dir "${out_dir}" \
      --model_name "${model_name}" \
      --embed_bs "${EMBED_BS}" \
      --num_workers "${NUM_WORKERS}" \
      --enlayer "${en}" \
      --delayer "${de}" \
      --encoder_name "${encoder_name}" \
      --denoiser_name "${denoiser_name}" \
      --ckpt_date "${ckpt_date}" \
      >> "${log_path}" 2>&1


  echo "# END $(date)" >> "${log_path}"
}

# -------------------------
# 3) Run one config: dataset-parallel with a max concurrency
# -------------------------
run_one_config() {
  local en="$1"
  local de="$2"
  local encoder_name="$3"
  local denoiser_name="$4"
  local ckpt_date="$5"

  local cfg_tag_raw="en${en}_de${de}_e_${encoder_name}_d_${denoiser_name}_${ckpt_date}"
  local cfg_tag
  cfg_tag="$(sanitize "${cfg_tag_raw}")"

  local out_dir="${RESULT_ROOT}/${cfg_tag}"
  local cfg_log_dir="${LOG_ROOT}/${cfg_tag}"
  mkdir -p "${out_dir}" "${cfg_log_dir}"

  # model_name 决定结果文件名，建议跟配置绑定，避免不同配置互相覆盖
  local model_name="GraphGPS_Encoder_${cfg_tag}"

  echo
  echo "[CONFIG-START] ${cfg_tag_raw}"
  echo "  out_dir: ${out_dir}"
  echo "  log_dir: ${cfg_log_dir}"
  echo "  model_name: ${model_name}"
  echo

  # Job control
  declare -a PIDS=()
  declare -a DESCS=()
  local active=0
  local launched=0
  local fail=0

  for ds_path in "${DATASETS[@]}"; do
    local ds_base
    ds_base="$(basename "${ds_path}" .json)"

    # pick GPU round-robin by launch index
    local gpu_id="${GPUS[$((launched % ${#GPUS[@]}))]}"
    local log_path="${cfg_log_dir}/${ds_base}.gpu${gpu_id}.log"

    # throttle: if too many active, wait for one to finish
    while [[ "${active}" -ge "${MAX_PARALLEL}" ]]; do
      if wait -n; then
        active=$((active-1))
      else
        active=$((active-1))
        fail=1
      fi
    done

    # launch job
    run_one_dataset_job \
      "${cfg_tag_raw}" \
      "${en}" "${de}" "${encoder_name}" "${denoiser_name}" "${ckpt_date}" \
      "${ds_path}" "${gpu_id}" \
      "${out_dir}" "${log_path}" \
      "${model_name}" &

    PIDS+=("$!")
    DESCS+=("${ds_base}@gpu${gpu_id}")
    active=$((active+1))
    launched=$((launched+1))
    echo "[LAUNCH] ${cfg_tag_raw} | ${ds_base} on gpu${gpu_id} | pid=${PIDS[-1]}"
  done

  # wait remaining
  for pid in "${PIDS[@]}"; do
    if wait "${pid}"; then
      :
    else
      fail=1
    fi
  done

  if [[ "${fail}" -ne 0 ]]; then
    echo "[CONFIG-FAIL] ${cfg_tag_raw} (some dataset jobs failed). Check logs: ${cfg_log_dir}"
    return 1
  fi

  echo "[CONFIG-DONE] ${cfg_tag_raw}"
  echo "  results: ${out_dir}"
  echo "  logs:    ${cfg_log_dir}"

  # ✅ merge all datasets into one CSV for this config
  merge_one_config_results \
    "${out_dir}" "${model_name}" \
    "${en}" "${de}" "${encoder_name}" "${denoiser_name}" "${ckpt_date}" "${cfg_tag}"

}

# -------------------------
# 4) Main loop over CONFIGS
# -------------------------
overall_fail=0
idx=0

for cfg in "${CONFIGS[@]}"; do
  idx=$((idx+1))
  IFS='|' read -r en de encoder_name denoiser_name ckpt_date <<< "${cfg}"

  echo
  echo "============================================================"
  echo "[RUN] CONFIG ${idx}/${#CONFIGS[@]}: en=${en} de=${de} encoder=${encoder_name} denoiser=${denoiser_name} ckpt_date=${ckpt_date}"
  echo "============================================================"

  if ! run_one_config "${en}" "${de}" "${encoder_name}" "${denoiser_name}" "${ckpt_date}"; then
    overall_fail=1
    break
  fi
done

if [[ "${overall_fail}" -ne 0 ]]; then
  echo "[ERROR] Sweep failed. Logs root: ${LOG_ROOT}"
  exit 1
fi

echo
echo "[INFO] All configs finished successfully."
echo "  Logs root:    ${LOG_ROOT}"
echo "  Results root: ${RESULT_ROOT}"
