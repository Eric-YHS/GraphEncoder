#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dataset-parallel sweep (config-serial):
# - For each CONFIG:
#     run ALL datasets with max parallelism (dataset-parallel)
# - Finish this CONFIG -> move to next CONFIG
#
# Supports --detach to run in background even if terminal closes.
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

GPUS=(0 1 2 3)
MAX_PARALLEL=8

EMBED_BS=64
NUM_WORKERS=0

RUN_TS="$(date +"%Y_%m%d-%H%M%S")"
ROOT_DIR="./logs_downstream_sweep_runs/${RUN_TS}"
LOG_ROOT="${ROOT_DIR}/logs"
RESULT_ROOT="${ROOT_DIR}/results"
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"

EXCLUDE_DATASETS=(
)

export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT=2
export PYTHONUNBUFFERED=1

# -------------------------
# CONFIGS
# format: enlayer|delayer|encoder_name|denoiser_name|ckpt_date|pearl_fuse
# -------------------------
  # "9|5|cls_graphormer|uni_o2_condition|20260204-163653|concat"
  # "9|5|cls_pearl|uni_o2_condition|20260204-175227|concat"
  # "9|5|cls_pearl|uni_o2_condition|20260204-180020|add"
  # "9|5|cls_graphormer_pearl|uni_o2_condition|20260204-181909|concat"
  # "9|5|cls_graphormer|uni_o2_cat|20260207-095648|concat"
  # "9|5|cls_pearl|uni_o2_cat|20260207-100118|add"
  # "9|5|cls_pearl|uni_o2_cat|20260207-100118|concat"
  # "9|5|cls_graphormer_pearl|uni_o2_cat|20260207-095648|concat"
CONFIGS=(
  "9|5|cls_graphormer_pearl|uni_o2_condition|20260301-121017|concat"
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
  local out_dir="$1"
  local model_name="$2"
  local en="$3"
  local de="$4"
  local encoder_name="$5"
  local denoiser_name="$6"
  local ckpt_date="$7"
  local pearl_fuse="$8"
  local cfg_tag="$9"

  local merged_csv="${out_dir}/ALL_${model_name}_results.csv"

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
      head -n 1 "$f" | awk -v OFS=',' '{print $0,"enlayer","delayer","encoder_name","denoiser_name","ckpt_date","pearl_fuse","config_tag"}' >> "$tmpfile"
      tail -n +2 "$f" | awk -v OFS=',' \
        -v en="$en" -v de="$de" -v e="$encoder_name" -v d="$denoiser_name" -v c="$ckpt_date" -v p="$pearl_fuse" -v t="$cfg_tag" \
        '{print $0,en,de,e,d,c,p,t}' >> "$tmpfile"
      first=0
    else
      tail -n +2 "$f" | awk -v OFS=',' \
        -v en="$en" -v de="$de" -v e="$encoder_name" -v d="$denoiser_name" -v c="$ckpt_date" -v p="$pearl_fuse" -v t="$cfg_tag" \
        '{print $0,en,de,e,d,c,p,t}' >> "$tmpfile"
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
  local pearl_fuse="$7"
  local ds_path="$8"
  local gpu_id="$9"
  local out_dir="${10}"
  local log_path="${11}"
  local model_name="${12}"

  local ds_base
  ds_base="$(basename "${ds_path}" .json)"

  mkdir -p "$(dirname "${log_path}")" "${out_dir}"

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
    echo "  --pearl_fuse ${pearl_fuse} \\"
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
      --pearl_fuse "${pearl_fuse}" \
      >> "${log_path}" 2>&1

  echo "# END $(date)" >> "${log_path}"
}

# -------------------------
# 3) Run one config
# -------------------------
run_one_config() {
  local en="$1"
  local de="$2"
  local encoder_name="$3"
  local denoiser_name="$4"
  local ckpt_date="$5"
  local pearl_fuse="$6"

  # pearl_fuse 进 tag，避免 add/concat 覆盖同一目录/同一 model_name
  local cfg_tag_raw="en${en}_de${de}_e_${encoder_name}_d_${denoiser_name}_pf_${pearl_fuse}_${ckpt_date}"
  local cfg_tag
  cfg_tag="$(sanitize "${cfg_tag_raw}")"

  local out_dir="${RESULT_ROOT}/${cfg_tag}"
  local cfg_log_dir="${LOG_ROOT}/${cfg_tag}"
  mkdir -p "${out_dir}" "${cfg_log_dir}"

  local model_name="GraphGPS_Encoder_${cfg_tag}"

  echo
  echo "[CONFIG-START] ${cfg_tag_raw}"
  echo "  out_dir: ${out_dir}"
  echo "  log_dir: ${cfg_log_dir}"
  echo "  model_name: ${model_name}"
  echo

  declare -a PIDS=()
  local active=0
  local launched=0
  local fail=0

  for ds_path in "${DATASETS[@]}"; do
    local ds_base
    ds_base="$(basename "${ds_path}" .json)"

    local gpu_id="${GPUS[$((launched % ${#GPUS[@]}))]}"
    local log_path="${cfg_log_dir}/${ds_base}.gpu${gpu_id}.log"

    while [[ "${active}" -ge "${MAX_PARALLEL}" ]]; do
      if wait -n; then
        active=$((active-1))
      else
        active=$((active-1))
        fail=1
      fi
    done

    run_one_dataset_job \
      "${cfg_tag_raw}" \
      "${en}" "${de}" "${encoder_name}" "${denoiser_name}" "${ckpt_date}" "${pearl_fuse}" \
      "${ds_path}" "${gpu_id}" \
      "${out_dir}" "${log_path}" \
      "${model_name}" &

    PIDS+=("$!")
    active=$((active+1))
    launched=$((launched+1))
    echo "[LAUNCH] ${cfg_tag_raw} | ${ds_base} on gpu${gpu_id} | pid=${PIDS[-1]}"
  done

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

  merge_one_config_results \
    "${out_dir}" "${model_name}" \
    "${en}" "${de}" "${encoder_name}" "${denoiser_name}" "${ckpt_date}" "${pearl_fuse}" "${cfg_tag}"
}

# -------------------------
# 4) Main loop over CONFIGS
# -------------------------
overall_fail=0
idx=0

for cfg in "${CONFIGS[@]}"; do
  idx=$((idx+1))
  IFS='|' read -r en de encoder_name denoiser_name ckpt_date pearl_fuse <<< "${cfg}"
  pearl_fuse="${pearl_fuse:-none}"   # 兼容老的 5 段格式

  echo
  echo "============================================================"
  echo "[RUN] CONFIG ${idx}/${#CONFIGS[@]}: en=${en} de=${de} encoder=${encoder_name} denoiser=${denoiser_name} ckpt_date=${ckpt_date} pearl_fuse=${pearl_fuse}"
  echo "============================================================"

  if ! run_one_config "${en}" "${de}" "${encoder_name}" "${denoiser_name}" "${ckpt_date}" "${pearl_fuse}"; then
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