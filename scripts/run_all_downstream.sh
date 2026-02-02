#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Detach wrapper: run this script in background even if terminal closes
# Usage:
#   ./downstream_sweep.sh --detach
#   ./downstream_sweep.sh --detach --daemon_log path --pidfile path
# Monitor:
#   tail -f downstream_sweep_daemon.log
# Stop:
#   kill -TERM "$(cat downstream_sweep_daemon.pid)"
# ============================================================

DETACH=0
DAEMON_LOG="./logs_downstream_sweep_daemon/log"
PIDFILE="./logs_downstream_sweep_daemon/pid"
REMAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      DETACH=1
      shift
      ;;
    --daemon_log)
      DAEMON_LOG="${2:?missing value for --daemon_log}"
      shift 2
      ;;
    --pidfile)
      PIDFILE="${2:?missing value for --pidfile}"
      shift 2
      ;;
    *)
      REMAIN_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${REMAIN_ARGS[@]}"

if [[ "${DETACH}" -eq 1 && "${DETACHED:-0}" -eq 0 ]]; then
  mkdir -p "$(dirname "$DAEMON_LOG")" "$(dirname "$PIDFILE")" 2>/dev/null || true
  echo "[DETACH] launching in background..."
  echo "  daemon_log: ${DAEMON_LOG}"
  echo "  pidfile:    ${PIDFILE}"

  setsid nohup env DETACHED=1 bash "$0" "$@" >>"$DAEMON_LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "[DETACH] ok. pid=$(cat "$PIDFILE")"
  exit 0
fi

cleanup() {
  echo "[SIGNAL] received, terminating process group..."
  kill -- -$$ 2>/dev/null || true
  exit 1
}
trap cleanup INT TERM

# =========================
# User config
# =========================
PREPARED_DIR="data/prepared"
SCRIPT="scripts/downstream_benchmark_port.py"
TRAIN_CONFIG="configs/training.yml"

# GPUs: run N configs in parallel each round (cuda:0..)
# GPUS=(0 1 2 3)
GPUS=(0)

# Optional: bind CPU cores per GPU worker (disable if you don't want)
# CPU_SETS=("0-23" "24-47" "48-71" "72-95")
CPU_SETS=("0-95")
USE_TASKSET=1   # 1=enable, 0=disable

# Logs root (we'll create per-config subdir)
RUN_LOGDIR="logs_downstream_sweep_per_gpu"
mkdir -p "${RUN_LOGDIR}"

# Results root
RESULT_ROOT="logs_embedding/embedded_cache_sweep"
mkdir -p "${RESULT_ROOT}"

# add current time to result folder name (once per script run)
RUN_TS="$(date +"%Y_%m%d-%H%M%S")"

# Extra args passed to downstream script (optional)
# Example: EXTRA_ARGS=(--override)
EXTRA_ARGS=()

# Perf knobs
EMBED_BS=256
NUM_WORKERS=8

# Optional: reduce warning spam
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# Limit CPU threads per process (recommended)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT=2

# ============================================================
# Sweep configs
# Format per item:
#   "enlayer,delayer,encoder_name,denoiser_name,ckpt_date"
#
# NOTE:
# - ckpt_date 要和你 python 脚本里 args.ckpt_date 一致（例如 20260127-143109）
# - encoder_name/denoiser_name 要和你训练/ckpt 目录命名一致
# - 你现在 python 脚本会自己拼：
#   ./outputs/checkpoints/training/en{en}_de{de}_e_{encoder}_d_{denoiser}_{ckpt_date}/best.pt
# ============================================================
SWEEPS=(
  # 例子（把下面替换成你真实要跑的组合）
  "9,5,cls_graphormer,uni_o2_condition,20260127-143109"
  # "9,5,cls_graphormer,uni_o2_cat,20260127-143134"
  # "9,5,cls_gps,uni_o2_condition,20260127-143204"
  # "9,5,cls_gps,uni_o2_cat,20260127-143209"
)

# =========================
# Helpers
# =========================
sanitize_name() {
  local s="$1"
  s="$(basename "$s")"
  s="${s// /_}"
  s="${s//\//_}"
  echo "$s"
}

cfg_tag() {
  # input: en,de,encoder,denoiser,ckpt_date
  local en="$1" de="$2" enc="$3" den="$4" date="$5"
  echo "en${en}_de${de}_e_${enc}_d_${den}_${date}"
}

# =========================
# 1) Collect datasets (dedup)
# =========================
declare -A BEST=()

for f in "${PREPARED_DIR}"/*.json; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  name="${base%.json}"
  BEST["$name"]="$f"
done

DATASETS=()
for k in "${!BEST[@]}"; do
  DATASETS+=("${BEST[$k]}")
done

IFS=$'\n' DATASETS=($(printf "%s\n" "${DATASETS[@]}" | sort))
unset IFS

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[ERROR] No prepared datasets found under: ${PREPARED_DIR}"
  exit 1
fi

echo "[INFO] Found ${#DATASETS[@]} datasets under ${PREPARED_DIR}"
echo "[INFO] GPUs:   ${GPUS[*]} (run ${#GPUS[@]} configs in parallel per round)"
echo "[INFO] RUN_TS: ${RUN_TS}"
echo "[INFO] SWEEPS: ${#SWEEPS[@]} configs"

# =========================
# 2) Worker: one GPU runs one config, loops all datasets sequentially
# =========================
run_one_cfg_on_one_gpu() {
  local cfg="$1"      # "en,de,encoder,denoiser,ckpt_date"
  local gpu_id="$2"

  IFS=',' read -r EN DE ENC_NAME DEN_NAME CKPT_DATE <<< "${cfg}"
  unset IFS

  local cpu_set=""
  if [[ "${USE_TASKSET}" -eq 1 ]]; then
    cpu_set="${CPU_SETS[$gpu_id]}"
  fi

  local tag
  tag="$(cfg_tag "${EN}" "${DE}" "${ENC_NAME}" "${DEN_NAME}" "${CKPT_DATE}")"

  # embedder name written into results
  local model_name="GraphGPS_Encoder_${tag}"

  # results dir includes tag + current time (once per run)
  local run_tag="${tag}_${RUN_TS}"
  local out_dir="${RESULT_ROOT}/${run_tag}"

  local model_logdir="${RUN_LOGDIR}/${tag}"
  mkdir -p "${out_dir}" "${model_logdir}"

  # pre-check ckpt existence (match python's ckpt_path rule)
  local ckpt_dir="./outputs/checkpoints/training/en${EN}_de${DE}_e_${ENC_NAME}_d_${DEN_NAME}_${CKPT_DATE}"
  local ckpt_path="${ckpt_dir}/best.pt"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[SKIP] ${tag} missing ckpt: ${ckpt_path}"
    return 0
  fi

  echo "[CFG-START] GPU ${gpu_id} -> ${tag} | ckpt=${ckpt_path} | out_dir=${out_dir}"

  for ds_path in "${DATASETS[@]}"; do
    local dataset_base
    dataset_base="$(basename "$ds_path" .json)"

    local ds_name
    ds_name="$(sanitize_name "$ds_path")"

    local log_path="${model_logdir}/${ds_name}.gpu${gpu_id}.log"
    local expected_csv="${out_dir}/${dataset_base}/${model_name}_results.csv"

    # ===== skip if already done (unless force rerun) =====
    if [[ -s "${expected_csv}" ]]; then
      echo "[SKIP] GPU ${gpu_id} | ${tag} | ${dataset_base} (exists)"
      continue
    fi

    echo "[RUN] GPU ${gpu_id} | ${tag} | ${ds_name}"

    {
      echo "# CMD:"
      echo "CUDA_VISIBLE_DEVICES=${gpu_id} python ${SCRIPT} \\"
      echo "  --prepared_path ${ds_path} \\"
      echo "  --ckpt_date ${CKPT_DATE} \\"
      echo "  --device cuda:0 \\"
      echo "  --train_config ${TRAIN_CONFIG} \\"
      echo "  --out_dir ${out_dir} \\"
      echo "  --model_name ${model_name} \\"
      echo "  --embed_bs ${EMBED_BS} \\"
      echo "  --num_workers ${NUM_WORKERS} \\"
      echo "  --enlayer ${EN} \\"
      echo "  --delayer ${DE} \\"
      echo "  --encoder_name ${ENC_NAME} \\"
      echo "  --denoiser_name ${DEN_NAME} \\"
      if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        echo "  ${EXTRA_ARGS[*]}"
      fi
      echo ""
    } > "${log_path}"

    if [[ "${USE_TASKSET}" -eq 1 ]]; then
      taskset -c "${cpu_set}" \
        env CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python "${SCRIPT}" \
          --prepared_path "${ds_path}" \
          --ckpt_date "${CKPT_DATE}" \
          --device "cuda:0" \
          --train_config "${TRAIN_CONFIG}" \
          --out_dir "${out_dir}" \
          --model_name "${model_name}" \
          --embed_bs "${EMBED_BS}" \
          --num_workers "${NUM_WORKERS}" \
          --enlayer "${EN}" \
          --delayer "${DE}" \
          --encoder_name "${ENC_NAME}" \
          --denoiser_name "${DEN_NAME}" \
          "${EXTRA_ARGS[@]}" >> "${log_path}" 2>&1
    else
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python "${SCRIPT}" \
          --prepared_path "${ds_path}" \
          --ckpt_date "${CKPT_DATE}" \
          --device "cuda:0" \
          --train_config "${TRAIN_CONFIG}" \
          --out_dir "${out_dir}" \
          --model_name "${model_name}" \
          --embed_bs "${EMBED_BS}" \
          --num_workers "${NUM_WORKERS}" \
          --enlayer "${EN}" \
          --delayer "${DE}" \
          --encoder_name "${ENC_NAME}" \
          --denoiser_name "${DEN_NAME}" \
          "${EXTRA_ARGS[@]}" >> "${log_path}" 2>&1
    fi
  done

  echo "[CFG-DONE] GPU ${gpu_id} <- ${tag}"
}

# =========================
# 3) Launch workers in rounds
# =========================
BATCH_SIZE="${#GPUS[@]}"
TOTAL_CFGS="${#SWEEPS[@]}"

fail=0
round=0

for ((start=0; start<TOTAL_CFGS; start+=BATCH_SIZE)); do
  round=$((round+1))
  echo "[ROUND ${round}] Launch cfgs ${start}..$((start+BATCH_SIZE-1)) on GPUs ${GPUS[*]}"

  declare -a PIDS=()
  declare -a DESC=()

  for ((j=0; j<BATCH_SIZE; j++)); do
    idx=$((start+j))
    [[ $idx -lt $TOTAL_CFGS ]] || break

    cfg="${SWEEPS[$idx]}"
    gpu="${GPUS[$j]}"

    IFS=',' read -r EN DE ENC_NAME DEN_NAME CKPT_DATE <<< "${cfg}"
    unset IFS
    tag="$(cfg_tag "${EN}" "${DE}" "${ENC_NAME}" "${DEN_NAME}" "${CKPT_DATE}")"

    run_one_cfg_on_one_gpu "${cfg}" "${gpu}" &
    PIDS+=("$!")
    DESC+=("gpu${gpu}:${tag}")
    echo "[LAUNCH] ${DESC[-1]} pid=${PIDS[-1]}"
  done

  for i in $(seq 0 $((${#PIDS[@]} - 1))); do
    pid="${PIDS[$i]}"
    d="${DESC[$i]}"
    if wait "${pid}"; then
      echo "[DONE] ${d}"
    else
      echo "[FAIL] ${d}"
      fail=1
    fi
  done

  if [[ "${fail}" -ne 0 ]]; then
    echo "[ERROR] Some cfg workers failed in ROUND ${round}. Check logs under: ${RUN_LOGDIR}/"
    exit 1
  fi
done

echo "[INFO] All cfg benchmarks finished."

# =========================
# 4) Merge results per config into one CSV each
# =========================
for cfg in "${SWEEPS[@]}"; do
  IFS=',' read -r EN DE ENC_NAME DEN_NAME CKPT_DATE <<< "${cfg}"
  unset IFS

  tag="$(cfg_tag "${EN}" "${DE}" "${ENC_NAME}" "${DEN_NAME}" "${CKPT_DATE}")"
  model_name="GraphGPS_Encoder_${tag}"
  tag_root="${RESULT_ROOT}/${tag}_${RUN_TS}"
  merged_csv="${tag_root}/ALL_${model_name}_results.csv"

  if [[ ! -d "${tag_root}" ]]; then
    echo "[WARN] Skip merge for ${tag}: missing dir ${tag_root}"
    continue
  fi

  mapfile -t CSV_FILES < <(find "${tag_root}" -type f -name "${model_name}_results.csv" | sort)
  if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "[WARN] No per-dataset result CSV for ${tag} under: ${tag_root}"
    continue
  fi

  tmpfile="$(mktemp)"
  first=1

  for f in "${CSV_FILES[@]}"; do
    [[ -s "$f" ]] || { echo "[WARN] Skip empty file: $f"; continue; }
    if [[ $first -eq 1 ]]; then
      head -n 1 "$f" | awk -v OFS=',' \
        '{print $0,"enlayer","delayer","encoder_name","denoiser_name","ckpt_date"}' >> "$tmpfile"
      tail -n +2 "$f" | awk -v en="${EN}" -v de="${DE}" -v enc="${ENC_NAME}" -v den="${DEN_NAME}" -v date="${CKPT_DATE}" -v OFS=',' \
        '{print $0,en,de,enc,den,date}' >> "$tmpfile"
      first=0
    else
      tail -n +2 "$f" | awk -v en="${EN}" -v de="${DE}" -v enc="${ENC_NAME}" -v den="${DEN_NAME}" -v date="${CKPT_DATE}" -v OFS=',' \
        '{print $0,en,de,enc,den,date}' >> "$tmpfile"
    fi
  done

  mv "$tmpfile" "${merged_csv}"
  echo "[DONE] Merged ${tag} -> ${merged_csv} | files=${#CSV_FILES[@]}"
done
