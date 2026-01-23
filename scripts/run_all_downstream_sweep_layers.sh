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

  # New session + nohup to ignore SIGHUP. Re-run this script with DETACHED=1.
  setsid nohup env DETACHED=1 bash "$0" "$@" >>"$DAEMON_LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "[DETACH] ok. pid=$(cat "$PIDFILE")"
  exit 0
fi

# If killed, try to kill the whole process group (safe when detached / job control).
cleanup() {
  echo "[SIGNAL] received, terminating process group..."
  # kill process group of this script (negative pid targets group)
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

CKPT_ROOT="outputs/checkpoints/training"
CKPT_NAME="best.pt"

# ✅ ckpt suffix: tag + "_20260109-180629"
CKPT_SUFFIX="20260121-191024"

# Sweep: en fixed, de range
EN=9
DE_MIN=4
DE_MAX=5

# GPUs: run 2 models in parallel each round (cuda:0-1)
GPUS=(0 1)

# Optional: bind CPU cores per GPU worker (disable if you don't want)
CPU_SETS=("0-47" "48-95")
USE_TASKSET=1   # 1=enable, 0=disable

# Logs root (we'll create per-model subdir)
RUN_LOGDIR="logs_downstream_sweep_per_gpu"
mkdir -p "${RUN_LOGDIR}"

# Results root
RESULT_ROOT="logs_embedding/embedded_cache_sweep_layers"
mkdir -p "${RESULT_ROOT}"

# ✅ add current time to result folder name (once per script run)
RUN_TS="$(date +"%Y_%m%d-%H%M%S")"   # e.g. 2026_0110-235959

# Extra args passed to downstream script (optional)
# Example: EXTRA_ARGS=(--override)
EXTRA_ARGS=()

# Perf knobs
EMBED_BS=64
NUM_WORKERS=1

# Optional: reduce warning spam
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# Limit CPU threads per process (recommended)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=MAX_PARALLEL
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT=2

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

# =========================
# 1) Collect datasets (dedup)
# =========================
declare -A BEST=()   # key=dataset_name_without_ext -> filepath

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

# =========================
# 2) Build model tags (en9_de3..en9_de6)
# =========================
MODEL_TAGS=()
for de in $(seq "${DE_MIN}" "${DE_MAX}"); do
  MODEL_TAGS+=("en${EN}_de${de}")
done

echo "[INFO] Models: ${MODEL_TAGS[*]}"
echo "[INFO] GPUs:   ${GPUS[*]} (run ${#GPUS[@]} models in parallel per round)"
echo "[INFO] RUN_TS: ${RUN_TS}"
echo "[INFO] CKPT_SUFFIX: ${CKPT_SUFFIX}"

# =========================
# 3) Worker: one GPU runs one model tag, loops all datasets sequentially
# =========================
run_one_model_on_one_gpu() {
  local tag="$1"      # e.g. en9_de3
  local gpu_id="$2"   # e.g. 0
  local cpu_set=""    # optional

  if [[ "${USE_TASKSET}" -eq 1 ]]; then
    cpu_set="${CPU_SETS[$gpu_id]}"
  fi

  local model_name="GraphGPS_Encoder_${tag}"

  # ✅ results dir includes tag + current time
  local run_tag="${tag}_${RUN_TS}"
  local out_dir="${RESULT_ROOT}/${run_tag}"

  local model_logdir="${RUN_LOGDIR}/${tag}"
  mkdir -p "${out_dir}" "${model_logdir}"

  # ✅ ckpt dir includes tag + CKPT_SUFFIX
  local ckpt_dir="${CKPT_ROOT}/${tag}_${CKPT_SUFFIX}"
  local ckpt_path="${ckpt_dir}/${CKPT_NAME}"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[SKIP] ${tag} missing ckpt: ${ckpt_path}"
    return 0
  fi

  echo "[MODEL-START] GPU ${gpu_id} -> ${tag} | ckpt=${ckpt_path} | out_dir=${out_dir}"

  for ds_path in "${DATASETS[@]}"; do
    local dataset_base
    dataset_base="$(basename "$ds_path" .json)"   # e.g. ogbg-moltoxcast

    local ds_name
    ds_name="$(sanitize_name "$ds_path")"

    local log_path="${model_logdir}/${ds_name}.gpu${gpu_id}.log"
    local expected_csv="${out_dir}/${dataset_base}/${model_name}_results.csv"
    local expected_joblib="${out_dir}/${dataset_base}/${model_name}.joblib"

    # ===== skip if already done (unless force rerun) =====
    if [[ "${dataset_base}" == "ogbg-moltoxcast" ]]; then
      echo "[FORCE] GPU ${gpu_id} | ${tag} | ${dataset_base} (rerun)"
      rm -f "${expected_csv}" || true
      # 如果你希望 toxcast 连 embedding 都重算，取消下一行注释：
      # rm -f "${expected_joblib}" || true
    else
      if [[ -s "${expected_csv}" ]]; then
        echo "[SKIP] GPU ${gpu_id} | ${tag} | ${dataset_base} (exists)"
        continue
      fi
    fi

    echo "[RUN] GPU ${gpu_id} | ${tag} | ${ds_name}"

    {
      echo "# CMD:"
      echo "CUDA_VISIBLE_DEVICES=${gpu_id} python ${SCRIPT} \\"
      echo "  --prepared_path ${ds_path} \\"
      echo "  --ckpt ${ckpt_path} \\"
      echo "  --device cuda:0 \\"
      echo "  --train_config ${TRAIN_CONFIG} \\"
      echo "  --out_dir ${out_dir} \\"
      echo "  --model_name ${model_name} \\"
      echo "  --embed_bs ${EMBED_BS} \\"
      echo "  --num_workers ${NUM_WORKERS} \\"
      echo "  --enlayer ${EN} \\"
      echo "  --delayer ${tag#en${EN}_de} \\"
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
          --ckpt "${ckpt_path}" \
          --device "cuda:0" \
          --train_config "${TRAIN_CONFIG}" \
          --out_dir "${out_dir}" \
          --model_name "${model_name}" \
          --embed_bs "${EMBED_BS}" \
          --num_workers "${NUM_WORKERS}" \
          --enlayer "${EN}" \
          --delayer "${tag#en${EN}_de}" \
          "${EXTRA_ARGS[@]}" >> "${log_path}" 2>&1
    else
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python "${SCRIPT}" \
          --prepared_path "${ds_path}" \
          --ckpt "${ckpt_path}" \
          --device "cuda:0" \
          --train_config "${TRAIN_CONFIG}" \
          --out_dir "${out_dir}" \
          --model_name "${model_name}" \
          --embed_bs "${EMBED_BS}" \
          --num_workers "${NUM_WORKERS}" \
          --enlayer "${EN}" \
          --delayer "${tag#en${EN}_de}" \
          "${EXTRA_ARGS[@]}" >> "${log_path}" 2>&1
    fi
  done

  echo "[MODEL-DONE] GPU ${gpu_id} <- ${tag}"
}

# =========================
# 4) Launch workers in rounds:
#    each round uses cuda0-1 to run 2 model tags in parallel, then wait
# =========================
BATCH_SIZE="${#GPUS[@]}"
TOTAL_MODELS="${#MODEL_TAGS[@]}"

fail=0
round=0

for ((start=0; start<TOTAL_MODELS; start+=BATCH_SIZE)); do
  round=$((round+1))
  echo "[ROUND ${round}] Launch models ${start}..$((start+BATCH_SIZE-1)) on GPUs ${GPUS[*]}"

  declare -a PIDS=()
  declare -a DESC=()

  for ((j=0; j<BATCH_SIZE; j++)); do
    idx=$((start+j))
    [[ $idx -lt $TOTAL_MODELS ]] || break

    tag="${MODEL_TAGS[$idx]}"
    gpu="${GPUS[$j]}"

    run_one_model_on_one_gpu "${tag}" "${gpu}" &
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
    echo "[ERROR] Some model workers failed in ROUND ${round}. Check logs under: ${RUN_LOGDIR}/en9_de*/"
    exit 1
  fi
done

echo "[INFO] All model benchmarks finished."

# =========================
# 5) Merge results per model into one CSV each
#    (merge under tag + RUN_TS)
# =========================
for tag in "${MODEL_TAGS[@]}"; do
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
  de="${tag#en${EN}_de}"

  for f in "${CSV_FILES[@]}"; do
    [[ -s "$f" ]] || { echo "[WARN] Skip empty file: $f"; continue; }
    if [[ $first -eq 1 ]]; then
      head -n 1 "$f" | awk -v OFS=',' '{print $0,"enlayer","delayer"}' >> "$tmpfile"
      tail -n +2 "$f" | awk -v en="${EN}" -v de="${de}" -v OFS=',' '{print $0,en,de}' >> "$tmpfile"
      first=0
    else
      tail -n +2 "$f" | awk -v en="${EN}" -v de="${de}" -v OFS=',' '{print $0,en,de}' >> "$tmpfile"
    fi
  done

  mv "$tmpfile" "${merged_csv}"
  echo "[DONE] Merged ${tag} -> ${merged_csv} | files=${#CSV_FILES[@]}"
done
