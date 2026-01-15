#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config
# =========================
PREPARED_DIR="data/prepared"
SCRIPT="scripts/downstream_benchmark_port.py"

# GPUs to use (CUDA_VISIBLE_DEVICES will be set per job)
GPUS=(0 1 2 3)
MAX_PARALLEL=${#GPUS[@]}

# Logs
RUN_LOGDIR="logs_downstream"
mkdir -p "${RUN_LOGDIR}"

# Results (your current layout)
RESULT_ROOT="logs_embedding/embedded_cache"
RESULT_GLOB="${RESULT_ROOT}/*/GraphGPS_Encoder_results.csv"
MERGED_CSV="${RESULT_ROOT}/ALL_GraphGPS_Encoder_results.csv"

# Extra args passed to downstream script (optional)
EXTRA_ARGS=()   # e.g. EXTRA_ARGS=(--override)

# =========================
# Helpers
# =========================
sanitize_name() {
  local s="$1"
  s="$(basename "$s")"
  # replace spaces and weird chars to underscores
  s="${s// /_}"
  s="${s//\//_}"
  echo "$s"
}

# =========================
# 1) Collect datasets (dedup, prefer .joblib over .json)
# =========================
declare -A BEST=()   # key=dataset_name_without_ext -> filepath

# collect joblib first (preferred)
for f in "${PREPARED_DIR}"/*.json; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  name="${base%.joblib}"
  BEST["$name"]="$f"
done

# # fill missing with json
# for f in "${PREPARED_DIR}"/*.joblib; do
#   [[ -e "$f" ]] || continue
#   base="$(basename "$f")"
#   name="${base%.json}"
#   if [[ -z "${BEST[$name]+x}" ]]; then
#     BEST["$name"]="$f"
#   fi
# done

# materialize
DATASETS=()
for k in "${!BEST[@]}"; do
  DATASETS+=("${BEST[$k]}")
done

# sort for deterministic order
IFS=$'\n' DATASETS=($(printf "%s\n" "${DATASETS[@]}" | sort))
unset IFS

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[ERROR] No prepared datasets found under: ${PREPARED_DIR}"
  exit 1
fi

echo "[INFO] Found ${#DATASETS[@]} unique datasets under ${PREPARED_DIR} (prefer json)"


# =========================
# 2) Run jobs with GPU round-robin + concurrency limit
# =========================
# We'll implement a simple job queue:
# - launch at most MAX_PARALLEL jobs concurrently
# - each job uses one GPU via CUDA_VISIBLE_DEVICES
# - record PIDs and wait them out

declare -a PIDS=()
declare -a JOB_NAMES=()
declare -a JOB_GPU=()
declare -a JOB_LOG=()

job_count=0
idx_gpu=0

launch_job() {
  local ds_path="$1"
  local gpu_id="$2"

  local ds_name
  ds_name="$(sanitize_name "$ds_path")"
  local log_path="${RUN_LOGDIR}/${ds_name}.log"

  echo "[LAUNCH] GPU ${gpu_id} -> ${ds_name}"
  echo "# CMD: CUDA_VISIBLE_DEVICES=${gpu_id} python ${SCRIPT} --prepared_path ${ds_path} ${EXTRA_ARGS[*]}" > "${log_path}"
  echo "" >> "${log_path}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    python "${SCRIPT}" --prepared_path "${ds_path}" "${EXTRA_ARGS[@]}" >> "${log_path}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  JOB_NAMES+=("${ds_name}")
  JOB_GPU+=("${gpu_id}")
  JOB_LOG+=("${log_path}")
}

# Wait one job slot (waits for the oldest pid)
wait_one() {
  local pid="${PIDS[0]}"
  local name="${JOB_NAMES[0]}"
  local gpu="${JOB_GPU[0]}"
  local log_path="${JOB_LOG[0]}"

  if wait "${pid}"; then
    echo "[DONE] GPU ${gpu} <- ${name}"
  else
    echo "[FAIL] GPU ${gpu} <- ${name} | log: ${log_path}"
  fi

  # pop front
  PIDS=("${PIDS[@]:1}")
  JOB_NAMES=("${JOB_NAMES[@]:1}")
  JOB_GPU=("${JOB_GPU[@]:1}")
  JOB_LOG=("${JOB_LOG[@]:1}")
}

for ds_path in "${DATASETS[@]}"; do
  gpu_id="${GPUS[$idx_gpu]}"
  idx_gpu=$(( (idx_gpu + 1) % ${#GPUS[@]} ))

  # if slots full, wait one
  while [[ ${#PIDS[@]} -ge ${MAX_PARALLEL} ]]; do
    wait_one
  done

  launch_job "${ds_path}" "${gpu_id}"
  job_count=$((job_count + 1))
done

# wait remaining
while [[ ${#PIDS[@]} -gt 0 ]]; do
  wait_one
done

echo "[INFO] All downstream jobs finished."

# =========================
# 3) Merge results to a single CSV
# =========================
echo "[INFO] Merging per-dataset CSVs from: ${RESULT_GLOB}"

# Safety: ensure result root exists
mkdir -p "${RESULT_ROOT}"

# find csvs
mapfile -t CSV_FILES < <(ls -1 ${RESULT_GLOB} 2>/dev/null || true)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
  echo "[WARN] No per-dataset result CSV found at: ${RESULT_GLOB}"
  echo "[WARN] Nothing to merge. Exiting."
  exit 0
fi

# merge with single header
tmpfile="$(mktemp)"
first=1
for f in "${CSV_FILES[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "[WARN] Skip empty file: $f"
    continue
  fi
  if [[ $first -eq 1 ]]; then
    cat "$f" >> "$tmpfile"
    first=0
  else
    # skip header line
    tail -n +2 "$f" >> "$tmpfile"
  fi
done

mv "$tmpfile" "${MERGED_CSV}"
echo "[DONE] Merged CSV saved to: ${MERGED_CSV}"
echo "[INFO] Total CSV files merged: ${#CSV_FILES[@]}"
