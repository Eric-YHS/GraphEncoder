#!/usr/bin/env bash
set -eo pipefail

cd /root/graph_research/huangjh/encoder
source /root/miniconda3/etc/profile.d/conda.sh

DATA_ROOT=${DATA_ROOT:-/mnt2/luyifeng/diff4MoleculeRepresentation/data/PCQM4M}
BENCH_ROOT=${BENCH_ROOT:-/root/graph_research/benchmarking_molecular_models-master}
OUT_DIR=${OUT_DIR:-${DATA_ROOT}/pcqm4m-v2/grale_official_cache}
POS_IDX=${POS_IDX:-${DATA_ROOT}/pcqm4m-v2/pos_cache/train_idx_with_pos.npy}
DATA_CSV=${DATA_CSV:-${DATA_ROOT}/pcqm4m-v2/raw/data.csv.gz}
NUM_SHARDS=${NUM_SHARDS:-8}
GRALE_BATCH_SIZE=${GRALE_BATCH_SIZE:-64}

mkdir -p logs_condition_ladder_daemon "${OUT_DIR}/shards"
TS=$(date +"%Y%m%d-%H%M%S")

for GPU in $(seq 0 $((NUM_SHARDS - 1))); do
  LOG="logs_condition_ladder_daemon/grale_cache_shard${GPU}_${TS}.log"
  nohup bash -lc "
set -eo pipefail
cd /root/graph_research/huangjh/encoder
source /root/miniconda3/etc/profile.d/conda.sh
conda activate grale_bmm
export PYTHONPATH=${BENCH_ROOT}:${BENCH_ROOT}/model_wrappers/grale:${BENCH_ROOT}/model_wrappers/grale/vendor/GRALE-main:\${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${GPU}
python -u scripts/build_pcqm4m_grale_cache.py \\
  --benchmark_root \"${BENCH_ROOT}\" \\
  --data_csv \"${DATA_CSV}\" \\
  --pos_idx_path \"${POS_IDX}\" \\
  --out_dir \"${OUT_DIR}\" \\
  --batch_size \"${GRALE_BATCH_SIZE}\" \\
  --shard_id \"${GPU}\" \\
  --num_shards \"${NUM_SHARDS}\"
" > "${LOG}" 2>&1 < /dev/null &
  echo "[STARTED] grale shard=${GPU} log=${LOG}"
done

echo "[INFO] after all shards finish, merge with:"
echo "source /root/miniconda3/etc/profile.d/conda.sh && conda activate hjhencoder && cd /root/graph_research/huangjh/encoder && python scripts/build_pcqm4m_grale_cache.py --merge --benchmark_root ${BENCH_ROOT} --data_csv ${DATA_CSV} --pos_idx_path ${POS_IDX} --out_dir ${OUT_DIR}"
