#!/usr/bin/env bash
set -euo pipefail

# =========================
# 0) 基本配置（按需修改）
# =========================
ROOT="/mnt2/luyifeng/ZINC"
PY="python"
SCRIPT="scripts/ZINC_preprocess.py"

# 目标：先下载 1000 万 raw 分子，再处理到 600 万成功入库
TARGET_RAW_MOLS=10000000
TARGET_OK_MOLS=6000000

# 更快/更稳的源（推荐）
BASE_URL="https://files2.docking.org/3D/"

# 下载筛选
TOP_TRANCHES="ALL"
PURCH="ABCDE"
REACTIVITY=""
PH=""
CHARGE=""

# 随机种子（固定可复现）
SEED=2025

# 索引/下载阶段超时：小一些，避免卡死
CONNECT_TIMEOUT=5
READ_TIMEOUT=20
RETRIES=1

# 下载并发：建议 16~32（太大可能被限流）
NUM_DL_WORKERS=32

# 预处理写 LMDB 参数
MAP_SIZE_GB=400
COMMIT_EVERY=10000

# 分子处理参数
REMOVE_HS=1
CENTER_POS=1
MAX_ATOMS=128
STORE_SMILES=0
STORE_ZINC_ID=1

# 可选压缩
COMPRESS=0
COMPRESS_LEVEL=3

# 预处理断点续跑（download_raw 是“存在即跳过”，preprocess_local 才有 resume）
RESUME=1

# 线程环境变量（注意不要用反斜杠续行）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

RAW_DIR="${ROOT}/raw_zinc20_3d"
export RAW_DIR

# =========================
# 工具：统计 raw 分子数（数 $$$$，不依赖 RDKit）
# =========================
count_raw_mols() {
  ${PY} - <<'PY'
import os, gzip, pathlib, sys
raw_dir = pathlib.Path(os.environ["RAW_DIR"])
if not raw_dir.exists():
    print(0); sys.exit(0)
n = 0
files = list(raw_dir.rglob("*.sdf.gz"))
for p in files:
    try:
        with gzip.open(p, "rb") as f:
            for line in f:
                if line.startswith(b"$$$$"):
                    n += 1
    except Exception:
        # skip corrupted/incomplete file
        pass
print(n)
PY
}

# =========================
# 1) 下载 raw（随机），逐步加大 max_files，直到 raw >= 1000 万
# =========================
echo "==== [1/2] Download RAW first (randomized) ===="
mkdir -p "${ROOT}"

# 逐步扩大下载文件数：你可按磁盘/网络情况调整
MAX_FILES_LIST=(500 1000 2000 4000 6000 8000 10000 12000 15000)

for MAX_FILES in "${MAX_FILES_LIST[@]}"; do
  echo ""
  echo "[download_raw] max_files=${MAX_FILES}"

  ${PY} ${SCRIPT} download_raw \
    --root "${ROOT}" \
    --base_url "${BASE_URL}" \
    --seed "${SEED}" \
    --shuffle 1 \
    --top_tranches "${TOP_TRANCHES}" \
    --reactivity "${REACTIVITY}" \
    --purch "${PURCH}" \
    --ph "${PH}" \
    --charge "${CHARGE}" \
    --max_files "${MAX_FILES}" \
    --num_download_workers "${NUM_DL_WORKERS}" \
    --connect_timeout "${CONNECT_TIMEOUT}" \
    --read_timeout "${READ_TIMEOUT}" \
    --retries "${RETRIES}"

  echo ""
  echo "[count] counting raw molecules in ${RAW_DIR} ..."
  RAW_MOLS=$(count_raw_mols)
  echo "[count] raw_mols=${RAW_MOLS}"

  if [ "${RAW_MOLS}" -ge "${TARGET_RAW_MOLS}" ]; then
    echo "[OK] reached target_raw_mols=${TARGET_RAW_MOLS}"
    break
  else
    echo "[INFO] not enough raw mols yet; will increase max_files..."
  fi
done

echo ""
echo "Raw dir: ${RAW_DIR}"
echo "Index cache (if enabled): ${RAW_DIR}/subdir_index_cache.json"
echo ""

# =========================
# 2) 离线处理到 LMDB（成功 600 万为止）
# =========================
echo "==== [2/2] Offline preprocess until ${TARGET_OK_MOLS} valid molecules ===="

${PY} ${SCRIPT} preprocess_local \
  --root "${ROOT}" \
  --max_mols "${TARGET_OK_MOLS}" \
  --map_size_gb "${MAP_SIZE_GB}" \
  --commit_every "${COMMIT_EVERY}" \
  --seed "${SEED}" \
  --shuffle_files 1 \
  --resume "${RESUME}" \
  --remove_hs "${REMOVE_HS}" \
  --center_pos "${CENTER_POS}" \
  --max_atoms "${MAX_ATOMS}" \
  --store_smiles "${STORE_SMILES}" \
  --store_zinc_id "${STORE_ZINC_ID}" \
  --compress "${COMPRESS}" \
  --compress_level "${COMPRESS_LEVEL}"

echo ""
echo "==== Done ===="
echo "LMDB: ${ROOT}/zinc20_3d_${TARGET_OK_MOLS}_k1/zinc_confs.lmdb"



