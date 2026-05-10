import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_target_smiles(data_csv: str, target_idx: np.ndarray):
    target_set = set(int(x) for x in target_idx.tolist())
    smiles_by_idx = {}
    try:
        import pandas as pd

        for chunk in pd.read_csv(data_csv, usecols=["idx", "smiles"], chunksize=200000):
            sub = chunk[chunk["idx"].isin(target_set)]
            for row in sub.itertuples(index=False):
                smiles_by_idx[int(row.idx)] = str(row.smiles)
            if len(smiles_by_idx) >= len(target_set):
                break
    except Exception:
        with gzip.open(data_csv, "rt", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split(",")
            idx_col = header.index("idx")
            smiles_col = header.index("smiles")
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) <= max(idx_col, smiles_col):
                    continue
                idx = int(parts[idx_col])
                if idx in target_set:
                    smiles_by_idx[idx] = parts[smiles_col]
                    if len(smiles_by_idx) >= len(target_set):
                        break
    missing = [int(i) for i in target_idx.tolist() if int(i) not in smiles_by_idx]
    if missing:
        raise RuntimeError("missing %d target SMILES, first=%s" % (len(missing), missing[:5]))
    return [smiles_by_idx[int(i)] for i in target_idx.tolist()]


def load_grale_embedder(benchmark_root: str, batch_size: int):
    root = Path(benchmark_root).resolve()
    wrapper_dir = root / "model_wrappers" / "grale"
    vendor_dir = wrapper_dir / "vendor" / "GRALE-main"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(wrapper_dir))
    sys.path.insert(0, str(vendor_dir))
    from wrapper import GRALEEmbedder

    return GRALEEmbedder(model_name="GRALE-128-32", batch_size=batch_size)


def shard_indices(pos_idx: np.ndarray, shard_id: int, num_shards: int, max_mols: int = None) -> np.ndarray:
    arr = pos_idx[int(shard_id)::int(num_shards)]
    if max_mols is not None:
        arr = arr[: int(max_mols)]
    return arr.astype(np.int64, copy=False)


def run_shard(args):
    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pos_idx = np.load(args.pos_idx_path).astype(np.int64)
    target_idx = shard_indices(pos_idx, args.shard_id, args.num_shards, args.max_mols)
    logging.info("shard=%d/%d target=%d", args.shard_id, args.num_shards, len(target_idx))
    smiles = read_target_smiles(args.data_csv, target_idx)
    embedder = load_grale_embedder(args.benchmark_root, args.batch_size)

    emb_rows = []
    valid_rows = []
    start = time.time()
    for start_i in range(0, len(smiles), int(args.batch_size)):
        stop_i = min(len(smiles), start_i + int(args.batch_size))
        batch_smiles = smiles[start_i:stop_i]
        emb = embedder.forward(batch_smiles).astype(np.float32, copy=False)
        valid = np.isfinite(emb).all(axis=1).astype(np.uint8)
        emb_rows.append(emb)
        valid_rows.append(valid)
        done = stop_i
        rate = done / max(time.time() - start, 1e-6)
        logging.info(
            "shard=%d progress=%d/%d valid=%d rate=%.1f/s",
            args.shard_id,
            done,
            len(smiles),
            int(sum(v.sum() for v in valid_rows)),
            rate,
        )

    emb_all = np.concatenate(emb_rows, axis=0) if emb_rows else np.zeros((0, 128), dtype=np.float32)
    valid_all = np.concatenate(valid_rows, axis=0) if valid_rows else np.zeros((0,), dtype=np.uint8)
    prefix = shard_dir / ("shard_%03d" % int(args.shard_id))
    np.save(str(prefix) + "_idx.npy", target_idx)
    np.save(str(prefix) + "_emb.npy", emb_all)
    np.save(str(prefix) + "_valid.npy", valid_all)
    with open(str(prefix) + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "shard_id": int(args.shard_id),
                "num_shards": int(args.num_shards),
                "count": int(target_idx.size),
                "valid": int(valid_all.sum()),
                "embedding_dim": int(emb_all.shape[1]) if emb_all.ndim == 2 else 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )
    logging.info("wrote shard prefix=%s valid=%d/%d", prefix, int(valid_all.sum()), int(target_idx.size))


def merge_shards(args):
    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    idx_files = sorted(shard_dir.glob("shard_*_idx.npy"))
    if not idx_files:
        raise FileNotFoundError("no shard idx files under %s" % shard_dir)

    all_idx = []
    all_emb = []
    all_valid = []
    for idx_path in idx_files:
        stem = idx_path.name.replace("_idx.npy", "")
        emb_path = shard_dir / (stem + "_emb.npy")
        valid_path = shard_dir / (stem + "_valid.npy")
        idx = np.load(idx_path).astype(np.int64)
        emb = np.load(emb_path).astype(np.float32)
        valid = np.load(valid_path).astype(np.uint8)
        if idx.shape[0] != emb.shape[0] or idx.shape[0] != valid.shape[0]:
            raise ValueError("shard shape mismatch for %s" % stem)
        all_idx.append(idx)
        all_emb.append(emb)
        all_valid.append(valid)

    idx_cat = np.concatenate(all_idx, axis=0)
    emb_cat = np.concatenate(all_emb, axis=0)
    valid_cat = np.concatenate(all_valid, axis=0)
    n_index = int(args.n_index) if args.n_index is not None else int(idx_cat.max()) + 1
    dim = int(emb_cat.shape[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_out = np.lib.format.open_memmap(
        out_dir / "grale_embedding_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_index, dim),
    )
    valid_out = np.lib.format.open_memmap(
        out_dir / "valid_uint8.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(n_index,),
    )
    emb_out[:] = np.nan
    valid_out[:] = 0
    emb_out[idx_cat] = emb_cat
    valid_out[idx_cat] = valid_cat
    mask = valid_cat > 0
    if mask.any():
        valid_emb = emb_cat[mask]
        mean = np.nanmean(valid_emb, axis=0).astype(np.float32)
        std = np.maximum(np.nanstd(valid_emb, axis=0).astype(np.float32), 1e-6)
    else:
        mean = np.zeros((dim,), dtype=np.float32)
        std = np.ones((dim,), dtype=np.float32)
    meta = {
        "version": "pcqm4m_grale_official_1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_root": os.path.abspath(args.benchmark_root),
        "data_csv": os.path.abspath(args.data_csv) if args.data_csv else None,
        "pos_idx_path": os.path.abspath(args.pos_idx_path) if args.pos_idx_path else None,
        "n_index": int(n_index),
        "target_total": int(idx_cat.size),
        "num_valid": int(valid_cat.sum()),
        "coverage": float(valid_cat.sum() / max(1, idx_cat.size)),
        "embedding_dim": int(dim),
        "embedding_mean": mean.astype(float).tolist(),
        "embedding_std": std.astype(float).tolist(),
        "model_name": "GRALE-128-32",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logging.info("merged target=%d valid=%d coverage=%.4f dir=%s", idx_cat.size, int(valid_cat.sum()), meta["coverage"], out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str)
    parser.add_argument("--pos_idx_path", type=str)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--benchmark_root", type=str, default="/root/graph_research/benchmarking_molecular_models-master")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_mols", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--n_index", type=int, default=None)
    args = parser.parse_args()
    setup_logging()
    if args.merge:
        merge_shards(args)
    else:
        if not args.data_csv or not args.pos_idx_path:
            raise ValueError("--data_csv and --pos_idx_path are required for shard mode")
        run_shard(args)


if __name__ == "__main__":
    main()
