#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import logging
import os
import random
from pathlib import Path

import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_smiles(data_csv: str):
    opener = gzip.open if data_csv.endswith(".gz") else open
    smiles = []
    with opener(data_csv, "rt", newline="") as f:
        reader = csv.DictReader(f)
        if "smiles" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'smiles' column, got %s" % reader.fieldnames)
        for row in reader:
            smiles.append(row["smiles"])
    return smiles


def main():
    parser = argparse.ArgumentParser("Check GRALE cache idx/SMILES/embedding alignment.")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--pos_idx_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    setup_logging()
    cache_dir = Path(args.cache_dir)
    emb_path = cache_dir / "grale_embedding_float32.npy"
    valid_path = cache_dir / "valid_uint8.npy"
    meta_path = cache_dir / "meta.json"
    for path in [emb_path, valid_path, meta_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    emb = np.load(emb_path, mmap_mode="r")
    valid = np.load(valid_path, mmap_mode="r")
    pos_idx = np.load(args.pos_idx_path, mmap_mode="r")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if emb.shape[0] != valid.shape[0]:
        raise ValueError("embedding rows %d != valid rows %d" % (emb.shape[0], valid.shape[0]))
    if emb.ndim != 2:
        raise ValueError("embedding cache must be rank-2, got %s" % (emb.shape,))

    logging.info("loading SMILES CSV for alignment checks: %s", args.data_csv)
    smiles = read_smiles(args.data_csv)

    valid_idx = np.flatnonzero(np.asarray(valid) > 0)
    if len(valid_idx) == 0:
        raise RuntimeError("GRALE cache has no valid rows")
    rng = random.Random(args.seed)
    sample = rng.sample(list(map(int, valid_idx)), k=min(args.num_samples, len(valid_idx)))

    missing_from_pos = 0
    nonfinite = 0
    bad_smiles = 0
    pos_idx_set = set(map(int, np.asarray(pos_idx)))
    norms = []
    examples = []
    for idx in sample:
        if idx >= len(smiles):
            bad_smiles += 1
            continue
        if idx not in pos_idx_set:
            missing_from_pos += 1
        row = np.asarray(emb[idx], dtype=np.float32)
        finite = bool(np.isfinite(row).all())
        if not finite:
            nonfinite += 1
        norms.append(float(np.linalg.norm(row)))
        if len(examples) < 5:
            examples.append({"idx": idx, "smiles": smiles[idx], "norm": norms[-1] if finite else float("nan")})

    if bad_smiles or missing_from_pos or nonfinite:
        raise RuntimeError(
            "alignment check failed bad_smiles=%d missing_from_pos=%d nonfinite=%d"
            % (bad_smiles, missing_from_pos, nonfinite)
        )

    logging.info(
        "alignment ok samples=%d emb_shape=%s valid=%d/%d coverage=%.6f norm_mean=%.4f norm_std=%.4f meta_coverage=%s",
        len(sample),
        tuple(emb.shape),
        int(np.asarray(valid).sum()),
        len(valid),
        float(np.asarray(valid).mean()),
        float(np.mean(norms)),
        float(np.std(norms)),
        meta.get("coverage"),
    )
    logging.info("examples=%s", examples)


if __name__ == "__main__":
    main()
