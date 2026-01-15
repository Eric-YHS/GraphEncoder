#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZINC_preprocess.py (two-phase)

Phase A: download_raw
  - Crawl ZINC20 3D tranche directory listing under https://files.docking.org/3D/
  - Download *.sdf.gz files to local raw cache FIRST
  - No RDKit parsing at this stage

Phase B: preprocess_local
  - Read local *.sdf.gz files (offline)
  - For each molecule (protomer with 3D coords), build aligned 2D+3D record:
      x: [N,9] OGB atom features (PCQM4M-style, non-negative ints)
      edge_attr: [E,3] OGB bond features (non-negative ints)
      pos: [N,3] float16 (from SDF conformer 0, optional centering)
  - Save into LMDB

Output layout (root default /mnt2/datasets/ZINC):
  {root}/
    raw_zinc20_3d/                      <-- downloaded raw files
      AA/AAML/XXXXXX.xaa.sdf.gz ...
      manifest_urls.txt                 <-- planned URLs (deterministic after crawl+filter)
    zinc20_3d_{max_mols}_k1/            <-- processed dataset dir
      zinc_confs.lmdb
      meta.json
      state.json

Important:
- Use OGB features to match typical AtomEncoder/BondEncoder pipelines.
  Install: pip install ogb
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import pickle
import zlib

import lmdb
import numpy as np
import requests
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import threading
from requests.exceptions import HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed




try:
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
    _HAS_OGB = True
except Exception:
    _HAS_OGB = False


# -----------------------------
# Constants / HTTP utils
# -----------------------------
BASE_URL_DEFAULT = "https://files2.docking.org/3D/"
UA = {"User-Agent": "zinc20-3d-preprocess/2.0"}
HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(p: Path, obj: dict) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def http_get_text(url: str, timeout: Tuple[int, int], retries: int) -> str:
    last = None
    for _ in range(max(1, retries)):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
        except Exception as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"GET failed: {url} ({last})")

def parse_links(html: str) -> List[str]:
    return HREF_RE.findall(html)

def join_url(base: str, path: str) -> str:
    if not base.endswith("/"):
        base += "/"
    return base + path

def http_download(url: str, out_path: Path, timeout: Tuple[int, int], retries: int, chunk_size: int = 1 << 20) -> None:
    """
    Download with .part file + atomic rename. If file exists and >0, skip.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    last = None
    for _ in range(max(1, retries)):
        try:
            with requests.get(url, headers=UA, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                ensure_dir(out_path.parent)
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return
        except Exception as e:
            last = e
            if tmp.exists():
                tmp.unlink()
            time.sleep(1.0)
    raise RuntimeError(f"DOWNLOAD failed: {url} ({last})")


# -----------------------------
# Crawl ZINC 3D tree
# -----------------------------
def list_top_tranches(base_url: str, timeout: Tuple[int, int], retries: int) -> List[str]:
    html = http_get_text(base_url, timeout=timeout, retries=retries)
    links = parse_links(html)
    dirs = sorted({x for x in links if re.fullmatch(r"[A-Z]{2}/", x)})
    if not dirs:
        raise RuntimeError(f"No top tranches found at {base_url}")
    return dirs

def list_second_level_dirs(base_url: str, top: str, timeout: Tuple[int, int], retries: int) -> List[str]:
    url = join_url(join_url(base_url, top), "")
    html = http_get_text(url, timeout=timeout, retries=retries)
    links = parse_links(html)
    dirs = sorted({x for x in links if re.fullmatch(r"[A-Z]{4}/", x)})
    return dirs

def match_subdir_filters(sub4: str,
                         reactivity: str,
                         purch: str,
                         ph: str,
                         charge: str) -> bool:
    """
    sub4 like AAML: [reactivity][purch][pH][charge]
    Empty filter means allow all.
    """
    sub4 = sub4.rstrip("/")
    if len(sub4) != 4:
        return False
    r, p, h, c = sub4[0], sub4[1], sub4[2], sub4[3]
    if reactivity and (r not in set(reactivity)):
        return False
    if purch and (p not in set(purch)):
        return False
    if ph and (h not in set(ph)):
        return False
    if charge and (c not in set(charge)):
        return False
    return True



# -----------------------------
# Phase A: download raw
# -----------------------------
def local_path_for_url(raw_dir: Path, url: str) -> Path:
    """
    Map URL .../3D/AA/AAML/XXXXXX.xaa.sdf.gz -> raw_dir/AA/AAML/XXXXXX.xaa.sdf.gz
    """
    parts = url.split("/")
    # expect tail ... /3D/{top}/{sub4}/{file}
    fname = parts[-1]
    sub4 = parts[-2]
    top = parts[-3]
    return raw_dir / top / sub4 / fname

# ---------- NEW: fast randomized downloader (no per-subdir file listing) ----------

_thread_local = threading.local()

def _get_session():
    # one Session per thread for connection reuse
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s

def part_name(i: int) -> str:
    """
    ZINC uses split-like suffix: xaa, xab, ..., xaz, xba, ...
    Equivalent to prefix 'x' + two base-26 letters.
    Total 26*26 = 676 possible parts per directory.
    """
    if i < 0 or i >= 26 * 26:
        return ""
    hi = i // 26
    lo = i % 26
    return "x{}{}".format(chr(ord("a") + hi), chr(ord("a") + lo))

def http_download_maybe_404(url: str, out_path: Path, timeout: Tuple[int, int], retries: int, chunk_size: int = 1 << 20) -> str:
    """
    Return status: 'ok' | 'skip' | 'not_found' | 'fail'
    - skip: already exists
    - not_found: 404 (no retry)
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    last = None
    for attempt in range(max(1, retries)):
        try:
            sess = _get_session()
            with sess.get(url, headers=UA, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return "not_found"
                r.raise_for_status()

                ensure_dir(out_path.parent)
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            tmp.replace(out_path)
            return "ok"

        except HTTPError as e:
            # if 404 bubbles up here, treat as not_found
            try:
                if e.response is not None and e.response.status_code == 404:
                    return "not_found"
            except Exception:
                pass
            last = e
        except Exception as e:
            last = e

        # clean partial
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        time.sleep(0.5)

    return "fail"

def build_subdir_index(
    base_url: str,
    timeout: Tuple[int, int],
    retries: int,
    seed: int,
    shuffle: bool,
    top_tranches: str,
    reactivity: str,
    purch: str,
    ph: str,
    charge: str,
    index_workers: int = 16,
) -> List[Tuple[str, str]]:
    """
    并行索引二级目录：
      /3D/ -> AA/, AB/...
      /3D/AA/ -> AAML/, ...
    任何一个 top tranche 超时/失败 -> 直接跳过，不阻塞整体。
    """
    rnd = random.Random(seed)

    tops = list_top_tranches(base_url, timeout=timeout, retries=retries)
    if top_tranches.strip().upper() != "ALL":
        want = {x.strip().upper() for x in top_tranches.split(",") if x.strip()}
        tops = [t for t in tops if t.rstrip("/") in want]
    if not tops:
        raise RuntimeError("No matching top tranches. Check --top_tranches.")

    if shuffle:
        rnd.shuffle(tops)

    def fetch_one(top_dir: str) -> Tuple[str, List[str]]:
        # top_dir like "AA/"
        try:
            subdirs = list_second_level_dirs(base_url, top_dir, timeout=timeout, retries=retries)
            subdirs = [s for s in subdirs if match_subdir_filters(s, reactivity, purch, ph, charge)]
            return top_dir.rstrip("/"), [s.rstrip("/") for s in subdirs]
        except Exception:
            # 超时/限流/偶发失败 -> 直接返回空
            return top_dir.rstrip("/"), []

    pairs: List[Tuple[str, str]] = []
    # tqdm 进度条：你会看到索引在跑
    with ThreadPoolExecutor(max_workers=max(1, int(index_workers))) as ex:
        futures = [ex.submit(fetch_one, t) for t in tops]
        for fut in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True, desc="Index top tranches"):
            top, subs = fut.result()
            for sub4 in subs:
                pairs.append((top, sub4))

    if shuffle:
        rnd.shuffle(pairs)
    return pairs

def download_raw(
    root: Path,
    base_url: str,
    seed: int,
    shuffle: bool,
    timeout: Tuple[int, int],
    retries: int,
    top_tranches: str,
    reactivity: str,
    purch: str,
    ph: str,
    charge: str,
    max_files: int,
    num_download_workers: int,
) -> None:
    """
    Fast randomized download:
    - crawl only top tranche and second-level dirs once
    - for each (top, sub4) directory, try chunks in order: xaa, xab, ...
    - randomly sample directories so downloaded pool is mixed across tranches
    """
    raw_dir = root / "raw_zinc20_3d"
    ensure_dir(raw_dir)

    meta_path = raw_dir / "download_meta.json"
    urls_downloaded = raw_dir / "urls_downloaded.txt"

    # 1) build (top, sub4) index (THIS IS FAST ENOUGH)
    print("[download_raw] indexing tranche subdirs (no file listing) ...", flush=True)

    index_cache = raw_dir / "subdir_index_cache.json"
    if index_cache.exists():
        obj = json.loads(index_cache.read_text(encoding="utf-8"))
        pairs = [tuple(x) for x in obj.get("pairs", [])]
        print(f"[download_raw] loaded cached index: {len(pairs)} dirs", flush=True)
    else:
        # 让索引阶段并行，默认用 min(16, num_download_workers)
        pairs = build_subdir_index(
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            seed=seed,
            shuffle=shuffle,
            top_tranches=top_tranches,
            reactivity=reactivity,
            purch=purch,
            ph=ph,
            charge=charge,
            index_workers=min(16, int(num_download_workers)),
        )
        index_cache.write_text(json.dumps({"pairs": pairs}, indent=2), encoding="utf-8")
        print(f"[download_raw] cached index to: {index_cache}", flush=True)

    print(f"[download_raw] found {len(pairs)} tranche dirs", flush=True)

    print("[download_raw] found {} tranche dirs".format(len(pairs)), flush=True)

    if max_files <= 0:
        raise ValueError("--max_files must be > 0 in this downloader (number of .sdf.gz chunks to fetch).")

    # per directory next part index to try
    next_part: Dict[Tuple[str, str], int] = {p: 0 for p in pairs}
    active = list(pairs)

    rnd = random.Random(seed)

    save_json(meta_path, {
        "base_url": base_url,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "shuffle": shuffle,
        "filters": {
            "top_tranches": top_tranches,
            "reactivity": reactivity,
            "purch": purch,
            "ph": ph,
            "charge": charge,
            "max_files": max_files,
        },
        "raw_dir": str(raw_dir),
        "note": "Randomized tranche-dir sampling; no per-subdir file listing.",
    })

    # 2) download loop with thread pool
    done = 0
    skipped = 0
    failed = 0
    not_found = 0

    pbar = tqdm(total=max_files, dynamic_ncols=True, desc="Download raw .sdf.gz (randomized)")

    def make_task(pair: Tuple[str, str]) -> Optional[Tuple[Tuple[str, str], str, Path]]:
        top, sub4 = pair
        i = next_part.get(pair, 0)
        if i >= 26 * 26:
            return None
        pn = part_name(i)
        prefix = "{}{}".format(top, sub4)  # 6 letters
        url = "{}/{}/{}/{}.{}.sdf.gz".format(base_url.rstrip("/"), top, sub4, prefix, pn)
        lp = raw_dir / top / sub4 / "{}.{}.sdf.gz".format(prefix, pn)
        return (pair, url, lp)

    # avoid infinite loops if server throttles heavily
    max_attempts = max_files * 50
    attempts = 0

    with ThreadPoolExecutor(max_workers=max(1, int(num_download_workers))) as ex:
        while (done + skipped) < max_files and active and attempts < max_attempts:
            # sample a batch of distinct dirs
            batch = []
            used = set()
            target_batch = min(len(active), int(num_download_workers) * 2)

            while len(batch) < target_batch:
                pair = active[rnd.randrange(len(active))]
                if pair in used:
                    continue
                task = make_task(pair)
                if task is None:
                    used.add(pair)
                    continue
                used.add(pair)
                batch.append(task)

            if not batch:
                break

            futures = {}
            for pair, url, lp in batch:
                futures[ex.submit(http_download_maybe_404, url, lp, timeout, retries)] = (pair, url)

            for fut in as_completed(futures):
                pair, url = futures[fut]
                attempts += 1
                try:
                    status = fut.result()
                except Exception:
                    status = "fail"

                if status == "ok":
                    done += 1
                    next_part[pair] += 1
                    with urls_downloaded.open("a", encoding="utf-8") as f:
                        f.write(url + "\n")
                    pbar.update(1)

                elif status == "skip":
                    skipped += 1
                    next_part[pair] += 1
                    with urls_downloaded.open("a", encoding="utf-8") as f:
                        f.write(url + "\n")
                    pbar.update(1)

                elif status == "not_found":
                    # this directory has no more chunks -> deactivate
                    not_found += 1
                    # do NOT advance next_part; just remove dir
                    try:
                        active.remove(pair)
                    except ValueError:
                        pass

                else:
                    failed += 1
                    # transient failure: keep dir active; retry later without advancing part

                pbar.set_postfix(done=done, skipped=skipped, failed=failed, not_found=not_found, active_dirs=len(active))

                if (done + skipped) >= max_files:
                    break

    pbar.close()
    print("[OK] download finished. ok={}, skipped={}, failed={}, not_found_dirs_events={}".format(done, skipped, failed, not_found))
    print("[OK] raw_dir: {}".format(raw_dir))
    print("[OK] urls_downloaded: {}".format(urls_downloaded))

# -----------------------------
# Phase B: preprocess local raw -> LMDB
# -----------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    remove_hs: bool = True
    center_pos: bool = True
    max_atoms: int = 128
    store_smiles: bool = False
    store_zinc_id: bool = False
    compress: bool = False
    compress_level: int = 3

def open_lmdb(path: Path, map_size_gb: int, readonly: bool = False) -> lmdb.Environment:
    ensure_dir(path.parent)
    return lmdb.open(
        str(path),
        subdir=False,
        readonly=readonly,
        lock=not readonly,
        readahead=False,
        meminit=False,
        map_size=int(map_size_gb * (1024**3)) if not readonly else 0,
        max_dbs=1,
    )

def lmdb_get_len(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        v = txn.get(b"__len__")
        return int(v.decode("utf-8")) if v else 0

def lmdb_set_len(txn: lmdb.Transaction, n: int) -> None:
    txn.put(b"__len__", str(int(n)).encode("utf-8"))

def key_of(i: int) -> bytes:
    return f"{i:09d}".encode("ascii")

def sanitize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return mol
        except Exception:
            return None

def get_zinc_id_from_props(mol: Chem.Mol) -> Optional[str]:
    keys = ["zinc_id", "ZINC_ID", "ZINC", "ID", "Name", "_Name"]
    for k in keys:
        if mol.HasProp(k):
            v = mol.GetProp(k).strip()
            if v:
                return v
    try:
        v = mol.GetProp("_Name").strip()
        return v if v else None
    except Exception:
        return None

def mol_to_coords(mol: Chem.Mol, center: bool) -> Optional[np.ndarray]:
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(0)
    pos = np.asarray(conf.GetPositions(), dtype=np.float32)
    if center:
        pos = pos - pos.mean(axis=0, keepdims=True)
    return pos[None, ...].astype(np.float16)  # [1,N,3]

def mol_to_ogb_x9(mol: Chem.Mol) -> np.ndarray:
    if not _HAS_OGB:
        raise RuntimeError("ogb is required for OGB/PCQM4M-style features. Install: pip install ogb")
    x = [atom_to_feature_vector(a) for a in mol.GetAtoms()]  # len=9, non-negative
    return np.asarray(x, dtype=np.int16)

def mol_to_ogb_bonds_undir(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_OGB:
        raise RuntimeError("ogb is required for OGB/PCQM4M-style features. Install: pip install ogb")
    us, vs, feats = [], [], []
    for b in mol.GetBonds():
        us.append(b.GetBeginAtomIdx())
        vs.append(b.GetEndAtomIdx())
        feats.append(bond_to_feature_vector(b))  # len=3, non-negative
    if len(us) == 0:
        bonds = np.zeros((2, 0), dtype=np.uint16)
        ea = np.zeros((0, 3), dtype=np.uint8)
        return bonds, ea
    bonds = np.vstack([np.asarray(us, dtype=np.uint16), np.asarray(vs, dtype=np.uint16)])
    ea = np.asarray(feats, dtype=np.uint8)
    return bonds, ea

def mol_to_record(mol: Chem.Mol, cfg: PreprocessConfig) -> Optional[bytes]:
    mol = sanitize_mol(mol)
    if mol is None:
        return None

    if mol.GetNumConformers() == 0:
        return None

    if cfg.remove_hs:
        try:
            mol = Chem.RemoveHs(mol, sanitize=True)
        except Exception:
            return None

    n = mol.GetNumAtoms()
    if n == 0 or n > cfg.max_atoms:
        return None

    coords = mol_to_coords(mol, center=cfg.center_pos)
    if coords is None or coords.shape[1] != n:
        return None

    x = mol_to_ogb_x9(mol)
    bonds, edge_attr_undir = mol_to_ogb_bonds_undir(mol)

    smi = Chem.MolToSmiles(mol, canonical=True) if cfg.store_smiles else None
    zid = get_zinc_id_from_props(mol) if cfg.store_zinc_id else None

    payload = (x, bonds, edge_attr_undir, coords, smi, zid)
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if cfg.compress:
        raw = zlib.compress(raw, level=int(cfg.compress_level))
    return raw

def iter_mols_from_sdf_gz(path: Path) -> Iterator[Chem.Mol]:
    with gzip.open(path, "rb") as f:
        supplier = Chem.ForwardSDMolSupplier(f, sanitize=False, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            yield mol

def list_local_raw_files(raw_dir: Path) -> List[Path]:
    files = [p for p in raw_dir.rglob("*.sdf.gz") if p.is_file()]
    files.sort()
    return files

def preprocess_local(
    root: Path,
    max_mols: int,
    out_name: str,
    map_size_gb: int,
    commit_every: int,
    seed: int,
    shuffle_files: bool,
    resume: bool,
    part_idx: int,
    num_parts: int,
    cfg: PreprocessConfig,
) -> Path:
    raw_dir = root / "raw_zinc20_3d"
    if not raw_dir.exists():
        raise RuntimeError(f"raw_dir not found: {raw_dir}. Run download_raw first.")

    out_dir = root / (out_name.strip() or f"zinc20_3d_{max_mols}_k1")
    ensure_dir(out_dir)

    lmdb_path = out_dir / "zinc_confs.lmdb"
    meta_path = out_dir / "meta.json"
    state_path = out_dir / "state.json"

    state = load_json(state_path) if (resume and state_path.exists()) else {}
    rnd = random.Random(seed)

    files = list_local_raw_files(raw_dir)
    if not files:
        raise RuntimeError(f"No *.sdf.gz found under {raw_dir}.")

    # Optional partitioning: each process handles a subset of files by hash modulo
    if num_parts > 1:
        def pick(p: Path) -> bool:
            h = int.from_bytes(hashlib.blake2b(str(p).encode("utf-8"), digest_size=4).digest(), "little")
            return (h % num_parts) == part_idx
        files = [p for p in files if pick(p)]

    if shuffle_files:
        rnd.shuffle(files)

    RDLogger.DisableLog("rdApp.*")

    env = open_lmdb(lmdb_path, map_size_gb=map_size_gb, readonly=False)
    n_written = lmdb_get_len(env) if resume else 0

    # Resume pointers
    file_i = int(state.get("file_i", 0))
    mol_i_in_file = int(state.get("mol_i_in_file", 0))
    seen = int(state.get("seen", 0))
    fail = int(state.get("fail", 0))

    txn = env.begin(write=True)
    pbar = tqdm(total=max_mols, initial=n_written, dynamic_ncols=True, desc="Preprocess -> LMDB")

    def flush():
        save_json(state_path, {
            "file_i": file_i,
            "mol_i_in_file": mol_i_in_file,
            "n_written": n_written,
            "seen": seen,
            "fail": fail,
        })
        save_json(meta_path, {
            "root": str(root),
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
            "lmdb_path": str(lmdb_path),
            "max_mols": max_mols,
            "n_written": n_written,
            "cfg": asdict(cfg),
            "partition": {"part_idx": part_idx, "num_parts": num_parts},
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    try:
        for fi in range(file_i, len(files)):
            file_i = fi
            path = files[fi]

            mol_idx = 0
            for mol in iter_mols_from_sdf_gz(path):
                if n_written >= max_mols:
                    break

                if mol_i_in_file and mol_idx < mol_i_in_file:
                    mol_idx += 1
                    continue

                seen += 1
                rec = mol_to_record(mol, cfg)
                if rec is None:
                    fail += 1
                    mol_idx += 1
                    mol_i_in_file = mol_idx
                    if (seen % 5000) == 0:
                        pbar.set_postfix(seen=seen, fail=fail)
                    continue

                txn.put(key_of(n_written), rec)
                n_written += 1
                pbar.update(1)

                mol_idx += 1
                mol_i_in_file = mol_idx

                if (n_written % commit_every) == 0:
                    lmdb_set_len(txn, n_written)
                    txn.commit()
                    txn = env.begin(write=True)
                    flush()
                    pbar.set_postfix(seen=seen, fail=fail)

            # finished a file
            mol_i_in_file = 0
            flush()

            if n_written >= max_mols:
                break

        lmdb_set_len(txn, n_written)
        txn.commit()
        flush()
    finally:
        pbar.close()
        env.sync()
        env.close()

    print(f"[OK] wrote {n_written:,} molecules to {lmdb_path}")
    print(f"[OK] out_dir: {out_dir}")
    return out_dir


# -----------------------------
# Dataset (same contract as before)
# -----------------------------
class Zinc20_3D_LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str | Path, compressed: bool = False):
        super().__init__()
        self.lmdb_path = str(lmdb_path)
        self.compressed = compressed
        self._env = None
        self._length = None

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = open_lmdb(Path(self.lmdb_path), map_size_gb=1, readonly=True)
        return self._env

    def __len__(self) -> int:
        if self._length is None:
            self._length = lmdb_get_len(self._get_env())
        return int(self._length)

    def __getitem__(self, idx: int):
        env = self._get_env()
        with env.begin(write=False) as txn:
            raw = txn.get(key_of(int(idx)))
        if raw is None:
            raise IndexError(idx)
        if self.compressed:
            raw = zlib.decompress(raw)

        x, bonds, edge_attr_undir, coords, smi, zid = pickle.loads(raw)
        pos = coords[0].astype(np.float32)

        if bonds.shape[1] > 0:
            u = bonds[0].astype(np.int64)
            v = bonds[1].astype(np.int64)
            edge_index = np.concatenate([np.stack([u, v], 0), np.stack([v, u], 0)], axis=1)
            edge_attr = np.concatenate([edge_attr_undir, edge_attr_undir], axis=0).astype(np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 3), dtype=np.int64)

        data = Data(
            x=torch.from_numpy(x.astype(np.int64)),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            pos=torch.from_numpy(pos),
            y=torch.zeros((1,), dtype=torch.long),
        )
        if smi is not None:
            data.smiles = smi
        if zid is not None:
            data.zinc_id = zid
        return data


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download_raw", help="crawl & download raw .sdf.gz first")
    p_dl.add_argument("--root", type=str, default="/mnt2/datasets/ZINC")
    p_dl.add_argument("--base_url", type=str, default=BASE_URL_DEFAULT)
    p_dl.add_argument("--seed", type=int, default=2025)
    p_dl.add_argument("--shuffle", type=int, default=0)
    p_dl.add_argument("--top_tranches", type=str, default="ALL")
    p_dl.add_argument("--reactivity", type=str, default="")
    p_dl.add_argument("--purch", type=str, default="ABCDE")
    p_dl.add_argument("--ph", type=str, default="")
    p_dl.add_argument("--charge", type=str, default="")
    p_dl.add_argument("--max_files", type=int, default=0, help="0=all matched files")
    p_dl.add_argument("--num_download_workers", type=int, default=16)
    p_dl.add_argument("--connect_timeout", type=int, default=10)
    p_dl.add_argument("--read_timeout", type=int, default=180)
    p_dl.add_argument("--retries", type=int, default=3)

    p_pp = sub.add_parser("preprocess_local", help="offline preprocess local raw files -> LMDB")
    p_pp.add_argument("--root", type=str, default="/mnt2/datasets/ZINC")
    p_pp.add_argument("--max_mols", type=int, default=10000)
    p_pp.add_argument("--out_name", type=str, default="", help="optional output dir name under root")
    p_pp.add_argument("--map_size_gb", type=int, default=50)
    p_pp.add_argument("--commit_every", type=int, default=10000)
    p_pp.add_argument("--seed", type=int, default=2025)
    p_pp.add_argument("--shuffle_files", type=int, default=0)
    p_pp.add_argument("--resume", type=int, default=1)

    # optional partitioning for multi-process offline preprocess (still simple)
    p_pp.add_argument("--num_parts", type=int, default=1)
    p_pp.add_argument("--part_idx", type=int, default=0)

    p_pp.add_argument("--remove_hs", type=int, default=1)
    p_pp.add_argument("--center_pos", type=int, default=1)
    p_pp.add_argument("--max_atoms", type=int, default=128)
    p_pp.add_argument("--store_smiles", type=int, default=0)
    p_pp.add_argument("--store_zinc_id", type=int, default=1)
    p_pp.add_argument("--compress", type=int, default=0)
    p_pp.add_argument("--compress_level", type=int, default=3)

    args = ap.parse_args()
    root = Path(args.root)
    ensure_dir(root)

    if args.cmd == "download_raw":
        download_raw(
            root=root,
            base_url=args.base_url.rstrip("/") + "/",
            seed=int(args.seed),
            shuffle=bool(int(args.shuffle)),
            timeout=(int(args.connect_timeout), int(args.read_timeout)),
            retries=int(args.retries),
            top_tranches=args.top_tranches,
            reactivity=args.reactivity.strip().upper(),
            purch=args.purch.strip().upper(),
            ph=args.ph.strip().upper(),
            charge=args.charge.strip().upper(),
            max_files=int(args.max_files),
            num_download_workers=int(args.num_download_workers),
        )
        return

    if args.cmd == "preprocess_local":
        cfg = PreprocessConfig(
            remove_hs=bool(int(args.remove_hs)),
            center_pos=bool(int(args.center_pos)),
            max_atoms=int(args.max_atoms),
            store_smiles=bool(int(args.store_smiles)),
            store_zinc_id=bool(int(args.store_zinc_id)),
            compress=bool(int(args.compress)),
            compress_level=int(args.compress_level),
        )
        out_dir = preprocess_local(
            root=root,
            max_mols=int(args.max_mols),
            out_name=args.out_name,
            map_size_gb=int(args.map_size_gb),
            commit_every=int(args.commit_every),
            seed=int(args.seed),
            shuffle_files=bool(int(args.shuffle_files)),
            resume=bool(int(args.resume)),
            part_idx=int(args.part_idx),
            num_parts=int(args.num_parts),
            cfg=cfg,
        )
        print("\nTraining usage:")
        print("  from ZINC_preprocess import Zinc20_3D_LMDBDataset")
        print(f"  ds = Zinc20_3D_LMDBDataset('{out_dir / 'zinc_confs.lmdb'}', compressed={bool(int(args.compress))})")
        return


if __name__ == "__main__":
    main()
