# downstream_benchmark_port.py
# ------------------------------------------------------------
# Use prepared benchmark datasets (*.json / *.joblib) and a frozen
# GraphGPSEncoder (loaded from your best.pt) to produce embeddings,
# then train downstream heads (rf / ridge / knn) with GridSearchCV
# aligned to benchmarking_molecular_models.
# ------------------------------------------------------------

import os
import json
import lmdb
import zlib
import base64
import shutil
import argparse
import logging as log
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
from ogb.utils.features import get_bond_feature_dims

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.utils import to_undirected

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, roc_auc_score

from utils.builder import build_encoder, build_ckpt

# ====== optional skfp multioutput auroc (same as benchmark) ======
try:
    from skfp.metrics import multioutput_auroc_score
    _HAS_SKFP = True
except Exception:
    _HAS_SKFP = False
    multioutput_auroc_score = None



# ============================================================
#  Constants (match benchmark const.py)
# ============================================================
CV_SPLITS = 5
N_JOBS = 8
DEFAULT_MEMORY_WEIGHT = 1
VERBOSITY = 10

AVAILABLE_HEADS = ["rf", "ridge", "knn"]


# ============================================================
#  Legacy JSON hooks (match benchmark Dataset.deserialize_legacy)
# ============================================================

def proba_to_pos_matrix(y_pred_proba, n_tasks: int) -> np.ndarray:
    """
    Convert sklearn predict_proba output to [N, T] positive-class probability matrix.
    - single task: ndarray [N,2] -> [N,1]
    - multi task: list of [N,2] (len=T) -> [N,T]
    - sometimes ndarray [T,N,2] or [N,T,2] -> [N,T]
    """
    if isinstance(y_pred_proba, list):
        # list length T, each [N,2]
        return np.stack([p[:, 1] for p in y_pred_proba], axis=1)

    if isinstance(y_pred_proba, np.ndarray):
        if y_pred_proba.ndim == 2:
            # [N,2] or [N] (rare)
            if y_pred_proba.shape[1] == 2:
                return y_pred_proba[:, 1:2]
            return y_pred_proba.reshape(-1, 1)

        if y_pred_proba.ndim == 3:
            # [T,N,2] or [N,T,2]
            if y_pred_proba.shape[-1] != 2:
                raise ValueError(f"Expected last dim=2 for proba, got {y_pred_proba.shape}")
            if y_pred_proba.shape[0] == n_tasks:      # [T,N,2]
                return np.stack([y_pred_proba[i, :, 1] for i in range(n_tasks)], axis=1)
            if y_pred_proba.shape[1] == n_tasks:      # [N,T,2]
                return y_pred_proba[:, :, 1]

    raise ValueError(f"Unsupported y_pred_proba type/shape: {type(y_pred_proba)} / {getattr(y_pred_proba,'shape',None)}")

def to_jsonable(obj):
    # numpy scalar -> python scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # dict -> dict
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    # list/tuple -> list
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def _decode_base64_array(b64: str, dtype: str, shape: List[int]) -> np.ndarray:
    raw = base64.b64decode(b64.encode("utf-8"))
    arr = np.frombuffer(raw, dtype=np.dtype(dtype))
    return arr.reshape(shape)

def json_numpy_obj_hook(d: Dict[str, Any]) -> Any:
    if "__ndarray__" in d:
        return _decode_base64_array(d["__ndarray__"], d["dtype"], d["shape"])
    if "__torch_tensor__" in d:
        return _decode_base64_array(d["__torch_tensor__"], d["dtype"], d["shape"])
    if "__dataframe__" in d:
        payload = d["__dataframe__"]
        return pd.DataFrame(
            payload["data"],
            columns=payload["columns"],
            index=payload.get("index", None),
        )
    return d


# ============================================================
#  Dataset + EmbeddedDataset types (minimal benchmark-compatible)
# ============================================================
@dataclass
class Dataset:
    name: str
    task: Any
    data: Any     # pd.DataFrame
    splits: Any   # dict

    @classmethod
    def deserialize_legacy(cls, path: str) -> "Dataset":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f, object_hook=json_numpy_obj_hook)
        return cls(**obj)

    def filter_out_problematic_molecules(self, illegal_smiles: List[str]):
        if "smiles" not in self.data.columns:
            raise ValueError("Dataset does not contain a 'smiles' column.")

        # ensure split column exists (benchmark behavior)
        if "split" not in self.data.columns:
            log.warning("Dataset does not contain split in dataframe, fixing")
            self.data["split"] = "UNKNOWN"
            for k, indices in self.splits.items():
                idx_list = list(indices) if not isinstance(indices, list) else indices
                if len(idx_list) > 0:
                    self.data.iloc[idx_list, self.data.columns.get_loc("split")] = k

        initial = len(self.data)
        self.data = self.data[~self.data["smiles"].isin(illegal_smiles)].reset_index(drop=True)
        after = len(self.data)

        log.info(f"Filtered out {initial - after} problematic molecules from dataset '{self.name}'.")
        log.info("Fixing splits indices")

        self.splits = {
            k: np.where(self.data["split"] == k)[0].tolist()
            for k in self.splits.keys()
        }
        if "UNKNOWN" in self.splits:
            raise ValueError("Dataset contains 'UNKNOWN' split, which should not happen.")

    @property
    def labels(self) -> pd.DataFrame:
        id_cols = [c for c in self.data.columns if ("id" in c.lower() or "split" in c.lower())]
        return self.data.drop(columns=(["smiles", "graph"] + id_cols), errors="ignore")


@dataclass
class EmbeddedDataset:
    name: str
    task: str
    embedder: str
    X: np.ndarray
    y_np: np.ndarray
    splits: Dict[str, Union[List[int], np.ndarray]]


# ============================================================
#  Load prepared dataset (*.json / *.joblib)
# ============================================================
def load_prepared_dataset(prepared_path: str,
                          illegal_smiles_txt: Optional[str] = None) -> Dataset:
    if prepared_path.endswith(".json"):
        ds = Dataset.deserialize_legacy(prepared_path)
    elif prepared_path.endswith(".joblib"):
        ds = joblib.load(prepared_path)
        if not isinstance(ds, Dataset):
            # benchmark joblib usually stores Dataset object; if it's a dict, wrap
            if isinstance(ds, dict) and all(k in ds for k in ["name", "task", "data", "splits"]):
                ds = Dataset(**ds)
            else:
                raise TypeError(f"Unsupported joblib content type: {type(ds)}")
    else:
        raise ValueError("prepared_path must end with .json or .joblib")

    if illegal_smiles_txt is not None and os.path.exists(illegal_smiles_txt):
        with open(illegal_smiles_txt, "r", encoding="utf-8") as f:
            illegal = [line.strip() for line in f if line.strip()]
        log.info(f"Filtering illegal SMILES: {len(illegal)}")
        ds.filter_out_problematic_molecules(illegal)

    # normalize splits to list
    for k in ["train", "valid", "test"]:
        if k in ds.splits and not isinstance(ds.splits[k], list):
            ds.splits[k] = list(ds.splits[k])

    return ds


# ============================================================
#  Graph dict -> PyG Data
#  IMPORTANT: Prefer OGB smiles2graph fields (node_feat=9, edge_feat=3)
# ============================================================
def graph_dict_to_pyg_data(g: Dict[str, Any], row_id: int = -1) -> Data:
    # prefer ogb schema
    if "node_feat" in g and "edge_feat" in g and "edge_index" in g:
        node_feat = np.asarray(g["node_feat"])
        edge_feat = np.asarray(g["edge_feat"])
        edge_index = np.asarray(g["edge_index"])
        x = torch.from_numpy(node_feat).long()
        edge_attr = torch.from_numpy(edge_feat).long()
        edge_index = torch.from_numpy(edge_index).long()
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data._row_id = int(row_id)
        return data

    # fallback (not recommended if you want strict PCQM4M alignment)
    if "x" in g and "edge_index" in g:
        x = g["x"]
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = torch.from_numpy(np.asarray(x)).float()

        if "edge_attr" in g:
            ea = g["edge_attr"]
        elif "edge_feat" in g:
            ea = g["edge_feat"]
        else:
            ea = np.zeros((g["edge_index"].shape[1], 0), dtype=np.float32)

        if torch.is_tensor(ea):
            ea = ea.detach().cpu().numpy()
        edge_attr = torch.from_numpy(np.asarray(ea)).float()

        ei = g["edge_index"]
        if torch.is_tensor(ei):
            ei = ei.detach().cpu().numpy()
        edge_index = torch.from_numpy(np.asarray(ei)).long()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data._row_id = int(row_id)
        return data

    raise ValueError("Graph dict missing required fields.")

def _spd_np_to_bytes(a: np.ndarray) -> bytes:
    # a: uint8/uint16 etc.
    raw = a.tobytes(order="C")
    return zlib.compress(raw, level=3)

def _spd_bytes_to_np(blob: bytes, dtype: np.dtype, shape: Tuple[int, int]) -> np.ndarray:
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)

def shortest_path_dense_unweighted(num_nodes: int, edge_index: torch.Tensor, inf: int) -> np.ndarray:
    """
    返回 [n,n] 的最短路径长度（无权图），不可达为 inf。
    建议 inf = spd_max_dist + 1（Graphormer 的做法是 clamp 到 max+1）
    """
    # edge_index: [2, E] on CPU
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        if s != d:
            adj[s].append(d)

    dist = np.full((num_nodes, num_nodes), inf, dtype=np.uint16)
    for i in range(num_nodes):
        dist[i, i] = 0
        q = deque([i])
        while q:
            v = q.popleft()
            dv = dist[i, v]
            nv = dv + 1
            for nb in adj[v]:
                if dist[i, nb] == inf:
                    dist[i, nb] = nv
                    q.append(nb)
    return dist

def shortest_path_spd_and_edge_input(
    num_nodes: int,
    edge_index_undirected: torch.Tensor,  # [2,Eu] CPU
    edge_type_map: dict,                  # (u,v)->edge_type_id, both dirs filled
    spd_max_dist: int,
):
    """
    returns:
      spd: [n,n] uint16, INF=spd_max_dist+1
      edge_input: [n,n,K] uint16, K=spd_max_dist, 0 padding
    """
    n = int(num_nodes)
    INF = int(spd_max_dist) + 1
    K = int(spd_max_dist)

    src = edge_index_undirected[0].tolist()
    dst = edge_index_undirected[1].tolist()
    adj = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u].append(v)
    for u in range(n):
        adj[u].sort()

    spd = np.full((n, n), INF, dtype=np.uint16)
    edge_input = np.zeros((n, n, K), dtype=np.uint16)

    for s in range(n):
        dist = [INF] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])

        while q:
            u = q.popleft()
            du = dist[u]
            if du >= INF:
                continue
            nd = du + 1
            if nd > INF:
                continue
            for v in adj[u]:
                if dist[v] > nd:
                    dist[v] = nd
                    parent[v] = u
                    q.append(v)

        spd[s, :] = np.asarray(dist, dtype=np.uint16)

        # reconstruct one shortest path (BFS tree path)
        for t in range(n):
            d = dist[t]
            if d <= 0 or d > K or d >= INF:
                continue

            edges = []
            cur = t
            while cur != s:
                p = parent[cur]
                if p < 0:
                    edges = []
                    break
                edges.append((p, cur))
                cur = p
            edges.reverse()

            for k, (u, v) in enumerate(edges):
                edge_input[s, t, k] = np.uint16(edge_type_map.get((u, v), 0))

    return spd, edge_input
def pack_edge_type_id(edge_attr: torch.Tensor) -> torch.Tensor:
    """
    edge_attr: [E,3] long
    return: edge_type_id [E] long, in [1..num_edge_types], 0 reserved for padding.
    """
    dims = get_bond_feature_dims()  # len=3
    d0, d1, d2 = int(dims[0]), int(dims[1]), int(dims[2])

    ea = edge_attr.long()
    a0 = ea[:, 0].clamp(0, d0 - 1)
    a1 = ea[:, 1].clamp(0, d1 - 1)
    a2 = ea[:, 2].clamp(0, d2 - 1)

    # mixed radix packing (stable)
    et = a0 + a1 * d0 + a2 * (d0 * d1)   # 0..(d0*d1*d2-1)
    return (et + 1).long()              # 1..num_edge_types

def _edge_np_to_bytes(arr: np.ndarray) -> bytes:
    # arr: uint16 [n,n,K]
    raw = arr.tobytes(order="C")
    return zlib.compress(raw, level=3)

def _edge_bytes_to_np(blob: bytes, dtype, shape) -> np.ndarray:
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)

def get_spd_lmdb_path(prepared_path: str, spd_max_dist: int, cache_root: str = "data/prepared_spd_cache") -> str:
    os.makedirs(cache_root, exist_ok=True)
    name = os.path.splitext(os.path.basename(prepared_path))[0]  # AMES
    import hashlib
    h = hashlib.md5(os.path.abspath(prepared_path).encode()).hexdigest()[:8]
    return os.path.join(cache_root, f"{name}_{h}_spd_md{spd_max_dist}.lmdb")


def build_spd_lmdb_if_missing(prepared_path: str, ds: Dataset, spd_max_dist: int,
                              cache_root: str = "data/prepared_spd_cache",
                              map_size: int = 1 << 40):
    log.info("constructing spd + edge_input...")
    lmdb_path = get_spd_lmdb_path(prepared_path, spd_max_dist, cache_root)
    data_mdb = os.path.join(lmdb_path, "data.mdb")

    if os.path.exists(data_mdb):
        try:
            env_chk = lmdb.open(lmdb_path, subdir=True, readonly=True, lock=False)
            with env_chk.begin(write=False) as txn_chk:
                ok = (txn_chk.get(b"0") is not None and
                      txn_chk.get(b"edge:0") is not None and
                      txn_chk.get(b"__done__") == b"1" and
                      txn_chk.get(b"__edge_done__") == b"1")
            env_chk.close()
            if ok:
                return lmdb_path
        except Exception:
            pass
        shutil.rmtree(lmdb_path, ignore_errors=True)

    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=True, lock=True)

    INF = int(spd_max_dist) + 1
    dims = get_bond_feature_dims()
    num_edge_types = int(dims[0]) * int(dims[1]) * int(dims[2])

    txn = env.begin(write=True)
    txn.put(b"__meta_spd_max_dist__", str(int(spd_max_dist)).encode("utf-8"))
    txn.put(b"__meta_num_edge_types__", str(int(num_edge_types)).encode("utf-8"))
    txn.put(b"__meta_edge_max_hops__", str(int(spd_max_dist)).encode("utf-8"))  # K

    for i in range(len(ds.data)):
        g = ds.data.iloc[i]["graph"]
        data = graph_dict_to_pyg_data(g)

        ei = data.edge_index
        ea = data.edge_attr
        if not torch.is_tensor(ei):
            ei = torch.as_tensor(ei, dtype=torch.long)
        if not torch.is_tensor(ea):
            ea = torch.as_tensor(ea, dtype=torch.long)
        ei = ei.long().cpu()
        ea = ea.long().cpu()

        n = int(data.num_nodes)

        # edge type ids (1..T)
        edge_type_id = pack_edge_type_id(ea)

        # map (u,v)->type, both directions
        edge_type_map = {}
        for k in range(ei.size(1)):
            u = int(ei[0, k])
            v = int(ei[1, k])
            et = int(edge_type_id[k])
            edge_type_map.setdefault((u, v), et)
            edge_type_map.setdefault((v, u), et)

        ei_u = to_undirected(ei, num_nodes=n).cpu()

        spd, edge_input = shortest_path_spd_and_edge_input(
            num_nodes=n,
            edge_index_undirected=ei_u,
            edge_type_map=edge_type_map,
            spd_max_dist=int(spd_max_dist),
        )
        spd = np.minimum(spd, INF).astype(np.uint16)  # [n,n]
        # edge_input: [n,n,K] uint16, 0 padding

        # ---- sanity check: edge ids must fit vocab
        if edge_input.max() > num_edge_types:
            raise ValueError(f"edge_input id overflow at row {i}: max_id={edge_input.max()} > num_edge_types={num_edge_types}")

        spd_blob = _spd_np_to_bytes(spd)
        edge_blob = _edge_np_to_bytes(edge_input)

        txn.put(f"__shape__:{i}".encode("utf-8"), f"{n}".encode("utf-8"))
        txn.put(str(i).encode("utf-8"), spd_blob)
        txn.put(f"edge:{i}".encode("utf-8"), edge_blob)

        if (i + 1) % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)

        if (i + 1) % 5000 == 0:
            print(f"[SPD+EDGE-LMDB] built {i+1}/{len(ds.data)}")

    txn.put(b"__done__", b"1")
    txn.put(b"__edge_done__", b"1")
    txn.commit()
    env.sync()
    env.close()
    return lmdb_path

class CollateWithSPDLmdbDownstream:
    """
    collate_fn附加
      batch.spatial_pos_dense: [B,L,L]
      batch.edge_input_dense:  [B,L,L,K]  K=spd_max_dist
    """
    def __init__(self, spd_lmdb_path: str, spd_max_dist: int):
        self.spd_lmdb_path = spd_lmdb_path
        self.spd_max_dist = int(spd_max_dist)
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.spd_lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def __call__(self, batch_list: List[Data]):
        env = self._get_env()
        inf = self.spd_max_dist + 1
        K = self.spd_max_dist

        n_list = [int(d.num_nodes) for d in batch_list]
        L = max(n_list)
        B = len(batch_list)

        spd_dense = torch.full((B, L, L), fill_value=inf, dtype=torch.long)
        edge_dense = torch.zeros((B, L, L, K), dtype=torch.long)  # 0 padding

        with env.begin(write=False) as txn:
            for bi, d in enumerate(batch_list):
                i = int(d._row_id)
                n = int(d.num_nodes)

                blob = txn.get(str(i).encode("utf-8"))
                if blob is None:
                    raise KeyError(f"SPD not found in LMDB for row {i}")

                n_str = txn.get(f"__shape__:{i}".encode("utf-8"))
                if n_str is None:
                    raise KeyError(f"shape missing for row {i}")
                n2 = int(n_str.decode("utf-8"))
                if n2 != n:
                    raise ValueError(f"shape mismatch row {i}: lmdb_n={n2}, data_n={n}")

                spd = _spd_bytes_to_np(blob, dtype=np.uint16, shape=(n, n))
                spd = np.minimum(spd, inf)
                spd_dense[bi, :n, :n] = torch.from_numpy(spd.astype(np.int64))

                eblob = txn.get(f"edge:{i}".encode("utf-8"))
                if eblob is None:
                    raise KeyError(f"EDGE_INPUT not found in LMDB for row {i} (key=edge:{i})")

                edge_in = _edge_bytes_to_np(eblob, dtype=np.uint16, shape=(n, n, K))
                edge_dense[bi, :n, :n, :] = torch.from_numpy(edge_in.astype(np.int64))

        batch = Batch.from_data_list(batch_list)
        batch.spatial_pos_dense = spd_dense
        batch.edge_input_dense = edge_dense
        return batch

def infer_in_dims_from_dataset(ds: Dataset) -> Tuple[int, int]:
    g0 = ds.data.iloc[0]["graph"]
    # prefer ogb
    if "node_feat" in g0 and "edge_feat" in g0:
        return int(np.asarray(g0["node_feat"]).shape[1]), int(np.asarray(g0["edge_feat"]).shape[1])
    # fallback
    x0 = g0["x"]
    if torch.is_tensor(x0):
        x0 = x0.detach().cpu().numpy()
    e0 = g0.get("edge_attr", g0.get("edge_feat", None))
    if e0 is None:
        edge_in = 0
    else:
        if torch.is_tensor(e0):
            e0 = e0.detach().cpu().numpy()
        edge_in = int(np.asarray(e0).shape[1])
    return int(np.asarray(x0).shape[1]), edge_in


# ============================================================
#  Embed all molecules (frozen encoder)
# ============================================================
@torch.no_grad()
def embed_dataset_graph_level(
    encoder: torch.nn.Module,
    ds: Dataset,
    device: torch.device,
    prepared_path: str,
    batch_size: int = 256,
    num_workers: int = 8,
    spd_max_dist: int = 8,
    spd_cache_root: str = "data/prepared_spd_cache",
) -> np.ndarray:
    encoder.eval()

    # 1) build/load SPD lmdb
    spd_lmdb_path = build_spd_lmdb_if_missing(
        prepared_path=prepared_path,
        ds=ds,
        spd_max_dist=spd_max_dist,
        cache_root=spd_cache_root,
    )
    log.info(f"[SPD] lmdb_path={spd_lmdb_path} exists={os.path.exists(os.path.join(spd_lmdb_path,'data.mdb'))}")

    collate_fn = CollateWithSPDLmdbDownstream(spd_lmdb_path, spd_max_dist=spd_max_dist)

    # 2) build data_list with row_id
    data_list: List[Data] = []
    for i in range(len(ds.data)):
        g = ds.data.iloc[i]["graph"]
        data_list.append(graph_dict_to_pyg_data(g, row_id=i))

    loader = TorchDataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    X_chunks: List[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        node_emb, graph_emb = encoder(batch)   # (N,H), (B,H)
        X_chunks.append(graph_emb.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(X_chunks, axis=0)

# ============================================================
#  Benchmark heads (copy logic from benchmark models.py)
# ============================================================
def get_knn_distance(embeddings_dtype):
    if np.issubdtype(embeddings_dtype, np.integer):
        try:
            from skfp.distances import tanimoto_count_distance
            return tanimoto_count_distance
        except Exception:
            return "cosine"
    elif np.issubdtype(embeddings_dtype, np.floating):
        return "cosine"
    else:
        raise ValueError(f"Unsupported embeddings dtype: {embeddings_dtype}")

RF_CLF = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["entropy"],
}
RF_REG = {
    "clf__min_samples_split": np.arange(2, 11, 2),
    "clf__n_estimators": [500],
    "clf__criterion": ["squared_error"],
}
RIDGE__MULTIOUTPUT_CLF = {
    "clf__estimator__C": 1 / np.logspace(-2, 3, 10),
    "clf__estimator__penalty": ["l2"],
    "clf__estimator__solver": ["lbfgs"],
    "clf__estimator__max_iter": [5000],
}
RIDGE_CLF = {
    "clf__C": 1 / np.logspace(-2, 3, 10),
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"],
    "clf__max_iter": [5000],
}
RIDGE_REG = {
    "clf__alpha": np.logspace(-2, 3, 10),
    "clf__max_iter": [5000],
    "clf__solver": ["lbfgs"],
}
KNN_CLF = {"clf__n_neighbors": np.arange(1, 11, 2)}
KNN_REG = {"clf__n_neighbors": np.arange(1, 11, 2)}

def get_clf_models(no_output: int, embeddings_dtype):
    if no_output == 1:
        lr_clf = LogisticRegression(n_jobs=1)
        lr_params = RIDGE_CLF
    else:
        lr_clf = MultiOutputClassifier(LogisticRegression(n_jobs=1))
        lr_params = RIDGE__MULTIOUTPUT_CLF

    return {
        "rf": {"model": Pipeline([("clf", RandomForestClassifier(n_jobs=1))]), "params": RF_CLF.copy()},
        "ridge": {"model": Pipeline([("scaler", StandardScaler()), ("clf", lr_clf)]), "params": lr_params.copy()},
        "knn": {"model": Pipeline([("scaler", StandardScaler()),
                                  ("clf", KNeighborsClassifier(n_jobs=1, metric=get_knn_distance(embeddings_dtype)))]),
                "params": KNN_CLF.copy()},
    }

def get_reg_models(embeddings_dtype):
    return {
        "rf": {"model": Pipeline([("clf", RandomForestRegressor(n_jobs=1))]), "params": RF_REG.copy()},
        "ridge": {"model": Pipeline([("scaler", StandardScaler()), ("clf", Ridge())]), "params": RIDGE_REG.copy()},
        "knn": {"model": Pipeline([("scaler", StandardScaler()),
                                  ("clf", KNeighborsRegressor(n_jobs=1, metric=get_knn_distance(embeddings_dtype)))]),
                "params": KNN_REG.copy()},
    }


# ============================================================
#  Scorers (multioutput AUROC aligned to benchmark)
# ============================================================
def multioutput_auroc_from_proba(y_true, y_pred_proba) -> float:
    """
    Robust mean AUROC over tasks.
    - skips tasks with <2 classes in y_true (within this fold)
    - skips tasks where predicted proba is NaN
    - NEVER raises -> safe for GridSearchCV
    """
    y_true = np.asarray(y_true, dtype=float)

    # OGB-style missing label sometimes is -1
    y_true[y_true < 0] = np.nan

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    T = y_true.shape[1]
    probs = proba_to_pos_matrix(y_pred_proba, n_tasks=T)  # [N,T]

    scores = []
    for t in range(T):
        m = ~np.isnan(y_true[:, t])  # ignore missing labels if any
        if m.sum() == 0:
            continue

        yt = y_true[m, t]
        yp = probs[m, t]
        if np.all(np.isnan(yp)):
            continue

        # AUROC undefined if only one class in this fold
        if np.unique(yt).size < 2:
            continue

        try:
            scores.append(roc_auc_score(yt, yp))
        except Exception:
            # never crash CV
            continue

    return float(np.mean(scores)) if len(scores) else float("nan")

  
def fallback_multioutput_auroc(y_true: np.ndarray, y_pred_proba) -> float:
    """
    Robust multi-task AUROC:
    - Accept y_pred_proba: list([N,2])*T, or ndarray [T,N,2]/[N,T,2]/[N,T]/[N,2]
    - Handle missing labels: treat y_true < 0 as missing (OGB style), ignore in scoring
    """
    y_true = y_true.astype(float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    # OGB multi-task commonly uses -1 for missing labels
    y_true[y_true < 0] = np.nan

    probs = proba_to_pos_matrix(y_pred_proba, n_tasks=y_true.shape[1])  # [N,T]
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    scores = []
    T = y_true.shape[1]
    for k in range(T):
        yt = y_true[:, k]
        pk = probs[:, k]

        mask = ~np.isnan(yt)
        if mask.sum() < 2:
            continue

        yt_valid = yt[mask]
        pk_valid = pk[mask]

        # need at least 2 classes among valid labels
        uniq = np.unique(yt_valid)
        if uniq.size < 2:
            continue

        scores.append(roc_auc_score(yt_valid, pk_valid))

    return float(np.mean(scores)) if len(scores) else float("nan")


def get_train_data(dataset: EmbeddedDataset) -> Tuple[np.ndarray, np.ndarray]:
    tr = dataset.splits["train"]
    va = dataset.splits["valid"]
    tr = tr if isinstance(tr, list) else tr.tolist()
    va = va if isinstance(va, list) else va.tolist()
    idx = tr + va
    return dataset.X[idx], dataset.y_np[idx]

def get_test_data(dataset: EmbeddedDataset) -> Tuple[np.ndarray, np.ndarray]:
    te = dataset.splits["test"]
    te = te if isinstance(te, list) else te.tolist()
    return dataset.X[te], dataset.y_np[te]

def _get_clf_from_estimator(estimator):
    # Pipeline -> last step clf
    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        return estimator.named_steps["clf"]
    return estimator

def _pos_proba_from_estimator(estimator, y_pred_proba, n_tasks: int) -> np.ndarray:
    """
    返回 [N,T]，每列是 P(y=1)。
    关键：用 estimator 的 classes_ 找到 class==1 对应的列（解决三分类 {-1,0,1} 的坑）。
    """
    clf = _get_clf_from_estimator(estimator)

    # multi-task: y_pred_proba is list[T] of (N, C_t)
    if isinstance(y_pred_proba, list):
        T = n_tasks
        N = np.asarray(y_pred_proba[0]).shape[0]
        out = np.full((N, T), np.nan, dtype=float)

        # classes per task
        classes_list = None
        if hasattr(clf, "estimators_"):  # MultiOutputClassifier
            classes_list = [est.classes_ for est in clf.estimators_]
        elif hasattr(clf, "classes_"):
            # RandomForest multioutput: classes_ is list of arrays
            classes_list = clf.classes_ if isinstance(clf.classes_, (list, tuple)) else [clf.classes_]

        for t in range(min(len(y_pred_proba), T)):
            p = np.asarray(y_pred_proba[t])
            if p.ndim != 2 or p.shape[1] < 2:
                continue
            if classes_list is not None and t < len(classes_list):
                cls = np.asarray(classes_list[t])
                idx = np.where(cls == 1)[0]
                j = int(idx[0]) if len(idx) else (p.shape[1] - 1)
            else:
                j = p.shape[1] - 1
            out[:, t] = p[:, j]
        return out

    # single-task: ndarray (N, C)
    arr = np.asarray(y_pred_proba)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return np.zeros((arr.shape[0], 1), dtype=float)

        cls = None
        if hasattr(clf, "classes_"):
            cls = clf.classes_
            if isinstance(cls, (list, tuple)):  # sometimes list even for 1 task
                cls = cls[0]
            cls = np.asarray(cls)

        if cls is not None:
            idx = np.where(cls == 1)[0]
            j = int(idx[0]) if len(idx) else (arr.shape[1] - 1)
        else:
            j = arr.shape[1] - 1
        return arr[:, j:j+1].astype(float)

    raise ValueError(f"Unsupported proba shape: {arr.shape}")

def auroc_scorer(estimator, X, y_true) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    # missing label: -1 -> nan
    y_true[y_true < 0] = np.nan
    T = y_true.shape[1]

    y_pred_proba = estimator.predict_proba(X)
    proba_pos = _pos_proba_from_estimator(estimator, y_pred_proba, n_tasks=T)  # [N,T]

    scores = []
    for t in range(T):
        m = ~np.isnan(y_true[:, t])
        if m.sum() == 0:
            continue
        yt = y_true[m, t]
        yp = proba_pos[m, t]
        if np.unique(yt).size < 2:
            continue
        try:
            scores.append(roc_auc_score(yt, yp))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else float("nan")

def fit_and_eval_embedding_per_task(
    embedded: EmbeddedDataset,
    model_head: str,
    memory_weight: int = DEFAULT_MEMORY_WEIGHT,
    cv_splits: int = CV_SPLITS,
    n_jobs: int = N_JOBS,
    verbosity: int = VERBOSITY,
):
    X = embedded.X
    y = embedded.y_np.astype(float)

    # 单任务：复用原逻辑
    if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
        return fit_and_eval_embedding(
            embedded,
            model_head=model_head,
            memory_weight=memory_weight,
            cv_splits=cv_splits,
            n_jobs=n_jobs,
            verbosity=verbosity,
        )

    # -------- 多任务逐 task --------
    N, T = y.shape

    # train+valid 的全局索引
    trainval_idx = np.asarray(
        list(embedded.splits["train"]) + list(embedded.splits["valid"]),
        dtype=int
    )

    # test 子集（注意：X_test/y_test 都是 test 子集视角）
    X_test, y_test = get_test_data(embedded)
    y_test = y_test.astype(float)
    y_test[y_test < 0] = np.nan  # 统一 missing: -1 -> NaN

    N_test = X_test.shape[0]

    # ✅ 关键：预测矩阵必须是 (N_test, T)，否则 te_mask 长度对不上
    y_pred_pos = np.full((N_test, T), np.nan, dtype=float)

    hyperparams = [None] * T
    cv_scores, test_scores = [], []
    skipped = 0

    for t in range(T):
        # 全局 y 的第 t 个任务
        y_t = y[:, t].astype(float)
        y_t[y_t < 0] = np.nan

        # ---- train/valid：只保留该任务有标签的样本（全局索引）----
        m_all = ~np.isnan(y_t)                      # length N
        tr_mask = m_all[trainval_idx]              # length len(trainval_idx)
        tr_idx_t = trainval_idx[tr_mask]           # 全局 index

        # train 端有效性
        if tr_idx_t.size < 10:
            skipped += 1
            continue
        yt_tr = y_t[tr_idx_t]
        if np.unique(yt_tr).size < 2:
            skipped += 1
            continue

        # ---- test：只在 test 子集内有标签的位置上评估 ----
        te_mask = ~np.isnan(y_test[:, t])          # length N_test
        if te_mask.sum() < 2:
            skipped += 1
            continue
        if np.unique(y_test[te_mask, t]).size < 2:
            skipped += 1
            continue

        # ---- 拟合（单任务 1D y）----
        try:
            best = fit_model(
                X=X[tr_idx_t],
                y=y_t[tr_idx_t],   # 1D + 已剔除 NaN
                task=embedded.task,
                model_head=model_head,
                memory_weight=memory_weight,
                cv_splits=cv_splits,
                n_jobs=n_jobs,
                verbosity=verbosity,
            )
        except ValueError:
            skipped += 1
            continue

        hyperparams[t] = best["best_params"]
        cv_scores.append(best["best_score"])

        # ---- 预测（只对 test 子集里该任务有标签的样本预测）----
        proba = best["model_obj"].predict_proba(X_test[te_mask])  # shape (n_labeled_test, 2) usually
        pos = _pos_proba_from_estimator(best["model_obj"], proba, n_tasks=1)[:, 0]  # (n_labeled_test,)

        # ✅ 关键：写回 y_pred_pos 也必须用 test 子集 mask（长度 N_test）
        y_pred_pos[te_mask, t] = pos

        test_scores.append(float(roc_auc_score(y_test[te_mask, t], pos)))

    return {
        "model": model_head,
        "hyperparams": hyperparams,  # list[T]，可能含 None
        "cv_metric_name": "roc_auc",
        "cv_metric": float(np.nanmean(cv_scores)) if cv_scores else float("nan"),
        "test_metric_name": "roc_auc",
        "test_metric": float(np.nanmean(test_scores)) if test_scores else float("nan"),
        "y_test_true": y_test,       # (N_test, T)
        "y_test_pred": y_pred_pos,   # ✅ (N_test, T)
        "n_tasks": int(T),
        "n_tasks_used": int(len(test_scores)),
        "n_tasks_skipped": int(skipped),
    }

def fit_model(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    model_head: str,
    memory_weight: int = DEFAULT_MEMORY_WEIGHT,
    cv_splits: int = CV_SPLITS,
    n_jobs: int = N_JOBS,
    verbosity: int = VERBOSITY,
):
    # --- normalize y to 1D ---
    y = np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(
            f"fit_model expects single-task y as 1D. Got shape={y.shape}. "
            f"For multi-task, split per task and call fit_model per task."
        )

    # --- treat missing labels ---
    # 如果你不确定是否存在 -1，可以先不加这一行；
    # 但加了通常更安全：-1/其他负数当缺失。
    y_clean = y.copy()
    y_clean[y_clean < 0] = np.nan

    mask = ~np.isnan(y_clean)
    X_fit = X[mask]
    y_fit = y_clean[mask]

    # --- sanity checks ---
    if X_fit.shape[0] < 10:
        raise ValueError(f"Too few labeled samples after filtering: {X_fit.shape[0]}")

    if task == "classification":
        # AUROC 需要至少两类
        if np.unique(y_fit).size < 2:
            raise ValueError("Only one class present after filtering; AUROC is undefined.")
        # 单任务分类：让 sklearn 吃 (N,) 并且最好是 int
        y_fit = y_fit.astype(int)

        models = get_clf_models(no_output=1, embeddings_dtype=X.dtype)
        scorer = "roc_auc"
    else:
        models = get_reg_models(embeddings_dtype=X.dtype)
        scorer = "neg_mean_squared_error"  

    model = models[model_head]
    grid_search = GridSearchCV(
        model["model"],
        model["params"],
        cv=cv_splits,
        scoring=scorer,
        n_jobs=int(n_jobs / memory_weight),
        verbose=verbosity,
        refit=True,
    )

    try:
        grid_search.fit(X_fit, y_fit)
    except ValueError as e:
        log.error(f"Error fitting model {model_head}: {e}")
        if "lbfgs" not in str(e):
            raise
        log.error("L-BFG-S failed, replacing with SVD")
        if "clf__estimator__solver" in model["params"]:
            model["params"]["clf__estimator__solver"] = ["svd"]
        elif "clf__solver" in model["params"]:
            model["params"]["clf__solver"] = ["svd"]
        else:
            raise ValueError("Cannot replace solver with SVD; missing solver key in params")

        grid_search = GridSearchCV(
            model["model"],
            model["params"],
            cv=cv_splits,
            scoring=scorer,
            n_jobs=int(n_jobs / memory_weight),
            verbose=verbosity,
            refit=True,
        )
        grid_search.fit(X_fit, y_fit)

    return {
        "model": model_head,
        "model_obj": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "n_fit": int(X_fit.shape[0]),
    }




def eval_on_test(task: str, y_test: np.ndarray, y_pred_proba) -> float:
    # 注意：这里不要把缺失标签随便变成 0（OGB 常见 -1 表示 missing）
    y_test_ = y_test.astype(float)

    if task == "classification":
        # OGB 多任务常见 missing label = -1，转换为 nan，评估时会跳过
        y_test_[y_test_ < 0] = np.nan

        if y_test_.ndim == 1:
            y_test_ = y_test_.reshape(-1, 1)

        n_tasks = y_test_.shape[1]

        # 多任务：需要 [N,T] 形式的正类概率
        if n_tasks > 1:
            return float(fallback_multioutput_auroc(y_test_, y_pred_proba))

        # 单任务
        if isinstance(y_pred_proba, list):
            prob1 = y_pred_proba[0][:, 1]
        else:
            prob1 = y_pred_proba[:, 1]

        mask = ~np.isnan(y_test_[:, 0])
        return float(roc_auc_score(y_test_[mask, 0], prob1[mask]))

    raise NotImplementedError("Regression metrics not wired here yet.")


def fit_and_eval_embedding(dataset: EmbeddedDataset,
                           model_head: str,
                           memory_weight: int = DEFAULT_MEMORY_WEIGHT,
                           cv_splits: int = CV_SPLITS,
                           n_jobs: int = N_JOBS,
                           verbosity: int = VERBOSITY):

    X_train, y_train = get_train_data(dataset)
    best = fit_model(
        X=X_train,
        y=y_train,
        task=dataset.task,
        model_head=model_head,
        memory_weight=memory_weight,
        cv_splits=cv_splits,
        n_jobs=n_jobs,
        verbosity=verbosity,
    )

    X_test, y_test = get_test_data(dataset)
    y_pred = best["model_obj"].predict_proba(X_test)

    test_score = eval_on_test(dataset.task, y_test, y_pred)

    return {
        "model": model_head,
        "hyperparams": best["best_params"],
        "cv_metric_name": "roc_auc",
        "cv_metric": best["best_score"],
        "test_metric_name": "roc_auc",
        "test_metric": test_score,
        "y_test_true": y_test,
        "y_test_pred": y_pred,
        "model_obj": best["model_obj"],  # optional
    }


# ============================================================
#  Main procedure (prepared -> embed cache -> score heads)
# ============================================================

def run_downstream_from_prepared(
    prepared_path: str,
    prepared_spd_path: str,
    ckpt_path: str,
    cfg: Any,
    device: torch.device,
    out_dir: str,
    model_name: str,
    illegal_smiles_txt: Optional[str] = None,
    embed_batch_size: int = 256,
    num_workers: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)
    encoder_cfg = cfg.encoder

    # 1) load prepared dataset
    ds = load_prepared_dataset(prepared_path, illegal_smiles_txt=illegal_smiles_txt)
    dataset_name = ds.name

    # 2) load encoder ckpt
    node_in_dim, edge_in_dim = infer_in_dims_from_dataset(ds)
    log.info(f"Inferred dims: node_in_dim={node_in_dim}, edge_in_dim={edge_in_dim}")

    encoder = build_encoder(cfg, device, node_in_dim, edge_in_dim)
    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=True)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    encoder.eval()
    log.info(f"Load encoder finished")

    # 3) embedding cache path (benchmark style)
    save_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    log.info(f"create emb...")
    X = embed_dataset_graph_level(
        encoder=encoder,
        ds=ds,
        device=device,
        prepared_path=prepared_path,
        batch_size=embed_batch_size,
        num_workers=num_workers,
        spd_max_dist=int(encoder_cfg.spd_max_dist),
        spd_cache_root=prepared_spd_path,
    )

    y = ds.labels.to_numpy(dtype=float)
    embedded = EmbeddedDataset(
        name=dataset_name,
        task=ds.task,
        embedder=model_name,
        X=X,
        y_np=y,
        splits=ds.splits,
    )

    # 4) train downstream heads
    results = []
    for head in AVAILABLE_HEADS:
        log.info(f"Training head={head} on dataset={dataset_name}, embedder={model_name}")
        res = fit_and_eval_embedding_per_task(embedded, model_head=head)
        res_row = {
            "dataset": dataset_name,
            "task": embedded.task,
            "embedder": model_name,
            "model": head,
            "hyperparams": json.dumps(to_jsonable(res["hyperparams"]), ensure_ascii=False),
            "cv_metric_name": res["cv_metric_name"],
            "cv_metric": res["cv_metric"],
            "test_metric_name": res["test_metric_name"],
            "test_metric": res["test_metric"],
            "key": f"{dataset_name}_{model_name}_{head}",
        }
        results.append(res_row)

        # optional: save predictions
        pred_path = os.path.join(save_dir, f"{model_name}_{head}_preds.joblib")
        joblib.dump(
            {"y_true": res["y_test_true"], "y_pred": res["y_test_pred"]},
            pred_path
        )

    # 5) save result csv
    out_csv = os.path.join(save_dir, f"{model_name}_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    log.info(f"Saved results CSV: {out_csv}")

    return results


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument("--prepared_path", type=str, default="/mnt2/luyifeng/diff4MoleculeRepresentation/data/prepared/ogbg-molsider.json",
                        help="Path to prepared dataset (.json or .joblib), e.g. prepared/DILI.json")
    parser.add_argument("--prepared_spd_path", type=str, default="/mnt2/luyifeng/diff4MoleculeRepresentation/data/prepared_spd_cache",
                        help="Path to prepared dataset (.json or .joblib), e.g. prepared/DILI.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default="./logs_embedding/embedded_cache",
                        help="Where to store embeddings/preds/results (benchmark-like)")
    parser.add_argument("--illegal_smiles", type=str, default='./configs/illegal_smiles.txt',
                        help="Path to illegal_smiles.txt (optional)")

    parser.add_argument("--ckpt_date", type=str, default="20260204-181909",help="Path to your training checkpoint best.pt")
    parser.add_argument("--enlayer",default=9)
    parser.add_argument("--delayer",default=5)
    parser.add_argument("--encoder_name", default="cls_graphormer_pearl")
    parser.add_argument("--denoiser_name", default="uni_o2_condition")
    parser.add_argument("--embed_bs", type=int, default=64)
    parser.add_argument("--pearl_fuse",type=str, default="concat")

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_name",type=str, default=None)


    # encoder config: easiest is to load the SAME training yaml and pass cfg.encoder
    parser.add_argument("--train_config", type=str, required=False,default="configs/training.yml",
                        help="Your training yaml path; we will use cfg.encoder to build GraphGPSEncoder")

    args = parser.parse_args()

    logging_level = os.environ.get("LOGLEVEL", "INFO").upper()
    log.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device(args.device)

    # load encoder cfg from your training config (OmegaConf)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.train_config)
    cfg.model.num_layers = int(args.delayer)
    cfg.encoder.num_layers = int(args.enlayer)
    cfg.encoder.name = args.encoder_name
    cfg.model.model_type = args.denoiser_name
    cfg.encoder.pearl_fuse = args.pearl_fuse

    ckpt_path, meta = build_ckpt(args, cfg, train=False, run_time = args.ckpt_date)
    ckpt_path = os.path.join(ckpt_path,"best.pt")
    if args.model_name is None:
        model_name = f"en{args.enlayer}_de{args.delayer}_e_{args.encoder_name}_d_{args.denoiser_name}"
    else:
        model_name = args.model_name
        
    run_downstream_from_prepared(
        prepared_path=args.prepared_path,
        prepared_spd_path=args.prepared_spd_path,
        ckpt_path=ckpt_path,
        cfg=cfg,
        device=device,
        out_dir=args.out_dir,
        model_name=model_name,
        illegal_smiles_txt=args.illegal_smiles,
        embed_batch_size=args.embed_bs,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
