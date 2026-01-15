# downstream_benchmark_port.py
# ------------------------------------------------------------
# Use prepared benchmark datasets (*.json / *.joblib) and a frozen
# GraphGPSEncoder (loaded from your best.pt) to produce embeddings,
# then train downstream heads (rf / ridge / knn) with GridSearchCV
# aligned to benchmarking_molecular_models.
# ------------------------------------------------------------

import os
import json
import base64
import argparse
import logging as log
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import argparse
import os
import shutil

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, roc_auc_score

# ====== optional skfp multioutput auroc (same as benchmark) ======
try:
    from skfp.metrics import multioutput_auroc_score
    _HAS_SKFP = True
except Exception:
    _HAS_SKFP = False
    multioutput_auroc_score = None

# ====== your encoder import path (as you said) ======
from models import GraphGPSEncoder


# ============================================================
#  Constants (match benchmark const.py)
# ============================================================
CV_SPLITS = 5
N_JOBS = 32
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
    task: Literal["classification", "regression"]
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
def graph_dict_to_pyg_data(g: Dict[str, Any]) -> Data:
    # prefer ogb schema
    if "node_feat" in g and "edge_feat" in g and "edge_index" in g:
        node_feat = np.asarray(g["node_feat"])
        edge_feat = np.asarray(g["edge_feat"])
        edge_index = np.asarray(g["edge_index"])
        x = torch.from_numpy(node_feat).long()
        edge_attr = torch.from_numpy(edge_feat).long()
        edge_index = torch.from_numpy(edge_index).long()
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    raise ValueError("Graph dict missing required fields.")


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
    batch_size: int = 256,
    num_workers: int = 8,
) -> np.ndarray:
    encoder.eval()

    data_list: List[Data] = []
    for i in range(len(ds.data)):
        g = ds.data.iloc[i]["graph"]
        data_list.append(graph_dict_to_pyg_data(g))

    loader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    X_chunks: List[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)

        # IMPORTANT: your GraphGPSEncoder uses Linear, requires float
        batch.x = batch.x.float()
        batch.edge_attr = batch.edge_attr.float()

        node_emb, graph_emb = encoder(batch)   # (N,H), (B,H)
        X_chunks.append(graph_emb.detach().cpu().numpy().astype(np.float32))

    X = np.concatenate(X_chunks, axis=0)
    return X


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
        lr_clf = LogisticRegression(n_jobs=-1)
        lr_params = RIDGE_CLF
    else:
        lr_clf = MultiOutputClassifier(LogisticRegression(n_jobs=-1))
        lr_params = RIDGE__MULTIOUTPUT_CLF

    return {
        "rf": {"model": Pipeline([("clf", RandomForestClassifier(n_jobs=-1))]), "params": RF_CLF.copy()},
        "ridge": {"model": Pipeline([("scaler", StandardScaler()), ("clf", lr_clf)]), "params": lr_params.copy()},
        "knn": {"model": Pipeline([("scaler", StandardScaler()),
                                  ("clf", KNeighborsClassifier(n_jobs=-1, metric=get_knn_distance(embeddings_dtype)))]),
                "params": KNN_CLF.copy()},
    }

def get_reg_models(embeddings_dtype):
    return {
        "rf": {"model": Pipeline([("clf", RandomForestRegressor(n_jobs=-1))]), "params": RF_REG.copy()},
        "ridge": {"model": Pipeline([("scaler", StandardScaler()), ("clf", Ridge())]), "params": RIDGE_REG.copy()},
        "knn": {"model": Pipeline([("scaler", StandardScaler()),
                                  ("clf", KNeighborsRegressor(n_jobs=-1, metric=get_knn_distance(embeddings_dtype)))]),
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

def proba_to_pos_matrix(y_pred_proba, n_tasks: int) -> np.ndarray:
    """
    Convert sklearn predict_proba output to a dense [N, T] matrix of P(y=1).
    Supports:
      - list length T: each item (N, Ck)
      - ndarray (T, N, C)
      - ndarray (N, 2) for single-task
      - ndarray (N, T) already positive probs
      - ndarray (N, 1) (degenerate / single-class) -> treat as 0 for pos prob
    Always returns float matrix shape [N, n_tasks], padding missing cols with NaN.
    """
    try:
        if isinstance(y_pred_proba, list):
            # list[t] is (N, C_t)
            N = np.asarray(y_pred_proba[0]).shape[0]
            out = np.full((N, n_tasks), np.nan, dtype=float)

            for t in range(min(len(y_pred_proba), n_tasks)):
                p = np.asarray(y_pred_proba[t])
                if p.ndim == 2 and p.shape[1] >= 2:
                    out[:, t] = p[:, 1]
                elif p.ndim == 2 and p.shape[1] == 1:
                    # can't know which class; usually means model saw single class
                    # set pos-prob to 0; AUROC for this task will be 0.5 if labels have both classes
                    out[:, t] = 0.0
                elif p.ndim == 1:
                    out[:, t] = p.astype(float)
                else:
                    # unexpected shape -> keep NaN for this task
                    pass
            return out

        arr = np.asarray(y_pred_proba)

        # case: (T, N, C)
        if arr.ndim == 3:
            # could be (T,N,2) or (N,T,2) depending on how it was packed
            if arr.shape[0] == n_tasks:
                # (T,N,C)
                T, N, C = arr.shape
                out = np.full((N, n_tasks), np.nan, dtype=float)
                for t in range(n_tasks):
                    if C >= 2:
                        out[:, t] = arr[t, :, 1]
                    else:
                        out[:, t] = 0.0
                return out
            # if it's (N,T,C)
            if arr.shape[1] == n_tasks:
                N, T, C = arr.shape
                out = np.full((N, n_tasks), np.nan, dtype=float)
                for t in range(n_tasks):
                    if C >= 2:
                        out[:, t] = arr[:, t, 1]
                    else:
                        out[:, t] = 0.0
                return out

        # case: (N,2) single task
        if arr.ndim == 2 and arr.shape[1] == 2 and n_tasks == 1:
            return arr[:, 1:2].astype(float)

        # case: already (N,T) positive probs
        if arr.ndim == 2 and arr.shape[1] == n_tasks:
            return arr.astype(float)

        # case: (N,1) but multi-task requested -> pad
        if arr.ndim == 2 and arr.shape[1] == 1:
            N = arr.shape[0]
            out = np.full((N, n_tasks), np.nan, dtype=float)
            out[:, 0] = 0.0
            return out

    except Exception:
        pass
    
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


def fit_model(X: np.ndarray, y: np.ndarray,
              task: str, model_head: str,
              memory_weight: int = DEFAULT_MEMORY_WEIGHT,
              cv_splits: int = CV_SPLITS,
              n_jobs: int = N_JOBS,
              verbosity: int = VERBOSITY):

    if task == "classification":
        no_outputs = y.shape[1] if len(y.shape) > 1 else 1
        models = get_clf_models(no_outputs, X.dtype)
    elif task == "regression":
        models = get_reg_models(X.dtype)
    else:
        raise ValueError(f"Unknown task: {task}")

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # scorer selection aligned to benchmark train.py
    if y.shape[1] > 1:
        scorer = make_scorer(multioutput_auroc_from_proba, response_method="predict_proba")
    else:
        scorer = "roc_auc"
        
    y = np.nan_to_num(y, nan=0.0)

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
        grid_search.fit(X, y)
    except ValueError as e:
        # replicate benchmark lbfgs->svd fallback
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
        grid_search.fit(X, y)

    return {
        "model": model_head,
        "model_obj": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
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
            proba_pos = proba_to_pos_matrix(y_pred_proba, n_tasks=n_tasks)  # [N,T]

            if _HAS_SKFP:
                # skfp 的 multioutput_auroc_score 期望 y_true/y_pred 同形状 [N,T]
                return float(multioutput_auroc_score(y_test_, proba_pos))
            else:
                # 你原来的 fallback 如果是吃 list[(N,2)]，这里也可以改成吃 [N,T]
                return float(fallback_multioutput_auroc(y_test_, proba_pos))

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
    ckpt_path: str,
    encoder_cfg: Any,
    device: torch.device,
    out_dir: str,
    model_name: str,
    illegal_smiles_txt: Optional[str] = None,
    override: bool = False,
    embed_batch_size: int = 256,
    num_workers: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) load prepared dataset
    ds = load_prepared_dataset(prepared_path, illegal_smiles_txt=illegal_smiles_txt)
    dataset_name = ds.name

    # 2) load encoder ckpt
    node_in_dim, edge_in_dim = infer_in_dims_from_dataset(ds)
    log.info(f"Inferred dims: node_in_dim={node_in_dim}, edge_in_dim={edge_in_dim}")

    encoder = GraphGPSEncoder(encoder_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    encoder.eval()

    # 3) embedding cache path (benchmark style)
    save_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    embedded_path = os.path.join(save_dir, f"{model_name}.joblib")

    if (not override) and os.path.exists(embedded_path):
        embedded: EmbeddedDataset = joblib.load(embedded_path)
        log.info(f"Loaded cached embeddings: {embedded_path}")
    else:
        X = embed_dataset_graph_level(
            encoder=encoder,
            ds=ds,
            device=device,
            batch_size=embed_batch_size,
            num_workers=num_workers,
        )

        y = ds.labels.to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0)

        embedded = EmbeddedDataset(
            name=dataset_name,
            task=ds.task,
            embedder=model_name,
            X=X,
            y_np=y,
            splits=ds.splits,
        )
        joblib.dump(embedded, embedded_path)
        log.info(f"Saved embeddings: {embedded_path}")

    # 4) train downstream heads
    results = []
    for head in AVAILABLE_HEADS:
        log.info(f"Training head={head} on dataset={dataset_name}, embedder={model_name}")
        res = fit_and_eval_embedding(embedded, model_head=head)
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
    parser.add_argument("--prepared_path", type=str, required=True,
                        help="Path to prepared dataset (.json or .joblib), e.g. prepared/DILI.json")
    parser.add_argument("--ckpt", type=str, required=False, default="./logs_diffusion/training_2025_12_30__14_12_07/checkpoints/best.pt",
                        help="Path to your training checkpoint best.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default="./logs_embedding/embedded_cache",
                        help="Where to store embeddings/preds/results (benchmark-like)")
    parser.add_argument("--model_name", type=str, default="GraphGPS_Encoder",
                        help="Embedder name used in result table")
    parser.add_argument("--illegal_smiles", type=str, default='./configs/illegal_smiles.txt',
                        help="Path to illegal_smiles.txt (optional)")
    parser.add_argument("--override", action="store_true",
                        help="Recompute embeddings even if cached exists")
    parser.add_argument("--enlayer",default=3)
    parser.add_argument("--delayer",default=3)
    parser.add_argument("--embed_bs", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

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

    encoder_cfg = cfg.encoder
    encoder_cfg

    run_downstream_from_prepared(
        prepared_path=args.prepared_path,
        ckpt_path=args.ckpt,
        encoder_cfg=encoder_cfg,
        device=device,
        out_dir=args.out_dir,
        model_name=args.model_name,
        illegal_smiles_txt=args.illegal_smiles,
        override=args.override,
        embed_batch_size=args.embed_bs,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
