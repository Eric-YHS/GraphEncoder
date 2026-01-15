# bench_prepared_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import base64
import json
import numpy as np
import pandas as pd
import logging as log

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# --------- JSON hooks (match benchmark legacy format) ----------
def _decode_base64_array(b64: str, dtype: str, shape: List[int]) -> np.ndarray:
    raw = base64.b64decode(b64.encode("utf-8"))
    arr = np.frombuffer(raw, dtype=np.dtype(dtype))
    return arr.reshape(shape)

def json_numpy_obj_hook(d: Dict[str, Any]) -> Any:
    # numpy array
    if "__ndarray__" in d:
        return _decode_base64_array(d["__ndarray__"], d["dtype"], d["shape"])

    # torch tensor stored as raw bytes (benchmark legacy)
    if "__torch_tensor__" in d:
        # decode to numpy (you can convert to torch later)
        return _decode_base64_array(d["__torch_tensor__"], d["dtype"], d["shape"])

    # pandas dataframe wrapper
    if "__dataframe__" in d:
        payload = d["__dataframe__"]
        return pd.DataFrame(
            payload["data"],
            columns=payload["columns"],
            index=payload.get("index", None),
        )

    return d


@dataclass
class Dataset:
    name: str
    task: Literal["classification", "regression"]
    data: Any          # pd.DataFrame
    splits: Any        # dict(train/valid/test -> indices)

    @classmethod
    def deserialize_legacy(cls, path: str) -> "Dataset":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f, object_hook=json_numpy_obj_hook)
        return cls(**obj)

    def filter_out_problematic_molecules(self, illegal_smiles: List[str]):
        if "smiles" not in self.data.columns:
            raise ValueError("Dataset does not contain a 'smiles' column.")

        # ensure split column exists (benchmark does this)
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

        # rebuild splits based on split column
        log.info("Fixing splits indices")
        self.splits = {
            k: np.where(self.data["split"] == k)[0].tolist()
            for k in self.splits.keys()
        }
        if "UNKNOWN" in self.splits:
            raise ValueError("Dataset contains 'UNKNOWN' split, which should not happen.")

    @property
    def labels(self) -> pd.DataFrame:
        # same rule as benchmark: drop id/split columns + smiles/graph
        id_cols = [c for c in self.data.columns if ("id" in c.lower() or "split" in c.lower())]
        drop_cols = ["smiles", "graph"] + id_cols
        return self.data.drop(columns=drop_cols, errors="ignore")


def load_prepared_dataset_from_json(
    json_path: str,
    illegal_smiles_txt: Optional[str] = None,
) -> Dataset:
    ds = Dataset.deserialize_legacy(json_path)
    if illegal_smiles_txt is not None:
        with open(illegal_smiles_txt, "r", encoding="utf-8") as f:
            illegal = [line.strip() for line in f if line.strip()]
        log.info(f"Filtering illegal SMILES: {len(illegal)}")
        ds.filter_out_problematic_molecules(illegal)
    return ds


def graph_dict_to_pyg_data(g: Dict[str, Any]) -> Data:
    """
    Prefer OGB smiles2graph fields to match your PCQM4M training:
      - node_feat: [n, 9] int
      - edge_feat: [E, 3] int
      - edge_index: [2, E] int
    """
    # choose ogb schema first
    if "node_feat" in g and "edge_feat" in g and "edge_index" in g:
        x = torch.from_numpy(np.asarray(g["node_feat"])).long()
        edge_attr = torch.from_numpy(np.asarray(g["edge_feat"])).long()
        edge_index = torch.from_numpy(np.asarray(g["edge_index"])).long()
    else:
        # fallback (not recommended if you want strict alignment)
        x = torch.from_numpy(np.asarray(g["x"])).float()
        edge_attr = torch.from_numpy(np.asarray(g["edge_attr"])).float()
        edge_index = torch.from_numpy(np.asarray(g["edge_index"])).long()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_pyg_dataloader_from_prepared(
    ds: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle: bool = False,
) -> Tuple[DataLoader, np.ndarray, Dict[str, List[int]]]:
    """
    Returns:
      loader over ALL molecules in ds.data order
      y_np aligned with ds.data order
      splits indices (train/valid/test) aligned with ds.data order
    """
    # graphs
    data_list: List[Data] = []
    for i in range(len(ds.data)):
        g = ds.data.iloc[i]["graph"]
        data_list.append(graph_dict_to_pyg_data(g))

    # labels
    y = ds.labels.to_numpy(dtype=float)
    y = np.nan_to_num(y, nan=0.0)

    # splits
    splits = ds.splits
    # make sure lists
    for k in ["train", "valid", "test"]:
        if k in splits and not isinstance(splits[k], list):
            splits[k] = list(splits[k])

    loader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, y, splits
