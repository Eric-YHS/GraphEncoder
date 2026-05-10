import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _idx_to_numpy(idx: torch.Tensor) -> np.ndarray:
    return idx.detach().cpu().view(-1).numpy().astype(np.int64, copy=False)


def _as_float_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device, non_blocking=True)


def graph_stats(z: torch.Tensor, max_rows: int = 512) -> Dict[str, float]:
    if z is None or z.numel() == 0:
        return {
            "cond_norm": 0.0,
            "cond_std": 0.0,
            "cond_mean_cos": 0.0,
            "cond_effective_rank": 0.0,
        }
    zs = z.detach().float()
    if zs.size(0) > max_rows:
        zs = zs[:max_rows]
    norm = zs.norm(dim=-1).mean()
    std = zs.std(dim=0, unbiased=False).mean()
    if zs.size(0) <= 1:
        mean_cos = zs.new_tensor(0.0)
        erank = zs.new_tensor(1.0)
    else:
        zn = torch.nn.functional.normalize(zs, dim=-1)
        cos = zn @ zn.t()
        off = cos[~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)]
        mean_cos = off.mean()
        zc = zs - zs.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(zc)
        p = s / s.sum().clamp_min(1e-12)
        erank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return {
        "cond_norm": float(norm.item()),
        "cond_std": float(std.item()),
        "cond_mean_cos": float(mean_cos.item()),
        "cond_effective_rank": float(erank.item()),
    }


class GraphConditionAdapter(nn.Module):
    def __init__(self, input_dim: int, graph_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.graph_dim = int(graph_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.graph_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class BaseConditionSource(nn.Module):
    name = "base"

    @property
    def input_dim(self) -> int:
        return 0

    @property
    def valid_mask(self) -> Optional[np.ndarray]:
        return None

    def get_features(self, batch, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class NullConditionSource(BaseConditionSource):
    name = "none"

    def get_features(self, batch, device: torch.device) -> torch.Tensor:
        idx = batch.idx.view(-1) if hasattr(batch, "idx") else torch.arange(int(batch.batch.max().item()) + 1)
        return torch.empty((int(idx.numel()), 0), device=device, dtype=torch.float32)


class PropertyConditionSource(BaseConditionSource):
    name = "property"

    def __init__(self, semantic_cache_dir: str, y_mean: float, y_std: float, include_y: bool = True):
        super().__init__()
        self.cache_dir = Path(semantic_cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError("semantic cache meta not found: %s" % meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.desc = np.load(self.cache_dir / "descriptors_float32.npy", mmap_mode="r")
        self.valid = np.load(self.cache_dir / "valid_uint8.npy", mmap_mode="r")
        self.desc_mean = np.asarray(self.meta["descriptor_mean"], dtype=np.float32)
        self.desc_std = np.maximum(np.asarray(self.meta["descriptor_std"], dtype=np.float32), 1e-6)
        self.y_mean = float(y_mean)
        self.y_std = max(float(y_std), 1e-6)
        self.include_y = bool(include_y)
        self._input_dim = int(self.desc.shape[1]) + (1 if self.include_y else 0)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def valid_mask(self) -> Optional[np.ndarray]:
        return np.asarray(self.valid, dtype=np.uint8)

    def get_features(self, batch, device: torch.device) -> torch.Tensor:
        idx_np = _idx_to_numpy(batch.idx)
        desc = np.array(self.desc[idx_np], copy=True).astype(np.float32, copy=False)
        desc = (desc - self.desc_mean.reshape(1, -1)) / self.desc_std.reshape(1, -1)
        desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        feat = _as_float_tensor(desc, device)
        if self.include_y:
            y = batch.y.view(-1).to(device=device, dtype=torch.float32)
            y = torch.nan_to_num((y - self.y_mean) / self.y_std, nan=0.0, posinf=0.0, neginf=0.0)
            feat = torch.cat([feat, y.unsqueeze(-1)], dim=-1)
        return feat


class GraleConditionSource(BaseConditionSource):
    name = "grale_official"

    def __init__(self, cache_dir: str, normalize: bool = True):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError("GRALE cache meta not found: %s" % meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.emb = np.load(self.cache_dir / "grale_embedding_float32.npy", mmap_mode="r")
        self.valid = np.load(self.cache_dir / "valid_uint8.npy", mmap_mode="r")
        self.normalize = bool(normalize)
        self.mean = np.asarray(self.meta.get("embedding_mean", np.zeros((self.emb.shape[1],), dtype=np.float32)), dtype=np.float32)
        self.std = np.maximum(
            np.asarray(self.meta.get("embedding_std", np.ones((self.emb.shape[1],), dtype=np.float32)), dtype=np.float32),
            1e-6,
        )

    @property
    def input_dim(self) -> int:
        return int(self.emb.shape[1])

    @property
    def valid_mask(self) -> Optional[np.ndarray]:
        return np.asarray(self.valid, dtype=np.uint8)

    def get_features(self, batch, device: torch.device) -> torch.Tensor:
        idx_np = _idx_to_numpy(batch.idx)
        emb = np.array(self.emb[idx_np], copy=True).astype(np.float32, copy=False)
        if self.normalize:
            emb = (emb - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        return _as_float_tensor(emb, device)


class ConditionSystem(nn.Module):
    def __init__(
        self,
        source: BaseConditionSource,
        node_dim: int,
        graph_dim: int,
        adapter_hidden_dim: int,
        adapter_dropout: float,
    ):
        super().__init__()
        self.source = source
        self.node_dim = int(node_dim)
        self.graph_dim = int(graph_dim)
        if source.input_dim > 0:
            self.adapter = GraphConditionAdapter(
                input_dim=source.input_dim,
                graph_dim=self.graph_dim,
                hidden_dim=int(adapter_hidden_dim),
                dropout=float(adapter_dropout),
            )
        else:
            self.adapter = None

    @property
    def valid_mask(self) -> Optional[np.ndarray]:
        return self.source.valid_mask

    def forward(self, batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.source.get_features(batch, device)
        num_nodes = int(batch.x.size(0))
        num_graphs = int(batch.idx.view(-1).numel())
        cond_node = torch.zeros((num_nodes, self.node_dim), device=device, dtype=torch.float32)
        if self.adapter is None:
            graph_cond = torch.zeros((num_graphs, self.graph_dim), device=device, dtype=torch.float32)
        else:
            graph_cond = self.adapter(features)
        return cond_node, graph_cond, features


def build_condition_system(cfg) -> ConditionSystem:
    cond_cfg = cfg.condition
    source_name = str(cond_cfg.source).lower()
    if source_name == "none":
        source = NullConditionSource()
    elif source_name == "property":
        source = PropertyConditionSource(
            semantic_cache_dir=str(cond_cfg.semantic_cache_dir),
            y_mean=float(cond_cfg.property_mean),
            y_std=float(cond_cfg.property_std),
            include_y=bool(getattr(cond_cfg, "include_pcqm_y", True)),
        )
    elif source_name == "grale_official":
        source = GraleConditionSource(
            cache_dir=str(cond_cfg.grale_cache_dir),
            normalize=bool(getattr(cond_cfg, "normalize_grale", True)),
        )
    else:
        raise ValueError("Unsupported condition.source=%s" % source_name)
    return ConditionSystem(
        source=source,
        node_dim=int(cond_cfg.node_dim),
        graph_dim=int(cond_cfg.graph_dim),
        adapter_hidden_dim=int(cond_cfg.adapter_hidden_dim),
        adapter_dropout=float(cond_cfg.adapter_dropout),
    )
