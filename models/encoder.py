import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from torch_geometric.nn import GINEConv, global_mean_pool, GINConv
from torch_geometric.utils import degree
from torch_geometric.utils import to_dense_batch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class DegreeEncoder(nn.Module):
    def __init__(self, max_degree: int, hidden_dim: int):
        super().__init__()
        self.max_degree = max_degree
        self.emb = nn.Embedding(max_degree + 1, hidden_dim)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        deg = degree(edge_index[0], num_nodes=num_nodes).long().clamp_(0, self.max_degree)
        return self.emb(deg)

####################################
# GraphGPS, no CLS
####################################

class GraphGlobalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        spd_max_dist: int = 8,
        use_spd_bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_spd_bias = use_spd_bias
        self.spd_max_dist = spd_max_dist

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,   # (B, L, D)
        )

        # Graphormer 风格：SPD -> bias（离散距离 embedding，再映射到每个 head 的 bias）
        if use_spd_bias:
            self.spatial_emb = nn.Embedding(spd_max_dist + 2, num_heads)  
            # dist in [0..spd_max_dist]，其余归为 spd_max_dist+1 (unreachable/too far)

    def forward(
        self,
        h: torch.Tensor,                 # [N, D]
        batch: torch.Tensor,             # [N]
        spatial_pos: Optional[torch.Tensor] = None,  # [N, N] per-graph 不现实；我们希望 batch 里带 dense 的 (B, L, L)
    ) -> torch.Tensor:
        # dense: [B, L, D], mask: [B, L] True for valid nodes
        h_dense, mask = to_dense_batch(h, batch=batch)  # mask=True 表示有效 token
        B, L, D = h_dense.shape

        # key_padding_mask: True 表示要被 mask 掉（PyTorch 的定义）
        key_padding_mask = ~mask  # [B, L]

        attn_mask = None
        if self.use_spd_bias:
            if spatial_pos is None:
                raise ValueError("use_spd_bias=True but spatial_pos is None. Please provide batch.spatial_pos.")
            # spatial_pos 期望是 dense 的 [B, L, L]，无效位置可填 spd_max_dist+1
            # -> 先 clamp，再查 embedding 得到 [B, L, L, H]
            spd = spatial_pos.clamp(0, self.spd_max_dist + 1)
            bias = self.spatial_emb(spd)  # [B, L, L, H]
            # MultiheadAttention 的 attn_mask 形状可为 (B*num_heads, L, L)
            bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, L, L]
            attn_mask = bias.view(B * self.num_heads, L, L)

            # 同时把 padding 的位置 mask 掉（否则 bias 会让 padding 参与）
            # 用一个非常小的负数近似 -inf
            neg_inf = torch.finfo(h_dense.dtype).min
            pad2d = key_padding_mask.unsqueeze(1) | key_padding_mask.unsqueeze(2)  # [B, L, L]
            pad2d = pad2d.unsqueeze(1).expand(B, self.num_heads, L, L)             # [B, H, L, L]
            attn_mask = attn_mask.masked_fill(pad2d.reshape(B * self.num_heads, L, L), neg_inf)

        out, _ = self.mha(
            h_dense, h_dense, h_dense,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        # 折回稀疏：把 padding 去掉
        out = out[mask]  # [N, D]
        return out

class GraphGPSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_spd_bias: bool = False,
        spd_max_dist: int = 8,
    ):
        super().__init__()
        self.dropout = dropout

        # Local branch: GINEConv 支持 edge_attr
        nn_local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(nn_local, edge_dim=edge_dim)

        # Global branch
        self.global_attn = GraphGlobalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            use_spd_bias=use_spd_bias,
            spd_max_dist=spd_max_dist,
        )

        # Combine + FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,                # [N, D]
        edge_index: torch.Tensor,       # [2, E]
        edge_attr: torch.Tensor,        # [E, edge_dim]
        batch: torch.Tensor,            # [N]
        spatial_pos: Optional[torch.Tensor] = None,  # [B, L, L]（dense）
    ) -> torch.Tensor:
        # local
        h_local = self.local_conv(h, edge_index, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # global
        h_global = self.global_attn(h, batch=batch, spatial_pos=spatial_pos)
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)

        # residual + norm
        h = self.norm1(h + h_local + h_global)

        # ffn
        h_ffn = self.ffn(h)
        h_ffn = F.dropout(h_ffn, p=self.dropout, training=self.training)
        h = self.norm2(h + h_ffn)
        return h

class GraphGPSEncoder(nn.Module):
    def __init__(self, cfg, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg

        self.atom_encoder = AtomEncoder(cfg.hidden_dim)      # -> [N, hidden_dim]
        self.bond_encoder = BondEncoder(cfg.hidden_dim)      # -> [E, hidden_dim]

        if cfg.edge_emb_dim == cfg.hidden_dim:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(cfg.hidden_dim, cfg.edge_emb_dim)

        self.degree_enc = DegreeEncoder(cfg.max_degree, cfg.hidden_dim) if cfg.use_degree else None

        self.blocks = nn.ModuleList([
            GraphGPSBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                edge_dim=cfg.edge_emb_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                use_spd_bias=cfg.use_spd_bias,
                spd_max_dist=cfg.spd_max_dist,
            )
            for _ in range(cfg.num_layers)
        ])

        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

        self._expect_node_dim = node_in_dim
        self._expect_edge_dim = edge_in_dim

    def forward(self, batch):
        """
        return:
          node_emb: [N, hidden_dim]
          graph_emb: [B, hidden_dim]
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_id = batch.batch
        N = x.size(0)

        if x.dtype != torch.long:
            x = x.long()
        if edge_attr.dtype != torch.long:
            edge_attr = edge_attr.long()
        if x.size(-1) != self._expect_node_dim:
            raise ValueError(f"Unexpected node feature dim: got {x.size(-1)}, expect {self._expect_node_dim}")
        if edge_attr.size(-1) != self._expect_edge_dim:
            raise ValueError(f"Unexpected edge feature dim: got {edge_attr.size(-1)}, expect {self._expect_edge_dim}")

        h = self.atom_encoder(x)                 # [N, H]
        e = self.edge_proj(self.bond_encoder(edge_attr))  # [E, edge_emb_dim]

        if self.degree_enc is not None:
            h = h + self.degree_enc(edge_index, num_nodes=N)

        spatial_pos_dense = getattr(batch, "spatial_pos_dense", None)
        if self.cfg.use_spd_bias and spatial_pos_dense is None:
            raise ValueError("cfg.use_spd_bias=True but batch.spatial_pos_dense is missing.")

        for blk in self.blocks:
            h = blk(h, edge_index, e, batch_id, spatial_pos=spatial_pos_dense)

        h = self.out_norm(h)
        g = global_mean_pool(h, batch_id)
        return h, g


####################################
# GraphGPS_CLS, with CLS, no PE
####################################

class GraphGlobalSelfAttention_CLS(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        spd_max_dist: int = 8,
        use_spd_bias: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_spd_bias = use_spd_bias
        self.spd_max_dist = spd_max_dist

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        if use_spd_bias:
            self.spatial_emb = nn.Embedding(spd_max_dist + 2, num_heads)

    def forward(
        self,
        h: torch.Tensor,                 # [N, D]
        batch: torch.Tensor,             # [N]
        spatial_pos: Optional[torch.Tensor] = None,  # [B, L, L] dense（仅节点）
        cls: Optional[torch.Tensor] = None,          # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          out_cat:  [B, L+1, D]  （包含 CLS + nodes 的 attention 输出）
          mask:     [B, L]       （原 nodes 的 mask，用于 block 里还原稀疏）
        """
        if cls is None:
            raise ValueError("GraphGlobalSelfAttention forward() requires cls of shape [B, D].")

        # sparse -> dense nodes
        h_dense, mask = to_dense_batch(h, batch=batch)  # [B, L, D], [B, L]
        B, L, D = h_dense.shape

        if cls.size(0) != B or cls.size(1) != D:
            raise ValueError(f"cls shape mismatch: got {cls.shape}, expected [{B}, {D}]")

        # cat = [CLS; nodes]
        cls_tok = cls.unsqueeze(1)                       # [B,1,D]
        h_cat = torch.cat([cls_tok, h_dense], dim=1)     # [B,L+1,D]

        # mask: CLS always valid
        cls_mask = h_dense.new_ones(B, 1, dtype=torch.bool)  # [B,1]
        mask_cat = torch.cat([cls_mask, mask], dim=1)        # [B,L+1]
        key_padding_mask = ~mask_cat                         # True=pad

        attn_mask = None
        if self.use_spd_bias:
            if spatial_pos is None:
                raise ValueError("use_spd_bias=True but spatial_pos is None.")

            if spatial_pos.shape[:2] != (B, L) or spatial_pos.shape[2] != L:
                raise ValueError(f"spatial_pos shape mismatch: got {spatial_pos.shape}, expected [{B},{L},{L}]")

            # [B,L,L] -> [B,L+1,L+1]
            spd = spatial_pos.clamp(0, self.spd_max_dist + 1)
            spd_full = spd.new_full((B, L + 1, L + 1), self.spd_max_dist + 1)
            spd_full[:, 1:, 1:] = spd
            spd_full[:, 0, 0] = 0

            # ✅ 更合理：CLS 与所有节点距离设为 0（或 1 也行）
            spd_full[:, 0, 1:] = 0
            spd_full[:, 1:, 0] = 0

            bias = self.spatial_emb(spd_full)                 # [B,L+1,L+1,H]
            bias = bias.permute(0, 3, 1, 2).contiguous()      # [B,H,L+1,L+1]
            bias = bias.view(B * self.num_heads, L + 1, L + 1)

            neg_inf = torch.finfo(h_cat.dtype).min
            pad2d = key_padding_mask.unsqueeze(1) | key_padding_mask.unsqueeze(2)  # [B,L+1,L+1]
            pad2d = pad2d.unsqueeze(1).expand(B, self.num_heads, L + 1, L + 1)     # [B,H,L+1,L+1]
            attn_mask = bias.masked_fill(pad2d.reshape(B * self.num_heads, L + 1, L + 1), neg_inf)

        out_cat, _ = self.mha(
            h_cat, h_cat, h_cat,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )  # [B, L+1, D]

        return out_cat, mask

class GraphGPSBlock_CLS(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_spd_bias: bool = False,
        spd_max_dist: int = 8,
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        nn_local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(nn_local, edge_dim=edge_dim)

        self.global_attn = GraphGlobalSelfAttention_CLS(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            use_spd_bias=use_spd_bias,
            spd_max_dist=spd_max_dist,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,                # [N, D]
        edge_index: torch.Tensor,       # [2, E]
        edge_attr: torch.Tensor,        # [E, edge_dim]
        batch: torch.Tensor,            # [N]
        spatial_pos: Optional[torch.Tensor] = None,  # [B, L, L]（dense nodes）
        cls: Optional[torch.Tensor] = None,          # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if cls is None:
            raise ValueError("GraphGPSBlock_CLS.forward requires cls [B,D].")

        # ---- local branch (nodes only, sparse)
        h_local = self.local_conv(h, edge_index, edge_attr)     # [N,D]
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # ---- prepare dense nodes (for cat-level aligned processing)
        h_dense, mask = to_dense_batch(h, batch=batch)          # [B,L,D], [B,L]
        h_local_dense, _ = to_dense_batch(h_local, batch=batch) # [B,L,D]

        B, L, D = h_dense.shape
        if cls.size(0) != B or cls.size(1) != D:
            raise ValueError(f"cls shape mismatch: got {cls.shape}, expected [{B},{D}]")

        # cat_in = [CLS; nodes]
        cls_tok = cls.unsqueeze(1)                               # [B,1,D]
        cat_in = torch.cat([cls_tok, h_dense], dim=1)            # [B,L+1,D]

        # local_cat: CLS 没有 local 分支 => 0；nodes 加 local
        zeros_cls = h_dense.new_zeros(B, 1, D)                   # [B,1,D]
        local_cat = torch.cat([zeros_cls, h_local_dense], dim=1) # [B,L+1,D]

        # ---- global branch (attention on cat)
        attn_out_cat, _mask_nodes = self.global_attn(
            h=h, batch=batch, spatial_pos=spatial_pos, cls=cls
        )  # [B,L+1,D]
        attn_out_cat = F.dropout(attn_out_cat, p=self.dropout, training=self.training)

        # ---- ✅ aligned residual + norm (CLS 和 nodes 同构)
        cat = self.norm1(cat_in + local_cat + attn_out_cat)

        # ---- ✅ aligned FFN + norm
        cat_ffn = self.ffn(cat)
        cat_ffn = F.dropout(cat_ffn, p=self.dropout, training=self.training)
        cat = self.norm2(cat + cat_ffn)

        # ---- split back
        cls_out = cat[:, 0, :]             # [B,D]
        node_out_dense = cat[:, 1:, :]     # [B,L,D]
        h_out = node_out_dense[mask]       # [N,D]

        return h_out, cls_out

class GraphGPSEncoder_CLS(nn.Module):
    def __init__(self, cfg, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg

        self.atom_encoder = AtomEncoder(cfg.hidden_dim)      # -> [N, hidden_dim]
        self.bond_encoder = BondEncoder(cfg.hidden_dim)      # -> [E, hidden_dim]

        if cfg.edge_emb_dim == cfg.hidden_dim:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(cfg.hidden_dim, cfg.edge_emb_dim)

        self.degree_enc = DegreeEncoder(cfg.max_degree, cfg.hidden_dim) if cfg.use_degree else None

        self.blocks = nn.ModuleList([
            GraphGPSBlock_CLS(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                edge_dim=cfg.edge_emb_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                use_spd_bias=cfg.use_spd_bias,
                spd_max_dist=cfg.spd_max_dist,
            )
            for _ in range(cfg.num_layers)
        ])

        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.graph_norm = nn.LayerNorm(cfg.hidden_dim)

        # 图级 CLS token 参数（共享所有层）
        self.graph_token = nn.Parameter(torch.zeros(1, cfg.hidden_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self._expect_node_dim = node_in_dim
        self._expect_edge_dim = edge_in_dim

    def forward(self, batch):
        """
        return:
          node_emb:  [N, hidden_dim]
          graph_emb: [B, hidden_dim]  （图级 CLS 表征）
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_id = batch.batch
        N = x.size(0)

        if x.dtype != torch.long:
            x = x.long()
        if edge_attr.dtype != torch.long:
            edge_attr = edge_attr.long()

        if x.size(-1) != self._expect_node_dim:
            raise ValueError(f"Unexpected node feature dim: got {x.size(-1)}, expect {self._expect_node_dim}")
        if edge_attr.size(-1) != self._expect_edge_dim:
            raise ValueError(f"Unexpected edge feature dim: got {edge_attr.size(-1)}, expect {self._expect_edge_dim}")

        # node / edge encoding
        h = self.atom_encoder(x)                           # [N, H]
        e = self.edge_proj(self.bond_encoder(edge_attr))   # [E, edge_emb_dim]

        # degree encoding on nodes
        if self.degree_enc is not None:
            h = h + self.degree_enc(edge_index, num_nodes=N)

        spatial_pos_dense = getattr(batch, "spatial_pos_dense", None)
        if self.cfg.use_spd_bias and spatial_pos_dense is None:
            raise ValueError("cfg.use_spd_bias=True but batch.spatial_pos_dense is missing.")

        # init CLS per graph
        B = int(batch_id.max().item()) + 1
        cls = self.graph_token.expand(B, -1).contiguous()  # [B, H]

        # stacked blocks
        for blk in self.blocks:
            h, cls = blk(
                h=h,
                edge_index=edge_index,
                edge_attr=e,
                batch=batch_id,
                spatial_pos=spatial_pos_dense,
                cls=cls,
            )

        h = self.out_norm(h)
        # graph_emb = self.graph_norm(cls)
        graph_emb = self.graph_norm(cls)

        return h, graph_emb


####################################
# GraphGPS_CLS, with CLS, Graphormer's style PE
####################################
class GraphGlobalSelfAttention_CLS_Graphormer(nn.Module):
    """
    Graphormer-style global self-attention with SPD-based bias.

    - 输入：稀疏节点特征 h[N,D] + batch[N] + CLS[B,D]
    - 在 logits 上加 pairwise SPD bias（per-head）
    - CLS 不参与 SPD，CLS<->nodes 的 bias = 0
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        spd_max_dist: int = 8,
        num_edge_types: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spd_max_dist = spd_max_dist
        self.num_edge_types = num_edge_types

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.spatial_emb = nn.Embedding(spd_max_dist + 3, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        vocab = (self.spd_max_dist + 1) * (self.num_edge_types + 1)
        self.edge_path_emb = nn.Embedding(vocab, num_heads, padding_idx=0)

        # hop ids 1..max_dist (buffer)
        hop = torch.arange(1, self.spd_max_dist + 1, dtype=torch.long)
        self.register_buffer("_hop_ids", hop, persistent=False)

    def forward(
        self,
        h_dense: torch.Tensor,                 # [B,L,D]
        mask: torch.Tensor,              # [B,L]
        batch: torch.Tensor,             # [N]
        spatial_pos: torch.Tensor,       # [B, L, L] (SPD indices, 0..spd_max+1)
        edge_input: torch.Tensor,        # [B, L, L, max_dist] (edge type ids)
        cls: torch.Tensor,               # [B, D]
    ):
        B, L, D = h_dense.shape

        if cls.dim() != 2:
            raise ValueError(f"cls must be [B,D], got {cls.shape}")
        if cls.size(0) != B or cls.size(1) != D:
            raise ValueError(f"cls shape mismatch: got {cls.shape}, expected [{B},{D}]")

        if spatial_pos.dim() != 3:
            raise ValueError(f"spatial_pos must be [B,L,L], got {spatial_pos.shape}")
        if spatial_pos.shape[0] != B or spatial_pos.shape[1] != L or spatial_pos.shape[2] != L:
            raise ValueError(
                f"spatial_pos shape mismatch: got {spatial_pos.shape}, expected [{B},{L},{L}]"
            )
        if spatial_pos.dtype != torch.long:
            spatial_pos = spatial_pos.long()
        assert edge_input.size(-1) == self.spd_max_dist

        ########### [CLS; nodes]
        cls_tok = cls.unsqueeze(1)                    # [B,1,D]
        h_cat = torch.cat([cls_tok, h_dense], dim=1)  # [B,L+1,D]

        # token mask: True=valid token
        cls_valid = mask.new_ones(B, 1)               # [B,1] CLS always valid
        tok_valid = torch.cat([cls_valid, mask], dim=1)     # [B,L+1]
        key_padding_mask = ~tok_valid                       # [B,L+1], True=pad

        ########### spatial_pos: [B,L,L]，值域 0..spd_max+1
        spd = spatial_pos.clamp(0, self.spd_max_dist + 1)
        spd = spd + 1
        pair_valid = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B,L,L]
        spd = spd.masked_fill(~pair_valid, 0)

        spd_full = spd.new_zeros((B, L + 1, L + 1))       # [B,L+1,L+1]
        spd_full[:, 1:, 1:] = spd                         # 只对 nodes 部分加 SPD

        # embedding -> per-head bias
        bias = self.spatial_emb(spd_full)                 # [B,L+1,L+1,H]
        bias = bias.permute(0, 3, 1, 2).contiguous()      # [B,H,L+1,L+1]

        ########### Path/edge bias
        hop = self._hop_ids.view(1, 1, 1, self.spd_max_dist)
        M = self.num_edge_types + 1
        # idx: [B,L,L,max_dist]
        idx = hop * M + edge_input
        valid = edge_input != 0
        idx = idx.masked_fill(~valid, 0)

        # lookup: [B,L,L,max_dist,H]
        edge_bias = self.edge_path_emb(idx)  # per-head bias [B,L,L,max_dist,H]
        valid_f = valid.unsqueeze(-1).to(edge_bias.dtype) # [B,L,L,max_dist,1]
        denom = valid_f.sum(dim=3).clamp(min=1.0)         # [B,L,L,1]
        edge_bias = (edge_bias * valid_f).sum(dim=3) / denom  # [B,L,L,H]

        # add into nodes×nodes block only (CLS excluded)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + edge_bias.permute(0, 3, 1, 2).contiguous()


        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + t   # nodes -> CLS
        bias[:, :, 0, :]  = bias[:, :, 0, :]  + t   # CLS  -> all (含 CLS->CLS)
        bias = bias.view(B * self.num_heads, L + 1, L + 1)

        # 把 pad 的行/列全部设为 -inf
        neg_inf = torch.finfo(h_cat.dtype).min
        pad2d = key_padding_mask.unsqueeze(1) | key_padding_mask.unsqueeze(2)  # [B,L+1,L+1]
        pad2d = pad2d.unsqueeze(1).expand(B, self.num_heads, L + 1, L + 1)     # [B,H,L+1,L+1]
        attn_mask = bias.masked_fill(
            pad2d.reshape(B * self.num_heads, L + 1, L + 1), neg_inf
        )

        out_cat, _ = self.mha(
            h_cat, h_cat, h_cat,
            key_padding_mask=key_padding_mask,  # [B,L+1]
            attn_mask=attn_mask,               # [B*H,L+1,L+1] or None
            need_weights=False,
        )  # [B,L+1,D]

        return out_cat, mask

class GraphGPSBlock_CLS_GraphormerSPD(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        spd_max_dist: int = 8,
        num_edge_types: int = 0,
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types

        nn_local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(nn_local, edge_dim=edge_dim)

        self.global_attn = GraphGlobalSelfAttention_CLS_Graphormer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            spd_max_dist=spd_max_dist,
            num_edge_types=num_edge_types,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,                # [N, D]
        edge_index: torch.Tensor,       # [2, E]
        edge_attr: torch.Tensor,        # [E, edge_dim]
        batch: torch.Tensor,            # [N]
        spatial_pos: torch.Tensor,      # [B, L, L]
        edge_input: torch.Tensor,       # [B, L, L, max_dist]
        cls: torch.Tensor,              # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ---- local branch
        h_local = self.local_conv(h, edge_index, edge_attr)     # [N,D]
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # ---- dense nodes（只是为了 residual 对齐）
        h_dense, mask = to_dense_batch(h, batch=batch)          # [B,L,D], [B,L]
        h_local_dense, _ = to_dense_batch(h_local, batch=batch) # [B,L,D]

        B, L, D = h_dense.shape
        if cls.size(0) != B or cls.size(1) != D:
            raise ValueError(f"cls shape mismatch: got {cls.shape}, expected [{B},{D}]")

        # cat_in = [CLS; nodes]
        cls_tok = cls.unsqueeze(1)                               # [B,1,D]
        cat_in = torch.cat([cls_tok, h_dense], dim=1)            # [B,L+1,D]

        # local_cat: CLS 没有 local 分支 => 0；nodes 有 local
        zeros_cls = h_dense.new_zeros(B, 1, D)                   # [B,1,D]
        local_cat = torch.cat([zeros_cls, h_local_dense], dim=1) # [B,L+1,D]

        # ---- global branch: Graphormer-style SPD bias
        attn_out_cat, _ = self.global_attn(
            h_dense=h_dense,
            mask=mask,
            batch=batch,
            spatial_pos=spatial_pos,
            edge_input=edge_input,
            cls=cls,
        )  # [B,L+1,D]
        attn_out_cat = F.dropout(attn_out_cat, p=self.dropout, training=self.training)

        # residual + norm
        # tok_valid: [B,L+1]，True=valid
        tok_valid = torch.cat([mask.new_ones(B, 1), mask], dim=1)
        cat_in = cat_in * tok_valid.unsqueeze(-1)
        local_cat = local_cat * tok_valid.unsqueeze(-1)
        attn_out_cat = attn_out_cat * tok_valid.unsqueeze(-1)
        cat = self.norm1(cat_in + local_cat + attn_out_cat)

        # FFN + residual + norm
        cat_ffn = self.ffn(cat)
        cat_ffn = F.dropout(cat_ffn, p=self.dropout, training=self.training)
        cat = self.norm2(cat + cat_ffn)

        # split back
        cls_out = cat[:, 0, :]             # [B,D]
        node_out_dense = cat[:, 1:, :]     # [B,L,D]
        h_out = node_out_dense[mask]       # [N,D]

        return h_out, cls_out

class GraphGPSEncoder_CLS_GraphormerSPD(nn.Module):
    """
    GraphGPS + CLS，但 global attention 使用 Graphormer-style SPD bias；
    SPD (shortest-path distance) 在 encoder 内部根据 edge_index + batch 现算。
    """
    def __init__(self, cfg, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg

        self.atom_encoder = AtomEncoder(cfg.hidden_dim)      # [N,H]
        self.bond_encoder = BondEncoder(cfg.hidden_dim)      # [E,H]

        if cfg.edge_emb_dim == cfg.hidden_dim:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(cfg.hidden_dim, cfg.edge_emb_dim)

        self.degree_enc = DegreeEncoder(cfg.max_degree, cfg.hidden_dim) if cfg.use_degree else None

        if hasattr(cfg, "num_edge_types") and int(cfg.num_edge_types) > 0:
            num_edge_types = int(cfg.num_edge_types)
        else:
            from ogb.utils.features import get_bond_feature_dims
            dims = get_bond_feature_dims()  # len=3
            num_edge_types = int(dims[0]) * int(dims[1]) * int(dims[2])
        self.num_edge_types = num_edge_types

        self.blocks = nn.ModuleList([
            GraphGPSBlock_CLS_GraphormerSPD(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                edge_dim=cfg.edge_emb_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                spd_max_dist=cfg.spd_max_dist,
                num_edge_types=num_edge_types,
            )
            for _ in range(cfg.num_layers)
        ])

        self.out_norm   = nn.LayerNorm(cfg.hidden_dim)
        self.graph_norm = nn.LayerNorm(cfg.hidden_dim)

        # graph-level CLS token（共享）
        self.graph_token = nn.Parameter(torch.zeros(1, cfg.hidden_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self._expect_node_dim = node_in_dim
        self._expect_edge_dim = edge_in_dim

        self.spd_max_dist = cfg.spd_max_dist



    # ---------- forward ----------
    def forward(self, batch):
        """
        return:
          node_emb:  [N, hidden_dim]
          graph_emb: [B, hidden_dim]
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_id = batch.batch
        N = x.size(0)

        if x.dtype != torch.long:
            x = x.long()
        if edge_attr.dtype != torch.long:
            edge_attr = edge_attr.long()

        if x.size(-1) != self._expect_node_dim:
            raise ValueError(f"Unexpected node feature dim: got {x.size(-1)}, expect {self._expect_node_dim}")
        if edge_attr.size(-1) != self._expect_edge_dim:
            raise ValueError(f"Unexpected edge feature dim: got {edge_attr.size(-1)}, expect {self._expect_edge_dim}")

        # node / edge encoding
        h = self.atom_encoder(x)                           # [N,H]
        e = self.edge_proj(self.bond_encoder(edge_attr))   # [E,edge_emb_dim]

        # degree encoding
        if self.degree_enc is not None:
            h = h + self.degree_enc(edge_index, num_nodes=N)
        if not hasattr(batch, "spatial_pos_dense"):
            raise ValueError("use_spd_bias=True but batch has no spatial_pos_dense. Did you set collate_fn?")
        spatial_pos_dense = batch.spatial_pos_dense
        if spatial_pos_dense is None:
            raise ValueError("use_spd_bias=True but batch.spatial_pos_dense is None.")
        if spatial_pos_dense.dtype != torch.long:
            spatial_pos_dense = spatial_pos_dense.long()

        # edge_input from collate
        if not hasattr(batch, "edge_input_dense"):
            raise ValueError("use_spd_bias=True but batch has no edge_input_dense. Did you build SPD+EDGE cache and collate it?")
        edge_input_dense = batch.edge_input_dense
        if edge_input_dense is None:
            raise ValueError("batch.edge_input_dense is None.")
        if edge_input_dense.dtype != torch.long:
            edge_input_dense = edge_input_dense.long()


        # init CLS for each graph
        B = int(batch_id.max().item()) + 1
        cls = self.graph_token.expand(B, -1).contiguous()  # [B,H]

        # stacked blocks
        for blk in self.blocks:
            h, cls = blk(
                h=h,
                edge_index=edge_index,
                edge_attr=e,
                batch=batch_id,
                spatial_pos=spatial_pos_dense,
                edge_input=edge_input_dense,
                cls=cls,
            )

        h = self.out_norm(h)
        graph_emb = self.graph_norm(cls)

        return h, graph_emb


####################################
# GraphGPS_CLS, with CLS, add Pearl PE
####################################

class PearlAbsolutePE(nn.Module):
    """
    R-PEARL (stronger): Random identifiers -> structure-only ResGIN Φ -> mean over M samples.

    - 结构-only：只用 edge_index，不看节点/边属性
    - M 个随机标识样本并行展开 (M copies)
    - Φ 使用 GINConv(MLP) + residual + pre-LN
    """
    def __init__(
        self,
        hidden_dim: int,
        q_dim: int = 16,
        num_samples: int = 8,          # M
        num_layers: int = 4,           # 建议比原来 2 稍深一点
        mlp_hidden_mult: int = 2,      # GIN 内部 MLP 宽度倍率
        dropout: float = 0.1,
        deterministic_eval: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.q_dim = int(q_dim)
        self.num_samples = int(num_samples)
        self.num_layers = int(num_layers)
        self.mlp_hidden_mult = int(mlp_hidden_mult)
        self.dropout = float(dropout)
        self.deterministic_eval = bool(deterministic_eval)

        self.in_proj = nn.Linear(self.q_dim, self.hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.mlp_hidden_mult * self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.mlp_hidden_mult * self.hidden_dim, self.hidden_dim),
            )
            # train_eps=True => eps 可学习（更强）
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def _sample_q(self, N: int, device, dtype):
        if (not self.training) and self.deterministic_eval:
            # 局部固定 seed，不污染外部 RNG
            devs = []
            if device.type == "cuda":
                devs = [device.index] if device.index is not None else list(range(torch.cuda.device_count()))
            with torch.random.fork_rng(devices=devs):
                torch.manual_seed(0)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(0)

                q = torch.randn(self.num_samples, N, self.q_dim, device=device, dtype=dtype)
        else:
            q = torch.randn(self.num_samples, N, self.q_dim, device=device, dtype=dtype)
        return q

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        edge_index: [2, E]，整个 batch 的稀疏边（各图互不连通）
        num_nodes: N
        return:
          pe: [N, hidden_dim]
        """
        N = int(num_nodes)
        device = edge_index.device
        dtype = torch.float32

        # 1) 随机节点标识 q: [M,N,q_dim]
        q = self._sample_q(N, device=device, dtype=dtype)

        # 2) 向量化展开：把 M 个样本当作 M 个“独立图副本”
        M = self.num_samples
        q_flat = q.reshape(M * N, self.q_dim)
        x = self.in_proj(q_flat)  # [M*N, H]

        # 复制 edge_index 并按样本偏移节点编号
        E = edge_index.size(1)
        offsets = (torch.arange(M, device=device, dtype=edge_index.dtype) * N).view(1, M, 1)  # [1,M,1]
        edge_rep = edge_index.view(2, 1, E).repeat(1, M, 1) + offsets                         # [2,M,E]
        edge_rep = edge_rep.reshape(2, M * E)                                                 # [2,M*E]

        # 3) 结构 MPNN Φ：ResGIN + pre-LN
        for conv, ln in zip(self.convs, self.norms):
            h_in = x
            x = ln(x)
            x = conv(x, edge_rep)                 # GINConv 内部会 sum 聚合 + MLP
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + h_in                          # residual

        x = self.out_norm(x)                      # [M*N, H]
        x = x.view(M, N, self.hidden_dim)         # [M,N,H]

        # 4) 聚合（ρ）：mean over samples -> [N,H]
        pe = x.mean(dim=0)
        return pe

class GraphGPSEncoder_CLS_PEARL(nn.Module):
    def __init__(self, cfg, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg

        self.atom_encoder = AtomEncoder(cfg.hidden_dim)      # -> [N, hidden_dim]
        self.bond_encoder = BondEncoder(cfg.hidden_dim)      # -> [E, hidden_dim]

        if cfg.edge_emb_dim == cfg.hidden_dim:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(cfg.hidden_dim, cfg.edge_emb_dim)

        self.degree_enc = DegreeEncoder(cfg.max_degree, cfg.hidden_dim) if cfg.use_degree else None

        # pearl part
        self.pearl_pe = PearlAbsolutePE(
                hidden_dim=cfg.hidden_dim,
                q_dim=int(getattr(cfg, "pearl_q_dim", 16)),
                num_samples=int(getattr(cfg, "pearl_num_samples", 8)),
                num_layers=int(getattr(cfg, "pearl_num_layers", 2)),
                dropout=float(getattr(cfg, "pearl_dropout", 0.0)),
                deterministic_eval=bool(getattr(cfg, "pearl_deterministic_eval", True)),
            )
        self.pearl_scale = nn.Parameter(torch.tensor(0.1))

        self.pearl_fuse = str(getattr(cfg, "pearl_fuse", "add")).lower()
        if self.pearl_fuse not in ("add", "concat"):
            raise ValueError(f"cfg.pearl_fuse must be 'add' or 'concat', got {self.pearl_fuse}")

        if self.pearl_fuse == "concat":
            self.pearl_fuse_proj = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim)
        else:
            self.pearl_fuse_proj = None

        # blocks part
        self.blocks = nn.ModuleList([
            GraphGPSBlock_CLS(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                edge_dim=cfg.edge_emb_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                use_spd_bias=False,
            )
            for _ in range(cfg.num_layers)
        ])

        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.graph_norm = nn.LayerNorm(cfg.hidden_dim)

        # cls token
        self.graph_token = nn.Parameter(torch.zeros(1, cfg.hidden_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self._expect_node_dim = node_in_dim
        self._expect_edge_dim = edge_in_dim

    def forward(self, batch):
        """
        return:
          node_emb:  [N, hidden_dim]
          graph_emb: [B, hidden_dim]  （图级 CLS 表征）
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_id = batch.batch
        N = x.size(0)

        if x.dtype != torch.long:
            x = x.long()
        if edge_attr.dtype != torch.long:
            edge_attr = edge_attr.long()

        if x.size(-1) != self._expect_node_dim:
            raise ValueError(f"Unexpected node feature dim: got {x.size(-1)}, expect {self._expect_node_dim}")
        if edge_attr.size(-1) != self._expect_edge_dim:
            raise ValueError(f"Unexpected edge feature dim: got {edge_attr.size(-1)}, expect {self._expect_edge_dim}")

        # node / edge encoding
        h = self.atom_encoder(x)                           # [N, H]
        e = self.edge_proj(self.bond_encoder(edge_attr))   # [E, edge_emb_dim]

        # degree encoding on nodes
        if self.degree_enc is not None:
            h = h + self.degree_enc(edge_index, num_nodes=N)

        pe = self.pearl_pe(edge_index=edge_index, num_nodes=N)  # [N,H]
        pe = pe.to(dtype=h.dtype, device=h.device)
        if self.pearl_fuse == "add":
            h = h + pe * self.pearl_scale
        else:  # concat
            h = self.pearl_fuse_proj(torch.cat([h, pe], dim=-1))

        # init CLS per graph
        B = int(batch_id.max().item()) + 1
        cls = self.graph_token.expand(B, -1).contiguous()  # [B, H]

        # stacked blocks
        for blk in self.blocks:
            h, cls = blk(
                h=h,
                edge_index=edge_index,
                edge_attr=e,
                batch=batch_id,
                spatial_pos=None,
                cls=cls,
            )

        h = self.out_norm(h)
        # graph_emb = self.graph_norm(cls)
        graph_emb = self.graph_norm(cls)

        return h, graph_emb

####################################
# GraphGPS_CLS, with CLS, add both spd and Pearl PE
####################################
class GraphGPSEncoder_CLS_GraphormerSPD_Pearl(nn.Module):
    def __init__(self, cfg, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg

        self.atom_encoder = AtomEncoder(cfg.hidden_dim)      # [N,H]
        self.bond_encoder = BondEncoder(cfg.hidden_dim)      # [E,H]

        if cfg.edge_emb_dim == cfg.hidden_dim:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(cfg.hidden_dim, cfg.edge_emb_dim)

        self.degree_enc = DegreeEncoder(cfg.max_degree, cfg.hidden_dim) if cfg.use_degree else None

        if hasattr(cfg, "num_edge_types") and int(cfg.num_edge_types) > 0:
            num_edge_types = int(cfg.num_edge_types)
        else:
            from ogb.utils.features import get_bond_feature_dims
            dims = get_bond_feature_dims()  # len=3
            num_edge_types = int(dims[0]) * int(dims[1]) * int(dims[2])
        self.num_edge_types = num_edge_types


        # pearl part
        self.pearl_pe = PearlAbsolutePE(
                hidden_dim=cfg.hidden_dim,
                q_dim=int(getattr(cfg, "pearl_q_dim", 16)),
                num_samples=int(getattr(cfg, "pearl_num_samples", 8)),
                num_layers=int(getattr(cfg, "pearl_num_layers", 2)),
                dropout=float(getattr(cfg, "pearl_dropout", 0.0)),
                deterministic_eval=bool(getattr(cfg, "pearl_deterministic_eval", True)),
            )
        self.pearl_scale = nn.Parameter(torch.tensor(0.1))

        self.pearl_fuse = str(getattr(cfg, "pearl_fuse", "add")).lower()
        if self.pearl_fuse not in ("add", "concat"):
            raise ValueError(f"cfg.pearl_fuse must be 'add' or 'concat', got {self.pearl_fuse}")

        if self.pearl_fuse == "concat":
            self.pearl_fuse_proj = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim)
        else:
            self.pearl_fuse_proj = None

        # block part
        self.blocks = nn.ModuleList([
            GraphGPSBlock_CLS_GraphormerSPD(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                edge_dim=cfg.edge_emb_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                spd_max_dist=cfg.spd_max_dist,
                num_edge_types=num_edge_types,
            )
            for _ in range(cfg.num_layers)
        ])

        self.out_norm   = nn.LayerNorm(cfg.hidden_dim)
        self.graph_norm = nn.LayerNorm(cfg.hidden_dim)

        # graph-level CLS token（共享）
        self.graph_token = nn.Parameter(torch.zeros(1, cfg.hidden_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self._expect_node_dim = node_in_dim
        self._expect_edge_dim = edge_in_dim

        self.spd_max_dist = cfg.spd_max_dist

    # ---------- forward ----------
    def forward(self, batch):
        """
        return:
          node_emb:  [N, hidden_dim]
          graph_emb: [B, hidden_dim]
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_id = batch.batch
        N = x.size(0)

        if x.dtype != torch.long:
            x = x.long()
        if edge_attr.dtype != torch.long:
            edge_attr = edge_attr.long()

        if x.size(-1) != self._expect_node_dim:
            raise ValueError(f"Unexpected node feature dim: got {x.size(-1)}, expect {self._expect_node_dim}")
        if edge_attr.size(-1) != self._expect_edge_dim:
            raise ValueError(f"Unexpected edge feature dim: got {edge_attr.size(-1)}, expect {self._expect_edge_dim}")

        # node / edge encoding
        h = self.atom_encoder(x)                           # [N,H]
        e = self.edge_proj(self.bond_encoder(edge_attr))   # [E,edge_emb_dim]

        # degree encoding
        if self.degree_enc is not None:
            h = h + self.degree_enc(edge_index, num_nodes=N)
        pe = self.pearl_pe(edge_index=edge_index, num_nodes=N)  # [N,H]
        pe = pe.to(dtype=h.dtype, device=h.device)
        if self.pearl_fuse == "add":
            h = h + pe * self.pearl_scale
        else:  # concat
            h = self.pearl_fuse_proj(torch.cat([h, pe], dim=-1))

        if not hasattr(batch, "spatial_pos_dense"):
            raise ValueError("use_spd_bias=True but batch has no spatial_pos_dense. Did you set collate_fn?")
        spatial_pos_dense = batch.spatial_pos_dense
        if spatial_pos_dense is None:
            raise ValueError("use_spd_bias=True but batch.spatial_pos_dense is None.")
        if spatial_pos_dense.dtype != torch.long:
            spatial_pos_dense = spatial_pos_dense.long()

        # edge_input from collate
        if not hasattr(batch, "edge_input_dense"):
            raise ValueError("use_spd_bias=True but batch has no edge_input_dense. Did you build SPD+EDGE cache and collate it?")
        edge_input_dense = batch.edge_input_dense
        if edge_input_dense is None:
            raise ValueError("batch.edge_input_dense is None.")
        if edge_input_dense.dtype != torch.long:
            edge_input_dense = edge_input_dense.long()
        # edge_input_dense: [B,L,L,K]
        max_id = int(edge_input_dense.max().item())
        if max_id > self.num_edge_types:
            raise ValueError(f"edge_input_dense has id {max_id} > num_edge_types {self.num_edge_types}. Packing/dims mismatch.")



        # init CLS for each graph
        B = int(batch_id.max().item()) + 1
        cls = self.graph_token.expand(B, -1).contiguous()  # [B,H]

        # stacked blocks
        for blk in self.blocks:
            h, cls = blk(
                h=h,
                edge_index=edge_index,
                edge_attr=e,
                batch=batch_id,
                spatial_pos=spatial_pos_dense,
                edge_input=edge_input_dense,
                cls=cls,
            )

        h = self.out_norm(h)
        graph_emb = self.graph_norm(cls)

        return h, graph_emb
