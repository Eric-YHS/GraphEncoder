import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from torch_geometric.nn import GINEConv, global_mean_pool
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
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spd_max_dist = spd_max_dist

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )


        self.spatial_emb = nn.Embedding(spd_max_dist + 2, num_heads, padding_idx=0)

    def forward(
        self,
        h: torch.Tensor,                 # [N, D]
        batch: torch.Tensor,             # [N]
        spatial_pos: torch.Tensor,       # [B, L, L] (SPD indices, 0..spd_max+1)
        cls: torch.Tensor,               # [B, D]
    ):
        if cls is None:
            raise ValueError("GraphGlobalSelfAttention_CLS_Graphormer requires cls [B,D].")

        # sparse -> dense
        h_dense, mask = to_dense_batch(h, batch=batch)  # [B,L,D], [B,L]
        B, L, D = h_dense.shape

        if cls.size(0) != B or cls.size(1) != D:
            raise ValueError(f"cls shape mismatch: got {cls.shape}, expected [{B},{D}]")

        # [CLS; nodes]
        cls_tok = cls.unsqueeze(1)                       # [B,1,D]
        h_cat  = torch.cat([cls_tok, h_dense], dim=1)    # [B,L+1,D]

        # padding mask for tokens: True = pad
        cls_mask  = h_dense.new_ones(B, 1, dtype=torch.bool)  # CLS 永远有效
        mask_cat  = torch.cat([cls_mask, mask], dim=1)        # [B,L+1]
        key_padding_mask = ~mask_cat                          # [B,L+1], True = pad

        if spatial_pos is None:
            raise ValueError("spatial_pos is None.")

        if spatial_pos.shape[:2] != (B, L) or spatial_pos.shape[2] != L:
            raise ValueError(
                f"spatial_pos shape mismatch: got {spatial_pos.shape}, expected [{B},{L},{L}]"
            )

        # spatial_pos: [B,L,L]，值域 0..spd_max+1
        spd = spatial_pos.clamp(0, self.spd_max_dist + 1)

        # 扩展 CLS：CLS 行列 SPD=0（不加任何 bias）
        spd_full = spd.new_zeros((B, L + 1, L + 1))       # [B,L+1,L+1]
        spd_full[:, 1:, 1:] = spd                         # 只对 nodes 部分加 SPD

        # embedding -> per-head bias
        bias = self.spatial_emb(spd_full)                 # [B,L+1,L+1,H]
        bias = bias.permute(0, 3, 1, 2).contiguous()      # [B,H,L+1,L+1]
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

        self.global_attn = GraphGlobalSelfAttention_CLS_Graphormer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
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
        spatial_pos: torch.Tensor,      # [B, L, L]
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
            h=h,
            batch=batch,
            spatial_pos=spatial_pos,
            cls=cls,
        )  # [B,L+1,D]
        attn_out_cat = F.dropout(attn_out_cat, p=self.dropout, training=self.training)

        # residual + norm
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

        self.blocks = nn.ModuleList([
            GraphGPSBlock_CLS_GraphormerSPD(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                edge_dim=cfg.edge_emb_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
                spd_max_dist=cfg.spd_max_dist,
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

    # ---------- 最短路径(按 batch) 计算 SPD dense ----------
    def _compute_spatial_pos_dense(
        self,
        edge_index: torch.Tensor,  # [2,E]
        batch: torch.Tensor,       # [N]
    ) -> torch.Tensor:
        """
        返回 Graphormer 风格的 SPD index：
          spatial_pos[b, i, j] ∈ {0..spd_max+1}
          0: padding / unreachable
          1: self (dist=0)
          2..spd_max+1: hop = 1..spd_max
        """
        device = edge_index.device
        N = batch.size(0)
        B = int(batch.max().item()) + 1

        # 使用 to_dense_batch 来固定每个图里节点的顺序
        idx = torch.arange(N, device=device)
        idx_dense, mask = to_dense_batch(idx.unsqueeze(-1), batch=batch)  # [B,L,1], [B,L]
        idx_dense = idx_dense.squeeze(-1)                                  # [B,L]
        _, L = mask.shape

        spd = idx_dense.new_zeros((B, L, L), dtype=torch.long)            # 全 0（padding）

        for b in range(B):
            node_ids = idx_dense[b, mask[b]]  # 当前图真实节点的全局 index，shape [Lb]
            Lb = node_ids.size(0)
            if Lb == 0:
                continue

            # global -> local 映射
            global2local = -torch.ones(N, device=device, dtype=torch.long)
            global2local[node_ids] = torch.arange(Lb, device=device, dtype=torch.long)

            # 选出属于该图的边
            edge_mask_b = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
            ei = edge_index[:, edge_mask_b]          # [2, E_b]
            if ei.numel() == 0:
                dist_idx = torch.eye(Lb, device=device, dtype=torch.long) + 1  # diag=1, off=0
                spd[b, :Lb, :Lb] = dist_idx
                continue

            src = global2local[ei[0]]    # [E_b]
            dst = global2local[ei[1]]

            # adjacency list（无向）
            adj = [[] for _ in range(Lb)]
            src_list = src.tolist()
            dst_list = dst.tolist()
            for s, d in zip(src_list, dst_list):
                if s < 0 or d < 0:
                    continue
                adj[s].append(d)
                adj[d].append(s)

            INF = self.spd_max_dist + 1
            dist_mat = torch.full((Lb, Lb), INF, device=device, dtype=torch.long)

            # BFS from each node
            for u in range(Lb):
                dist = [-1] * Lb
                dist[u] = 0
                q = deque([u])
                while q:
                    v = q.popleft()
                    if dist[v] >= self.spd_max_dist:
                        continue
                    for w in adj[v]:
                        if dist[w] == -1:
                            dist[w] = dist[v] + 1
                            q.append(w)
                for v in range(Lb):
                    d = dist[v]
                    if d == -1:
                        continue
                    if d > self.spd_max_dist:
                        d = self.spd_max_dist
                    dist_mat[u, v] = d

            # d ∈ [0..spd_max] -> index = d+1；INF -> 0
            dist_idx = dist_mat.clone()
            unreachable = (dist_idx == INF)
            dist_idx[unreachable] = -1
            dist_idx = dist_idx + 1                # -1 -> 0, 0 -> 1, ...
            dist_idx = dist_idx.clamp(min=0, max=self.spd_max_dist + 1)

            spd[b, :Lb, :Lb] = dist_idx

        return spd  # [B,L,L]

    # ---------- forward ----------
    def forward(self, batch):
        """
        return:
          node_emb:  [N, hidden_dim]
          graph_emb: [B, hidden_dim]  （graph-level CLS）
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
                cls=cls,
            )

        h = self.out_norm(h)
        graph_emb = self.graph_norm(cls)

        return h, graph_emb

####################################
# GraphGPS_CLS, with CLS, GraphGPS's style PE
####################################
class GraphGlobalSelfAttention_CLS_NoBias(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(
        self,
        h: torch.Tensor,     # [N, D]
        batch: torch.Tensor, # [N]
        cls: torch.Tensor,   # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return:
          out_cat: [B, L+1, D]  (CLS + nodes)
          mask:    [B, L]       node mask
        """
        h_dense, mask = to_dense_batch(h, batch=batch)  # [B,L,D], [B,L]
        B, L, D = h_dense.shape
        if cls.size(0) != B or cls.size(1) != D:
            raise ValueError(f"cls shape mismatch: got {cls.shape}, expected [{B},{D}]")

        cls_tok = cls.unsqueeze(1)                   # [B,1,D]
        h_cat = torch.cat([cls_tok, h_dense], dim=1) # [B,L+1,D]

        cls_mask = h_dense.new_ones(B, 1, dtype=torch.bool)   # [B,1]
        mask_cat = torch.cat([cls_mask, mask], dim=1)         # [B,L+1]
        key_padding_mask = ~mask_cat                          # True=pad

        out_cat, _ = self.mha(
            h_cat, h_cat, h_cat,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out_cat, mask

class GraphGPSBlock_CLS_GPSStyle(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout

        nn_local = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_conv = GINEConv(nn_local, edge_dim=edge_dim)

        self.global_attn = GraphGlobalSelfAttention_CLS_NoBias(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
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
        h: torch.Tensor,          # [N,D]
        edge_index: torch.Tensor, # [2,E]
        edge_attr: torch.Tensor,  # [E,edge_dim]
        batch: torch.Tensor,      # [N]
        cls: torch.Tensor,        # [B,D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # local
        h_local = self.local_conv(h, edge_index, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)

        # dense alignment for "cat-level" residual
        h_dense, mask = to_dense_batch(h, batch=batch)           # [B,L,D]
        h_local_dense, _ = to_dense_batch(h_local, batch=batch)  # [B,L,D]
        B, L, D = h_dense.shape

        cls_tok = cls.unsqueeze(1)                                # [B,1,D]
        cat_in = torch.cat([cls_tok, h_dense], dim=1)             # [B,L+1,D]
        zeros_cls = h_dense.new_zeros(B, 1, D)
        local_cat = torch.cat([zeros_cls, h_local_dense], dim=1)  # [B,L+1,D]

        # global
        attn_out_cat, _ = self.global_attn(h=h, batch=batch, cls=cls)
        attn_out_cat = F.dropout(attn_out_cat, p=self.dropout, training=self.training)

        # residual + norm
        cat = self.norm1(cat_in + local_cat + attn_out_cat)

        # ffn
        cat_ffn = self.ffn(cat)
        cat_ffn = F.dropout(cat_ffn, p=self.dropout, training=self.training)
        cat = self.norm2(cat + cat_ffn)

        # split back
        cls_out = cat[:, 0, :]          # [B,D]
        node_out_dense = cat[:, 1:, :]  # [B,L,D]
        h_out = node_out_dense[mask]    # [N,D]
        return h_out, cls_out

class GraphGPSEncoder_CLS_GPSSPD(nn.Module):
    """
    Strict GraphGPS-style:
      - SPD is used to create node-wise structural encoding (SE)
      - SE is fused into node features BEFORE blocks
      - global attention DOES NOT consume SPD bias
      - CLS token kept (graph token)
    """
    def __init__(self, cfg, node_in_dim: int, edge_in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim

        # ====== your existing encoders ======
        self.atom_encoder = AtomEncoder(cfg.hidden_dim)
        self.bond_encoder = BondEncoder(cfg.hidden_dim)

        if cfg.edge_emb_dim == cfg.hidden_dim:
            self.edge_proj = nn.Identity()
        else:
            self.edge_proj = nn.Linear(cfg.hidden_dim, cfg.edge_emb_dim)

        self.degree_enc = DegreeEncoder(cfg.max_degree, cfg.hidden_dim) if cfg.use_degree else None

        # ====== GraphGPS blocks (NO SPD BIAS) ======
        self.blocks = nn.ModuleList([
            GraphGPSBlock_CLS_GPSStyle(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                edge_dim=cfg.edge_emb_dim,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.graph_norm = nn.LayerNorm(cfg.hidden_dim)

        # CLS graph token
        self.graph_token = nn.Parameter(torch.zeros(1, cfg.hidden_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self._expect_node_dim = node_in_dim
        self._expect_edge_dim = edge_in_dim

        # ====== SPD node structural encoding ======
        # use cfg.use_spd_node_pe as switch; if you don't have it, reuse cfg.use_spd_bias but semantics change
        self.use_spd_node_pe = bool(getattr(cfg, "use_spd_node_pe", True))
        self.spd_max_dist = int(getattr(cfg, "spd_max_dist", 8))
        self.spd_inf = self.spd_max_dist + 1
        spd_bins = self.spd_max_dist + 2  # 0..max+1

        # mode: "sum" (strict, simple) or "concat" (often stronger but slightly deviates from minimalism)
        self.spd_fuse_mode = str(getattr(cfg, "spd_fuse_mode", "concat"))  # "sum" or "concat"

        self.spd_hist_mlp = nn.Sequential(
            nn.Linear(spd_bins, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        if self.spd_fuse_mode == "concat":
            self.spd_cat_proj = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        else:
            self.spd_cat_proj = None

    def _inject_spd_node_se(self, h: torch.Tensor, batch_id: torch.Tensor, spatial_pos_dense: torch.Tensor) -> torch.Tensor:
        """
        spatial_pos_dense: [B,L,L] (padded with inf=max+1)
        Return: updated h [N,H] with SE fused.
        """
        device = h.device
        spd = spatial_pos_dense.to(device=device, dtype=torch.long)  # [B,L,L]
        h_dense, mask = to_dense_batch(h, batch=batch_id)            # [B,L,H], [B,L]
        B, L, H = h_dense.shape

        if spd.shape[0] != B:
            raise ValueError(f"spatial_pos_dense batch mismatch: spd.B={spd.shape[0]}, expected B={B}")
        if spd.shape[1] < L or spd.shape[2] < L:
            raise ValueError(f"spatial_pos_dense too small: got {spd.shape}, need at least [{B},{L},{L}]")

        # crop to current max L (safe)
        spd = spd[:, :L, :L].clamp(0, self.spd_inf)  # [B,L,L]

        # ---- IMPORTANT: avoid padding pollution ----
        # mask_j indicates which columns (j) are real nodes
        # mask: [B,L] -> mask_j: [B,1,L,1] broadcast to [B,L,L,C]
        mask_j = mask.unsqueeze(1).unsqueeze(-1)  # [B,1,L,1]

        C = self.spd_max_dist + 2  # bins
        oh = F.one_hot(spd, num_classes=C).float()  # [B,L,L,C]
        oh = oh * mask_j  # ignore padded j columns

        hist = oh.sum(dim=2)  # [B,L,C] (for each i, count over valid j)

        # normalize by number of valid nodes in the graph (n)
        n_nodes = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # [B,1,1]
        hist = hist / n_nodes

        pe_dense = self.spd_hist_mlp(hist)  # [B,L,H]

        if self.spd_fuse_mode == "sum":
            h_dense = h_dense + pe_dense
        elif self.spd_fuse_mode == "concat":
            h_dense = self.spd_cat_proj(torch.cat([h_dense, pe_dense], dim=-1))
        else:
            raise ValueError(f"Unknown spd_fuse_mode={self.spd_fuse_mode}")

        h_out = h_dense[mask]  # [N,H]
        return h_out

    def forward(self, batch):
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

        # node/edge encoding
        h = self.atom_encoder(x)                           # [N,H]
        e = self.edge_proj(self.bond_encoder(edge_attr))   # [E,edge_emb_dim]

        # degree
        if self.degree_enc is not None:
            h = h + self.degree_enc(edge_index, num_nodes=N)

        # ---- GraphGPS-style: SPD -> node SE injected into h ----
        if self.use_spd_node_pe:
            spatial_pos_dense = getattr(batch, "spatial_pos_dense", None)
            if spatial_pos_dense is None:
                raise ValueError("use_spd_node_pe=True but batch.spatial_pos_dense is missing. Did you set collate_fn?")
            h = self._inject_spd_node_se(h, batch_id, spatial_pos_dense)

        # init CLS per graph
        B = int(batch_id.max().item()) + 1
        cls = self.graph_token.expand(B, -1).contiguous()  # [B,H]

        # blocks
        for blk in self.blocks:
            h, cls = blk(
                h=h,
                edge_index=edge_index,
                edge_attr=e,
                batch=batch_id,
                cls=cls,
            )

        h = self.out_norm(h)
        graph_emb = self.graph_norm(cls)
        return h, graph_emb
