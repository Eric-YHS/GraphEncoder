import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    x: [N, D] or [B, N, D]
    shift/scale broadcastable to x
    GraphDiT style: x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift

class GraphCondEmbedder(nn.Module):
    """
    把 graph_embedding (每个图一个向量) -> hidden_dim 的条件向量，
    并支持 condition dropout（CFG 训练）与 unconditioned 强制 drop（CFG 采样）。
    """
    def __init__(self, in_dim: int, hidden_dim: int, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # unconditional / dropped condition 的可学习向量（GraphDiT 的 dropped embedding 思路）
        self.null = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        graph_emb: torch.Tensor,      # [B, in_dim]
        batch: torch.Tensor,          # [N]  node->graph mapping
        training: bool,
        unconditioned: bool = False,
    ) -> torch.Tensor:
        assert graph_emb.dim() == 2, "graph_embedding must be [num_graphs, graph_emb_dim]"
        B = graph_emb.size(0)
        device = graph_emb.device

        # 如果某个图的 graph_emb 含 NaN，当作“缺条件”，强制 drop
        force_drop = torch.isnan(graph_emb).any(dim=-1)  # [B] bool

        if unconditioned:
            drop_mask = torch.ones(B, device=device, dtype=torch.bool)
        else:
            drop_mask = force_drop
            if training and self.drop_prob > 0:
                rand_drop = (torch.rand(B, device=device) < self.drop_prob)
                drop_mask = drop_mask | rand_drop

        emb = self.proj(graph_emb)  # [B, hidden_dim]
        null = self.null.unsqueeze(0).expand(B, -1)  # [B, hidden_dim]
        emb = torch.where(drop_mask.unsqueeze(-1), null, emb)  # drop -> null

        # broadcast to nodes
        cond_node = emb[batch]  # [N, hidden_dim]
        return cond_node

class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True,
                 cond_dim=None):  # ✅ NEW: cond_dim
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # ✅ NEW: AdaLN style conditioning (GraphDiT-like)
        self.cond_dim = input_dim if cond_dim is None else cond_dim
        self.norm_h = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.cond_dim, 3 * input_dim, bias=True),  # shift, scale, gate (all [N, D])
        )
        # ✅ GraphDiT/DiT 常用的 zero-init：初始近似恒等，训练更稳
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None, cond_node=None):  # ✅ NEW: cond_node
        """
        h: [N, D]
        cond_node: [N, D] (每个节点对应其所属图的条件向量)
        """
        N = h.size(0)
        src, dst = edge_index

        # ---- ✅ AdaLN inject condition into h used for q/k/v ----
        if cond_node is None:
            h_mod = h
            gate = None
        else:
            shift, scale, gate = self.adaLN_modulation(cond_node).chunk(3, dim=-1)  # all [N, D]
            h_mod = modulate(self.norm_h(h), shift, scale)  # [N, D]

        hi, hj = h_mod[dst], h_mod[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        q = self.hq_func(h_mod).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)
        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N).view(-1, self.output_dim)

        if self.out_fc:
            # 这里 concat 仍用原 h，避免条件改变 residual 分支的“基准”
            msg = self.node_output(torch.cat([output, h], -1))  # [N, D]
        else:
            msg = output

        # ---- ✅ gated residual (GraphDiT-like) ----
        if gate is None:
            h_new = h + msg
        else:
            h_new = h + gate * msg  # [N, D] elementwise gating

        return h_new

class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r',
                 cond_dim=None):  # ✅ NEW
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

        # ✅ NEW: AdaLN for h + scalar gate for delta_x
        self.cond_dim = input_dim if cond_dim is None else cond_dim
        self.norm_h = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.cond_dim, 2 * input_dim + 1, bias=True),  # shift[D], scale[D], gate_scalar[1]
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None, cond_node=None):  # ✅ NEW
        N = h.size(0)
        src, dst = edge_index
        device = h.device

        if cond_node is None:
            h_mod = h
            gate_x = None
        else:
            out = self.adaLN_modulation(cond_node)  # [N, 2D+1]
            shift, scale, gate_x = torch.split(out, [self.input_dim, self.input_dim, 1], dim=-1)
            h_mod = modulate(self.norm_h(h), shift, scale)  # [N, D]
            # gate_x: [N,1]

        hi, hj = h_mod[dst], h_mod[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # [E, heads, 3]
        q = self.xq_func(h_mod).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)
        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # [N, heads, 3]
        delta_x = output.mean(1)  # [N, 3]

        # ✅ gated delta_x
        if gate_x is not None:
            delta_x = delta_x * gate_x  # broadcast [N,3] * [N,1]

        return delta_x

class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 2,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc, cond_dim=hidden_dim)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 2,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, cond_dim=hidden_dim)
            )

    def forward(self, h, x, edge_attr, edge_type_feat, edge_index, mask_ligand, e_w=None, fix_x=False, cond_node=None):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        base_dist_feat = self.distance_expansion(dist)
        base_dist_feat = outer_product(edge_type_feat, base_dist_feat)

        for i in range(self.num_x2h):
            h_out = self.x2h_layers[i](
                h_in, base_dist_feat, edge_feat, edge_index, e_w=e_w, cond_node=cond_node  # ✅ NEW
            )
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_type_feat, dist_feat)

            delta_x = self.h2x_layers[i](
                new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w, cond_node=cond_node  # ✅ NEW
            )
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]

            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class UniTransformerO2TwoUpdateGeneral_CFG(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1,
                 r_max=10., x2h_out_fc=True, sync_twoup=False, graph_cond_dim:int=0, drop_graph_cond: float = 0.0,):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types

        self.cutoff_mode = cutoff_mode   # ['radius','knn']
        self.k = k
        self.ew_net_type = ew_net_type   # ['r','m','none','global']

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x

        self.r_max = r_max
        self.r = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)

        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.base_block = self._build_share_blocks()

        self.graph_cond_dim = int(graph_cond_dim)
        self.drop_graph_cond = float(drop_graph_cond)
        if self.graph_cond_dim > 0:
            self.graph_cond_embedder = GraphCondEmbedder(
                in_dim=self.graph_cond_dim,
                hidden_dim=self.hidden_dim,
                drop_prob=self.drop_graph_cond
            )
        else:
            self.graph_cond_embedder = None

    def _build_share_blocks(self):
        base_block = []
        for _ in range(self.num_layers):
            base_block.append(
                AttentionLayerO2TwoUpdateNodeGeneral(
                    self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim,
                    act_fn=self.act_fn, norm=self.norm,
                    num_x2h=self.num_x2h, num_h2x=self.num_h2x,
                    r_max=self.r_max, num_node_types=self.num_node_types,
                    ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
                )
            )
        return nn.ModuleList(base_block)

    def _connect_radius_edge(self, x, batch):
        if self.cutoff_mode == 'radius':
            return radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            return knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        else:
            raise ValueError(f'Not supported cutoff_mode={self.cutoff_mode}')

    def forward(self, h, x, mask_ligand, batch,
                bond_edge_index=None,
                bond_edge_attr=None,
                graph_embedding=None,      # ✅ NEW: [num_graphs, graph_cond_dim]
                unconditioned: bool = False,  # ✅ NEW: CFG unconditional branch
                return_delta: bool = False,   # ✅ NEW: return delta_x for CFG mixing
                return_all=False, fix_x=False):

        if bond_edge_attr is not None and bond_edge_index is None:
            raise ValueError("Not allowed bond_edge_index is empty but bond_edge_attr is not empty")

        x_in = x  # ✅ keep original noisy x for delta

        # ---- ✅ build per-node condition (like GraphDiT: a single c used for all layers) ----
        cond_node = None
        if self.graph_cond_embedder is not None:
            if graph_embedding is None:
                # no condition provided -> treat as unconditional
                unconditioned = True
                # make a dummy tensor to satisfy embedder shape, if you want strict behavior
                # but simplest: just skip cond_node, model behaves like original
                cond_node = None
            else:
                cond_node = self.graph_cond_embedder(
                    graph_emb=graph_embedding,
                    batch=batch,
                    training=self.training,
                    unconditioned=unconditioned
                )  # [N, hidden_dim]

        all_x = [x]
        all_h = [h]

        for _ in range(self.num_blocks):
            edge_index_r = self._connect_radius_edge(x, batch)
            E_r = edge_index_r.size(1)

            if bond_edge_index is not None:
                E_b = bond_edge_index.size(1)
                edge_index = torch.cat([bond_edge_index, edge_index_r], dim=1)
                edge_type_idx = torch.cat([
                    torch.zeros(E_b, device=x.device, dtype=torch.long),
                    torch.ones(E_r,  device=x.device, dtype=torch.long),
                ], dim=0)
            else:
                E_b = 0
                edge_index = edge_index_r
                edge_type_idx = torch.ones(E_r, device=x.device, dtype=torch.long)

            src, dst = edge_index
            edge_type_feat = F.one_hot(edge_type_idx, num_classes=2).float()

            if bond_edge_attr is not None:
                bond_edge_attr = bond_edge_attr.to(x.device).float()
                bond_dim = bond_edge_attr.size(-1)
                bond_pad = torch.zeros((E_r, bond_dim), device=x.device, dtype=bond_edge_attr.dtype)
                bond_feat_union = torch.cat([bond_edge_attr, bond_pad], dim=0)
                edge_attr = torch.cat([edge_type_feat, bond_feat_union], dim=-1)
                assert edge_attr.shape[-1] == self.edge_feat_dim
            else:
                edge_attr = edge_type_feat

            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for layer in self.base_block:
                h, x = layer(
                    h, x, edge_attr, edge_type_feat, edge_index, mask_ligand,
                    e_w=e_w, fix_x=fix_x, cond_node=cond_node  # ✅ NEW
                )

            all_x.append(x)
            all_h.append(h)

        outputs = {'x': x, 'h': h}
        if return_delta:
            outputs['delta_x'] = x - x_in  # ✅ NEW
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs

