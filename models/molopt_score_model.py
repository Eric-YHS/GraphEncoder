import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral


def get_refine_net(refine_net_type, config):
    if refine_net_type == 'uni_o2':
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup
        )
    elif refine_net_type == 'egnn':
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# %% categorical diffusion related
def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    # permute_order = (0, -1) + tuple(range(1, len(x.size())))
    # x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


# %%


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.float()
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class MolPosDiffusion(nn.Module):
    """
    Molecule-only, position-only diffusion.
    Denoiser = UniTransformerO2TwoUpdateGeneral (SE(3)-equivariant H<->X updates)
    Condition = encoder node embedding (2D-only) + time
    """
    def __init__(self, config, node_in_dim: int, cond_dim: int):
        super().__init__()
        self.config = config

        # ---- diffusion schedule (pos only)
        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)

        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))

        # ---- model choices
        self.model_mean_type = config.model_mean_type  # 建议用 'C0'（直接预测 x0）
        self.sample_time_method = config.sample_time_method  # ['importance','symmetric']
        self.center_pos_mode = getattr(config, "center_pos_mode", "graph")  # ['none','graph']

        self.hidden_dim = config.hidden_dim

        # ---- time embedding
        self.time_emb_dim = getattr(config, "time_emb_dim", 64)
        self.time_emb_mode = getattr(config, "time_emb_mode", "sin")

        if self.time_emb_dim > 0 and self.time_emb_mode == "sin":
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            self.time_emb = None

        # ---- condition fusion:  time + encoder node_emb
        # self.x_proj = nn.Linear(node_in_dim, self.hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, self.hidden_dim)

        # fuse_in = self.hidden_dim + self.hidden_dim + (self.time_emb_dim if self.time_emb_dim > 0 else 0)
        fuse_in = self.hidden_dim + (self.time_emb_dim if self.time_emb_dim > 0 else 0)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # ---- denoiser (equivariant refine net)
        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)

        # importance sampling bookkeeping（可选）
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))

    def q_pos_sample(self, x0: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        """
        x_t = sqrt(a_hat) * x0 + sqrt(1-a_hat) * eps
        """
        a = self.alphas_cumprod.index_select(0, t)          # [B]
        a_node = a[batch].unsqueeze(-1)                     # [N,1]
        eps = torch.randn_like(x0)
        xt = a_node.sqrt() * x0 + (1. - a_node).sqrt() * eps
        return xt, eps

    def q_pos_posterior_mean(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        """
        mean of q(x_{t-1} | x_t, x_0)
        """
        return extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
               extract(self.posterior_mean_ct_coef, t, batch) * xt

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 1e-4
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        if method == 'symmetric':
            t = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        raise ValueError(method)

    def forward(self, pos_t, x, batch, cond_node_emb, time_step,
                bond_edge_index=None, bond_edge_attr=None, return_all=False, fix_x=False):
        """
        pos_t: [N,3] noisy coords
        x: [N,node_in_dim] 2D atom feats (no 3D)
        cond_node_emb: [N,cond_dim] from encoder
        time_step: [B] long
        bond_edge_index: [2,E_b] (from batch.edge_index)
        """
        # fuse node features
        # hx = self.x_proj(x.float())
        hc = self.cond_proj(cond_node_emb)

        # feats = [hx, hc]
        feats = [hc]
        if self.time_emb_dim > 0:
            if self.time_emb_mode == "sin":
                t_feat = self.time_emb(time_step)          # [B,Dt]
                t_feat_node = t_feat[batch]                # [N,Dt]
            else:
                t_feat_node = (time_step / self.num_timesteps)[batch].unsqueeze(-1)  # [N,1]
            feats.append(t_feat_node)

        h0 = self.fuse(torch.cat(feats, dim=-1))            # [N,H]

        # mask: molecule-only => all ones (update all nodes)
        mask = torch.ones((x.size(0),), device=x.device, dtype=torch.float)

        out = self.refine_net(
            h0, pos_t, mask, batch,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            return_all=return_all,
            fix_x=fix_x
        )
        # out['x'] is predicted positions (we interpret as x0 if model_mean_type='C0')
        return out

    def get_diffusion_loss(self, batch, cond_node_emb, time_step=None):
        """
        batch: PyG Batch with .pos .x .edge_index .batch
        cond_node_emb: encoder output aligned with nodes
        """
        pos0 = batch.pos
        x = batch.x
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr.float()
        batch_id = batch.batch

        pos0, offset_per_node, scale_per_node = center_pos_mol(pos0, batch_id, mode=self.center_pos_mode)

        num_graphs = batch_id.max().item() + 1
        if time_step is None:
            t, _ = self.sample_time(num_graphs, pos0.device, self.sample_time_method)
        else:
            t = time_step

        pos_t, eps = self.q_pos_sample(pos0, t, batch_id)

        out = self.forward(
            pos_t=pos_t,
            x=x,
            batch=batch_id,
            cond_node_emb=cond_node_emb,
            time_step=t,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            return_all=False,
            fix_x=False
        )
        pred = out['x']  # [N,3]

        if self.model_mean_type == 'C0':
            target = pos0
            loss_node = ((pred - target) ** 2).sum(-1)  # [N]
        elif self.model_mean_type == 'noise':
            # 如果你想切到噪声预测，需要把 refine_net 输出解释成 eps；这里先给接口，不建议一开始用
            target = eps
            loss_node = ((pred - target) ** 2).sum(-1)
        else:
            raise ValueError(self.model_mean_type)

        loss_graph = scatter_mean(loss_node, batch_id, dim=0)  # [B]
        loss = loss_graph.mean()

        return {
            "loss": loss,
            "t": t,
            "pos0": pos0,
            "pos_t": pos_t,
            "pred_x0": pred,
        }

def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)

# def center_pos_mol(pos: torch.Tensor, batch: torch.Tensor, mode: str = "graph"):
#     """
#     把每个图去中心化（平移不变性更稳）。
#     return: pos_centered, offset_per_node
#     """
#     if mode == "none":
#         offset = torch.zeros((batch.max().item() + 1, 3), device=pos.device, dtype=pos.dtype)
#         return pos, offset[batch]

#     if mode == "graph":
#         offset = scatter_mean(pos, batch, dim=0)  # [B,3]
#         return pos - offset[batch], offset[batch]

#     raise ValueError(mode)

def center_pos_mol(pos: torch.Tensor,
                   batch: torch.Tensor,
                   mode: str = "graph",
                   normalize: bool = True,
                   eps: float = 1e-8):
    """
    去中心 + （可选）per-graph 归一化。
    return:
      pos_new: [N,3] 处理后的坐标
      offset_per_node: [N,3] 每个节点被减去的质心
      scale_per_node:  [N]   每个节点被除以的尺度（若 normalize=False，则全1）
    """
    B = batch.max().item() + 1

    if mode == "none":
        offset = torch.zeros((B, 3), device=pos.device, dtype=pos.dtype)
        scale = torch.ones(B, device=pos.device, dtype=pos.dtype)
        return pos, offset[batch], scale[batch]

    if mode == "graph":
        # 1) per-graph 质心
        offset = scatter_mean(pos, batch, dim=0)          # [B,3]
        pos_center = pos - offset[batch]                  # [N,3]

        # 2) 可选：per-graph RMS 归一化
        if not normalize:
            scale = torch.ones(B, device=pos.device, dtype=pos.dtype)
            return pos_center, offset[batch], scale[batch]

        # 每个节点到质心的平方距离
        dist2 = (pos_center ** 2).sum(dim=-1)             # [N]
        # 按图平均 -> 每个图的均方距离
        mean_dist2 = scatter_mean(dist2, batch, dim=0)    # [B]
        scale = torch.sqrt(mean_dist2 + eps)              # [B]

        # 避免 scale 太小（极端情况，单原子）
        scale = torch.clamp(scale, min=eps)

        pos_norm = pos_center / scale[batch].unsqueeze(-1)  # [N,3]
        return pos_norm, offset[batch], scale[batch]

    raise ValueError(mode)
