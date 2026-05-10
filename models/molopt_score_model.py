import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from models.uni_transformer_condition import UniTransformerO2TwoUpdateGeneral_CFG


def get_refine_net(refine_net_type, config):
    if refine_net_type in ['uni_o2', 'uni_o2_cat']:
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
    elif refine_net_type == 'uni_o2_condition':
        refine_net = UniTransformerO2TwoUpdateGeneral_CFG(
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
            sync_twoup=config.sync_twoup,
            graph_cond_dim = config.graph_emb_dim,
            drop_graph_cond = config.graph_dropout,
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


def sample_time_symmetric_middle_bias(
    num_graphs: int,
    num_timesteps: int,
    device: torch.device,
    min_frac: float = 0.1,
    max_frac: float = 0.9,
    beta_alpha: float = 2.0,
):
    """
    Symmetric middle-biased timestep sampling.

    - First restrict timesteps to a broad middle band [min_frac, max_frac].
- Then bias samples toward the center with a symmetric Beta(alpha, alpha).
    - alpha=1 reduces to uniform sampling within the band.
    """
    if not (0.0 <= min_frac < max_frac <= 1.0):
        raise ValueError(f"Expected 0 <= min_frac < max_frac <= 1, got {min_frac}, {max_frac}")
    if beta_alpha <= 0:
        raise ValueError(f"beta_alpha must be > 0, got {beta_alpha}")

    max_step = num_timesteps - 1
    t_min = int(round(max_step * min_frac))
    t_max = int(round(max_step * max_frac))
    t_min = max(0, min(t_min, max_step))
    t_max = max(t_min, min(t_max, max_step))

    if t_min == t_max:
        t = torch.full((num_graphs,), t_min, device=device, dtype=torch.long)
        pt = torch.ones_like(t, dtype=torch.float) / float(num_timesteps)
        return t, pt

    half = num_graphs // 2 + 1
    if abs(beta_alpha - 1.0) < 1e-8:
        u_half = torch.rand(half, device=device)
    else:
        alpha = torch.tensor(float(beta_alpha), device=device)
        beta_dist = torch.distributions.Beta(alpha, alpha)
        u_half = beta_dist.sample((half,)).to(device)

    u = torch.cat([u_half, 1.0 - u_half], dim=0)[:num_graphs]
    t = t_min + torch.round(u * (t_max - t_min)).long()
    t = t.clamp_(t_min, t_max)
    pt = torch.ones_like(t, dtype=torch.float) / float(num_timesteps)
    return t, pt


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
        self.sample_time_min_frac = float(getattr(config, "sample_time_min_frac", 0.1))
        self.sample_time_max_frac = float(getattr(config, "sample_time_max_frac", 0.9))
        self.sample_time_beta_alpha = float(getattr(config, "sample_time_beta_alpha", 2.0))

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
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

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
            return sample_time_symmetric_middle_bias(
                num_graphs=num_graphs,
                num_timesteps=self.num_timesteps,
                device=device,
                min_frac=self.sample_time_min_frac,
                max_frac=self.sample_time_max_frac,
                beta_alpha=self.sample_time_beta_alpha,
            )

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

        pos0, _ = center_pos_mol(pos0, batch_id, mode=self.center_pos_mode)

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
        pred, _ = center_pos_mol(pred, batch_id, mode=self.center_pos_mode)

        if self.model_mean_type == 'C0':
            target = pos0
            loss_node = ((pred - target) ** 2).sum(-1)  # [N]
        elif self.model_mean_type == 'noise':
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

    def _predict_x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        """
        x0 = (x_t - sqrt(1-a_hat) * eps) / sqrt(a_hat)
        t: [B], batch: [N]
        """
        sqrt_a = extract(self.sqrt_alphas_cumprod, t, batch)                 # [N,1]
        sqrt_1m = extract(self.sqrt_one_minus_alphas_cumprod, t, batch)      # [N,1]
        x0 = (x_t - sqrt_1m * eps) / (sqrt_a + 1e-12)
        return x0

    def _model_pred_to_x0(self, x_t: torch.Tensor, model_out_x: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        """
        explain x0_pred from model_out_x
        - mean_type=C0: model_out_x is x0
        - mean_type=noise: model_out_x is eps
        """
        if self.model_mean_type == "C0":
            return model_out_x
        elif self.model_mean_type == "noise":
            return self._predict_x0_from_eps(x_t, model_out_x, t, batch)
        else:
            raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")

    @torch.no_grad()
    def p_mean_variance(
        self,
        batch_obj,               # PyG Batch
        x_t: torch.Tensor,       # [N,3]
        cond_node_emb: torch.Tensor,
        t: torch.Tensor,         # [B] long
    ):
        """
        计算 p(x_{t-1} | x_t) 的 mean/logvar（DDPM：用 q posterior + x0_pred）
        返回：
          mean: [N,3]
          logvar: [N,1]
          x0_pred: [N,3]
        """
        batch_id = batch_obj.batch
        x = batch_obj.x
        bond_edge_index = batch_obj.edge_index
        bond_edge_attr = batch_obj.edge_attr.float() if hasattr(batch_obj, "edge_attr") and batch_obj.edge_attr is not None else None

        out = self.forward(
            pos_t=x_t,
            x=x,
            batch=batch_id,
            cond_node_emb=cond_node_emb,
            time_step=t,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            return_all=False,
            fix_x=False,
        )
        model_out = out["x"]  # [N,3]（解释为 x0 或 eps）

        x0_pred = self._model_pred_to_x0(x_t, model_out, t, batch_id)       # [N,3]
        x0_pred, _ = center_pos_mol(x0_pred, batch_id, mode=self.center_pos_mode)
        mean = self.q_pos_posterior_mean(x0_pred, x_t, t, batch_id)         # [N,3]
        logvar = extract(self.posterior_logvar, t, batch_id)                # [N,1]

        return mean, logvar, x0_pred

    @torch.no_grad()
    def p_sample(
        self,
        batch_obj,
        x_t: torch.Tensor,            # [N,3]
        cond_node_emb: torch.Tensor,
        t: torch.Tensor,              # [B] long
    ):
        """
        单步采样: x_{t-1} ~ N(mean, var)
        """
        mean, logvar, x0_pred = self.p_mean_variance(
            batch_obj=batch_obj,
            x_t=x_t,
            cond_node_emb=cond_node_emb,
            t=t,
        )

        # t==0 时不加噪声
        if (t == 0).all():
            x_prev = mean
        else:
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.exp(0.5 * logvar) * noise

        return x_prev, x0_pred

    @torch.no_grad()
    def p_sample_loop(
        self,
        batch_obj,
        cond_node_emb: torch.Tensor,
        x_T: torch.Tensor = None,     # [N,3] 可选：给定初始噪声
        return_traj: bool = False,
    ):
        """
        从 t=T-1 迭代到 0，返回 x0（以及可选轨迹）
        """
        device = batch_obj.x.device
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1
        N = batch_obj.x.size(0)

        x_t = torch.randn((N, 3), device=device) if x_T is None else x_T

        traj = []
        for step in reversed(range(self.num_timesteps)):
            t_graph = torch.full((B,), step, device=device, dtype=torch.long)
            x_t, _ = self.p_sample(
                batch_obj=batch_obj,
                x_t=x_t,
                cond_node_emb=cond_node_emb,
                t=t_graph,
            )
            if return_traj:
                traj.append(x_t.detach().cpu())

        return (x_t, traj) if return_traj else x_t

    # =========================
    #  Public APIs
    # =========================

    @torch.no_grad()
    def sample(
        self,
        batch_obj,
        cond_node_emb: torch.Tensor,
        return_traj: bool = False,
    ):
        """
        纯噪声生成：x_T ~ N(0,I) -> x0
        """
        return self.p_sample_loop(
            batch_obj=batch_obj,
            cond_node_emb=cond_node_emb,
            x_T=None,
            return_traj=return_traj,
        )

    @torch.no_grad()
    def reconstruct_from_noisy(
        self,
        batch_obj,
        x_t: torch.Tensor,            # 给定某个噪声步的坐标 [N,3]
        t_start: int,                 # 从 t_start 开始反推到 0
        cond_node_emb: torch.Tensor,
        return_traj: bool = False,
    ):
        device = batch_obj.x.device
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1

        x_cur = x_t

        traj = []
        for step in reversed(range(t_start + 1)):
            t_graph = torch.full((B,), step, device=device, dtype=torch.long)
            x_cur, _ = self.p_sample(
                batch_obj=batch_obj,
                x_t=x_cur,
                cond_node_emb=cond_node_emb,
                t=t_graph,
            )
            if return_traj:
                traj.append(x_cur.detach().cpu())

        return (x_cur, traj) if return_traj else x_cur

    @torch.no_grad()
    def reconstruct_from_clean(
        self,
        batch_obj,
        x0: torch.Tensor,             # 干净坐标 [N,3]
        t_start: int,
        cond_node_emb: torch.Tensor,
        return_traj: bool = False,
    ):
        """
        先 q_sample 加噪到 x_t_start，然后反推回 x0
        """
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1
        t_graph = torch.full((B,), t_start, device=x0.device, dtype=torch.long)

        x0c = x0
        x0c, _ = center_pos_mol(x0c, batch_id, mode=self.center_pos_mode)

        x_t, _ = self.q_pos_sample(x0c, t_graph, batch_id)
        return self.reconstruct_from_noisy(
            batch_obj=batch_obj,
            x_t=x_t,
            t_start=t_start,
            cond_node_emb=cond_node_emb,
            return_traj=return_traj,
        )


class MolPosDiffusion_condition(nn.Module):
    """
    Molecule-only, position-only diffusion.
    Denoiser = UniTransformerO2TwoUpdateGeneral (SE(3)-equivariant H<->X updates)
    Condition:
      - encoder node embedding (2D-only): cond_node_emb  [N, cond_dim]
      - encoder graph embedding (CLS token): graph_emb   [B, cond_dim]  (or [N, cond_dim], will be pooled)
      - time
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
        self.sample_time_min_frac = float(getattr(config, "sample_time_min_frac", 0.1))
        self.sample_time_max_frac = float(getattr(config, "sample_time_max_frac", 0.9))
        self.sample_time_beta_alpha = float(getattr(config, "sample_time_beta_alpha", 2.0))

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

        # ---- condition projection
        # node-level condition: cond_node_emb -> hidden
        self.cond_proj = nn.Linear(cond_dim, self.hidden_dim)

        # ---- condition fusion: [node_cond, graph_cond(broadcast), time]
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
        
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # node dropout
        self.node_dropout = config.node_dropout

    def _ensure_graph_emb(self, graph_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        规范 graph_emb 为 [B, cond_dim]：
        - 若传入是 [B,cond_dim] 直接返回
        - 若误传成 [N,cond_dim]，用 scatter_mean pool 成 [B,cond_dim]
        """
        if graph_emb is None:
            return None
        if graph_emb.dim() != 2:
            raise ValueError(f"graph_emb must be 2D, got shape={tuple(graph_emb.shape)}")

        num_graphs = int(batch.max().item()) + 1
        if graph_emb.size(0) == num_graphs:
            return graph_emb

        if graph_emb.size(0) == batch.size(0):
            return scatter_mean(graph_emb, batch, dim=0, dim_size=num_graphs)

        raise ValueError(
            f"graph_emb first dim must be num_graphs={num_graphs} or num_nodes={batch.size(0)}, "
            f"got {graph_emb.size(0)}"
        )

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
            return sample_time_symmetric_middle_bias(
                num_graphs=num_graphs,
                num_timesteps=self.num_timesteps,
                device=device,
                min_frac=self.sample_time_min_frac,
                max_frac=self.sample_time_max_frac,
                beta_alpha=self.sample_time_beta_alpha,
            )

        raise ValueError(method)

    def forward(
        self,
        pos_t,
        x,
        batch,
        cond_node_emb,
        graph_emb,                
        time_step,
        bond_edge_index=None,
        bond_edge_attr=None,
        return_all=False,
        fix_x=False,
        unconditioned: bool = False,  # ✅ 可选：用于你做 ablation（不用 CFG 也可以不用它）
        apply_graph_dropout: bool = True,
    ):
        """
        pos_t: [N,3] noisy coords
        x: [N,node_in_dim] 2D atom feats (no 3D)  (当前没用到，可保留)
        cond_node_emb: [N,cond_dim] encoder node embeddings
        graph_emb: [B,cond_dim] encoder CLS embedding (or [N,cond_dim], will be pooled)
        time_step: [B] long
        bond_edge_index: [2,E_b]
        """
        graph_emb = self._ensure_graph_emb(graph_emb, batch)  # [B,cond_dim]
        hc_node = self.cond_proj(cond_node_emb)  # [N,H]
        
        # random dropout node embeddings
        if self.training and self.node_dropout > 0:
            keep_prob = 1.0 - self.node_dropout
            keep_mask = (torch.rand(hc_node.size(0), device=hc_node.device) < keep_prob).float().unsqueeze(-1)
            hc_node = hc_node * keep_mask / keep_prob

        feats = [hc_node]

        if self.time_emb_dim > 0:
            if self.time_emb_mode == "sin":
                t_feat = self.time_emb(time_step)      # [B,Dt]
                t_feat_node = t_feat[batch]            # [N,Dt]
            else:
                t_feat_node = (time_step / self.num_timesteps)[batch].unsqueeze(-1)  # [N,1]
            feats.append(t_feat_node)

        h0 = self.fuse(torch.cat(feats, dim=-1))  # [N,H]

        # mask: molecule-only => all ones (update all nodes)
        mask = torch.ones((x.size(0),), device=x.device, dtype=torch.float)

        out = self.refine_net(
            h0, pos_t, mask, batch,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            graph_embedding=graph_emb,
            unconditioned=unconditioned,
            apply_graph_dropout=apply_graph_dropout,
            return_all=return_all,
            fix_x=fix_x
        )
        return out

    def get_diffusion_loss(
        self,
        batch,
        cond_node_emb,
        graph_emb,
        time_step=None,
        apply_graph_dropout: bool = True,
    ):
        """
        batch: PyG Batch with .pos .x .edge_index .batch
        cond_node_emb: [N,cond_dim] encoder output aligned with nodes
        graph_emb: [B,cond_dim] encoder CLS embedding (or [N,cond_dim], will be pooled)
        """
        pos0 = batch.pos
        x = batch.x
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr.float() if getattr(batch, "edge_attr", None) is not None else None
        bond_edge_attr = None
        

        batch_id = batch.batch

        pos0, _ = center_pos_mol(pos0, batch_id, mode=self.center_pos_mode)

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
            graph_emb=graph_emb,                
            time_step=t,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            return_all=False,
            fix_x=False,
            unconditioned=False,
            apply_graph_dropout=apply_graph_dropout,
        )
        pred = out['x']  # [N,3]
        pred, _ = center_pos_mol(pred, batch_id, mode=self.center_pos_mode)

        if self.model_mean_type == 'C0':
            target = pos0
            loss_node = ((pred - target) ** 2).sum(-1)  # [N]
        elif self.model_mean_type == 'noise':
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
    def _center_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # x: [N,3]
        if self.center_pos_mode == "none":
            return x
        x, _ = center_pos_mol(x, batch, mode=self.center_pos_mode)
        return x

    # def _com_free_noise_like(self, ref: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    #     z = torch.randn_like(ref)
    #     return self._center_graph(z, batch)  # 让噪声本身每个图零均值（CoM-free）

    def _predict_x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        """
        x0 = (x_t - sqrt(1-a_hat) * eps) / sqrt(a_hat)
        """
        sqrt_a = extract(self.sqrt_alphas_cumprod, t, batch)                 # [N,1]
        sqrt_1m = extract(self.sqrt_one_minus_alphas_cumprod, t, batch)      # [N,1]
        x0 = (x_t - sqrt_1m * eps) / (sqrt_a + 1e-12)
        return x0

    def _model_pred_to_x0(self, x_t: torch.Tensor, model_out_x: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        if self.model_mean_type == "C0":
            return model_out_x
        elif self.model_mean_type == "noise":
            return self._predict_x0_from_eps(x_t, model_out_x, t, batch)
        else:
            raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")

    @torch.no_grad()
    def p_mean_variance(
        self,
        batch_obj,
        x_t: torch.Tensor,
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        t: torch.Tensor,                     # [B] long
        unconditioned: bool = False,
        clip_x0: bool = False,
    ):
        """
        返回:
          mean:    [N,3]   posterior mean
          logvar:  [N,1]   posterior log-variance
          x0_pred: [N,3]
        """
        batch_id = batch_obj.batch
        x = batch_obj.x
        bond_edge_index = batch_obj.edge_index
        bond_edge_attr = batch_obj.edge_attr.float() if getattr(batch_obj, "edge_attr", None) is not None else None
        bond_edge_attr = None

        out = self.forward(
            pos_t=x_t,
            x=x,
            batch=batch_id,
            cond_node_emb=cond_node_emb,
            graph_emb=graph_emb,
            time_step=t,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            return_all=False,
            fix_x=False,
            unconditioned=unconditioned,
        )
        model_out = out["x"]  # interpret according to model_mean_type

        x0_pred = self._model_pred_to_x0(x_t, model_out, t, batch_id)
        x0_pred, _ = center_pos_mol(x0_pred, batch_id, mode=self.center_pos_mode)

        if clip_x0:
            x0_pred = x0_pred.clamp(min=-20.0, max=20.0)

        mean = self.q_pos_posterior_mean(x0_pred, x_t, t, batch_id)
        logvar = extract(self.posterior_logvar, t, batch_id)  # [N,1]

        return mean, logvar, x0_pred

    @torch.no_grad()
    def p_sample(self, batch_obj, x_t, cond_node_emb, graph_emb, t,
                unconditioned: bool = False):
        # batch_id = batch_obj.batch
        mean, logvar, x0_pred = self.p_mean_variance(batch_obj, x_t, cond_node_emb, graph_emb, t, unconditioned)

        if (t == 0).all():
            x_prev = mean
        else:
            # z = self._com_free_noise_like(x_t, batch_id)
            z = torch.randn_like(x_t)
            x_prev = mean + torch.exp(0.5 * logvar) * z
        

        return x_prev, x0_pred


    @torch.no_grad()
    def p_sample_loop(
        self,
        batch_obj,
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        x_T = None,
        return_traj: bool = False,
        unconditioned: bool = False,
    ):
        device = batch_obj.x.device
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1
        N = batch_obj.x.size(0)

        x_t = x_T if x_T is not None else torch.randn((N, 3), device=device)

        traj = []
        for step in reversed(range(self.num_timesteps)):
            t_graph = torch.full((B,), step, device=device, dtype=torch.long)
            x_t, _ = self.p_sample(
                batch_obj, x_t, cond_node_emb, graph_emb, t_graph,
                unconditioned=unconditioned,
            )
            if return_traj:
                traj.append(x_t.detach().cpu())
        x_t, _ = center_pos_mol(x_t, batch_obj.batch, mode=self.center_pos_mode)
        return (x_t, traj) if return_traj else x_t

    @torch.no_grad()
    def sample(self, batch_obj, cond_node_emb, graph_emb, return_traj: bool = False):
        """
        纯生成：默认 stochastic（符合 DDPM）
        """
        return self.p_sample_loop(
            batch_obj, cond_node_emb, graph_emb,
            x_T=None,
            return_traj=return_traj,
        )

    @torch.no_grad()
    def reconstruct_from_noisy(
        self,
        batch_obj,
        x_t: torch.Tensor,
        t_start: int,
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        return_traj: bool = False,
    ):
        device = batch_obj.x.device
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1
        x_cur = x_t
        traj = []

        for step in reversed(range(t_start + 1)):
            t_graph = torch.full((B,), step, device=device, dtype=torch.long)
            x_cur, _ = self.p_sample(
                batch_obj, x_cur, cond_node_emb, graph_emb, t_graph,
            )
            if return_traj:
                traj.append(x_cur.detach().cpu())

        return (x_cur, traj) if return_traj else x_cur

    @torch.no_grad()
    def reconstruct_from_clean(
        self,
        batch_obj,
        x0: torch.Tensor,
        t_start: int,
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        return_traj: bool = False,
        deterministic: bool = True,
    ):
        """
        先 q_sample 得到 x_t_start，再反推。
        """
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1
        t_graph = torch.full((B,), t_start, device=x0.device, dtype=torch.long)

        x0c = self._center_graph(x0, batch_id)
        x_t, _ = self.q_pos_sample(x0c, t_graph, batch_id)

        return self.reconstruct_from_noisy(
            batch_obj=batch_obj,
            x_t=x_t,
            t_start=t_start,
            cond_node_emb=cond_node_emb,
            graph_emb=graph_emb,
            return_traj=return_traj,
        )



class MolPosDiffusion_cat(nn.Module):
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
        self.sample_time_min_frac = float(getattr(config, "sample_time_min_frac", 0.1))
        self.sample_time_max_frac = float(getattr(config, "sample_time_max_frac", 0.9))
        self.sample_time_beta_alpha = float(getattr(config, "sample_time_beta_alpha", 2.0))

        self.hidden_dim = config.hidden_dim
        self.cond_dim = cond_dim

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
        self.graph_cond_proj = nn.Linear(cond_dim, self.hidden_dim)

        # fuse_in = self.hidden_dim + self.hidden_dim + (self.time_emb_dim if self.time_emb_dim > 0 else 0)
        fuse_in = self.hidden_dim * 2 + (self.time_emb_dim if self.time_emb_dim > 0 else 0)
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
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # node dropout
        self.node_dropout = config.node_dropout

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
            return sample_time_symmetric_middle_bias(
                num_graphs=num_graphs,
                num_timesteps=self.num_timesteps,
                device=device,
                min_frac=self.sample_time_min_frac,
                max_frac=self.sample_time_max_frac,
                beta_alpha=self.sample_time_beta_alpha,
            )

        raise ValueError(method)

    def _ensure_graph_emb(self, graph_emb: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        规范 graph_emb 为 [B, cond_dim]：
        - 若传入是 [B,cond_dim] 直接返回
        - 若误传成 [N,cond_dim]，用 scatter_mean pool 成 [B,cond_dim]
        """
        if graph_emb is None:
            return None
        if graph_emb.dim() != 2:
            raise ValueError(f"graph_emb must be 2D, got shape={tuple(graph_emb.shape)}")

        num_graphs = int(batch.max().item()) + 1
        if graph_emb.size(0) == num_graphs:
            return graph_emb

        if graph_emb.size(0) == batch.size(0):
            return scatter_mean(graph_emb, batch, dim=0, dim_size=num_graphs)

        raise ValueError(
            f"graph_emb first dim must be num_graphs={num_graphs} or num_nodes={batch.size(0)}, "
            f"got {graph_emb.size(0)}"
        )

    def forward(self, pos_t, x, batch, cond_node_emb, graph_emb, time_step,
                bond_edge_index=None, bond_edge_attr=None, return_all=False, fix_x=False):
        """
        pos_t: [N,3] noisy coords
        x: [N,node_in_dim] 2D atom feats (no 3D)
        cond_node_emb: [N,cond_dim] from encoder
        time_step: [B] long
        bond_edge_index: [2,E_b] (from batch.edge_index)
        """

        graph_emb = self._ensure_graph_emb(graph_emb, batch)
        graph_emb = graph_emb[batch]

        # fuse node features
        # hx = self.x_proj(x.float())
        hc_node = self.cond_proj(cond_node_emb)
        hc_graph = self.graph_cond_proj(graph_emb)
        if self.training and self.node_dropout > 0:
            keep_prob = 1.0 - self.node_dropout
            if self.node_dropout_type in ["node", "both"]:
                keep_mask = (torch.rand(hc_node.size(0), device=hc_node.device) < keep_prob).float().unsqueeze(-1)
                # hc = hc * keep_mask / max(keep_prob, 1e-6)
                hc_node = hc_node * keep_mask
            
            if self.node_dropout_type in ["feature", "both"]:
                hc_node = F.dropout(hc_node, p = self.node_dropout, training=True)

        feats = [hc_node, hc_graph]
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

    def get_diffusion_loss(self, batch, cond_node_emb, graph_emb, time_step=None):
        """
        batch: PyG Batch with .pos .x .edge_index .batch
        cond_node_emb: encoder output aligned with nodes
        """
        pos0 = batch.pos
        x = batch.x
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr.float()
        batch_id = batch.batch

        pos0, _ = center_pos_mol(pos0, batch_id, mode=self.center_pos_mode)

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
            graph_emb = graph_emb,
            time_step=t,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            return_all=False,
            fix_x=False
        )
        pred = out['x']  # [N,3]
        pred, _ = center_pos_mol(pred, batch_id, mode=self.center_pos_mode)

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


    def _predict_x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        """
        x0 = (x_t - sqrt(1-a_hat) * eps) / sqrt(a_hat)
        t: [B], batch: [N]
        """
        sqrt_a = extract(self.sqrt_alphas_cumprod, t, batch)                 # [N,1]
        sqrt_1m = extract(self.sqrt_one_minus_alphas_cumprod, t, batch)      # [N,1]
        x0 = (x_t - sqrt_1m * eps) / (sqrt_a + 1e-12)
        return x0

    def _model_pred_to_x0(self, x_t: torch.Tensor, model_out_x: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        """
        将网络输出解释成 x0_pred
        - mean_type=C0: model_out_x 就是 x0
        - mean_type=noise: model_out_x 是 eps
        """
        if self.model_mean_type == "C0":
            return model_out_x
        elif self.model_mean_type == "noise":
            return self._predict_x0_from_eps(x_t, model_out_x, t, batch)
        else:
            raise ValueError(f"Unknown model_mean_type: {self.model_mean_type}")

    @torch.no_grad()
    def p_mean_variance(
        self,
        batch_obj,               # PyG Batch
        x_t: torch.Tensor,       # [N,3]
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        t: torch.Tensor,         # [B] long
    ):
        """
        计算 p(x_{t-1} | x_t) 的 mean/logvar（DDPM：用 q posterior + x0_pred）
        返回：
          mean: [N,3]
          logvar: [N,1]
          x0_pred: [N,3]
        """
        batch_id = batch_obj.batch
        x = batch_obj.x
        bond_edge_index = batch_obj.edge_index
        bond_edge_attr = batch_obj.edge_attr.float() if hasattr(batch_obj, "edge_attr") and batch_obj.edge_attr is not None else None

        out = self.forward(
            pos_t=x_t,
            x=x,
            batch=batch_id,
            cond_node_emb=cond_node_emb,
            graph_emb=graph_emb,               # 这里保持原样，forward 内部会 _ensure + [batch]
            time_step=t,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            return_all=False,
            fix_x=False,
        )
        model_out = out["x"]  # [N,3]（解释为 x0 或 eps）

        x0_pred = self._model_pred_to_x0(x_t, model_out, t, batch_id)       # [N,3]
        x0_pred, _ = center_pos_mol(x0_pred, batch_id, mode=self.center_pos_mode)
        mean = self.q_pos_posterior_mean(x0_pred, x_t, t, batch_id)         # [N,3]
        logvar = extract(self.posterior_logvar, t, batch_id)                # [N,1]

        return mean, logvar, x0_pred

    @torch.no_grad()
    def p_sample(
        self,
        batch_obj,
        x_t: torch.Tensor,            # [N,3]
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        t: torch.Tensor,              # [B] long
    ):
        """
        单步采样：x_{t-1} ~ N(mean, var)
        """
        mean, logvar, x0_pred = self.p_mean_variance(
            batch_obj=batch_obj,
            x_t=x_t,
            cond_node_emb=cond_node_emb,
            graph_emb=graph_emb,
            t=t,
        )

        # t==0 时不加噪声
        if (t == 0).all():
            x_prev = mean
        else:
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.exp(0.5 * logvar) * noise

        return x_prev, x0_pred

    @torch.no_grad()
    def p_sample_loop(
        self,
        batch_obj,
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        x_T: torch.Tensor = None,     # [N,3] 可选：给定初始噪声
        return_traj: bool = False,
    ):
        """
        从 t=T-1 迭代到 0，返回 x0（以及可选轨迹）
        """
        device = batch_obj.x.device
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1
        N = batch_obj.x.size(0)

        x_t = torch.randn((N, 3), device=device) if x_T is None else x_T

        traj = []
        for step in reversed(range(self.num_timesteps)):
            t_graph = torch.full((B,), step, device=device, dtype=torch.long)
            x_t, _ = self.p_sample(
                batch_obj=batch_obj,
                x_t=x_t,
                cond_node_emb=cond_node_emb,
                graph_emb=graph_emb,
                t=t_graph,
            )
            if return_traj:
                traj.append(x_t.detach().cpu())

        return (x_t, traj) if return_traj else x_t

    # =========================
    #  Public APIs
    # =========================

    @torch.no_grad()
    def sample(
        self,
        batch_obj,
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        return_traj: bool = False,
    ):
        """
        pure noise generation: x_T ~ N(0,I) -> x0
        """
        return self.p_sample_loop(
            batch_obj=batch_obj,
            cond_node_emb=cond_node_emb,
            graph_emb=graph_emb,
            x_T=None,
            return_traj=return_traj,
        )

    @torch.no_grad()
    def reconstruct_from_noisy(
        self,
        batch_obj,
        x_t: torch.Tensor,            # given noisy coords [N,3]
        t_start: int,                 # reverse from t_start to 0
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        return_traj: bool = False,
    ):
        device = batch_obj.x.device
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1

        x_cur = x_t

        traj = []
        for step in reversed(range(t_start + 1)):
            t_graph = torch.full((B,), step, device=device, dtype=torch.long)
            x_cur, _ = self.p_sample(
                batch_obj=batch_obj,
                x_t=x_cur,
                cond_node_emb=cond_node_emb,
                graph_emb=graph_emb,
                t=t_graph,
            )
            if return_traj:
                traj.append(x_cur.detach().cpu())

        return (x_cur, traj) if return_traj else x_cur

    @torch.no_grad()
    def reconstruct_from_clean(
        self,
        batch_obj,
        x0: torch.Tensor,             # clean coords [N,3]
        t_start: int,
        cond_node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        return_traj: bool = False,
    ):
        """
        first q_sample to x_t_start, then reverse to x0
        """
        batch_id = batch_obj.batch
        B = int(batch_id.max().item()) + 1
        t_graph = torch.full((B,), t_start, device=x0.device, dtype=torch.long)

        x0c = x0
        x0c, _ = center_pos_mol(x0c, batch_id, mode=self.center_pos_mode)

        x_t, _ = self.q_pos_sample(x0c, t_graph, batch_id)
        return self.reconstruct_from_noisy(
            batch_obj=batch_obj,
            x_t=x_t,
            t_start=t_start,
            cond_node_emb=cond_node_emb,
            graph_emb=graph_emb,
            return_traj=return_traj,
        )


def extract(coef, t, batch):
    out = coef[t][batch]
    return out.unsqueeze(-1)


def center_pos_mol(pos: torch.Tensor,
                   batch: torch.Tensor,
                   mode: str = "graph",
                   ):

    B = batch.max().item() + 1

    if mode == "none":
        offset = torch.zeros((B, 3), device=pos.device, dtype=pos.dtype)
        return pos, offset[batch]

    if mode == "graph":
        offset = scatter_mean(pos, batch, dim=0)          # [B,3]
        pos_center = pos - offset[batch]   
        return pos_center, offset[batch] 

    raise ValueError(mode)
