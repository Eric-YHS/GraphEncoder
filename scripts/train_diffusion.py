import argparse
import os
import shutil
import sys
from pathlib import Path
import time

# ====== [新增] ======
import logging
import re
from glob import glob
# ====================

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from torch.utils.data import random_split

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from preprocess import get_pcqm4m_dataset
from models import GraphGPSEncoder, MolPosDiffusion, GraphGPSEncoder_CLS, MolPosDiffusion_condition
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from ZINC_preprocess import Zinc20_3D_LMDBDataset


def update_config(config, batch):
    config.data.node_in_dim = int(batch.x.shape[1])

    if getattr(batch, "edge_attr", None) is None:
        edge_in_dim = 0
    else:
        edge_in_dim = int(batch.edge_attr.shape[1])
    config.data.edge_in_dim = edge_in_dim
    config.model.edge_feat_dim = edge_in_dim + 2

    return config


def _to_device_and_cast(batch, device):
    batch = batch.to(device)

    if hasattr(batch, "x") and batch.x is not None:
        batch.x = batch.x.float()
    if hasattr(batch, "pos") and batch.pos is not None:
        batch.pos = batch.pos.float()
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr.float()

    return batch


def _get_node_emb(encoder_out) -> torch.Tensor:
    if torch.is_tensor(encoder_out):
        return encoder_out

    if isinstance(encoder_out, (tuple, list)):
        assert len(encoder_out) >= 1
        return encoder_out[0]

    if isinstance(encoder_out, dict):
        for k in ["node_emb", "node_repr", "h_node", "node", "node_out"]:
            if k in encoder_out:
                return encoder_out[k]
        raise KeyError(f"Cannot find node embedding key in encoder_out: {list(encoder_out.keys())}")

    raise TypeError(f"Unsupported encoder_out type: {type(encoder_out)}")


def forward_encoder(encoder: nn.Module, batch):
    try:
        return encoder(batch)
    except TypeError:
        return encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


@torch.no_grad()
def evaluate(encoder, diffusion, loader, device, config) -> float:
    encoder.eval()
    diffusion.eval()

    losses = []
    with torch.no_grad():
        for b in loader:
            b = _to_device_and_cast(b, device)
            enc_out = forward_encoder(encoder, b)
            node_emb, graph_emb = enc_out

            if config.model.model_type == 'uni_o2_condition':
                out = diffusion.get_diffusion_loss(b, cond_node_emb=node_emb, time_step=None, graph_emb=graph_emb)
            else:
                out = diffusion.get_diffusion_loss(b, cond_node_emb=node_emb, time_step=None)
            losses.append(out["loss"].item())

    encoder.train()
    diffusion.train()
    return float(sum(losses) / max(1, len(losses)))


def save_ckpt(path, step, encoder, diffusion, optimizer, scheduler, best_val):
    ckpt = {
        "step": step,
        "encoder": encoder.state_dict(),
        "diffusion": diffusion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val": best_val,
    }
    torch.save(ckpt, path)


# ====== [新增] resume 相关工具函数 ======
def resolve_ckpt_path(p: str) -> str:
    """p 可以是具体文件，也可以是目录；目录下优先 last.pt，其次 best.pt。"""
    if p is None:
        raise ValueError("resume_ckpt is None")
    if os.path.isdir(p):
        cand1 = os.path.join(p, "last.pt")
        cand2 = os.path.join(p, "best.pt")
        if os.path.isfile(cand1):
            return cand1
        if os.path.isfile(cand2):
            return cand2
        raise FileNotFoundError(f"Cannot find last.pt/best.pt under dir: {p}")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    return p


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """torch.load(map_location='cpu') 后把 optimizer state 的 tensor 挪回 device。"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def find_log_file_to_append(log_dir: str) -> str:
    """尽量找到历史日志文件，resume 时追加写同一个文件。"""
    preferred = ["log.txt", "train.log", "train.txt", "output.log"]
    for name in preferred:
        fp = os.path.join(log_dir, name)
        if os.path.isfile(fp):
            return fp

    candidates = glob(os.path.join(log_dir, "*.log")) + glob(os.path.join(log_dir, "*.txt"))
    if candidates:
        return max(candidates, key=os.path.getmtime)

    # 没找到就新建一个默认文件名
    return os.path.join(log_dir, "train.log")


def setup_logger(log_dir: str, resume: bool, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 清理旧 handler（避免重复打印）
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # console
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file (append / write)
    log_file = os.path.join(log_dir, "log.txt")
    fh = logging.FileHandler(log_file, mode="a" if resume else "w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger file: {log_file} (mode={'append' if resume else 'write'})")
    return logger
# =====================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--train_report_iter', type=int, default=100)
    parser.add_argument('--encoder_layers', type=int, default=None)
    parser.add_argument('--model_layers', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='GraphGPS_Encoder')

    parser.add_argument('--resume', action='store_true', help='resume from existing log+ckpt')
    parser.add_argument('--resume_log_dir', type=str, default=None, help='existing log_dir to continue writing')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='checkpoint file or dir (contains last.pt)')


    args = parser.parse_args()

    if args.config is None:
        if args.resume and args.resume_log_dir:
            yamls = glob(os.path.join(args.resume_log_dir, "*.yaml")) + glob(os.path.join(args.resume_log_dir, "*.yml"))
            if len(yamls) == 0:
                raise ValueError("No --config given and no *.yaml/*.yml found in resume_log_dir.")
            args.config = yamls[0]
        else:
            raise ValueError("--config is required unless --resume with a resume_log_dir containing yaml.")
    # ==========================================================================

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)
    device = torch.device(args.device)
    if args.encoder_layers is not None:
        config.encoder.num_layers = int(args.encoder_layers)
    if args.model_layers is not None:
        config.model.num_layers = int(args.model_layers)

    # Logging / dirs
    tag = f"en{config.encoder.num_layers}_de{config.model.num_layers}"
    if args.resume:
        if args.resume_log_dir is None or args.resume_ckpt is None:
            raise ValueError("When --resume, you must provide --resume_log_dir and --resume_ckpt")

        log_dir = args.resume_log_dir
        ckpt_path = resolve_ckpt_path(args.resume_ckpt)
        ckpt_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else args.resume_ckpt
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        run_time = time.localtime()
        log_ts  = time.strftime('%Y_%m_%d__%H_%M_%S', run_time)  # 给 log_dir 用
        ckpt_ts = time.strftime('%Y%m%d-%H%M%S', run_time) 
        log_dir = os.path.join('logs_diffusion', f"{config_name}_{log_ts}_{tag}")
        ckpt_dir = os.path.join('outputs', 'checkpoints', config_name, tag + f"_{ckpt_ts}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = setup_logger(log_dir, resume=args.resume, name='train')
    writer = SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)

    if not args.resume:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    # ==========================================================

    # Datasets and loaders
    logger.info('Loading dataset...')
    ds = Zinc20_3D_LMDBDataset(config.data.path, compressed=False)
    n = len(ds)
    logger.info(f"datasets['train'] size (filtered): {n}")

    n_train = int(0.9 * n)
    n_val = int(0.09 * n)
    n_test = n - n_train - n_val

    train_diff, val_diff, test_diff = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(2025)
    )

    train_loader = DataLoader(train_diff, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_diff,   batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_diff,  batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, pin_memory=True)

    train_iterator = utils_train.inf_iterator(train_loader)
    val_iterator   = utils_train.inf_iterator(val_loader)
    test_iterator  = utils_train.inf_iterator(test_loader)

    batch0 = next(train_iterator)
    config = update_config(config, batch0)

    logger.info(f"Auto inferred dims: node_in_dim={config.data.node_in_dim}, "
                f"edge_in_dim={config.data.edge_in_dim}, model.edge_feat_dim={config.model.edge_feat_dim}")

    # Encoder
    encoder = GraphGPSEncoder_CLS(
        config.encoder,
        node_in_dim=config.data.node_in_dim,
        edge_in_dim=config.data.edge_in_dim
    ).to(device)

    # Diffusion (pos-only)
    if config.model.model_type == 'uni_o2':
        diffusion = MolPosDiffusion(
            config.model,
            node_in_dim=config.data.node_in_dim,
            cond_dim=config.encoder.hidden_dim
        ).to(device)
    elif config.model.model_type == 'uni_o2_condition':
        diffusion = MolPosDiffusion_condition(
                    config.model,
                    node_in_dim=config.data.node_in_dim,
                    cond_dim=config.encoder.hidden_dim
                ).to(device)
    else:
        raise ValueError("model type error")
    # Optimizer and scheduler
    params = list(encoder.parameters()) + list(diffusion.parameters())
    opt_cfg = config.train.optimizer

    optimizer = Adam(
        params,
        lr=float(opt_cfg.lr),
        weight_decay=float(opt_cfg.weight_decay),
        betas=(float(opt_cfg.beta1), float(opt_cfg.beta2)),
    )

    sch_cfg = config.train.scheduler
    scheduler = None
    if sch_cfg.type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sch_cfg.factor),
            patience=int(sch_cfg.patience),
            min_lr=float(sch_cfg.min_lr),
            verbose=True,
        )

    # ====== [新增] resume：加载 ckpt，恢复 step/best_val/优化器等 ======
    best_val = float("inf")
    start_step = 1
    if args.resume:
        ckpt_path = resolve_ckpt_path(args.resume_ckpt)
        logger.info(f"Resuming from checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"])
        diffusion.load_state_dict(ckpt["diffusion"])

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            move_optimizer_to_device(optimizer, device)

        if scheduler is not None and ckpt.get("scheduler", None) is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                logger.warning(f"Failed to load scheduler state, will continue with fresh scheduler. err={e}")

        best_val = float(ckpt.get("best_val", float("inf")))
        last_step = int(ckpt.get("step", 0))
        start_step = last_step + 1

        logger.info(f"Resume done: last_step={last_step}, start_step={start_step}, best_val={best_val:.6f}")
    # ======================================================================

    # Train loop
    max_iters = int(config.train.max_iters)
    val_freq = int(config.train.val_freq)
    acc_steps = int(getattr(config.train, "n_acc_batch", 1))
    max_grad_norm = float(getattr(config.train, "max_grad_norm", 0.0))

    t0 = time.time()
    logger.info("Start training...")

    # ====== [替换] 从 start_step 开始 ======
    if start_step > max_iters:
        logger.info(f"start_step({start_step}) > max_iters({max_iters}), nothing to do. Will run final test only.")
    else:
        for step in range(start_step, max_iters + 1):
            encoder.train()
            diffusion.train()

            optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0
            for _ in range(acc_steps):
                b = next(train_iterator)
                b = _to_device_and_cast(b, device)

                enc_out = forward_encoder(encoder, b)
                node_emb, graph_emb = enc_out

                if config.model.model_type == 'uni_o2_condition':
                    out = diffusion.get_diffusion_loss(b, cond_node_emb=node_emb, time_step=None, graph_emb=graph_emb)
                else:
                    out = diffusion.get_diffusion_loss(b, cond_node_emb=node_emb, time_step=None)
                loss = out["loss"] / acc_steps
                loss.backward()
                total_loss += float(out["loss"].item())
            total_loss /= acc_steps

            if max_grad_norm and max_grad_norm > 0:
                nn.utils.clip_grad_norm_(params, max_grad_norm)

            optimizer.step()

            # logging
            if step % args.train_report_iter == 0:
                lr = optimizer.param_groups[0]["lr"]
                dt = time.time() - t0
                t0 = time.time()
                logger.info(f"[step {step}] train_loss={total_loss:.6f} lr={lr:.2e} dt={dt:.2f}s")
                writer.add_scalar("train/loss", total_loss, step)
                writer.add_scalar("train/lr", lr, step)

            # validation
            if step % val_freq == 0:
                val_loss = evaluate(encoder, diffusion, val_loader, device, config)
                logger.info(f"[step {step}] val_loss={val_loss:.6f}")
                writer.add_scalar("val/loss", val_loss, step)

                if scheduler is not None:
                    scheduler.step(val_loss)

                print(f"epoch {step} | train_loss: {total_loss:.6f} | val_loss: {val_loss:.6f}")

                # save best
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(
                        os.path.join(ckpt_dir, "best.pt"),
                        step, encoder, diffusion, optimizer, scheduler, best_val
                    )
                    logger.info(f"Saved best checkpoint at step={step}, best_val={best_val:.6f}")

                # save last
                save_ckpt(
                    os.path.join(ckpt_dir, "last.pt"),
                    step, encoder, diffusion, optimizer, scheduler, best_val
                )
    # ====================================

    # optional: final test eval
    test_loss = evaluate(encoder, diffusion, test_loader, device)
    logger.info(f"Final test_loss={test_loss:.6f}")
    writer.add_scalar("test/loss", test_loss, max_iters)

    writer.flush()
    writer.close()
