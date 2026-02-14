import argparse
import os
import shutil
import sys
from pathlib import Path
import time

import logging
from glob import glob

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import utils.misc as misc
from utils.builder import build_encoder, build_diffusion, build_datasetLoader, build_logger

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter



def _to_device_and_cast(batch, device):
    batch = batch.to(device)

    if hasattr(batch, "x") and batch.x is not None:
        batch.x = batch.x.float()
    if hasattr(batch, "pos") and batch.pos is not None:
        batch.pos = batch.pos.float()
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr.float()

    return batch

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

            if config.model.model_type in ['uni_o2_condition', 'uni_o2_cat']:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--train_report_iter', type=int, default=50)
    parser.add_argument('--encoder_layers', type=int, default=None)
    parser.add_argument('--model_layers', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default='GraphGPS_Encoder')
    parser.add_argument('--encoder_name', type=str, default=None)
    parser.add_argument('--denoiser_name', type=str, default=None)
    parser.add_argument('--pearl_fuse', type=str, default=None)
    

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
    config = misc.update_config_with_args(config, args)

    # Logging / dirs
    logger, writer, log_dir, ckpt_dir = build_logger(args, config, train=True)

    # Datasets and loaders
    train_loader, val_loader, test_loader, train_iterator = build_datasetLoader(config, logger)

    batch0 = next(train_iterator)
    config = misc.update_config_with_data(config, batch0)

    logger.info(f"Auto inferred dims: node_in_dim={config.data.node_in_dim}, "
                f"edge_in_dim={config.data.edge_in_dim}, model.edge_feat_dim={config.model.edge_feat_dim}")

    # Encoder
    encoder = build_encoder(config, device)
    # Diffusion (pos-only)
    diffusion = build_diffusion(config, device)
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
            threshold=float(getattr(sch_cfg, "threshold", 0.0)),
            threshold_mode=str(getattr(sch_cfg, "threshold_mode", "rel")),
            cooldown=int(getattr(sch_cfg, "cooldown", 0)),
            verbose=True,
        )

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

                if config.model.model_type in ['uni_o2_condition', 'uni_o2_cat']:
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
    test_loss = evaluate(encoder, diffusion, test_loader, device, config)
    logger.info(f"Final test_loss={test_loss:.6f}")
    writer.add_scalar("test/loss", test_loss, max_iters)

    writer.flush()
    writer.close()
