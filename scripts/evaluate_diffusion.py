# scripts/evaluate_diffusion.py
from __future__ import annotations
import os
import sys
import json
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import utils.misc as misc
from torch_geometric.data import Batch
import time
from collections import deque
import torch

from utils.covmat import pairwise_rmsd_matrix, covmat_from_rmsd, aggregate_covmat, CovMatResult
from utils.builder import build_datasetLoader,build_diffusion,build_encoder, build_logger, build_ckpt

def _gpu_mem_gb(device):
    if device.type != "cuda":
        return None
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserv = torch.cuda.memory_reserved(device) / 1024**3
    return alloc, reserv

def ckpt_path_from_tag(
    ckpt_root: str,
    enlayer: int,
    delayer: int,
    encoder_name: str,
    denoiser_name: str,
    ckpt_date: str
) -> str:
    # 与你 downstream 脚本一致的命名拼接方式
    ckpt_dirname = f"en{enlayer}_de{delayer}_e_{encoder_name}_d_{denoiser_name}_{ckpt_date}"
    ckpt_path = os.path.join(ckpt_root, ckpt_dirname, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    return ckpt_path

def split_pos_by_graph(batch: Batch, pos: torch.Tensor) -> List[np.ndarray]:
    pos_np = pos.detach().cpu().numpy()

    if hasattr(batch, "ptr") and batch.ptr is not None:
        ptr = batch.ptr.detach().cpu().numpy().astype(int)  # [B+1]
    else:
        # fallback: from batch.batch
        counts = torch.bincount(batch.batch, minlength=batch.num_graphs).detach().cpu().numpy()
        ptr = np.concatenate([[0], np.cumsum(counts).astype(int)])

    out = []
    for i in range(batch.num_graphs):
        s, e = int(ptr[i]), int(ptr[i + 1])
        out.append(pos_np[s:e])
    return out

@torch.no_grad()
def generate_positions_for_batch(
    encoder: torch.nn.Module,
    diffusion: torch.nn.Module,
    batch: Batch,
    n_samples: int,
    mode: str,
    t_recon: Optional[int] = None,
) -> List[List[np.ndarray]]:

    device = next(encoder.parameters()).device
    batch = batch.to(device)

    encoder.eval()
    diffusion.eval()

    cond_node_emb, graph_emb = encoder(batch)  # node_emb, graph_emb

    out: List[List[np.ndarray]] = [[] for _ in range(batch.num_graphs)]

    for _ in range(n_samples):
        if mode == "sample":
            pos_pred = diffusion.sample(
                batch_obj=batch,
                cond_node_emb=cond_node_emb,
                graph_emb=graph_emb,
            )
        elif mode == "reconstruct":
            t_start = (diffusion.num_timesteps - 1) if t_recon is None else int(t_recon)
            pos_pred = diffusion.reconstruct_from_clean(
                batch_obj=batch,
                x0=batch.pos,
                t_start=t_start,
                cond_node_emb=cond_node_emb,
                graph_emb=graph_emb,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        per_graph = split_pos_by_graph(batch, pos_pred)
        for g in range(batch.num_graphs):
            out[g].append(per_graph[g])

    return out

def evaluate_covmat(
    preds_per_graph: List[List[np.ndarray]],
    refs_per_graph: List[np.ndarray],
    threshold: float,
) -> Tuple[Dict[str, float], List[CovMatResult]]:
    """
    每个分子只有 1 个 ref 构象：refs_per_graph[g] 是 [n,3]。
    preds_per_graph[g] 是 K 个 [n,3]。
    """
    per_mol: List[CovMatResult] = []
    for g in range(len(refs_per_graph)):
        refs = [refs_per_graph[g]]
        preds = preds_per_graph[g]
        rmsd = pairwise_rmsd_matrix(preds, refs)  # [K,1]
        per_mol.append(covmat_from_rmsd(rmsd, threshold=threshold))
    agg = aggregate_covmat(per_mol)
    return agg, per_mol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", type=str, default="outputs/checkpoints/training")
    ap.add_argument("--ckpt_date", type=str, default="20260302-154952")
    ap.add_argument("--encoder_layers", type=int, default=9)
    ap.add_argument("--model_layers", type=int, default=5)
    ap.add_argument("--encoder_name", type=str, default="cls_graphormer_pearl")
    ap.add_argument("--denoiser_name", type=str, default="uni_o2_condition")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--n_samples", type=int, default=2)  # GeoDiff 系通常 Sg=2*Sr；你这 Sr=1 -> 2
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mode", type=str, choices=["sample", "reconstruct"], default="reconstruct")
    ap.add_argument("--t_recon", type=int, default=None)

    ap.add_argument("--out_json", type=str, default="eval_covmat.json")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)

    # 数据集/配置：按你工程现有方式补齐
    ap.add_argument("--config", type=str, default="configs/training.yml")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--pearl_fuse", type=str, default=None)

    args = ap.parse_args()
    
    # config
    config = misc.load_config(args.config)
    misc.seed_all(config.train.seed)

    device = torch.device(args.device)
    config = misc.update_config_with_args(config, args)
    config.train.batch_size = args.batch_size
    ckpt_path, meta = build_ckpt(args, config, train=False, run_time = args.ckpt_date)
    ckpt_path = os.path.join(ckpt_path, "best.pt")

    # logger
    logger, writer, log_dir = build_logger(args, config, train=False)

    # dataset
    train_loader, val_loader, test_loader, train_iterator = build_datasetLoader(config, logger, test_scale=0.1)
    batch0 = next(train_iterator)
    config = misc.update_config_with_data(config, batch0)
    logger.info(f"Auto inferred dims: node_in_dim={config.data.node_in_dim}, "
                f"edge_in_dim={config.data.edge_in_dim}, model.edge_feat_dim={config.model.edge_feat_dim}")
    logger.info(f"ckpt_path: {ckpt_path}")
    logger.info(f"split={args.split} mode={args.mode} n_samples={args.n_samples} thr={args.threshold} bs={args.batch_size}")


    # model
    encoder = build_encoder(config, device)
    diffusion = build_diffusion(config, device)

    # ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    diffusion.load_state_dict(ckpt["diffusion"], strict=True)

    # ========= 4) run eval =========
    all_per_mol: List[CovMatResult] = []

    if args.split == "train":
        loader = train_loader
    elif args.split == "eval":
        loader = val_loader
    else:
        loader = test_loader

    out_json = args.out_json
    if not os.path.isabs(out_json):
        out_json = os.path.join(log_dir, out_json)

    batch_times = deque(maxlen=20)
    n_seen = 0
    log_every=10

    for bi, batch in enumerate(tqdm(loader, desc=f"eval[{args.split}]")):
        tb0 = time.time()
        batch = batch.to(device)

        refs_per_graph = split_pos_by_graph(batch, batch.pos)

        preds_per_graph = generate_positions_for_batch(
            encoder=encoder,
            diffusion=diffusion,
            batch=batch,
            n_samples=args.n_samples,
            mode=args.mode,
            t_recon=args.t_recon,
        )

        # per-batch covmat
        _, per_mol = evaluate_covmat(preds_per_graph, refs_per_graph, threshold=args.threshold)
        all_per_mol.extend(per_mol)

        n_seen += int(batch.num_graphs)
        batch_times.append(time.time() - tb0)

        if (bi == 0) or ((bi + 1) % log_every == 0):
            # 在线聚合一下当前结果，便于观察是否在“跑着但没产出”
            cur = aggregate_covmat(all_per_mol)

            msg = (
            f"... "
            f"COV-R_mean={cur['COV-R_mean']:.4f} MAT-R_mean={cur['MAT-R_mean']:.4f} "
            f"COV-P_mean={cur['COV-P_mean']:.4f} MAT-P_mean={cur['MAT-P_mean']:.4f}"
            )

            if device.type == "cuda":
                alloc, reserv = _gpu_mem_gb(device)
                msg += f" | gpu_mem(alloc/reserv)={alloc:.2f}/{reserv:.2f}GB"

            logger.info(msg)
            # log sample scale statistics
            counts = torch.bincount(batch.batch).detach().cpu().numpy()
            logger.info(f"[batch stats] num_nodes: min={counts.min()} p50={np.median(counts):.0f} max={counts.max()}")

    final = aggregate_covmat(all_per_mol)
    payload = {
        "config": {
            "enlayer": args.encoder_layers,
            "delayer": args.model_layers,
            "encoder_name": args.encoder_name,
            "denoiser_name": args.denoiser_name,
            "ckpt_date": args.ckpt_date,
            "n_samples": args.n_samples,
            "threshold": args.threshold,
            "mode": args.mode,
            "t_recon": args.t_recon,
        },
        "metrics": final,
        "n_molecules": len(all_per_mol),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"[DONE] saved: {out_json}")
    logger.info(f"{final}")

if __name__ == "__main__":
    main()
