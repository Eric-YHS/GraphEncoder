import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.condition_sources import build_condition_system, graph_stats
from models.molopt_score_model import MolPosDiffusion_condition
from preprocess import get_pcqm4m_dataset
from utils.data import CollateWithSPDEdgeLmdb


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_rank0() -> bool:
    return rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def setup_distributed(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank)
    return torch.device(args.device if torch.cuda.is_available() else "cpu")


def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir: Path):
    logger = logging.getLogger("condition_ladder")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if is_rank0():
        log_dir.mkdir(parents=True, exist_ok=True)
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(log_dir / "train.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def reduce_sum(value: float, device: torch.device) -> float:
    t = torch.tensor(float(value), device=device)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def reduce_mean(value: float, device: torch.device) -> float:
    t = torch.tensor(float(value), device=device)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= float(world_size())
    return float(t.item())


def metric_value(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


def batch_to_device(batch, device):
    try:
        batch = batch.to(device, non_blocking=True)
    except TypeError:
        batch = batch.to(device)
    if hasattr(batch, "x") and batch.x is not None:
        batch.x = batch.x.float()
    if hasattr(batch, "pos") and batch.pos is not None:
        batch.pos = batch.pos.float()
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr.float()
    return batch


def filter_subset_by_valid_mask(dataset, subset: Subset, valid_mask: Optional[np.ndarray], name: str, logger) -> Subset:
    if valid_mask is None:
        return subset
    keep = []
    base_indices = getattr(dataset, "indices", None)
    if base_indices is None:
        raise AttributeError("PCQM dataset wrapper is expected to expose .indices")
    for rel_idx in subset.indices:
        global_idx = int(base_indices[int(rel_idx)])
        if global_idx < len(valid_mask) and int(valid_mask[global_idx]) > 0:
            keep.append(int(rel_idx))
    logger.info("%s valid-filtered size=%d/%d", name, len(keep), len(subset.indices))
    if not keep:
        raise RuntimeError("%s split is empty after condition valid-mask filtering" % name)
    return Subset(dataset, keep)


def load_compare_valid_mask(cfg, condition_system, logger) -> Optional[np.ndarray]:
    policy = str(getattr(cfg.condition, "valid_mask_source", "grale")).lower()
    if policy in ("none", "off", "false"):
        logger.info("condition valid-mask policy=%s; no mask will be applied", policy)
        return None

    if policy in ("grale", "compare"):
        cache_dir = str(getattr(cfg.condition, "compare_valid_cache_dir", getattr(cfg.condition, "grale_cache_dir", "")))
        mask_path = os.path.join(cache_dir, "valid_uint8.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path, mmap_mode="r")
            logger.info("condition valid-mask policy=%s path=%s valid=%d/%d", policy, mask_path, int(mask.sum()), len(mask))
            return mask
        if policy == "grale":
            raise FileNotFoundError(
                "condition.valid_mask_source=grale but %s is missing; build the GRALE cache first "
                "or set condition.require_valid_mask=false for smoke runs" % mask_path
            )

    valid_mask = condition_system.valid_mask
    if valid_mask is not None:
        logger.info("condition valid-mask policy=%s using source mask valid=%d/%d", policy, int(valid_mask.sum()), len(valid_mask))
        return valid_mask
    raise ValueError("condition.require_valid_mask=true but no valid mask is available")


def build_loaders(cfg, condition_system, logger):
    datasets = get_pcqm4m_dataset(
        root=str(cfg.data.path),
        sdf_path=os.path.join(str(cfg.data.path), "pcqm4m-v2", "pcqm4m-v2-train.sdf"),
        build_3d_cache_if_missing=False,
        mapping_mode="order",
        max_mols=getattr(cfg.data, "max_mols", None),
        map_size=1 << 40,
        build_spd_cache_if_missing=True,
        spd_max_dist=int(cfg.encoder.spd_max_dist),
    )
    dataset = datasets["train"]
    spd_lmdb_path = datasets["spd_lmdb_path"]
    if spd_lmdb_path is None:
        raise RuntimeError("PCQM4M SPD LMDB is missing.")

    n = len(dataset)
    n_train = int(float(cfg.data.train_ratio) * n)
    n_val = int(float(cfg.data.val_ratio) * n)
    n_test = n - n_train - n_val
    splits = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(int(cfg.train.seed)))
    train_ds, val_ds = splits[0], splits[1]
    require_mask = bool(getattr(cfg.condition, "require_valid_mask", False))
    if require_mask:
        valid_mask = load_compare_valid_mask(cfg, condition_system, logger)
        train_ds = filter_subset_by_valid_mask(dataset, train_ds, valid_mask, "train", logger)
        val_ds = filter_subset_by_valid_mask(dataset, val_ds, valid_mask, "val", logger)
    else:
        logger.info("condition valid-mask filtering disabled")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size(), rank=rank(), shuffle=True, drop_last=True) if is_dist() else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size(), rank=rank(), shuffle=False, drop_last=False) if is_dist() else None
    collate_fn = CollateWithSPDEdgeLmdb(spd_lmdb_path, spd_max_dist=int(cfg.encoder.spd_max_dist))
    num_workers = int(cfg.train.num_workers)
    common = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(cfg.train, "pin_memory", True)),
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        common["persistent_workers"] = bool(getattr(cfg.train, "persistent_workers", True))
        common["prefetch_factor"] = int(getattr(cfg.train, "prefetch_factor", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    logger.info(
        "PCQM4M filtered_pos=%d train=%d val=%d test=%d spd=%s",
        n,
        len(train_ds),
        len(val_ds),
        n_test,
        spd_lmdb_path,
    )
    return train_loader, val_loader, train_sampler


def build_models(cfg, device):
    diffusion = MolPosDiffusion_condition(
        cfg.model,
        node_in_dim=int(cfg.data.node_in_dim),
        cond_dim=int(cfg.condition.node_dim),
    ).to(device)
    condition_system = build_condition_system(cfg).to(device)
    if is_dist():
        diffusion = DDP(diffusion, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        if any(p.requires_grad for p in condition_system.parameters()):
            condition_system = DDP(condition_system, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    return diffusion, condition_system


def audit_v1(diffusion, cfg, logger):
    raw = unwrap(diffusion)
    hidden = int(cfg.model.hidden_dim)
    layers = int(cfg.model.num_layers)
    graph_dim = int(cfg.model.graph_emb_dim)
    expected = hidden * layers
    if graph_dim != expected:
        raise ValueError("model.graph_emb_dim=%d must equal num_layers*hidden_dim=%d" % (graph_dim, expected))
    refine = raw.refine_net
    embedder = getattr(refine, "graph_cond_embedder", None)
    if embedder is None:
        raise ValueError("V1 refine_net has no graph_cond_embedder; condition ladder requires graph_cond_dim > 0")
    logger.info(
        "V1 audit ok graph_dim=%d hidden=%d layers=%d graph_dropout=%.4f node_dropout=%.4f bond_edge_attr_forced_none=%s",
        graph_dim,
        hidden,
        layers,
        float(getattr(cfg.model, "graph_dropout", 0.0)),
        float(getattr(cfg.model, "node_dropout", 0.0)),
        bool(getattr(cfg.train, "log_bond_edge_attr_forced_none", True)),
    )


def adapter_grad_norm(condition_system) -> float:
    raw = unwrap(condition_system)
    if raw.adapter is None:
        return 0.0
    total = 0.0
    for p in raw.adapter.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def condition_forward(condition_system, batch, device):
    return condition_system(batch, device)


def should_apply_graph_dropout(cfg) -> bool:
    if str(cfg.condition.source).lower() == "none":
        return False
    return bool(getattr(cfg.condition, "apply_graph_dropout", True))


def fixed_rng_loss(diffusion, batch, cond_node, graph_cond, t, device, seed: int, apply_graph_dropout: bool = False):
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        out = unwrap(diffusion).get_diffusion_loss(
            batch,
            cond_node_emb=cond_node,
            graph_emb=graph_cond,
            time_step=t,
            apply_graph_dropout=apply_graph_dropout,
        )
    return float(out["loss"].detach().item())


@torch.no_grad()
def condition_probes(diffusion, condition_system, batch, device, fixed_t_values):
    diffusion.eval()
    condition_system.eval()
    cond_node, graph_cond, _ = condition_forward(condition_system, batch, device)
    num_graphs = graph_cond.size(0)
    results = {}
    for t_value in fixed_t_values:
        t = torch.full((num_graphs,), int(t_value), dtype=torch.long, device=device)
        seed = 1000003 + int(t_value)
        zero_graph = torch.zeros_like(graph_cond)
        if num_graphs > 1:
            shuf_graph = graph_cond[torch.randperm(num_graphs, device=device)]
        else:
            shuf_graph = graph_cond
        results[int(t_value)] = {
            "normal": fixed_rng_loss(diffusion, batch, cond_node, graph_cond, t, device, seed),
            "zero_graph": fixed_rng_loss(diffusion, batch, cond_node, zero_graph, t, device, seed),
            "shuf_graph": fixed_rng_loss(diffusion, batch, cond_node, shuf_graph, t, device, seed),
        }
    return results


@torch.no_grad()
def validate(diffusion, condition_system, val_loader, device, cfg):
    diffusion.eval()
    condition_system.eval()
    sums = defaultdict(float)
    count = 0.0
    max_batches = int(getattr(cfg.train, "validate_batches", 20))
    use_amp = bool(getattr(cfg.train, "amp", True)) and device.type == "cuda"
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        with autocast(enabled=use_amp):
            cond_node, graph_cond, _ = condition_forward(condition_system, batch, device)
            out = unwrap(diffusion).get_diffusion_loss(
                batch,
                cond_node_emb=cond_node,
                graph_emb=graph_cond,
                time_step=None,
                apply_graph_dropout=False,
            )
        bsz = float(graph_cond.size(0))
        sums["loss"] += metric_value(out["loss"]) * bsz
        count += bsz
    count = reduce_sum(count, device)
    metrics = {}
    for key, value in sums.items():
        total = reduce_sum(value, device)
        metrics["val_" + key] = total / max(count, 1.0)
    return metrics


def save_checkpoint(path: Path, diffusion, condition_system, optimizer, scaler, cfg, step: int, best_val: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "diffusion": unwrap(diffusion).state_dict(),
        "condition_system": unwrap(condition_system).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": int(step),
        "best_val": float(best_val),
    }
    torch.save(ckpt, path)


def train_one_step(diffusion, condition_system, optimizer, scaler, next_batch, device, cfg, use_amp: bool):
    acc_steps = int(cfg.train.n_acc_batch)
    metrics = defaultdict(float)
    graph_count = 0.0
    data_time = 0.0
    last_batch = None
    last_graph_cond = None
    optimizer.zero_grad(set_to_none=True)
    for _ in range(acc_steps):
        t0 = time.time()
        batch = batch_to_device(next_batch(), device)
        data_time += time.time() - t0
        last_batch = batch
        with autocast(enabled=use_amp):
            cond_node, graph_cond, _features = condition_forward(condition_system, batch, device)
            last_graph_cond = graph_cond
            out = unwrap(diffusion).get_diffusion_loss(
                batch,
                cond_node_emb=cond_node,
                graph_emb=graph_cond,
                time_step=None,
                apply_graph_dropout=should_apply_graph_dropout(cfg),
            )
            loss = out["loss"] / float(acc_steps)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()
        metrics["loss"] += metric_value(out["loss"]) / float(acc_steps)
        graph_count += float(graph_cond.size(0))
    scaler.unscale_(optimizer)
    grad_clip = float(getattr(cfg.train, "grad_clip", 0.0))
    params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(params, grad_clip)
    metrics["adapter_grad_norm"] = adapter_grad_norm(condition_system)
    scaler.step(optimizer)
    scaler.update()
    return metrics, graph_count, data_time, last_batch, last_graph_cond


def train(args):
    device = setup_distributed(args)
    cfg = OmegaConf.load(args.config)
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    if args.condition_source is not None:
        cfg.condition.source = args.condition_source
    if args.max_iters is not None:
        cfg.train.max_iters = int(args.max_iters)
    if args.no_amp:
        cfg.train.amp = False

    seed_all(int(cfg.train.seed) + rank())
    run_name = "%s_%s_%s" % (args.exp_name, str(cfg.condition.source), time.strftime("%Y%m%d-%H%M%S"))
    log_dir = Path(args.logdir) / run_name
    logger = setup_logger(log_dir)
    if is_rank0():
        OmegaConf.save(cfg, log_dir / "resolved_config.yml")
        logger.info("log_dir=%s", log_dir)
        logger.info("world_size=%d device=%s condition=%s", world_size(), device, cfg.condition.source)

    diffusion, condition_system = build_models(cfg, device)
    audit_v1(diffusion, cfg, logger)
    train_loader, val_loader, train_sampler = build_loaders(cfg, unwrap(condition_system), logger)

    params = [p for p in list(diffusion.parameters()) + list(condition_system.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg.train.max_iters)))
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    if train_sampler is not None:
        train_sampler.set_epoch(0)
    train_iter = iter(train_loader)
    epoch = 0

    def next_batch():
        nonlocal train_iter, epoch
        try:
            return next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    if args.audit_only:
        batch = batch_to_device(next_batch(), device)
        with torch.no_grad():
            cond_node, graph_cond, features = condition_forward(condition_system, batch, device)
            out = unwrap(diffusion).get_diffusion_loss(batch, cond_node_emb=cond_node, graph_emb=graph_cond, time_step=None, apply_graph_dropout=False)
        if is_rank0():
            logger.info(
                "audit_only ok cond_node=%s graph_cond=%s features=%s loss=%.6f stats=%s",
                tuple(cond_node.shape),
                tuple(graph_cond.shape),
                tuple(features.shape),
                float(out["loss"].item()),
                graph_stats(graph_cond),
            )
        cleanup_distributed()
        return

    best_val = float("inf")
    max_iters = int(cfg.train.max_iters)
    report_iter = int(cfg.train.report_iter)
    probe_freq = int(cfg.train.probe_freq)
    val_freq = int(cfg.train.val_freq)
    save_freq = int(cfg.train.save_freq)
    fixed_t_values = list(getattr(cfg.train, "fixed_t", [100, 500, 900]))
    fallback_on_nan = bool(getattr(cfg.train, "amp_fallback_on_nan", True))
    fallback_steps = int(getattr(cfg.train, "amp_fallback_steps", 100))
    last_report_time = time.time()
    last_report_step = 0
    interval_graph_count = 0.0
    interval_data_time = 0.0
    interval_step_time = 0.0

    for step in range(1, max_iters + 1):
        diffusion.train()
        condition_system.train()
        step_start = time.time()
        try:
            metrics, graph_count, data_time, last_batch, last_graph_cond = train_one_step(
                diffusion, condition_system, optimizer, scaler, next_batch, device, cfg, use_amp
            )
        except FloatingPointError:
            if use_amp and fallback_on_nan and step <= fallback_steps:
                if is_rank0():
                    logger.warning("non-finite AMP loss at step=%d; disabling AMP and retrying with next batch", step)
                use_amp = False
                scaler = GradScaler(enabled=False)
                optimizer.zero_grad(set_to_none=True)
                metrics, graph_count, data_time, last_batch, last_graph_cond = train_one_step(
                    diffusion, condition_system, optimizer, scaler, next_batch, device, cfg, use_amp
                )
            else:
                raise
        scheduler.step()
        step_time = time.time() - step_start
        interval_graph_count += graph_count
        interval_data_time += data_time
        interval_step_time += step_time

        if step % report_iter == 0 or step == 1:
            reduced = {k: reduce_mean(v, device) for k, v in metrics.items()}
            total_graphs = reduce_sum(graph_count, device)
            total_data_time = reduce_mean(data_time, device)
            graphs_per_sec = total_graphs / max(step_time, 1e-6)
            now = time.time()
            interval_wall = now - last_report_time
            interval_steps = max(1, step - last_report_step)
            interval_graphs = reduce_sum(interval_graph_count, device)
            interval_data = reduce_mean(interval_data_time, device)
            interval_step = reduce_mean(interval_step_time, device)
            interval_graphs_sec = interval_graphs / max(interval_wall, 1e-6)
            if is_rank0():
                stats = graph_stats(last_graph_cond)
                mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
                logger.info(
                    "step=%d loss=%.6f graphs_sec=%.1f interval_graphs_sec=%.1f interval_wall=%.3f interval_steps=%d data_time=%.3f interval_data_time=%.3f step_time=%.3f interval_step_time=%.3f amp=%s mem_gb=%.2f adapter_grad=%.4f cond_stats=%s",
                    step,
                    reduced["loss"],
                    graphs_per_sec,
                    interval_graphs_sec,
                    interval_wall,
                    interval_steps,
                    total_data_time,
                    interval_data,
                    step_time,
                    interval_step,
                    use_amp,
                    mem,
                    reduced.get("adapter_grad_norm", 0.0),
                    stats,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            last_report_time = now
            last_report_step = step
            interval_graph_count = 0.0
            interval_data_time = 0.0
            interval_step_time = 0.0

        if step % probe_freq == 0 or step == 1:
            probes = condition_probes(diffusion, condition_system, last_batch, device, fixed_t_values)
            if is_rank0():
                logger.info("probes step=%d %s", step, probes)

        if step % val_freq == 0 or step == max_iters:
            val_metrics = validate(diffusion, condition_system, val_loader, device, cfg)
            if is_rank0():
                logger.info("validation " + " ".join(["%s=%.6f" % (k, v) for k, v in val_metrics.items()]))
                val_loss = float(val_metrics["val_loss"])
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(log_dir / "best.pt", diffusion, condition_system, optimizer, scaler, cfg, step, best_val)
                    logger.info("saved best.pt step=%d best_val=%.6f", step, best_val)

        if is_rank0() and (step % save_freq == 0 or step == max_iters):
            save_checkpoint(log_dir / "last.pt", diffusion, condition_system, optimizer, scaler, cfg, step, best_val)

    if is_rank0():
        logger.info("done log_dir=%s", log_dir)
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_condition_ladder.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, default="logs_condition_ladder")
    parser.add_argument("--exp_name", type=str, default="condition_ladder")
    parser.add_argument("--condition_source", type=str, default=None, choices=["none", "property", "grale_official"])
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
