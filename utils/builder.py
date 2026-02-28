
import numpy as np
import torch
import os
import sys
import time
import shutil
import logging

from torch.utils.data import random_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, Dict, Any

from preprocess import get_pcqm4m_dataset
from .data import CollateWithSPDLmdb, CollateWithSPDEdgeLmdb
from .train import inf_iterator
from .ZINC_preprocess import Zinc20_3D_LMDBDataset
from models import GraphGPSEncoder, GraphGPSEncoder_CLS, GraphGPSEncoder_CLS_GraphormerSPD, GraphGPSEncoder_CLS_PEARL   
from models import GraphGPSEncoder_CLS_GraphormerSPD_Pearl
from models import MolPosDiffusion, MolPosDiffusion_condition, MolPosDiffusion_cat


def _build_tag(config) -> str:
    tag = f"en{config.encoder.num_layers}_de{config.model.num_layers}_e_{config.encoder.name}_d_{config.model.model_type}"
    if config.encoder.name in ['cls_pearl', 'cls_graphormer_pearl']:
        tag = tag + f"_{config.encoder.pearl_fuse}"
    return tag

def _parse_run_time(run_time: Optional[object]) -> time.struct_time:
    """
    run_time 支持：
    - None：使用当前时间 time.localtime()
    - time.struct_time：直接用
    - str：支持两种格式：
        1) 'YYYYmmdd-HHMMSS'      (ckpt_ts)
        2) 'YYYY_mm_dd__HH_MM_SS' (log_ts)
    """
    if run_time is None:
        return time.localtime()

    if isinstance(run_time, time.struct_time):
        return run_time

    if isinstance(run_time, str):
        s = run_time.strip()
        # ckpt_ts: 20260227-123045
        try:
            return time.strptime(s, "%Y%m%d-%H%M%S")
        except Exception:
            pass
        # log_ts: 2026_02_27__12_30_45
        try:
            return time.strptime(s, "%Y_%m_%d__%H_%M_%S")
        except Exception:
            pass
        raise ValueError(
            f"Unrecognized run_time str format: {run_time}. "
            f"Expect '%Y%m%d-%H%M%S' or '%Y_%m_%d__%H_%M_%S'."
        )

    raise TypeError(f"run_time must be None / time.struct_time / str, got {type(run_time)}")


def build_ckpt(
    args,
    config,
    *,
    train: bool = True,
    run_time: Optional[object] = None,
    create_dir: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    返回：
      ckpt_dir: str
      meta: dict，包含 tag / config_name / ckpt_ts / log_ts（方便 downstream 复用）

    说明：
    - 训练 fresh：用 run_time（或当前时间）生成 ckpt_ts
    - resume：忽略 run_time，直接从 resume_ckpt 推导 ckpt_dir
    - downstream/eval：可以传 create_dir=False，避免误创建新目录
    """
    tag = _build_tag(config)

    # ===== train branch =====
    if train:
        if getattr(args, "resume", False):
            if args.resume_ckpt is None:
                raise ValueError("When --resume, you must provide --resume_ckpt")
            ckpt_path = resolve_ckpt_path(args.resume_ckpt)
            ckpt_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else args.resume_ckpt
            if create_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            meta = {
                "tag": tag,
                "mode": "resume",
                "ckpt_path": ckpt_path,
                "ckpt_dir": ckpt_dir,
                "ckpt_ts": None,
                "log_ts": None,
            }
            return ckpt_dir, meta

        # fresh training
        run_tm = _parse_run_time(run_time)
        ckpt_ts = time.strftime('%Y%m%d-%H%M%S', run_tm)         # for ckpt dir
        log_ts  = time.strftime('%Y_%m_%d__%H_%M_%S', run_tm)    # optional (for pairing)

        config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
        ckpt_dir = os.path.join('outputs', 'checkpoints', config_name, f"{tag}_{ckpt_ts}")

        if create_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        meta = {
            "tag": tag,
            "mode": "fresh",
            "config_name": config_name,
            "ckpt_ts": ckpt_ts,
            "log_ts": log_ts,
        }
        return ckpt_dir, meta

    # ===== eval branch =====
    # 通常 eval 不“创建 ckpt_dir”，而是读取现有 ckpt_path；这里给一个可选的推导能力
    run_tm = _parse_run_time(run_time)
    ckpt_ts = time.strftime('%Y%m%d-%H%M%S', run_tm)
    log_ts  = time.strftime('%Y_%m_%d__%H_%M_%S', run_tm)

    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    ckpt_dir = os.path.join('outputs', 'checkpoints', config_name, f"{tag}_{ckpt_ts}")

    meta = {
        "tag": tag,
        "mode": "eval_infer",
        "config_name": config_name,
        "ckpt_ts": ckpt_ts,
        "log_ts": log_ts,
    }
    # eval 默认不建目录，除非你真的要写东西进去
    if create_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir, meta


def build_logger(
    args,
    config,
    *,
    train: bool = True,
    run_time: Optional[object] = None,
    resume: Optional[bool] = None,
) -> Tuple[object, object, str]:
    """
    返回：
      logger, writer, log_dir
    """
    tag = _build_tag(config)

    if resume is None:
        resume = bool(getattr(args, "resume", False))

    run_tm = _parse_run_time(run_time)
    log_ts = time.strftime('%Y_%m_%d__%H_%M_%S', run_tm)

    if train:
        if resume:
            if args.resume_log_dir is None:
                raise ValueError("When --resume, you must provide --resume_log_dir")
            log_dir = args.resume_log_dir
            os.makedirs(log_dir, exist_ok=True)

            logger = setup_logger(log_dir, resume=True, name='train')
            writer = SummaryWriter(log_dir)

            logger.info(args)
            logger.info(config)
            logger.info(f"[RESUME] log_dir={log_dir}")

            return logger, writer, log_dir

        # fresh training
        log_dir = os.path.join('logs_diffusion', f"{log_ts}_{tag}")
        os.makedirs(log_dir, exist_ok=True)

        logger = setup_logger(log_dir, resume=False, name='train')
        writer = SummaryWriter(log_dir)

        logger.info(args)
        logger.info(config)
        logger.info(f"[TRAIN] log_dir={log_dir}")

        # snapshot code/config (best-effort)
        try:
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        except Exception as e:
            logger.warning(f"Failed to copy config file: {e}")

        try:
            shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to copy models dir: {e}")

        return logger, writer, log_dir

    # ===== evaluation branch =====
    log_dir = os.path.join('logs_diffusion_evaluation', f"{log_ts}_{tag}")
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger(log_dir, resume=False, name='evaluate')
    writer = SummaryWriter(log_dir)

    logger.info(args)
    logger.info(config)
    logger.info(f"[EVAL] log_dir={log_dir}")

    return logger, writer, log_dir  
# def build_logger(args, config, train: bool = True):
#     """
#     train=True  -> train diffusion logs + ckpt dir
#     train=False -> eval diffusion logs (no resume semantics)
#     """
#     tag = f"en{config.encoder.num_layers}_de{config.model.num_layers}_e_{config.encoder.name}_d_{config.model.model_type}"
#     if config.encoder.name in ['cls_pearl', 'cls_graphormer_pearl']:
#         tag = tag + f"_{config.encoder.pearl_fuse}"

#     run_time = time.localtime()
#     log_ts = time.strftime('%Y_%m_%d__%H_%M_%S', run_time)  # for log dir

#     if train:
#         if getattr(args, "resume", False):
#             if args.resume_log_dir is None or args.resume_ckpt is None:
#                 raise ValueError("When --resume, you must provide --resume_log_dir and --resume_ckpt")

#             log_dir = args.resume_log_dir
#             ckpt_path = resolve_ckpt_path(args.resume_ckpt)
#             ckpt_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else args.resume_ckpt
#             os.makedirs(log_dir, exist_ok=True)
#             os.makedirs(ckpt_dir, exist_ok=True)

#             logger = setup_logger(log_dir, resume=True, name='train')
#             writer = SummaryWriter(log_dir)

#             logger.info(args)
#             logger.info(config)
#             logger.info(f"[RESUME] ckpt_path={ckpt_path}")
#             logger.info(f"[RESUME] ckpt_dir={ckpt_dir}")

#             return logger, writer, log_dir, ckpt_dir

#         # fresh training
#         config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
#         ckpt_ts = time.strftime('%Y%m%d-%H%M%S', run_time)   # for ckpt dir

#         log_dir = os.path.join('logs_diffusion', f"{log_ts}_{tag}")
#         ckpt_dir = os.path.join('outputs', 'checkpoints', config_name, f"{tag}_{ckpt_ts}")
#         os.makedirs(log_dir, exist_ok=True)
#         os.makedirs(ckpt_dir, exist_ok=True)

#         logger = setup_logger(log_dir, resume=False, name='train')
#         writer = SummaryWriter(log_dir)

#         logger.info(args)
#         logger.info(config)
#         logger.info(f"[TRAIN] log_dir={log_dir}")
#         logger.info(f"[TRAIN] ckpt_dir={ckpt_dir}")

#         # snapshot code/config (best-effort)
#         try:
#             shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
#         except Exception as e:
#             logger.warning(f"Failed to copy config file: {e}")

#         try:
#             shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)
#         except Exception as e:
#             logger.warning(f"Failed to copy models dir: {e}")

#         return logger, writer, log_dir, ckpt_dir

#     # ===== evaluation branch =====
#     log_dir = os.path.join('logs_diffusion_evaluation', f"{log_ts}_{tag}")
#     os.makedirs(log_dir, exist_ok=True)

#     logger = setup_logger(log_dir, resume=False, name='evaluate')
#     writer = SummaryWriter(log_dir)

#     logger.info(args)
#     logger.info(config)
#     logger.info(f"[EVAL] log_dir={log_dir}")

#     return logger, writer, log_dir
  

def build_datasetLoader(config, logger, test_scale = None):
    logger.info('Loading dataset...')
    if config.data.name == "ZINC":
        ds = Zinc20_3D_LMDBDataset(config.data.path, compressed=False)
        n = len(ds)
        n_train = int(0.9 * n)
        n_val = int(0.09 * n)
        n_test = n - n_train - n_val
        logger.info(f"datasets['train'] size (filtered): {n}")
        if test_scale != None and test_scale < 1:
            n_val += n_test *(1 - test_scale)
            n_test = n - n_train - n_val

        train_diff, val_diff, test_diff = random_split(
            ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(2025)
        )
    elif config.data.name == "PCQM4M":

        datasets = get_pcqm4m_dataset(
            root=config.data.path,
            sdf_path=os.path.join(config.data.path, "pcqm4m-v2", "pcqm4m-v2-train.sdf"),
            build_3d_cache_if_missing=False,
            mapping_mode="order",
            max_mols=None,
            map_size=1 << 40,
            build_spd_cache_if_missing=True,
            spd_max_dist=int(config.encoder.spd_max_dist),
        )
        spd_lmdb_path = datasets["spd_lmdb_path"]
        assert spd_lmdb_path is not None, "spd_lmdb_path is None; SPD cache missing and not built."

        datasets_diffusion = datasets["train"]
        n = len(datasets_diffusion)
        logger.info(f"PCQM4M size (filtered): {n}")

        n_train = int(0.9 * n)
        n_val = int(0.09 * n)
        n_test = n - n_train - n_val
        if test_scale != None and test_scale < 1:
            n_val += int(n_test *(1 - test_scale))
            n_test = n - n_train - n_val
        logger.info(f"PCQM4M size (filtered): {n} | train size: {n_train} | val size: {n_val} | test size: {n_test}")
        train_diff, val_diff, test_diff = random_split(
            datasets_diffusion,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(2025)
        )
    else:
        raise ValueError("dataset name error")

    collate_fn = CollateWithSPDEdgeLmdb(spd_lmdb_path, spd_max_dist=int(config.encoder.spd_max_dist))
    train_loader = TorchDataLoader(
        train_diff,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = TorchDataLoader(
        val_diff,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = TorchDataLoader(
        test_diff,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    train_iterator = inf_iterator(train_loader)

    return train_loader, val_loader, test_loader, train_iterator

def build_encoder(cfg,device, node_in_dim=None, edge_in_dim=None):
    if node_in_dim != None:
        cfg.data.node_in_dim = node_in_dim
    if edge_in_dim != None:
        cfg.data.edge_in_dim = edge_in_dim
    

    cfg_encoder = cfg.encoder
    if cfg_encoder.name == 'normal':
        encoder = GraphGPSEncoder(
            cfg_encoder,
            node_in_dim=cfg.data.node_in_dim,
            edge_in_dim=cfg.data.edge_in_dim
        ).to(device)
    elif cfg_encoder.name == 'cls':
        encoder = GraphGPSEncoder_CLS(
            cfg_encoder,
            node_in_dim=cfg.data.node_in_dim,
            edge_in_dim=cfg.data.edge_in_dim
        ).to(device)
    elif cfg_encoder.name == 'cls_graphormer':
        encoder = GraphGPSEncoder_CLS_GraphormerSPD(
            cfg_encoder,
            node_in_dim=cfg.data.node_in_dim,
            edge_in_dim=cfg.data.edge_in_dim
        ).to(device)
    elif cfg_encoder.name == 'cls_pearl':
        encoder = GraphGPSEncoder_CLS_PEARL(
            cfg_encoder,
            node_in_dim=cfg.data.node_in_dim,
            edge_in_dim=cfg.data.edge_in_dim
        ).to(device)
    elif cfg_encoder.name == 'cls_graphormer_pearl':
        encoder = GraphGPSEncoder_CLS_GraphormerSPD_Pearl(
            cfg_encoder,
            node_in_dim=cfg.data.node_in_dim,
            edge_in_dim=cfg.data.edge_in_dim
        ).to(device)
    else:
        raise ValueError("encoder name error!")
    return encoder

def build_diffusion(cfg, device):
    cfg_model = cfg.model
    if cfg_model.model_type == 'uni_o2':
        diffusion = MolPosDiffusion(
                cfg_model,
                node_in_dim=cfg.data.node_in_dim,
                cond_dim=cfg.encoder.hidden_dim
            ).to(device)
    elif cfg_model.model_type == 'uni_o2_condition':
        diffusion = MolPosDiffusion_condition(
                    cfg_model,
                    node_in_dim=cfg.data.node_in_dim,
                    cond_dim=cfg.encoder.hidden_dim
                ).to(device)
    elif cfg_model.model_type == 'uni_o2_cat':
        diffusion = MolPosDiffusion_cat(
                    cfg_model,
                    node_in_dim=cfg.data.node_in_dim,
                    cond_dim=cfg.encoder.hidden_dim
                ).to(device)
    else:
        raise ValueError("model type error")
    return diffusion


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
