import argparse
import itertools
import os
import subprocess
import time
import re
from glob import glob
from collections import deque
from datetime import datetime


def parse_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def get_config_name(config_path: str) -> str:
    # training.yml -> training
    base = os.path.basename(config_path)
    return os.path.splitext(base)[0]


def log_ts_to_ckpt_ts(log_ts: str) -> str:
    # 'YYYY_MM_DD__HH_MM_SS' -> 'YYYYMMDD-HHMMSS'
    m = re.match(r"^(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})$", log_ts)
    if not m:
        raise ValueError(f"bad log_ts: {log_ts}")
    y, mo, d, hh, mm, ss = m.groups()
    return f"{y}{mo}{d}-{hh}{mm}{ss}"


def parse_logdir_timestamp(log_run_dir: str, config_name: str, en: int, de: int):
    """
    log_run_dir example:
      logs_diffusion/training_2026_01_13__10_17_06_en9_de6
    """
    base = os.path.basename(log_run_dir.rstrip("/"))
    pat = re.compile(
        rf"^{re.escape(config_name)}_(\d{{4}}_\d{{2}}_\d{{2}}__\d{{2}}_\d{{2}}_\d{{2}})_en{en}_de{de}$"
    )
    m = pat.match(base)
    if not m:
        return None
    return m.group(1)  # log_ts


def parse_dt_from_log_ts(log_ts: str):
    # 'YYYY_MM_DD__HH_MM_SS' -> datetime
    m = re.match(r"^(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})$", log_ts)
    if not m:
        return None
    y, mo, d, hh, mm, ss = map(int, m.groups())
    return datetime(y, mo, d, hh, mm, ss)


def find_latest_log_run(log_root: str, config_name: str, en: int, de: int):
    """
    找该 (en,de) 最新的 log run dir：
      {log_root}/{config_name}_YYYY_MM_DD__HH_MM_SS_en{en}_de{de}
    """
    pattern = os.path.join(log_root, f"{config_name}_*_en{en}_de{de}")
    candidates = [p for p in glob(pattern) if os.path.isdir(p)]
    print(pattern)
    print(candidates)
    if not candidates:
        return None

    scored = []
    for p in candidates:
        log_ts = parse_logdir_timestamp(p, config_name, en, de)
        dt = parse_dt_from_log_ts(log_ts) if log_ts else None
        score = dt.timestamp() if dt is not None else os.path.getmtime(p)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def build_ckpt_run_from_log_run(ckpt_root: str, config_name: str, en: int, de: int, log_run: str):
    """
    给定 log_run，精确构造 ckpt_run：
      outputs/checkpoints/{config_name}/en{en}_de{de}_{YYYYMMDD-HHMMSS}
    """
    log_ts = parse_logdir_timestamp(log_run, config_name, en, de)
    if log_ts is None:
        return None

    ckpt_ts = log_ts_to_ckpt_ts(log_ts)
    tag = f"en{en}_de{de}"
    ckpt_run = os.path.join(ckpt_root, f"{tag}_{ckpt_ts}")
    return ckpt_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=str, default="scripts/train_diffusion.py")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--logdir", type=str, default="./logs_diffusion")

    # 你的真实 ckpt 根目录：outputs/checkpoints/training/...
    # 这里默认自动拼上 config_name
    ap.add_argument("--ckpt_base", type=str, default="./outputs/checkpoints")

    ap.add_argument("--gpus", type=str, default="0,1,2,3")

    # resume
    ap.add_argument("--resume", action="store_true", help="resume each (en,de) by matching log_ts -> ckpt_ts")
    ap.add_argument(
        "--resume_policy",
        type=str,
        default="if_exists",
        choices=["require", "if_exists", "skip"],
        help="require: 找不到匹配就报错；if_exists: 找到就resume否则新开；skip: 找不到就跳过该组合",
    )

    # grid
    ap.add_argument("--enc_range", type=str, default="9,9")
    ap.add_argument("--dec_range", type=str, default="3,6")
    ap.add_argument("--grid", action="store_true")

    # fixed total layers
    ap.add_argument("--base_enc", type=int, default=8)
    ap.add_argument("--base_dec", type=int, default=9)
    ap.add_argument("--delta", type=int, default=4)

    ap.add_argument("--max_iters", type=int, default=0)
    args = ap.parse_args()

    config_name = get_config_name(args.config)  # training.yml -> training
    ckpt_root = os.path.join(args.ckpt_base, config_name)  # outputs/checkpoints/training

    gpus = parse_list(args.gpus)
    free_gpus = deque(gpus)
    running = []

    cpu_map = {0: "0-23", 1: "24-47", 2: "48-71", 3: "72-95"}

    def maybe_build_resume_args(en, de):
        if not args.resume:
            return []

        log_run = find_latest_log_run(args.logdir, config_name, en, de)
        if log_run is None:
            msg = f"[RESUME NOT FOUND] en={en} de={de} (no log_run)"
            if args.resume_policy == "require":
                raise RuntimeError(msg)
            elif args.resume_policy == "skip":
                print(msg + " -> SKIP", flush=True)
                return None
            else:
                print(msg + " -> START NEW", flush=True)
                return []

        ckpt_run = build_ckpt_run_from_log_run(ckpt_root, config_name, en, de, log_run)
        if ckpt_run is None:
            msg = f"[RESUME NOT FOUND] en={en} de={de} (cannot parse log_ts from {log_run})"
            if args.resume_policy == "require":
                raise RuntimeError(msg)
            elif args.resume_policy == "skip":
                print(msg + " -> SKIP", flush=True)
                return None
            else:
                print(msg + " -> START NEW", flush=True)
                return []

        last_pt = os.path.join(ckpt_run, "last.pt")
        if not os.path.isfile(last_pt):
            msg = f"[RESUME NOT FOUND] en={en} de={de} log_run={log_run} ckpt_run={ckpt_run} (missing last.pt)"
            if args.resume_policy == "require":
                raise RuntimeError(msg)
            elif args.resume_policy == "skip":
                print(msg + " -> SKIP", flush=True)
                return None
            else:
                print(msg + " -> START NEW", flush=True)
                return []

        # ✅ 精确对齐：log_run 的 ts -> ckpt_run 的 ts
        return ["--resume", "--resume_log_dir", log_run, "--resume_ckpt", ckpt_run]

    def launch_one(en, de):
        resume_args = maybe_build_resume_args(en, de)
        if resume_args is None:
            return False  # skip

        gpu = free_gpus.popleft()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        cmd = [
            "taskset", "-c", cpu_map.get(gpu, "0-23"),
            "python", args.train_script,
            "--config", args.config,
            "--logdir", args.logdir,
            "--device", "cuda:0",
            "--encoder_layers", str(en),
            "--model_layers", str(de),
        ]

        cmd += resume_args

        if args.max_iters and args.max_iters > 0:
            cmd += ["--max_iters", str(args.max_iters)]

        print(f"[LAUNCH gpu={gpu}] en={en} de={de}\n  {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(cmd, env=env)
        running.append((p, gpu, (en, de)))
        return True

    # 生成组合
    if args.grid:
        enc_lo, enc_hi = [int(x) for x in args.enc_range.split(",")]
        dec_lo, dec_hi = [int(x) for x in args.dec_range.split(",")]
        pairs = list(itertools.product(range(enc_lo, enc_hi + 1), range(dec_lo, dec_hi + 1)))
    else:
        pairs = []
        for d in range(-args.delta, args.delta + 1):
            en = args.base_enc + d
            de = args.base_dec - d
            pairs.append((en, de))

    pairs = [(en, de) for (en, de) in pairs if en >= 1 and de >= 1]

    idx = 0
    while idx < len(pairs) or running:
        while idx < len(pairs) and free_gpus:
            en, de = pairs[idx]
            idx += 1
            launch_one(en, de)

        time.sleep(2)
        still = []
        for p, gpu, (en, de) in running:
            ret = p.poll()
            if ret is None:
                still.append((p, gpu, (en, de)))
            else:
                print(f"[DONE gpu={gpu}] en={en} de={de} ret={ret}", flush=True)
                free_gpus.append(gpu)
        running = still

    print("All sweeps finished.")


if __name__ == "__main__":
    main()
