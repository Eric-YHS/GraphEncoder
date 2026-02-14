import argparse
import itertools
import os
import subprocess
import time
import re
import sys
import signal
from glob import glob
from collections import deque
from datetime import datetime


def parse_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def get_config_name(config_path: str) -> str:
    base = os.path.basename(config_path)
    return os.path.splitext(base)[0]


def log_ts_to_ckpt_ts(log_ts: str) -> str:
    m = re.match(r"^(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})$", log_ts)
    if not m:
        raise ValueError(f"bad log_ts: {log_ts}")
    y, mo, d, hh, mm, ss = m.groups()
    return f"{y}{mo}{d}-{hh}{mm}{ss}"


def parse_logdir_timestamp(log_run_dir: str, config_name: str, en: int, de: int):
    base = os.path.basename(log_run_dir.rstrip("/"))
    pat = re.compile(
        rf"^{re.escape(config_name)}_(\d{{4}}_\d{{2}}_\d{{2}}__\d{{2}}_\d{{2}}_\d{{2}})_en{en}_de{de}$"
    )
    m = pat.match(base)
    if not m:
        return None
    return m.group(1)


def parse_dt_from_log_ts(log_ts: str):
    m = re.match(r"^(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})$", log_ts)
    if not m:
        return None
    y, mo, d, hh, mm, ss = map(int, m.groups())
    return datetime(y, mo, d, hh, mm, ss)


def find_latest_log_run(log_root: str, config_name: str, en: int, de: int):
    pattern = os.path.join(log_root, f"{config_name}_*_en{en}_de{de}")
    candidates = [p for p in glob(pattern) if os.path.isdir(p)]
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
    log_ts = parse_logdir_timestamp(log_run, config_name, en, de)
    if log_ts is None:
        return None
    ckpt_ts = log_ts_to_ckpt_ts(log_ts)
    tag = f"en{en}_de{de}"
    ckpt_run = os.path.join(ckpt_root, f"{tag}_{ckpt_ts}")
    return ckpt_run


def daemonize(daemon_log: str, pidfile: str = None):
    """
    让当前进程脱离终端：关闭 Cursor / 断开 SSH / 关掉集成终端后仍可继续运行
    - 双 fork
    - setsid
    - 忽略 SIGHUP
    - stdout/stderr 重定向到 daemon_log
    - 写 pidfile（可选）
    """
    if os.name != "posix":
        raise RuntimeError("--detach 仅支持 Linux/WSL/macOS（posix），Windows 原生不支持 fork。")

    os.makedirs(os.path.dirname(os.path.abspath(daemon_log)) or ".", exist_ok=True)

    # 第一次 fork：让父进程直接退出（这样命令行立刻返回）
    pid = os.fork()
    if pid > 0:
        print(f"[DETACH] 已转入后台。daemon_log={daemon_log}", flush=True)
        if pidfile:
            print(f"[DETACH] pidfile={pidfile}（后台进程会写入真实 PID）", flush=True)
        sys.exit(0)

    # 子进程：成为新会话 leader，脱离控制终端
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # 第二次 fork：避免未来重新获得控制终端
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # 重定向 stdin/stdout/stderr
    devnull = os.open(os.devnull, os.O_RDONLY)
    log_fd = os.open(os.path.abspath(daemon_log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    os.dup2(devnull, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)

    if devnull > 2:
        os.close(devnull)
    if log_fd > 2:
        os.close(log_fd)

    # 写 pidfile（写当前这个“最终后台进程”的 PID）
    if pidfile:
        pidfile_abs = os.path.abspath(pidfile)
        os.makedirs(os.path.dirname(pidfile_abs) or ".", exist_ok=True)
        with open(pidfile_abs, "w") as f:
            f.write(str(os.getpid()))
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=str, default="scripts/train_diffusion.py")
    ap.add_argument("--config", type=str, default="configs/training.yml")
    ap.add_argument("--logdir", type=str, default="./logs_diffusion")
    ap.add_argument("--ckpt_base", type=str, default="./outputs/checkpoints")
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument('--encoder_name', type=str, default=None)
    ap.add_argument('--denoiser_name', type=str, default=None)

    # ✅ 后台运行相关
    ap.add_argument("--detach", action="store_true", help="后台运行；关掉 Cursor/终端也能继续")
    ap.add_argument("--daemon_log", type=str, default="./sweep_daemon.log", help="后台总日志文件")
    ap.add_argument("--pidfile", type=str, default="./sweep_daemon.pid", help="写入后台 PID 的文件")
    ap.add_argument("--proc_logdir", type=str, default="./sweep_proc_logs", help="每个(en,de)子进程日志目录")

    # resume
    ap.add_argument("--resume", action="store_true")
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
    ap.add_argument("--pearl_fuse", type=str, default="concat")
    
    args = ap.parse_args()

    # 建议：保持 cwd 不变（训练脚本/配置里可能依赖相对路径）
    # 但把日志/脚本路径转成绝对路径，避免重定向/后台后找不到
    args.daemon_log = os.path.abspath(args.daemon_log)
    args.pidfile = os.path.abspath(args.pidfile)
    args.proc_logdir = os.path.abspath(args.proc_logdir)

    if args.detach:
        daemonize(args.daemon_log, args.pidfile)

    os.makedirs(args.proc_logdir, exist_ok=True)

    config_name = get_config_name(args.config)
    ckpt_root = os.path.join(args.ckpt_base, config_name)

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

        return ["--resume", "--resume_log_dir", log_run, "--resume_ckpt", ckpt_run]

    def launch_one(en, de):
        resume_args = maybe_build_resume_args(en, de)
        if resume_args is None:
            return False

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
            "--encoder_name", args.encoder_name,
            "--denoiser_name", args.denoiser_name,
            "--pearl_fuse", args.pearl_fuse,
        ]

        cmd += resume_args

        if args.max_iters and args.max_iters > 0:
            cmd += ["--max_iters", str(args.max_iters)]

        run_log = os.path.join(args.proc_logdir, f"en{en}_de{de}_gpu{gpu}.log")
        print(f"[LAUNCH gpu={gpu}] en={en} de={de}\n  {' '.join(cmd)}\n  -> {run_log}", flush=True)

        # 每个 run 单独日志；并且 start_new_session=True 让子进程不受终端挂断影响
        log_f = open(run_log, "a", buffering=1)
        p = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,
        )
        # 父进程持有 log_f 句柄没必要；关掉让文件由子进程持有即可
        log_f.close()

        running.append((p, gpu, (en, de)))
        return True

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

    print("All sweeps finished.", flush=True)


if __name__ == "__main__":
    main()
