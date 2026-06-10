#!/usr/bin/env python3
"""Lightweight RhythmMamba training monitor.

This script watches a training log and checkpoint directory, then writes a
compact status line every few seconds. It is intentionally independent from the
training process, so it can run in a normal terminal without touching training.
"""

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


LOSS_RE = re.compile(
    r"Train epoch\s+(?P<epoch>\d+):.*?(?P<progress>\d+)%.*?loss=(?P<loss>[0-9.]+)"
)


def read_tail(path, max_bytes=256_000):
    if not path or not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read().decode("utf-8", errors="replace")


def latest_training_state(train_log):
    text = read_tail(train_log)
    matches = list(LOSS_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return {
        "epoch": int(last.group("epoch")),
        "progress": int(last.group("progress")),
        "loss": float(last.group("loss")),
    }


def latest_checkpoint(checkpoint_dir, model_prefix):
    if not checkpoint_dir or not checkpoint_dir.exists():
        return None
    pattern = f"{model_prefix}_Epoch*.pth"
    checkpoints = sorted(
        checkpoint_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    match = re.search(r"_Epoch(\d+)\.pth$", latest.name)
    return {
        "epoch": int(match.group(1)) if match else None,
        "path": latest,
        "mtime": latest.stat().st_mtime,
    }


def gpu_snapshot():
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2)
    except Exception:
        return "gpu=n/a"
    parts = [p.strip() for p in out.strip().split(",")]
    if len(parts) < 5:
        return "gpu=n/a"
    util, mem_used, mem_total, power, temp = parts[:5]
    return f"gpu={util}% mem={mem_used}/{mem_total}MB power={power}W temp={temp}C"


def process_alive(pid):
    if pid is None:
        return None
    return os.path.exists(f"/proc/{pid}")


def append_line(path, line):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        state = latest_training_state(args.train_log)
        ckpt = latest_checkpoint(args.checkpoint_dir, args.model_prefix)
        alive = process_alive(args.pid)

        fields = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        if alive is not None:
            fields.append(f"pid={args.pid} alive={alive}")
        if state:
            fields.append(
                f"epoch={state['epoch']} progress={state['progress']}% loss={state['loss']:.4f}"
            )
        else:
            fields.append("epoch=n/a progress=n/a loss=n/a")
        if ckpt:
            fields.append(f"latest_ckpt_epoch={ckpt['epoch']}")
        else:
            fields.append("latest_ckpt_epoch=n/a")
        fields.append(gpu_snapshot())

        line = " | ".join(fields)
        print(line, flush=True)
        append_line(args.out, line)

        if args.once:
            break
        if alive is False:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
