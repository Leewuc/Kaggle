from __future__ import annotations

import json
from typing import Any

import torch


PROFILE_DEFAULTS = {
    "cpu": {
        "batch_size": 8,
        "eval_batch_size": 16,
        "max_length": 96,
        "gradient_accumulation_steps": 1,
    },
    "quality": {
        "batch_size": 8,
        "eval_batch_size": 16,
        "max_length": 128,
        "gradient_accumulation_steps": 1,
    },
    "safe": {
        "batch_size": 4,
        "eval_batch_size": 8,
        "max_length": 96,
        "gradient_accumulation_steps": 4,
    },
    "medium": {
        "batch_size": 8,
        "eval_batch_size": 16,
        "max_length": 112,
        "gradient_accumulation_steps": 2,
    },
    "full40": {
        "batch_size": 16,
        "eval_batch_size": 32,
        "max_length": 128,
        "gradient_accumulation_steps": 1,
    },
}


def get_gpu_memory_gb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = 1024**3
    return free_bytes / gib, total_bytes / gib


def choose_gpu_profile(device: torch.device, requested_profile: str) -> str:
    if requested_profile != "auto":
        return requested_profile
    if device.type != "cuda":
        return "cpu"
    return "quality"


def apply_runtime_profile(args: Any, device: torch.device) -> Any:
    profile = choose_gpu_profile(device, getattr(args, "gpu_profile", "auto"))
    defaults = PROFILE_DEFAULTS[profile]

    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    free_gb, total_gb = get_gpu_memory_gb(device)
    args.runtime_profile = profile
    args.gpu_free_gb = free_gb
    args.gpu_total_gb = total_gb

    payload = {
        "runtime_profile": profile,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gpu_free_gb": None if free_gb is None else round(free_gb, 2),
        "gpu_total_gb": None if total_gb is None else round(total_gb, 2),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return args
