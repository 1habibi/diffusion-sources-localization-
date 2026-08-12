"""Runtime metadata and accelerator resource measurements for experiment reports."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

import torch


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    """Return reproducibility metadata without failing outside a Git checkout."""
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "accelerator": properties.name,
                "total_vram_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    else:
        metadata.update(
            {
                "accelerator": platform.processor() or "CPU",
                "total_vram_bytes": None,
                "compute_capability": None,
            }
        )
    return metadata


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    return (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
