from __future__ import annotations

import torch

from diffusion_sources.runtime import peak_memory_bytes, reset_peak_memory, runtime_metadata


def test_cpu_runtime_metadata_is_reportable():
    device = torch.device("cpu")
    metadata = runtime_metadata(device)

    assert metadata["device"] == "cpu"
    assert metadata["torch"] == torch.__version__
    assert metadata["total_vram_bytes"] is None
    assert peak_memory_bytes(device) is None
    reset_peak_memory(device)
