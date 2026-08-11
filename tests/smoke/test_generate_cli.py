from __future__ import annotations

import subprocess
import sys


def test_generate_dataset_cli(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--config",
            "configs/pilot.yaml",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "train.npz").exists()
