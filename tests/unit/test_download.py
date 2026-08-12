from __future__ import annotations

import hashlib

import pytest

from diffusion_sources.download import download_file


def test_download_file_uses_existing_verified_file(tmp_path):
    output = tmp_path / "graph.txt"
    output.write_bytes(b"0 1\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()

    result = download_file(
        "https://invalid.example/not-used", output, expected_sha256=digest
    )

    assert result.downloaded is False
    assert result.sha256 == digest
    assert result.bytes == 4


def test_download_file_rejects_zero_retries(tmp_path):
    with pytest.raises(ValueError, match="retries must be positive"):
        download_file("https://invalid.example", tmp_path / "file", retries=0)
