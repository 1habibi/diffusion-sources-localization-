"""Download external graph datasets without silently replacing local files."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


FACEBOOK_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"


@dataclass(frozen=True)
class DownloadResult:
    url: str
    output: str
    bytes: int
    sha256: str
    downloaded: bool


def download_file(
    url: str,
    output: str | Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
    retries: int = 3,
    timeout: float = 120.0,
) -> DownloadResult:
    """Download to a temporary file, verify it, then atomically move it in place."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        digest = _sha256(output_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Existing file does not match expected SHA-256.")
        return DownloadResult(
            url=url,
            output=str(output_path),
            bytes=output_path.stat().st_size,
            sha256=digest,
            downloaded=False,
        )

    if retries < 1:
        raise ValueError("retries must be positive.")
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    with temporary_path.open("wb") as file:
                        while chunk := response.read(1024 * 1024):
                            file.write(chunk)
                break
            except (TimeoutError, urllib.error.URLError):
                temporary_path.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                time.sleep(attempt)
        digest = _sha256(temporary_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Downloaded file does not match expected SHA-256.")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return DownloadResult(
        url=url,
        output=str(output_path),
        bytes=output_path.stat().st_size,
        sha256=digest,
        downloaded=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=FACEBOOK_URL)
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/facebook_combined.txt.gz")
    )
    parser.add_argument("--sha256")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    result = download_file(
        args.url,
        args.output,
        expected_sha256=args.sha256,
        force=args.force,
        retries=args.retries,
        timeout=args.timeout,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
