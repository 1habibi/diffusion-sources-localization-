"""Hash-checked, append-only stage artifacts for the temporal pilot."""
from __future__ import annotations
import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _stage_path(root: Path, stage: str) -> Path:
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', stage):
        raise ValueError('Invalid stage name.')
    return Path(root) / stage


def _json(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_stage(output_dir: Path, stage: str, manifest: dict, payload: dict) -> None:
    target = _stage_path(output_dir, stage)
    if target.exists():
        raise FileExistsError(f'Result exists: {target}; no overwrite allowed.')
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    if list(root.glob(f'.{stage}-*')):
        raise ValueError(f'partial stage {stage}; inspect it or choose a new pilot output directory.')
    temporary = Path(tempfile.mkdtemp(prefix=f'.{stage}-', dir=root))
    _json(temporary / 'payload.json', payload)
    files = ['payload.json']
    if 'rows' in payload:
        _json(temporary / 'metrics.json', {k: v for k, v in payload.items() if k != 'rows'})
        files.append('metrics.json')
        flat = []
        for row in payload['rows']:
            item = {}
            for key, value in row.items():
                if isinstance(value, dict):
                    item.update({f'{key}_{metric}': val for metric, val in value.items()})
                else:
                    item[key] = json.dumps(value) if isinstance(value, (list, tuple)) else value
            flat.append(item)
        if flat:
            with (temporary / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
                writer.writeheader(); writer.writerows(flat)
            files.append('predictions.csv')
    saved = {**manifest, 'schema_version': 1,
             'file_hashes': {name: sha256_file(temporary / name) for name in files}}
    _json(temporary / 'manifest.json', saved)
    (temporary / 'complete').write_text('complete\n', encoding='utf-8')
    temporary.rename(target)


def read_stage(output_dir: Path, stage: str, expected_identity: dict) -> tuple[dict, dict]:
    target = _stage_path(output_dir, stage)
    if not (target / 'complete').is_file():
        raise ValueError(f'incomplete or missing stage {stage}')
    try:
        manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or manifest.get('identity') != expected_identity:
            raise ValueError(f'identity mismatch for {stage}; choose a new pilot output directory.')
        files = manifest['file_hashes']
        if 'payload.json' not in files:
            raise ValueError(f'missing payload hash for {stage}')
        for name, digest in files.items():
            if Path(name).name != name or sha256_file(target / name) != digest:
                raise ValueError(f'file hash mismatch for {stage}/{name}')
        return manifest, json.loads((target / 'payload.json').read_text(encoding='utf-8'))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid stage artifacts: {stage}: {exc}') from exc
