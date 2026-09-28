"""Generate once and seal without reading target values or model quality."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from .generation import generate_dataset
from .temporal_replay import FIELDS
from .temporal_pilot_artifacts import write_stage
from .temporal_independent_artifacts import (IndependentPaths, load_yaml, topology,
    metadata_seed_union, reference_archives, verify_freeze, verify_seal, stage_identity)

SIMULATION = {'source_counts': [1, 2, 3], 'probabilities': [.01, .02, .03],
              'max_steps': 3, 'distance_ranges': [{'min': 1, 'max': 2}, {'min': 3, 'max': 5}]}
OBSERVATION = {'fractions': [1., .75, .5], 'false_positive_count': 0, 'hide_source_count': 0}
DATASET = {'seed': 4007026, 'splits': {'independent_holdout': 1998},
           'min_candidates': 5, 'max_infected_fraction': .5, 'max_attempt_factor': 100,
           'distance_cache_size': 512, 'show_progress': True}


def validate_generation_config(config: dict, repo: Path) -> dict:
    result = copy.deepcopy(config)
    if set(result) != {'graph', 'simulation', 'observation', 'dataset'}:
        raise ValueError('Unsupported generation protocol sections.')
    result['observation'].setdefault('hide_source_count', 0)
    graph = result['graph']
    if (set(graph) != {'id', 'kind', 'path'} or graph['id'] != 'ego_facebook'
            or graph['kind'] != 'edge_list' or result['simulation'] != SIMULATION
            or result['observation'] != OBSERVATION or result['dataset'] != DATASET):
        raise ValueError('Independent generation protocol mismatch.')
    raw = Path(graph['path'])
    graph['path'] = str((raw if raw.is_absolute() else Path(repo) / raw).resolve())
    return result


def _schema(path: Path, nodes: int) -> None:
    """Read NPY headers only, not source/count/infected values."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate NPZ members.')
        for field in FIELDS:
            with archive.open(field + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f'Unsupported NPY header version: {version}')
            expected = ((1998, nodes, 2) if field == 'features'
                        else (1998, nodes) if field in ('source_labels', 'infected_masks', 'candidate_masks')
                        else (1998,))
            if shape != expected or dtype.hasobject:
                raise ValueError(f'Independent archive schema/size mismatch: {field}')


def generate_and_seal(paths: IndependentPaths) -> dict:
    frozen = verify_freeze(paths)
    if (paths.reports / 'seal').exists():
        return verify_seal(paths)
    config = validate_generation_config(load_yaml(paths.generation_config), paths.repo)
    reference_config = load_yaml(paths.reference / 'config.yaml')
    observation = {**reference_config['observation']}
    observation.setdefault('hide_source_count', 0)
    if reference_config['simulation'] != SIMULATION or observation != OBSERVATION:
        raise ValueError('Reference generation protocol differs.')
    for key in ('min_candidates', 'max_infected_fraction', 'max_attempt_factor', 'distance_cache_size'):
        if reference_config['dataset'][key] != DATASET[key]:
            raise ValueError(f'Reference acceptance protocol differs: {key}')
    references = metadata_seed_union(reference_archives(paths))
    if any(4007026 <= seed <= 4406625 for seed in references):
        raise ValueError('Reference seed overlap with generation window.')
    if paths.data.exists() or list(paths.data.parent.glob(f'.{paths.data.name}-*')):
        raise ValueError('Existing unsealed/partial generation; no overwrite or regeneration.')
    paths.data.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{paths.data.name}-', dir=paths.data.parent))
    print('Generating fixed independent dataset once; target metrics remain unopened.', flush=True)
    generate_dataset(config, temporary)
    if topology(temporary / 'graph.npz') != frozen['identity']['topology']:
        raise ValueError('Generated graph topology/node mapping mismatch.')
    if load_yaml(temporary / 'config.yaml') != config:
        raise ValueError('Generated config mismatch.')
    archive_path = temporary / 'independent_holdout.npz'
    _schema(archive_path, frozen['identity']['topology']['nodes'])
    seeds = metadata_seed_union([archive_path])
    if len(seeds) != 3996 or seeds & references:
        raise ValueError('Independent seeds duplicate or overlap references.')
    with np.load(archive_path, allow_pickle=False) as archive:
        simulation, observation = archive['simulation_seeds'], archive['observation_seeds']
    if (np.any(simulation < 4007026) or np.any(observation > 4406625)
            or np.any((simulation - 4007026) % 2) or not np.array_equal(observation, simulation + 1)):
        raise ValueError('Independent seeds do not follow frozen generator protocol.')
    verify_freeze(paths)  # Detect changed inputs during generation before publication.
    temporary.rename(paths.data)
    payload = {'evaluation_status': 'sealed_unopened', 'n': 1998,
               'dataset_seed': 4007026, 'target_metrics_computed': False,
               'seed_isolation': True,
               'seed_hashes': {name: hashlib.sha256(values.tobytes()).hexdigest()
                               for name, values in [('simulation', simulation), ('observation', observation)]}}
    write_stage(paths.reports, 'seal', {'identity': stage_identity(paths, 'seal')}, payload)
    print(f'Sealed: {paths.reports / "seal/manifest.json"}. STOP before opening.', flush=True)
    return payload
