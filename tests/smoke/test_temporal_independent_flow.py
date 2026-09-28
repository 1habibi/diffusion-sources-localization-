import pytest
import torch
from dataclasses import replace
from diffusion_sources.temporal_pilot_artifacts import sha256_file
from tests.unit.test_temporal_independent_inference import sealed_case, controlled_records


@pytest.mark.parametrize('positive', [True, False])
def test_complete_independent_flow_and_disk_summary(tmp_path, monkeypatch, positive):
    from diffusion_sources import temporal_independent_inference as inference
    from diffusion_sources.temporal_independent_summary import save_summary
    paths = sealed_case(tmp_path, monkeypatch)
    protected = [p for root in (paths.reference, paths.snapshot_holdout, paths.pilot, paths.baseline, *paths.runs.values())
                 for p in root.rglob('*') if p.is_file()]
    hashes = {p: sha256_file(p) for p in protected}
    inference.open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT')
    def collect(*args):
        records = controlled_records()
        if not positive:
            records = [replace(r, candidates=replace(r.candidates, early_observed=(False,)*10), early_empty=True)
                       for r in records]
        return records
    monkeypatch.setattr(inference, 'collect_independent_records', collect)
    for seed in (7026, 7027, 7028):
        inference.evaluate_seed(paths, seed, torch.device('cpu'))
    result = save_summary(paths)
    assert result['primary']['passed'] == positive
    assert result['secondary']['passed'] == positive
    assert result['bootstrap_n'] == 1998
    assert result['evaluation_role'] == 'independent_confirmation'
    monkeypatch.setattr(inference, 'collect_independent_records', lambda *a: pytest.fail('No inference on resume'))
    assert save_summary(paths) == result
    assert hashes == {p: sha256_file(p) for p in protected}
    with (paths.reports / 'seed_7027/predictions.csv').open('a') as f:
        f.write('corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        save_summary(paths)


def test_missing_reports_never_create_summary(tmp_path, monkeypatch):
    from diffusion_sources.temporal_independent_summary import save_summary
    from diffusion_sources.temporal_independent_inference import open_evaluation
    paths = sealed_case(tmp_path, monkeypatch)
    open_evaluation(paths, 'OPEN_INDEPENDENT_HOLDOUT')
    with pytest.raises(ValueError, match='stage'):
        save_summary(paths)
    assert not (paths.reports / 'summary').exists()
