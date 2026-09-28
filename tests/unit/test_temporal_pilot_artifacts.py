import json
import pytest


def test_roundtrip_identity_and_no_overwrite(tmp_path):
    from diffusion_sources.temporal_pilot_artifacts import write_stage, read_stage
    write_stage(tmp_path, 'select', {'identity': {'config': 'a'}}, {'beta': .25})
    _, payload = read_stage(tmp_path, 'select', {'config': 'a'})
    assert payload['beta'] == .25
    with pytest.raises(ValueError, match='identity'):
        read_stage(tmp_path, 'select', {'config': 'b'})
    with pytest.raises(FileExistsError):
        write_stage(tmp_path, 'select', {'identity': {}}, {'beta': .5})


def test_tampering_and_incomplete_result(tmp_path):
    from diffusion_sources.temporal_pilot_artifacts import write_stage, read_stage
    write_stage(tmp_path, 'select', {'identity': {}}, {'beta': .25})
    (tmp_path / 'select' / 'payload.json').write_text('{"beta": 1.0}')
    with pytest.raises(ValueError, match='hash'):
        read_stage(tmp_path, 'select', {})
    (tmp_path / 'select' / 'complete').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        read_stage(tmp_path, 'select', {})


def test_interrupted_temp_not_silently_reused(tmp_path):
    from diffusion_sources.temporal_pilot_artifacts import write_stage
    (tmp_path / '.select-partial').mkdir()
    with pytest.raises(ValueError, match='partial'):
        write_stage(tmp_path, 'select', {'identity': {}}, {'beta': .25})


@pytest.mark.parametrize('stage', ['../input', '/input', ''])
def test_invalid_stage(tmp_path, stage):
    from diffusion_sources.temporal_pilot_artifacts import write_stage
    with pytest.raises(ValueError):
        write_stage(tmp_path, stage, {'identity': {}}, {})


def test_paired_csv_and_metrics_are_written_and_verified(tmp_path):
    from diffusion_sources.temporal_pilot_artifacts import write_stage, read_stage
    row = {'index': 0, 'baseline_sources': [1], 'temporal_sources': [2],
           'baseline': {'f1': 0.}, 'temporal': {'f1': 1.}}
    write_stage(tmp_path, 'validation', {'identity': {}}, {'rows': [row], 'delta_f1': 1.})
    assert (tmp_path / 'validation' / 'predictions.csv').is_file()
    assert json.loads((tmp_path / 'validation' / 'metrics.json').read_text())['delta_f1'] == 1.
    (tmp_path / 'validation' / 'predictions.csv').write_text('modified')
    with pytest.raises(ValueError, match='hash'):
        read_stage(tmp_path, 'validation', {})
