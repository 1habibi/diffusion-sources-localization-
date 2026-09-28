import hashlib
import torch
import yaml


def test_real_smoke_no_training_or_closed_data_reads(temporal_dataset, tmp_path, monkeypatch):
    from diffusion_sources.models import JointSourceCountGCN
    from diffusion_sources.temporal_pilot_cli import smoke
    data, _ = temporal_dataset
    run = tmp_path / 'run'; run.mkdir()
    cfg = {'data': {'feature_names': ['observed_infected', 'log_degree_normalized']},
           'model': {'hidden_dim': 8, 'dropout': 0.}, 'training': {'seed': 13}}
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg))
    torch.save(JointSourceCountGCN(input_dim=2, hidden_dim=8, dropout=0.).state_dict(), run / 'best_model.pt')
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [data / 'train.npz', run / 'best_model.pt']}
    import numpy as np
    original = np.load
    def guarded(path, *args, **kwargs):
        assert str(path).endswith(('graph.npz', 'train.npz'))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    result = smoke(data, run, tmp_path / 'pilot', torch.device('cpu'))
    assert result['n'] == 6 and result['beta_zero_identity']
    assert (tmp_path / 'pilot' / 'smoke' / 'manifest.json').is_file()
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes}
    assert not (data / 'test.npz').exists()
    assert smoke(data, run, tmp_path / 'pilot', torch.device('cpu')) == result
