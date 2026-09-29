import json

import numpy as np
import pytest
import torch
import yaml

from diffusion_sources.temporal_learned_cli import LearnedPaths, run_stage
from diffusion_sources.temporal_pilot_artifacts import sha256_file
from tests.fixtures.temporal_learned import repeated_table
from tests.unit.test_temporal_learned_collect import partition_case


@pytest.fixture
def cli_case(tmp_path,monkeypatch):
    from pathlib import Path
    data=tmp_path/'data'; data.mkdir()
    generation,arrays=partition_case()
    (data/'config.yaml').write_text(yaml.safe_dump(generation),encoding='utf-8')
    np.savez(data/'graph.npz',graph_id=np.array('tiny'),node_count=np.array(6),edges=np.array([[i,i+1] for i in range(5)]))
    np.savez(data/'train.npz',**arrays)
    np.savez(data/'validation.npz',source_counts=np.tile([1,2,3],666))
    protocol=json.loads((Path(__file__).parents[2]/'configs/temporal_learned_reranker_v1.json').read_text())
    runs={}
    for seed in (7026,7027,7028):
        run=tmp_path/f'run_{seed}'; run.mkdir(); runs[seed]=run
        (run/'config.yaml').write_text('model: {}\ndata: {}\n',encoding='utf-8')
        (run/'best_model.pt').write_bytes(f'synthetic-{seed}'.encode())
        (run/'metrics.json').write_text(json.dumps({'validation_prediction_metrics':{'joint_estimated_k':{'all':{'f1':1.}}}}))
    protocol['expected_inputs']={name:sha256_file(data/name) for name in ('graph.npz','config.yaml','train.npz','validation.npz')}
    protocol['expected_inputs']['runs']={str(seed):{name:sha256_file(run/name) for name in ('config.yaml','best_model.pt','metrics.json')} for seed,run in runs.items()}
    cfg=tmp_path/'protocol.json'; cfg.write_text(json.dumps(protocol))
    paths=LearnedPaths(data,runs,tmp_path/'output',cfg)
    calls=[]
    options={'train_perfect':False,'validation_perfect':False}
    def collector(data_dir,run_dir,split,indices,device,**kwargs):
        calls.append((split,tuple(indices),run_dir))
        return repeated_table(indices,perfect_v3=options[f'{split}_perfect'])
    monkeypatch.setattr('diffusion_sources.temporal_learned_cli.collect_candidates',collector)
    return paths,calls,options


def stage(cli_case,name):
    return run_stage(name,cli_case[0],torch.device('cpu'))


def prepare_selection(cli_case):
    for name in ('smoke','cache','select'):
        result=stage(cli_case,name)
    return result


def test_smoke_has_no_fit_and_prints(cli_case,monkeypatch,capsys):
    monkeypatch.setattr('diffusion_sources.temporal_learned_cli.select_head',lambda *a,**kw:pytest.fail('smoke trained'))
    result=stage(cli_case,'smoke')
    assert result['n']==6 and result['passed']
    assert len(cli_case[1])==1 and len(cli_case[1][0][1])==6
    assert 'smoke' in capsys.readouterr().out


def test_train_cache_only_requested_once(cli_case):
    stage(cli_case,'smoke'); stage(cli_case,'cache'); stage(cli_case,'cache')
    assert [len(c[1]) for c in cli_case[1]] == [6,2160]


def test_failed_dev_never_reads_validation(cli_case):
    cli_case[2]['train_perfect']=True
    result=prepare_selection(cli_case)
    assert not result['gate']['passed']
    # Deleting a forbidden-before-gate input proves it was not even hashed.
    (cli_case[0].data_dir/'validation.npz').unlink()
    assert stage(cli_case,'validate')['status']=='blocked'
    assert all(c[0]=='train' for c in cli_case[1])


def test_failed_primary_never_collects_repeats(cli_case):
    prepare_selection(cli_case)
    cli_case[2]['validation_perfect']=True
    result=stage(cli_case,'validate')
    assert not result['gate']['passed']
    (cli_case[0].runs[7027]/'best_model.pt').unlink()
    assert stage(cli_case,'confirm')['status']=='blocked'
    assert len(cli_case[1])==3


def test_direct_confirm_cannot_bypass_gate(cli_case):
    with pytest.raises(ValueError):
        stage(cli_case,'confirm')
    assert cli_case[1]==[]


def test_restart_does_not_refit(cli_case,monkeypatch):
    selected=prepare_selection(cli_case)
    monkeypatch.setattr('diffusion_sources.temporal_learned_cli.select_head',lambda *a,**kw:pytest.fail('hidden refit'))
    assert stage(cli_case,'select')['head']==selected['head']
    assert len(cli_case[1])==2


def test_tampered_selection_and_partial_stop(cli_case):
    prepare_selection(cli_case)
    (cli_case[0].output_dir/'selection'/'payload.json').write_text('{}')
    with pytest.raises(ValueError):
        stage(cli_case,'validate')
    assert all(c[0]=='train' for c in cli_case[1])


def test_frozen_input_hashes_and_bad_paths(cli_case):
    p=cli_case[0]
    (p.runs[7026]/'best_model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        stage(cli_case,'smoke')
    bad=LearnedPaths(p.data_dir,p.runs,p.data_dir/'output',p.protocol_config)
    with pytest.raises(ValueError,match='overlap'):
        run_stage('smoke',bad,torch.device('cpu'))


def test_partial_stage_stops_before_collection(cli_case):
    root=cli_case[0].output_dir
    root.mkdir(); (root/'.smoke-partial').mkdir()
    with pytest.raises(ValueError,match='Partial'):
        stage(cli_case,'smoke')
    assert cli_case[1]==[]


def test_tampered_validation_cache_rejected_on_resume(cli_case):
    prepare_selection(cli_case)
    stage(cli_case,'validate')
    (cli_case[0].output_dir/'cache_validation_7026/cache.npz').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='hash'):
        stage(cli_case,'validate')
