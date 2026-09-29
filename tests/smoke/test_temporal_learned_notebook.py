import json
from pathlib import Path

import pytest

NOTEBOOK=Path(__file__).parents[2]/'notebooks/colab_temporal_learned_reranker.ipynb'


def codes():
    notebook=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']


def test_notebook_valid_python_and_stage_order():
    cells=codes()
    assert len(cells)==8
    for i,code in enumerate(cells):
        compile(code,f'notebook-cell-{i+1}','exec')
    seen=[]
    def run(name,*args):
        seen.append(name)
        return {'passed':True,'gate':{'passed':True},'status':'complete'}
    env={'run_stage':run,'PATHS':object(),'DEVICE':'cpu','print':lambda *a,**kw:None,
         'input':lambda _: 'RUN_LEARNED_RERANKER_PILOT'}
    for code in cells[2:]:
        exec(code,env)
    assert seen==['smoke','cache','select','validate','confirm','summary']


def test_setup_kernel_import_and_no_model_run(monkeypatch):
    import diffusion_sources.temporal_learned_cli as cli
    monkeypatch.setattr(cli,'run_stage',lambda *a,**kw:pytest.fail('setup launched a model'))
    env={}
    exec(codes()[0],env)
    assert Path(cli.__file__).resolve().is_relative_to(env['PROJECT'])
    assert env['DEVICE'] in ('cpu','cuda')


def test_false_gate_prints_and_does_not_advance(capsys):
    env={'SELECTION':{'gate':{'passed':False,'reasons':['not enough gain']}},
         'run_stage':lambda *a:pytest.fail('failed dev advanced'),'PATHS':object(),'DEVICE':'cpu'}
    exec(codes()[5],env)
    assert 'STOP' in capsys.readouterr().out
    env['VALIDATION']={'status':'blocked'}
    exec(codes()[6],env)
    assert 'STOP' in capsys.readouterr().out


def test_selection_requires_explicit_run_confirmation():
    env={'input':lambda _: 'no','PATHS':object(),'DEVICE':'cpu',
         'run_stage':lambda *a:pytest.fail('fit without confirmation')}
    with pytest.raises(ValueError,match='RUN_LEARNED_RERANKER_PILOT'):
        exec(codes()[4],env)


def test_output_is_not_captured_silently(capsys):
    def visible(stage,*args):
        print('VISIBLE STAGE OUTPUT')
        return {'passed':True}
    exec(codes()[2],{'run_stage':visible,'PATHS':object(),'DEVICE':'cpu'})
    assert 'VISIBLE STAGE OUTPUT' in capsys.readouterr().out
