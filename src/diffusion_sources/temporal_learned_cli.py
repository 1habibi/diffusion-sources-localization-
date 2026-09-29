"""Guarded, resumable exploratory stages. No holdout paths or automatic fit."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import yaml

from .dataset import load_graph_archive
from .temporal_learned_collect import Budget, check_train_partition, collect_candidates
from .temporal_learned_data import cache_identity, guard_output, load_cache, save_cache, subset_table
from .temporal_learned_evaluation import check_saved_snapshot_f1, dev_gate, evaluate_reranker, summarize_repeats, validation_gate
from .temporal_learned_features import FEATURE_NAMES, LinearHead
from .temporal_learned_training import select_head
from .temporal_pilot_artifacts import read_stage, sha256_file, write_stage

SEEDS=(7026,7027,7028)


@dataclass(frozen=True)
class LearnedPaths:
    data_dir: Path
    runs: dict[int,Path]
    output_dir: Path
    protocol_config: Path


def _protocol(paths):
    cfg=json.loads(Path(paths.protocol_config).read_text(encoding='utf-8'))
    constants=dict(schema_version=1,exploratory=True,beta=.5,t1=1,fit_n=1620,dev_n=540,
        train_n=9990,validation_n=1998,C_grid=[.1,1.,10.],budget_seconds=1800,
        bootstrap_repetitions=2000,bootstrap_seed=9282026,minimum_delta_f1=.02,
        maximum_subgroup_decline=.02,feature_names=list(FEATURE_NAMES))
    if set(cfg)!=set(constants)|{'expected_inputs'} or any(cfg[k]!=v for k,v in constants.items()):
        raise ValueError('Protocol does not match the fixed approved pilot')
    expected=cfg['expected_inputs']
    if set(expected)!={'graph.npz','config.yaml','train.npz','validation.npz','runs'} or set(expected['runs'])!={str(s) for s in SEEDS}:
        raise ValueError('Incomplete locked input hash schema')
    hashes=[v for k,v in expected.items() if k!='runs']
    for values in expected['runs'].values():
        if set(values)!={'config.yaml','best_model.pt','metrics.json'}:
            raise ValueError('Incomplete locked run hash schema')
        hashes.extend(values.values())
    if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in hashes):
        raise ValueError('Invalid locked SHA256')
    return cfg


def check_inputs(paths,device,*,validation=False,seeds=(7026,)):
    if set(paths.runs)!=set(SEEDS):
        raise ValueError('Explicit paths for all three frozen runs are required')
    protected=[paths.data_dir,paths.protocol_config,*paths.runs.values()]
    # A rollback bundle is protected as a whole, not only its individual inputs.
    for path in (Path(paths.data_dir),*map(Path,paths.runs.values())):
        protected.extend(parent for parent in (path,*path.parents) if (parent/'backup_manifest.json').is_file())
    guard_output(paths.output_dir,protected)
    cfg=_protocol(paths)
    locked=cfg['expected_inputs']
    for name in ('graph.npz','config.yaml','train.npz')+ (('validation.npz',) if validation else ()):
        if sha256_file(Path(paths.data_dir)/name)!=locked[name]:
            raise ValueError(f'Frozen input hash mismatch: {name}')
    for seed in seeds:
        if seed not in SEEDS:
            raise ValueError('Unexpected frozen seed')
        for name in ('config.yaml','best_model.pt')+ (('metrics.json',) if validation else ()):
            if sha256_file(Path(paths.runs[seed])/name)!=locked['runs'][str(seed)][name]:
                raise ValueError(f'Frozen run hash mismatch: {seed}/{name}')
    generation=yaml.safe_load((Path(paths.data_dir)/'config.yaml').read_text(encoding='utf-8'))
    with np.load(Path(paths.data_dir)/'train.npz',allow_pickle=False) as archive:
        metadata={name:archive[name] for name in ('source_counts','probabilities','observation_fractions','simulation_seeds','observation_seeds')}
    check_train_partition(generation,metadata)
    return cfg


def _identity(paths,device):
    names=sorted(Path(__file__).parent.glob('temporal_learned_*.py'))
    sources={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in names}
    return dict(protocol_hash=sha256_file(paths.protocol_config),
        train=cache_identity(paths.data_dir,paths.runs[7026],'train',list(range(2160)),device),sources=sources)


def _id(base,stage,**dependencies):
    return dict(base=base,stage=stage,dependencies=dependencies)


def _pending(paths,stage):
    root=Path(paths.output_dir)
    if list(root.glob(f'.{stage}-*')):
        raise ValueError(f'Partial {stage}; inspect it or choose a new output root')


def _read(paths,stage,identity):
    return read_stage(paths.output_dir,stage,identity)[1]


def _write(paths,stage,identity,payload):
    guard_output(paths.output_dir,[paths.data_dir,paths.protocol_config,*paths.runs.values()])
    write_stage(paths.output_dir,stage,{'identity':identity},payload)
    return payload


def _cache(paths,device,split,seed,indices,budget,*,require_existing=False):
    name=f'cache_{split}_{seed}'
    identity=cache_identity(paths.data_dir,paths.runs[seed],split,indices,device)
    budget.check()
    if (Path(paths.output_dir)/name).exists():
        print(f'Cache hit: {name}',flush=True)
        table=load_cache(paths.output_dir,name,identity)
    else:
        if require_existing:
            raise ValueError(f'Run cache stage first: {name}')
        _pending(paths,name)
        table=collect_candidates(paths.data_dir,paths.runs[seed],split,indices,device,
                                 budget_seconds=budget.seconds-(time.monotonic()-budget.start))
        if cache_identity(paths.data_dir,paths.runs[seed],split,indices,device)!=identity:
            raise ValueError('Inputs changed before cache publication')
        budget.check()
        save_cache(paths.output_dir,name,identity,table)
    if table.indices.tolist()!=list(indices):
        raise ValueError('Cache coverage/order differs from requested cascades')
    budget.check()
    return table,sha256_file(Path(paths.output_dir)/name/'cache.npz')


def _selection(paths,device,base,budget):
    _,digest=_cache(paths,device,'train',7026,list(range(2160)),budget,require_existing=True)
    identity=_id(base,'selection',train_cache=digest)
    result=_read(paths,'selection',identity)
    LinearHead.from_dict(result['head'])
    if result['gate']!=dev_gate(result['dev_report']):
        raise ValueError('Saved dev gate disagrees with saved metrics')
    if result['fit_indices']!=list(range(1620)) or result['dev_indices']!=list(range(1620,2160)) or result['refit'] is not False:
        raise ValueError('Selection split/refit provenance mismatch')
    return result,sha256_file(Path(paths.output_dir)/'selection/payload.json')


def _blocked(stage,reason):
    print(f'STOP {stage}: {reason}',flush=True)
    return dict(status='blocked',stage=stage,reason=reason,exploratory=True)


def _validation_identity(paths,device,base,selection_hash):
    return _id(base,'validation',selection=selection_hash,
        cache=cache_identity(paths.data_dir,paths.runs[7026],'validation',list(range(1998)),device),
        saved_metrics=sha256_file(Path(paths.runs[7026])/'metrics.json'))


def _run(stage,paths,device,budget):
    cfg=check_inputs(paths,device)
    budget.check()
    base=_identity(paths,device)
    budget.check()
    if stage=='smoke':
        identity=_id(base,'smoke')
        if (Path(paths.output_dir)/'smoke').exists():
            print('Stage hit: smoke',flush=True)
            return _read(paths,'smoke',identity)
        _pending(paths,'smoke')
        table=collect_candidates(paths.data_dir,paths.runs[7026],'train',list(range(6)),device,
                                 budget_seconds=budget.seconds-(time.monotonic()-budget.start))
        if table.indices.tolist()!=list(range(6)):
            raise ValueError('Smoke requires exactly the first six train cascades')
        budget.check()
        elapsed=time.monotonic()-budget.start
        return _write(paths,'smoke',identity,dict(exploratory=True,passed=True,n=6,
            training_performed=False,elapsed_seconds=elapsed,projected_cache2160_seconds=elapsed*360,
            projection_note='Rough linear extrapolation includes fixed load costs; not a guarantee.',
            early_mask_hash=table.early_mask_hash,status='complete'))
    if stage in ('cache','select'):
        smoke=_read(paths,'smoke',_id(base,'smoke'))
        if not smoke.get('passed') or smoke.get('n')!=6 or smoke.get('training_performed') is not False:
            raise ValueError('Successful authenticated smoke required')
        table,digest=_cache(paths,device,'train',7026,list(range(2160)),budget,require_existing=stage=='select')
        if stage=='cache':
            return dict(status='complete',exploratory=True,n=2160,cache=str(Path(paths.output_dir)/'cache_train_7026'))
        identity=_id(base,'selection',train_cache=digest)
        if (Path(paths.output_dir)/'selection').exists():
            print('Stage hit: selection (no refit)',flush=True)
            return _selection(paths,device,base,budget)[0]
        _pending(paths,'selection')
        _,graph=load_graph_archive(Path(paths.data_dir)/'graph.npz')
        head,result=select_head(subset_table(table,list(range(1620))),subset_table(table,list(range(1620,2160))),graph,
                                budget_seconds=budget.seconds-(time.monotonic()-budget.start))
        if result['head']!=head.to_dict():
            raise ValueError('Selection did not preserve the fitted head')
        budget.check()
        return _write(paths,'selection',identity,result)
    selection,selection_hash=_selection(paths,device,base,budget)
    if not selection['gate']['passed']:
        return _blocked(stage,'Dev gate failed; validation/repeats remain unread')
    check_inputs(paths,device,validation=True)
    primary_id=_validation_identity(paths,device,base,selection_hash)
    head=LinearHead.from_dict(selection['head'])
    if stage=='validate':
        if (Path(paths.output_dir)/'validation').exists():
            print('Stage hit: validation',flush=True)
            _cache(paths,device,'validation',7026,list(range(1998)),budget,require_existing=True)
            return _read(paths,'validation',primary_id)
        _pending(paths,'validation')
        table,_=_cache(paths,device,'validation',7026,list(range(1998)),budget)
        _,graph=load_graph_archive(Path(paths.data_dir)/'graph.npz')
        report=evaluate_reranker(table,head,graph,include_ci=True)
        metrics=json.loads((Path(paths.runs[7026])/'metrics.json').read_text(encoding='utf-8'))
        check_saved_snapshot_f1(report,metrics)
        report.update(seed=7026,gate=validation_gate(report),status='complete',elapsed_seconds=time.monotonic()-budget.start)
        budget.check()
        return _write(paths,'validation',primary_id,report)
    _cache(paths,device,'validation',7026,list(range(1998)),budget,require_existing=True)
    primary=_read(paths,'validation',primary_id)
    if primary['gate']!=validation_gate(primary):
        raise ValueError('Saved primary gate disagrees with metrics')
    if not primary['gate']['passed']:
        return _blocked(stage,'Primary gate failed; other checkpoints remain unread')
    check_inputs(paths,device,validation=True,seeds=SEEDS)
    repeat_ids={str(seed):dict(cache=cache_identity(paths.data_dir,paths.runs[seed],'validation',list(range(1998)),device),
                             saved_metrics=sha256_file(Path(paths.runs[seed])/'metrics.json')) for seed in (7027,7028)}
    confirm_id=_id(base,'confirm',selection=selection_hash,
        primary=sha256_file(Path(paths.output_dir)/'validation/payload.json'),runs=repeat_ids)
    if stage=='confirm':
        if (Path(paths.output_dir)/'confirm').exists():
            print('Stage hit: confirm',flush=True)
            for seed in (7027,7028):
                _cache(paths,device,'validation',seed,list(range(1998)),budget,require_existing=True)
            return _read(paths,'confirm',confirm_id)
        _pending(paths,'confirm')
        _,graph=load_graph_archive(Path(paths.data_dir)/'graph.npz')
        reports=[]
        for seed in (7027,7028):
            name=f'repeat_{seed}'
            identity=_id(base,name,selection=selection_hash,run=repeat_ids[str(seed)])
            if (Path(paths.output_dir)/name).exists():
                _cache(paths,device,'validation',seed,list(range(1998)),budget,require_existing=True)
                report=_read(paths,name,identity)
            else:
                _pending(paths,name)
                table,_=_cache(paths,device,'validation',seed,list(range(1998)),budget)
                report=evaluate_reranker(table,head,graph,include_ci=True)
                metrics=json.loads((Path(paths.runs[seed])/'metrics.json').read_text(encoding='utf-8'))
                check_saved_snapshot_f1(report,metrics)
                report.update(seed=seed,status='complete')
                budget.check()
                _write(paths,name,identity,report)
            reports.append(report)
            budget.check()
        return _write(paths,'confirm',confirm_id,dict(status='complete',exploratory=True,seeds=[7027,7028],
            reports=reports,delta_f1_by_seed={str(r['seed']):r['delta_f1'] for r in reports},elapsed_seconds=time.monotonic()-budget.start))
    for seed in (7027,7028):
        _cache(paths,device,'validation',seed,list(range(1998)),budget,require_existing=True)
    confirm=_read(paths,'confirm',confirm_id)
    identity=_id(base,'summary',confirm=sha256_file(Path(paths.output_dir)/'confirm/payload.json'))
    if (Path(paths.output_dir)/'summary').exists():
        return _read(paths,'summary',identity)
    _pending(paths,'summary')
    result=summarize_repeats([primary,*confirm['reports']])
    result.update(status='complete',elapsed_seconds=time.monotonic()-budget.start)
    budget.check()
    return _write(paths,'summary',identity,result)


def run_stage(stage,paths,device):
    if stage not in ('smoke','cache','select','validate','confirm','summary'):
        raise ValueError('Unknown pilot stage')
    device=torch.device(device)
    print(f'[{stage}] Start; device={device}; data={paths.data_dir}; output={paths.output_dir}',flush=True)
    budget=Budget(1800)
    result=_run(stage,paths,device,budget)
    budget.check()
    print(f'[{stage}] End in {time.monotonic()-budget.start:.1f}s; status={result.get("status","complete")}; gate={result.get("gate")}',flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['smoke','cache','select','validate','confirm','summary'])
    for name in ('data-dir','run-7026','run-7027','run-7028','output-dir','protocol-config'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(argv)
    paths=LearnedPaths(args.data_dir,{seed:getattr(args,f'run_{seed}') for seed in SEEDS},args.output_dir,args.protocol_config)
    result=run_stage(args.stage,paths,torch.device(args.device))
    print(json.dumps({key:value for key,value in result.items() if key not in ('rows','candidates','dev_report','reports','per_seed')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
