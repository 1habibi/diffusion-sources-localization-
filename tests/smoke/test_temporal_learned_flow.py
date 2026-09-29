from tests.unit.test_temporal_learned_cli import cli_case, stage


def test_complete_stage_flow_synthetic(cli_case):
    for name in ('smoke','cache','select','validate','confirm','summary'):
        result=stage(cli_case,name)
    assert result['gate']['passed']
    assert result['n_cascades']==1998
    assert len(cli_case[1])==5  # smoke, train cache, validation for three seeds
    assert result['mean_delta_f1'] > .02
    assert result['exploratory']
    assert stage(cli_case,'summary')==result
