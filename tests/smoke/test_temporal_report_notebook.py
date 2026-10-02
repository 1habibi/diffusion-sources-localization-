"""Execute the report notebook without GPU, project imports or a Jupyter server."""
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'thesis_report/notebooks/temporal_v3_report_figures.ipynb'
SUMMARY = ROOT / 'thesis_report/data/independent_summary.json'


def execute_notebook(output_dir):
    assert NOTEBOOK.exists(), 'Report notebook has not been created'
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    namespace = {'SOURCE_PATH': SUMMARY, 'OUTPUT_DIR': output_dir}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
    return namespace


@pytest.fixture(scope='module')
def executed(tmp_path_factory):
    return execute_notebook(tmp_path_factory.mktemp('report-figures'))


def test_notebook_exports_six_figures_and_plots_saved_values(executed):
    for stem in ('01_overall_f1', '02_paired_deltas', '03_f1_by_k',
                 '04_f1_by_candidates', '05_f1_by_seed', '06_localization'):
        for extension in ('png', 'svg'):
            path = executed['OUTPUT_DIR'] / f'{stem}.{extension}'
            assert path.exists() and path.stat().st_size > 1000
    bars = executed['FIGURES']['01_overall_f1'].axes[0].patches
    assert [bar.get_height() for bar in bars] == pytest.approx(
        [0.3441830719608497, 0.43487231350205374, 0.5237404070737403])
    groups = executed['FIGURES']['03_f1_by_k'].axes[0].patches
    assert [bar.get_height() for bar in groups[-3:]] == pytest.approx(
        [0.4069903236569903, 0.545045045045045, 0.6191858525191858])
    assert executed['FIGURES']['01_overall_f1'].axes[0].get_ylim()[0] == 0


def test_checked_out_summary_preserves_approved_hash():
    assert hashlib.sha256(SUMMARY.read_bytes()).hexdigest() == (
        '980e20c405503cde35ff8f89db934989c994390fd3d203db73b15ac303328030')


def test_archived_summary_matches_drive_manifest_and_used_file(executed):
    assert executed['SUMMARY_PATH'].resolve() == SUMMARY.resolve()


def test_group_labels_have_real_line_breaks(executed):
    for name in ('03_f1_by_k', '04_f1_by_candidates', '05_f1_by_seed'):
        labels = executed['FIGURES'][name].axes[0].get_xticklabels()
        assert all('\n(n=' in label.get_text() for label in labels)


def test_notebook_finds_local_summary_from_nested_notebook_folder(tmp_path, monkeypatch):
    assert NOTEBOOK.exists(), 'Report notebook has not been moved'
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    monkeypatch.chdir(NOTEBOOK.parent)
    namespace = {'OUTPUT_DIR': tmp_path}
    first_code = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
    exec(compile(''.join(first_code['source']), str(NOTEBOOK), 'exec'), namespace)
    assert namespace['SUMMARY_PATH'].resolve() == SUMMARY.resolve()


@pytest.mark.parametrize('mutation', ['count', 'n', 'seeds', 'group_n', 'delta'])
def test_notebook_rejects_misleading_protocol_or_inconsistent_summary(executed, mutation):
    summary = copy.deepcopy(executed['summary'])
    if mutation == 'count':
        summary['count_unchanged'] = False
    elif mutation == 'n':
        summary['bootstrap_n'] = 5994
    elif mutation == 'seeds':
        summary['seeds'] = [7026]
    elif mutation == 'group_n':
        summary['by_candidates']['51+']['n_per_seed'] += 1
    else:
        summary['means']['temporal_minus_early'] = 0.5
    with pytest.raises(ValueError):
        executed['validate_summary'](summary)
