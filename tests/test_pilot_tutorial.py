"""Scientific and artifact boundaries used by the Colab tutorial."""
from pathlib import Path
from types import SimpleNamespace
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from notebooks.pilot_support import (MODELS, choose_batch_size, summarize_scores,
                                     validate_probe_metadata, export_bundle,
                                     load_reference, calibrate_after_gate)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model', ['gemma', 'llama'])
def test_replay_recomputes_all_published_condition_and_mapping_statistics(model):
    bundle = load_reference(ROOT, model)
    actual, gaps, _ = summarize_scores(bundle['scores'])
    frozen = bundle['probe_summary']
    for row in actual.to_dict('records'):
        expected = frozen['changes'][row['comparison']][row['group']][row['probe']]
        assert row['mean_absolute_change'] == pytest.approx(expected['mean_absolute_z_change'], abs=1e-12)
        assert row['reference_signed_change'] == pytest.approx(expected['mean_reference_signed_z_change'], abs=1e-12)
    for row in gaps.to_dict('records'):
        assert row['mean_absolute_mapping_gap'] == pytest.approx(
            frozen['mapping_sensitivity'][row['probe']]['mean_absolute_z_gap'], abs=1e-12)


def test_missing_mapping_is_rejected_instead_of_silently_averaged():
    frame = load_reference(ROOT, 'gemma')['scores']
    with pytest.raises(ValueError, match='mapping'):
        summarize_scores(frame.iloc[1:])


@pytest.mark.parametrize('model,passed,expected', [
    ('gemma', True, 2), ('gemma', False, 1), ('llama', True, 1), ('llama', False, 1)])
def test_gate_never_uses_failed_batch_or_upgrades_llama(model, passed, expected):
    assert choose_batch_size(model, passed) == expected


def test_invalid_probe_revision_and_layer_are_rejected():
    spec = MODELS['gemma']
    direction = np.zeros(spec['width'], dtype=np.float32); direction[0] = 1
    weights = {'model_revision': spec['revision'], 'tokenizer_revision': spec['revision'],
               'selected_layer': spec['hf_layer']-1, 'difference_in_means_direction': direction,
               'difference_in_means_midpoint': 0., 'logistic_coef': direction[None, :],
               'logistic_intercept': np.array([0.]), 'logistic_c': 1.,
               'dim_scaler_mu': 0., 'dim_scaler_sigma': 1.,
               'logistic_scaler_mu': 0., 'logistic_scaler_sigma': 1.}
    validate_probe_metadata(weights, spec)
    with pytest.raises(ValueError, match='revision'):
        validate_probe_metadata(dict(weights, tokenizer_revision='wrong'), spec)
    with pytest.raises(ValueError, match='layer'):
        validate_probe_metadata(dict(weights, selected_layer=spec['hf_layer']), spec)
    with pytest.raises(ValueError, match='scale'):
        validate_probe_metadata(dict(weights, dim_scaler_sigma=0.), spec)


def test_failed_numerical_gate_runs_a_fresh_serial_calibration(tmp_path):
    calls = []
    def verify(backend, battery, path):
        path.mkdir()
        (path/'comparison.json').write_text(json.dumps({'passed': False}))
        return False
    def calibrate(model, battery, path, **kw):
        calls.append(kw)
        path.mkdir()
        (path/'summary.json').write_text(json.dumps({'requested_records':288,'valid_records':288}))
        return path
    cfg=SimpleNamespace(agent_model=object(), capture=object(), seed=0)
    result=calibrate_after_gate('gemma', object(), cfg, object(), tmp_path,
                                verify=verify, calibrate=calibrate,
                                coverage=lambda *_: {'calls':576,'observed_batch_sizes':[1]})
    assert result['batch_check_passed'] is False
    assert result['effective_batch_size'] == 1
    assert calls[0]['batch_size'] == 1
    assert calls[0]['n_personas'] == calls[0]['repeats'] == 3
    assert (tmp_path/'numerical-check/comparison.json').exists()


def test_export_includes_results_but_not_credentials_cache_or_symlinks(tmp_path):
    output=tmp_path/'run'; (output/'analysis').mkdir(parents=True)
    (output/'analysis/scores.csv').write_text('score\n1\n')
    (output/'.env').write_text('HF_TOKEN=do-not-export')
    (output/'token').write_text('do-not-export')
    (output/'hf-cache').mkdir(); (output/'hf-cache/weights.bin').write_bytes(b'weights')
    archive=export_bundle(output,tmp_path/'result.zip')
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'analysis/scores.csv','MANIFEST.sha256.json'}
    (output/'analysis/leak.txt').symlink_to(output/'.env')
    with pytest.raises(ValueError, match='symlink'):
        export_bundle(output,tmp_path/'bad.zip')


@pytest.mark.parametrize('remove', ['whole_trial','persona','repeat'])
def test_incomplete_full_design_is_rejected(remove):
    frame=load_reference(ROOT,'gemma')['scores']
    if remove=='whole_trial':frame=frame[frame.trial_id != frame.trial_id.iloc[0]]
    if remove=='persona':frame=frame[frame.agent != frame.agent.iloc[0]]
    if remove=='repeat':frame=frame[frame.repeat != 0]
    with pytest.raises(ValueError,match='design'):
        summarize_scores(frame)


def test_export_excludes_nested_credentials_and_model_cache(tmp_path):
    output=tmp_path/'run';(output/'analysis').mkdir(parents=True)
    (output/'analysis/.env').write_text('HF_TOKEN=secret')
    (output/'analysis/keep.csv').write_text('score\n1\n')
    (output/'live/hf-cache').mkdir(parents=True)
    (output/'live/hf-cache/token').write_text('secret')
    with zipfile.ZipFile(export_bundle(output,tmp_path/'results.zip')) as z:
        assert set(z.namelist())=={'analysis/keep.csv','MANIFEST.sha256.json'}


def test_batch_exception_cannot_be_reported_as_successful_batched_measurement(tmp_path):
    from mindvirus.calibrate import run_calibration
    from mindvirus.config import load_config
    from mindvirus.probes import Battery
    class BrokenBatch:
        name='fake';model='test';last_activation_path='synthetic.pt'
        def choice_logprobs(self, req, choices):return {'A':.6,'B':.4}
        def choice_logprobs_batch(self, reqs, choices):raise TypeError('batch failed')
    def verify(backend,battery,path):
        path.mkdir();(path/'comparison.json').write_text('{"passed":true}');return True
    cfg=load_config(ROOT/'config/runpod-gemma.yaml')
    battery=Battery.load(ROOT/'config/hand-relation-battery.json')
    with pytest.raises(RuntimeError,match='batch'):
        calibrate_after_gate('gemma',BrokenBatch(),cfg,battery,tmp_path,
                             verify=verify,calibrate=run_calibration)


@pytest.mark.parametrize('column',['call_id','trial_id','label','on_target'])
def test_null_metadata_cannot_silently_remove_measurements(column):
    frame=load_reference(ROOT,'gemma')['scores']
    frame[column]=frame[column].astype(object)
    frame.loc[0,column]=None
    with pytest.raises(ValueError,match='metadata'):
        summarize_scores(frame)


def test_dirty_source_is_rejected(tmp_path):
    import subprocess
    from notebooks.pilot_support import require_clean_repo
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'scorer.py').write_text('score = 1\n')
    subprocess.run(['git','add','scorer.py'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qm','Source'],cwd=tmp_path,check=True)
    require_clean_repo(tmp_path)
    (tmp_path/'scorer.py').write_text('score = -1\n')
    with pytest.raises(ValueError,match='modified'):
        require_clean_repo(tmp_path)
