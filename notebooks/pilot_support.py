"""Auditable support code for the moral-relation Colab tutorial.

CPU replay reads frozen scores; only run_live_model loads a language model.
The original calibration, numerical comparison, and Jeff scorer are reused.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import logging
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

JEFF_REVISION = 'ff3588c1d15ac89c6a3edd5da41c6fa121ad4157'
PAPER = Path('docs/papers/moral-relation-pilot/draft-v1')
MODELS = {
    'gemma': {'name': 'Gemma 2 9B IT', 'model': 'google/gemma-2-9b-it',
              'revision': '11c9b309abf73637e4b6f9a3fa1e92e615547819',
              'hf_layer': 28, 'width': 3584, 'batch_size': 2,
              'probe_file': 'gemma_layer27.npz',
              'probe_sha256': 'd3265c5dd23c20203c71ed79c8d84a789d36c1dafcca23f5f7b0b9409605a028'},
    'llama': {'name': 'Llama 3.1 8B Instruct', 'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
              'revision': '0e9e39f249a16976918f6564b8830bc894c89659',
              'hf_layer': 20, 'width': 4096, 'batch_size': 1,
              'probe_file': 'llama_layer19.npz',
              'probe_sha256': 'f18d891f3ef8b5bdd73ac10470329235e9c1844504e65df6b59433dc53910cf6'},
}
COMPARISONS = [('repeat', 'neutral'), ('seeded', 'neutral'),
               ('mentioned', 'neutral'), ('seeded', 'mentioned')]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_reference(repo, model):
    """Load the signed, frozen per-call data; verify every consumed evidence file."""
    base = Path(repo) / PAPER
    manifest = json.loads((base / 'checksums.json').read_text())['files']
    files = {'scores': f'{model}-per_call_scores.csv',
             'probe_summary': f'{model}-frozen-probe-context-2026-09-06.json',
             'native_report': f'{model}-calibration-2026-09-06.json',
             'design': f'{model}-design.json', 'battery': 'hand-relation-battery.json'}
    result = {'origin': 'saved original pilot; no new inference'}
    for key, name in files.items():
        path = base / 'evidence' / name
        if sha256(path) != manifest[f'evidence/{name}']['sha256']:
            raise ValueError(f'Frozen evidence checksum mismatch: {name}')
        result[key] = pd.read_csv(path) if name.endswith('.csv') else json.loads(path.read_text())
    return result


def summarize_scores(frame):
    """Recompute descriptive paired contrasts; keep the frozen semantic direction."""
    required = ['call_id','trial_id','agent','repeat','probe_id','condition','mapping',
                'on_target','label','dim_z','logistic_z']
    if not set(required) <= set(frame) or frame[required].isna().any().any():
        raise ValueError('Missing score or grouping metadata')
    if not frame.on_target.isin([True,False]).all() or not frame.label.isin([0,1]).all():
        raise ValueError('Invalid target/reference metadata')
    if frame.call_id.duplicated().any():
        raise ValueError('Duplicate capture call_id')
    if frame.duplicated(['trial_id', 'mapping']).any():
        raise ValueError('Duplicate answer mapping within a trial')
    mappings = frame.groupby('trial_id').mapping.agg(set)
    if not mappings.apply(lambda x: x == {'standard', 'reversed'}).all():
        raise ValueError('Both answer mappings are required for every trial')
    validate_full_design(frame)
    score_columns = ['dim_z', 'logistic_z']
    if not np.isfinite(frame[score_columns].to_numpy()).all():
        raise ValueError('Missing or non-finite probe scores')
    keys = ['agent', 'probe_id', 'condition', 'on_target', 'label']
    # Mapping reversal changes response symbols, not the semantic frozen probe.
    averaged = frame.groupby(keys, as_index=False)[score_columns].mean()
    conditions = averaged.groupby(['agent', 'probe_id']).condition.agg(set)
    if not conditions.apply(lambda x: x == {'neutral','repeat','seeded','mentioned'}).all():
        raise ValueError('Four journal conditions are required for each persona/item')
    rows = []
    for current, reference in COMPARISONS:
        paired = averaged[averaged.condition == current].merge(
            averaged[averaged.condition == reference],
            on=['agent','probe_id','on_target','label'],
            suffixes=('_current','_reference'), validate='one_to_one')
        for group, target in [('target', True), ('control', False)]:
            part = paired[paired.on_target == target]
            for probe in ('dim','logistic'):
                delta = part[f'{probe}_z_current'] - part[f'{probe}_z_reference']
                rows.append({'comparison': f'{current}_minus_{reference}', 'group': group,
                             'probe': probe, 'persona_item_pairs': len(part),
                             'mean_absolute_change': float(delta.abs().mean()),
                             'mean_signed_change': float(delta.mean()),
                             'reference_signed_change': float((delta*(2*part.label-1)).mean())})
    gaps = []
    for probe in ('dim','logistic'):
        mapped = frame.groupby(keys+['mapping'])[f'{probe}_z'].mean().unstack('mapping')
        gaps.append({'probe': probe, 'mean_absolute_mapping_gap':
                     float((mapped.standard-mapped.reversed).abs().mean())})
    return pd.DataFrame(rows), pd.DataFrame(gaps), averaged


def choose_batch_size(model, passed):
    """Preserve original per-model policy; any failed completed gate selects serial."""
    if not isinstance(passed, (bool, np.bool_)):
        raise ValueError('The numerical gate must return an explicit boolean')
    return MODELS[model]['batch_size'] if passed else 1


def validate_probe_metadata(weights, spec):
    for key in ('model_revision', 'tokenizer_revision'):
        if str(weights[key]) != spec['revision']:
            raise ValueError(f'Probe {key} differs from the requested checkpoint revision')
    if int(weights['selected_layer']) + 1 != spec['hf_layer']:
        raise ValueError('Probe layer does not match the hidden-state capture index')
    direction = np.asarray(weights['difference_in_means_direction'])
    if direction.shape != (spec['width'],) or not np.isfinite(direction).all():
        raise ValueError('Probe direction has incompatible width or non-finite values')
    if not np.isclose(np.linalg.norm(direction), 1, atol=1e-5):
        raise ValueError('DIM direction must have unit norm')
    if np.asarray(weights['logistic_coef']).shape != (1, spec['width']):
        raise ValueError('Logistic coefficients have incompatible width')
    for key in ('difference_in_means_midpoint','logistic_coef','logistic_intercept',
                'dim_scaler_mu','logistic_scaler_mu','logistic_c'):
        if not np.isfinite(np.asarray(weights[key])).all():
            raise ValueError(f'Non-finite frozen parameter: {key}')
    for key in ('dim_scaler_sigma','logistic_scaler_sigma'):
        if not np.isfinite(float(weights[key])) or float(weights[key]) <= 0:
            raise ValueError(f'Frozen scale must be finite and positive: {key}')


def check_probe(jeff_repo, model):
    spec = MODELS[model]
    path = Path(jeff_repo) / 'artifacts/probe_weights/m1_relation' / spec['probe_file']
    if sha256(path) != spec['probe_sha256']:
        raise ValueError(f'Frozen {model} probe checksum mismatch')
    with np.load(path, allow_pickle=False) as weights:
        validate_probe_metadata(weights, spec)
    return path


def validate_full_design(frame):
    dimensions = ['agent','repeat','probe_id','condition','mapping']
    if not set(dimensions+['trial_id']) <= set(frame) or frame[dimensions+['trial_id']].isna().any().any():
        raise ValueError('Missing experimental design metadata')
    expected = pd.MultiIndex.from_product([
        ['Maria','Frank','Ruth'], range(3),
        [f'rel-on-{i}' for i in range(4)]+[f'rel-ctrl-{i}' for i in range(4)],
        ['neutral','repeat','seeded','mentioned'], ['standard','reversed']], names=dimensions)
    actual = pd.MultiIndex.from_frame(frame[dimensions])
    if len(frame) != 576 or actual.has_duplicates or set(actual) != set(expected):
        raise ValueError('Incomplete or duplicate 3 × 3 × 8 × 4 × 2 experimental design')
    if not (frame.groupby('trial_id')[dimensions[:-1]].nunique() == 1).all().all():
        raise ValueError('Inconsistent trial metadata within the experimental design')


def validate_execution(calibration, batch_size):
    calls = pd.read_json(Path(calibration)/'calls.jsonl', lines=True)
    observations = pd.read_json(Path(calibration)/'observations.jsonl', lines=True)
    call_fields = ['call_id','round','agent','probe_id','answer_mapping','activation_path','logprobs']
    obs_fields = ['trial_id','agent','probe_id','repeat','condition','score_kind']
    if (not set(call_fields) <= set(calls) or not set(obs_fields) <= set(observations)
            or calls[call_fields].isna().any().any() or observations[obs_fields].isna().any().any()
            or len(observations) != 288 or observations.trial_id.duplicated().any()):
        raise RuntimeError('Incomplete execution metadata or observation coverage')
    observed_batches = calls.get('batch_size', pd.Series(1, index=calls.index)).fillna(1)
    if not (observed_batches == batch_size).all():
        raise RuntimeError('Actual batch execution differed from the requested batch size; inspect execution.log and retain this failed run')
    if (len(calls) != 576 or calls.call_id.duplicated().any()
            or calls.activation_path.isna().any() or calls.logprobs.isna().any()
            or (observations.score_kind != 'conditional_probability').any()):
        raise RuntimeError('Expected 576 captured probability calls and 288 native observations')
    joined = calls.merge(observations[['trial_id','agent','probe_id','repeat','condition']],
                         left_on=['round','agent','probe_id'],
                         right_on=['trial_id','agent','probe_id'],validate='many_to_one')
    validate_full_design(joined.rename(columns={'answer_mapping':'mapping'}))
    return {'calls':len(calls),'observed_batch_sizes':sorted(set(int(v) for v in observed_batches))}


def require_clean_repo(repo, expected_revision=None):
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if expected_revision is not None and head != expected_revision:
        raise ValueError('Repository revision differs from the pinned source')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo):
        raise ValueError('Tracked repository source is modified; use a clean pinned checkout')
    return head


def calibrate_after_gate(model, backend, cfg, battery, output, *, verify, calibrate,
                         coverage=validate_execution):
    """Retain failed batch evidence, then run the same full design serially."""
    output = Path(output)
    passed = verify(backend, battery, output / 'numerical-check')
    batch_size = choose_batch_size(model, passed)
    import torch
    torch.manual_seed(cfg.seed)
    calibration = calibrate(cfg.agent_model, battery, output / 'calibration',
                            n_personas=3, repeats=3, seed=cfg.seed, backend=backend,
                            capture=cfg.capture, batch_size=batch_size)
    summary = json.loads((calibration / 'summary.json').read_text())
    if summary['requested_records'] != 288 or summary['valid_records'] != 288:
        raise RuntimeError('Incomplete calibration: preserve outputs and inspect failures')
    execution = coverage(calibration, batch_size)
    return {'batch_check_passed': bool(passed), 'effective_batch_size': batch_size,
            'requested_observations': 288, 'native_summary': summary, **execution}


def gpu_preflight():
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Live mode requires a CUDA GPU with native BF16 support; select CPU replay on other runtimes.')
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory < 35:
        raise RuntimeError(f'This tutorial requires a 40 GB-class GPU (found {memory:.1f} GiB). Use an A100 or CPU replay.')
    return {'gpu': torch.cuda.get_device_name(0), 'gpu_memory_gib': memory,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'original_runtime': 'PyTorch 2.8.0+cu128 on RTX A6000 48 GB; new hardware may differ'}


def run_live_model(repo, jeff_repo, model, output):
    """One complete fresh-model measurement, numerical gate, frozen scoring, cleanup."""
    import torch
    from mindvirus.backends import build_backend
    from mindvirus.calibrate import run_calibration, runtime_metadata
    from mindvirus.config import load_config
    from mindvirus.probes import Battery
    from mindvirus.sweep import _validate_pilot_config

    gpu_preflight()
    repo, jeff_repo, output = Path(repo).resolve(), Path(jeff_repo).resolve(), Path(output).resolve()
    require_clean_repo(repo)
    require_clean_repo(jeff_repo, JEFF_REVISION)
    probe_path = check_probe(jeff_repo, model)
    cfg = load_config(repo / f'config/runpod-{model}.yaml')
    battery = Battery.load(repo / 'config/hand-relation-battery.json')
    _validate_pilot_config(cfg, battery)
    spec = MODELS[model]
    if (cfg.agent_model.model != spec['model'] or cfg.agent_model.revision != spec['revision']
            or cfg.capture.layers != [spec['hf_layer']] or cfg.agent_model.quantize_4bit
            or cfg.agent_model.dtype != 'bfloat16'):
        raise ValueError('Configuration differs from the frozen experiment contract')
    output.mkdir(parents=True, exist_ok=False)
    driver_path = repo / PAPER / 'evidence' / f'{model}-gpu-pilot.py'
    loader = importlib.util.spec_from_file_location('original_numerical_check', driver_path)
    driver = importlib.util.module_from_spec(loader); loader.loader.exec_module(driver)
    status = {'state':'loading', 'mode':'live', 'model_key':model,
              'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
              'jeff_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=jeff_repo,text=True).strip(),
              'probe_sha256':sha256(probe_path), 'numerical_driver_sha256':sha256(driver_path),
              'contagion_established':False}
    if status['jeff_commit'] != JEFF_REVISION:
        raise ValueError('Jeff checkout must be pinned to the recorded revision')
    write_json(output/'run-manifest.json', status)
    backend = None
    execution_logger = logging.getLogger('mindvirus.backends')
    execution_handler = logging.FileHandler(output/'execution.log')
    execution_logger.addHandler(execution_handler)
    try:
        backend = build_backend(cfg.agent_model, output, cfg.capture, seed=cfg.seed)
        if backend._model.dtype != torch.bfloat16:
            raise RuntimeError('Live inference must remain BF16')
        if not all(p.device.type == 'cuda' for p in backend._model.parameters()):
            raise RuntimeError('Model was offloaded; this run requires all parameters on GPU')
        if (backend._model.config._commit_hash != spec['revision'] or
                backend._model.config.hidden_size != spec['width']):
            raise RuntimeError('Loaded model revision/dimension differs from probe contract')
        backend._model.eval()
        status.update(state='measuring', runtime=runtime_metadata(backend))
        write_json(output/'run-manifest.json', status)
        torch.manual_seed(cfg.seed)
        measurement = calibrate_after_gate(model, backend, cfg, battery, output,
                                            verify=driver.verify, calibrate=run_calibration)
        status.update({k:v for k,v in measurement.items() if k!='native_summary'})
        status['state']='scoring'
        write_json(output/'run-manifest.json', status)
        # Same original scorer/scaler and artifact joins used in the published pilot.
        command = [sys.executable, str(repo/'audit/score_calibration_probes.py'),
                   '--calibration', str(output/'calibration'),
                   '--comparison', str(output/'numerical-check/comparison.json'),
                   '--probe', str(probe_path), '--jeff-repo', str(jeff_repo),
                   '--output', str(output/'frozen-scores')]
        with (output/'scoring.log').open('w') as stream:
            subprocess.run(command,cwd=repo,check=True,stdout=stream,
                           env=dict(os.environ,PYTHONPATH=str(repo),CUDA_VISIBLE_DEVICES=''))
        status['state']='completed'
        write_json(output/'run-manifest.json', status)
        return output
    except BaseException as exc:
        status.update(state='failed',error_type=type(exc).__name__)
        write_json(output/'run-manifest.json', status)
        raise
    finally:
        execution_logger.removeHandler(execution_handler)
        execution_handler.close()
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def load_live(output):
    output = Path(output)
    observations = pd.read_json(output/'calibration/observations.jsonl', lines=True)
    return {'origin':'new live checkpoint run',
            'scores':pd.read_csv(output/'frozen-scores/per_call_scores.csv'),
            'probe_summary':json.loads((output/'frozen-scores/summary.json').read_text()),
            'native_report':{'calibration':json.loads((output/'calibration/summary.json').read_text()),
                             'coverage':{'mapping_consistent':int(observations.mapping_consistent.sum())}},
            'design':json.loads((output/'calibration/design.json').read_text()),
            'battery':json.loads((output/'calibration/battery.json').read_text())}


def export_bundle(output, archive):
    """Export only the dedicated result areas, with checksums and no symlink traversal."""
    output, archive = Path(output).resolve(), Path(archive).resolve()
    allowed = {'analysis', 'live', 'replay', 'session.json', 'README.txt'}
    excluded = {'.env','token','tokens','hf_token','hf-cache','hf_cache','cache',
                '.cache','.git','.venv','venv','checkpoints','credentials'}
    paths = []
    for path in sorted(output.rglob('*')):
        relative = path.relative_to(output)
        if relative.parts[0] not in allowed:
            continue
        if any(part.lower() in excluded or part.lower().startswith('.env.') for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Refusing to export symlink: {relative}')
        if path.is_file():
            paths.append((path, relative.as_posix()))
    manifest = {name: {'bytes':p.stat().st_size,'sha256':sha256(p)} for p,name in paths}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path,name in paths:
            z.write(path,name)
        z.writestr('MANIFEST.sha256.json',json.dumps(manifest,indent=2)+'\n')
    return archive
