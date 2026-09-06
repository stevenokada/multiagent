"""Bounded exact-checkpoint numerical check followed by journal calibration."""
import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from mindvirus.backends import GenRequest, build_backend
from mindvirus.calibrate import NEUTRAL, run_calibration, runtime_metadata
from mindvirus.config import load_config
from mindvirus.payloads import PAYLOADS
from mindvirus.personas import PERSONAS
from mindvirus.probes import Battery
from mindvirus.relations import _relation_requests
from mindvirus.sweep import _validate_pilot_config


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def verify(backend, battery, output):
    output.mkdir()
    backend.capture_dir = output / 'activations'
    backend.capture_dir.mkdir()
    journals = [NEUTRAL, PAYLOADS[battery.payload_id].belief,
                "I encountered the following opinion and record it as someone else's view, without adopting it:\n"
                + PAYLOADS[battery.payload_id].belief]
    metadata = []
    for journal in journals:
        for item in battery.items:
            metadata.extend(_relation_requests(PERSONAS[0], journal, item, len(metadata)))
    random.Random(0).shuffle(metadata)  # mix input lengths inside each batch
    requests = [GenRequest(m['system'], m['messages'], 0, 1, f'serial-{i:04d}', 'probe')
                for i, m in enumerate(metadata)]
    assert all(len(backend.tokenizer.encode(c, add_special_tokens=False)) == 1 for c in ['A', 'B'])
    # Warm both paths without writing activation artifacts.
    warm = [replace(r, call_kind='warmup') for r in requests[:2]]
    backend.choice_logprobs(warm[0], ['A', 'B'])
    backend.choice_logprobs_batch(warm, ['A', 'B'])
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    serial = []
    for request in requests:
        probabilities = backend.choice_logprobs(request, ['A', 'B'])
        assert probabilities is not None and backend.last_activation_path is not None
        serial.append((probabilities, backend.last_activation_path))
    sync()
    serial_seconds = time.perf_counter() - started
    batched_requests = [replace(r, call_id=r.call_id.replace('serial-', 'batch-')) for r in requests]
    started = time.perf_counter()
    batched = []
    for start in range(0, len(requests), 2):
        batched.extend(backend.choice_logprobs_batch(batched_requests[start:start + 2], ['A', 'B']))
    sync()
    batch_seconds = time.perf_counter() - started
    layer = backend.capture.layers[0]
    pairs = []
    for index, (meta, request, (reference, path), current) in enumerate(zip(metadata, requests, serial, batched)):
        assert current.probabilities is not None and current.activation_path is not None
        a = torch.load(path, map_location='cpu', weights_only=True)[layer].float()
        b = torch.load(current.activation_path, map_location='cpu', weights_only=True)[layer].float()
        assert a.shape == b.shape and a.ndim == 1
        delta = max(abs(reference[c] - current.probabilities[c]) for c in ['A', 'B'])
        relative_l2 = float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a).clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        pairs.append({**meta, 'serial_call_id': request.call_id,
                      'batch_call_id': batched_requests[index].call_id,
                      'input_tokens': len(backend.tokenizer.encode(backend._render(request), add_special_tokens=False)),
                      'serial': reference, 'batch': current.probabilities,
                      'probability_max_abs_difference': delta, 'vector_relative_l2': relative_l2,
                      'vector_cosine': cosine,
                      'answer_changed': (reference['A'] > .5) != (current.probabilities['A'] > .5),
                      'serial_activation': str(Path(path).relative_to(output)),
                      'batch_activation': str(Path(current.activation_path).relative_to(output))})
    # Declared before inspecting exact-model outputs; report the observed errors too.
    tolerances = {'probability_max_abs_difference': .01, 'vector_relative_l2': .02,
                  'vector_min_cosine': .999, 'changed_answers_allowed': 0}
    passed = all(p['probability_max_abs_difference'] <= .01 and p['vector_relative_l2'] <= .02
                 and p['vector_cosine'] >= .999 and not p['answer_changed'] for p in pairs)
    result = {'passed': passed, 'tolerances': tolerances, 'requests': len(requests),
              'serial_seconds': serial_seconds, 'batch_seconds': batch_seconds,
              'observed_batch_speedup': serial_seconds / batch_seconds,
              'timing_scope': 'warmed forward inference including identical activation saves; single pass per method',
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              'pairs': pairs}
    write(output / 'comparison.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'pairs'}), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--battery', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg, battery = load_config(args.config), Battery.load(args.battery)
    _validate_pilot_config(cfg, battery)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {'state': 'loading', 'stage': 'numerical-check-then-calibration',
              'source_commit': 'c116f11acfe0ec98645c8db5a2987d542f798dc9',
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(output / 'status.json', status)
    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert torch.cuda.is_bf16_supported()
        assert torch.cuda.get_device_properties(0).total_memory >= 23000 * 1024**2
        started = time.perf_counter()
        backend = build_backend(cfg.agent_model, output, cfg.capture, seed=cfg.seed)
        assert backend._model.dtype == torch.bfloat16
        assert all(p.device.type == 'cuda' for p in backend._model.parameters())
        assert backend._model.config._commit_hash == cfg.agent_model.revision
        assert backend._model.config.hidden_size == 3584  # this invocation is Gemma
        status.update(state='checking', model_load_seconds=time.perf_counter()-started,
                      runtime=runtime_metadata(backend))
        write(output / 'status.json', status)
        passed = verify(backend, battery, output / 'numerical-check')
        status['batch_check_passed'] = passed
        if not passed:
            raise RuntimeError('Numerical batch check exceeded its declared tolerances')
        status['state'] = 'calibrating'
        write(output / 'status.json', status)
        torch.manual_seed(cfg.seed)
        run_calibration(cfg.agent_model, battery, output / 'calibration', n_personas=3,
                        repeats=3, seed=cfg.seed, backend=backend, capture=cfg.capture,
                        batch_size=cfg.probe_batch_size)
        status['state'] = 'completed'
    except BaseException as exc:
        status.update(state='failed', error_type=type(exc).__name__)
        raise
    finally:
        write(output / 'status.json', status)


if __name__ == '__main__':
    main()
