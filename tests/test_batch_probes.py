"""Verify batching preserves real transformer outputs and measurement identity."""
import importlib
import json
from dataclasses import replace

import pytest

torch = pytest.importorskip('torch')
from mindvirus.backends import CallLogger, GenRequest
from mindvirus.config import CaptureConfig, ModelConfig
from mindvirus.hf_backend import HFBackend
from mindvirus.personas import PERSONAS
from mindvirus.probes import hand_battery
from tests.test_hf_backend import local_tokenizer, CHAT_TEMPLATE, GEMMA_TEMPLATE
from tests.test_calibration import StableRelationBackend


def tiny_backend(tmp_path, family, padding_side, positions='last'):
    from transformers import LlamaConfig, LlamaForCausalLM, Gemma2Config, Gemma2ForCausalLM
    tokenizer = local_tokenizer(CHAT_TEMPLATE if family == 'llama' else GEMMA_TEMPLATE)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    common = dict(vocab_size=16, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=8, max_position_embeddings=256, pad_token_id=2,
                  bos_token_id=1, eos_token_id=2, attn_implementation='eager')
    torch.manual_seed(8)
    if family == 'llama':
        model = LlamaForCausalLM(LlamaConfig(**common)).eval()
    else:
        model = Gemma2ForCausalLM(Gemma2Config(**common, sliding_window=32,
                                             query_pre_attn_scalar=8)).eval()
    return HFBackend(ModelConfig('hf', 'local/test'), tokenizer=tokenizer, model=model,
                     capture=CaptureConfig(True, [1], positions, ['probe']),
                     capture_dir=tmp_path/'activations')


def requests():
    return [GenRequest('hi', [{'role':'user', 'content': text}], 0, 1,
                       f'c{i:06d}', 'probe')
            for i,text in enumerate(['hi', 'hi hi hi hi hi'],1)]


@pytest.mark.parametrize('family', ['llama','gemma'])
@pytest.mark.parametrize('padding_side', ['left','right'])
@pytest.mark.parametrize('positions', ['last','all'])
def test_batched_real_transformer_matches_serial_tokens_and_activations(
        tmp_path, family, padding_side, positions):
    backend = tiny_backend(tmp_path, family, padding_side, positions)
    original = requests()
    serial = [backend.choice_logprobs(r, ['1','hi']) for r in original]
    activations = [torch.load(backend.capture_dir/f'{r.call_id}.pt', weights_only=True)[1]
                   for r in original]
    assert hasattr(backend, 'choice_logprobs_batch'), 'HF batch inference is missing'
    batch_requests = [replace(r, call_id='batch-'+r.call_id) for r in original]
    batched = backend.choice_logprobs_batch(batch_requests, ['1','hi'])
    assert len(batched) == 2
    for reference, vector, result in zip(serial, activations, batched):
        assert result.probabilities == pytest.approx(reference, abs=1e-6)
        saved = torch.load(result.activation_path, weights_only=True)[1]
        assert saved.shape == vector.shape  # all-position output excludes padding
        torch.testing.assert_close(saved, vector, atol=1e-3, rtol=1e-3)


def test_batch_keeps_mappings_ids_and_native_scores(tmp_path):
    module = importlib.import_module('mindvirus.relations')
    assert hasattr(module, 'run_relation_probes'), 'relation batching is missing'
    from mindvirus.backends import ChoiceResult
    class BatchSemantic(StableRelationBackend):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []
        def choice_logprobs_batch(self, reqs, choices):
            self.batch_sizes.append(len(reqs))
            return [ChoiceResult(self.choice_logprobs(r,choices), f'{r.call_id}.pt') for r in reqs]
    backend = BatchSemantic()
    battery = hand_battery('honesty-absolutism',task='relation')
    trials = [(PERSONAS[0], 'unchanged journal', item, 3) for item in battery.items[:3]]
    out = module.run_relation_probes(CallLogger(backend,tmp_path/'calls.jsonl'), trials, batch_size=4)
    assert [r['score'] for r in out] == pytest.approx([.8,.8,.8])
    assert [r['probe_id'] for r in out] == [i.id for i in battery.items[:3]]
    assert backend.batch_sizes == [4,2]
    calls = [json.loads(x) for x in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert len(calls) == len({c['call_id'] for c in calls}) == 6
    assert [c['answer_mapping'] for c in calls] == ['standard','reversed']*3
    assert all(c['activation_path'] == f"{c['call_id']}.pt" for c in calls)
    assert all(c['system'].count('unchanged journal')==1 for c in calls)


def test_failed_batch_isolates_bad_requests(tmp_path):
    module = importlib.import_module('mindvirus.relations')
    assert hasattr(module, 'run_relation_probes'), 'relation batching is missing'
    class PartlyBroken(StableRelationBackend):
        def choice_logprobs_batch(self, reqs, choices):
            raise RuntimeError('batch cannot be evaluated')
        def choice_logprobs(self, req, choices):
            if 'BROKEN' in req.system:
                raise ValueError('one invalid input')
            return super().choice_logprobs(req,choices)
    item = hand_battery('honesty-absolutism',task='relation').items[0]
    trials = [(PERSONAS[0],journal,item,0) for journal in ['fine','BROKEN','fine']]
    result = module.run_relation_probes(CallLogger(PartlyBroken(default='invalid'),tmp_path/'calls.jsonl'),
                                       trials,batch_size=6)
    assert result[0]['score'] == pytest.approx(.8)
    assert result[1]['score'] is None
    assert result[2]['score'] == pytest.approx(.8)


def test_capture_write_failure_keeps_call_identity_and_valid_probabilities(tmp_path, monkeypatch):
    backend = tiny_backend(tmp_path, 'llama', 'left')
    save = torch.save
    saves = 0

    def fail_second(value, path, *args, **kwargs):
        nonlocal saves
        saves += 1
        if saves == 2:
            # Simulate a write that failed after creating a partial file.
            from pathlib import Path
            Path(path).write_bytes(b'incomplete')
            raise OSError('write failed')
        return save(value, path, *args, **kwargs)

    monkeypatch.setattr(torch, 'save', fail_second)
    logger = CallLogger(backend, tmp_path/'calls.jsonl')
    rows = [dict(system=r.system, messages=r.messages, call_kind=r.call_kind,
                 probe_id=f'item-{i}') for i, r in enumerate(requests())]
    results = logger.choice_logprobs_batch(rows, ['1', 'hi'])
    calls = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [c['call_id'] for c in calls] == ['c000001', 'c000002']
    assert all(result is not None for result in results)
    assert calls[0]['activation_path'] is not None
    assert calls[1]['activation_path'] is None
    assert calls[1]['activation_error_type'] == 'OSError'
    assert sorted(p.name for p in backend.capture_dir.iterdir()) == ['c000001.pt']
