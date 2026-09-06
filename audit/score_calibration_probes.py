"""Exploratory frozen-probe readouts on the exported calibration activations.

Uses Jeff's original scorer/scaler classes. Never refits a probe or its scaler.
The eight authored items are not a held-out validation of agent-context transfer.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--jeff-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.jeff_repo / 'src'))
    from geometry_of_truth.m1.probes import DifferenceInMeansProbe, LogisticProbe
    from geometry_of_truth.m1.support.metrics import Scaler

    design = json.loads((args.calibration / 'design.json').read_text())
    with np.load(args.probe, allow_pickle=False) as weights:
        assert str(weights['model_revision']) == design['model']['revision']
        assert str(weights['tokenizer_revision']) == design['model']['revision']
        layer = int(weights['selected_layer']) + 1
        assert design['capture']['layers'] == [layer]
        direction = weights['difference_in_means_direction'].copy()
        assert np.isclose(np.linalg.norm(direction), 1, atol=1e-5)
        dim = DifferenceInMeansProbe(direction, float(weights['difference_in_means_midpoint']))
        clf = LogisticRegression(C=float(weights['logistic_c']))
        clf.coef_ = weights['logistic_coef'].copy()
        clf.intercept_ = weights['logistic_intercept'].copy()
        clf.classes_ = np.array([0, 1])
        clf.n_features_in_ = len(direction)
        logistic = LogisticProbe(clf, float(weights['logistic_c']))
        scorers = {'dim': dim, 'logistic': logistic}
        scalers = {name: Scaler(float(weights[f'{name}_scaler_mu']),
                                float(weights[f'{name}_scaler_sigma']), 'Jeff frozen selection split')
                   for name in scorers}
        assert all(s.sigma > 0 for s in scalers.values())
    # Formula checks with the imported reference implementations, without data fitting.
    zero = np.zeros((1, len(direction)), dtype=np.float32)
    assert np.allclose(dim.score(zero), -dim.midpoint)
    assert np.allclose(logistic.score(zero), clf.intercept_)

    def vector(path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert set(saved) == {layer}
        array = saved[layer].float().numpy()
        assert array.shape == direction.shape and np.isfinite(array).all()
        return array

    calls = [json.loads(s) for s in (args.calibration / 'calls.jsonl').read_text().splitlines()]
    observations = [json.loads(s) for s in (args.calibration / 'observations.jsonl').read_text().splitlines()]
    battery = json.loads((args.calibration / 'battery.json').read_text())
    items = {i['id']: i for i in battery['items']}
    trials = {(r['agent'], r['trial_id'], r['probe_id']): r for r in observations}
    assert len(trials) == len(observations)
    metadata, vectors = [], []
    for call in calls:
        assert call['kind'] == 'probe' and call['activation_path']
        trial = trials[(call['agent'], call['round'], call['probe_id'])]
        filename = Path(call['activation_path']).name
        assert Path(filename).stem == call['call_id']
        vectors.append(vector(args.calibration / 'activations' / filename))
        item = items[call['probe_id']]
        metadata.append({k: trial[k] for k in ['trial_id', 'agent', 'repeat', 'condition', 'probe_id']}
                        | {'mapping': call['answer_mapping'], 'call_id': call['call_id'],
                           'on_target': item['on_target'], 'label': int(item['reference_label'] == 'Supports')})
    frame = pd.DataFrame(metadata)
    assert len(frame) == 2 * len(observations)
    assert frame.groupby('trial_id')['mapping'].agg(set).apply(lambda v: v == {'standard', 'reversed'}).all()
    features = np.stack(vectors)
    for name, scorer in scorers.items():
        frame[f'{name}_raw'] = scorer.score(features)
        frame[f'{name}_z'] = scalers[name](frame[f'{name}_raw'])
    # Collapse exact repeats; average the two mappings without reversing the
    # semantic Supports/Opposes probe direction when the A/B symbols are swapped.
    keys = ['agent', 'probe_id', 'condition', 'on_target', 'label']
    columns = [f'{name}_{suffix}' for name in scorers for suffix in ['raw', 'z']]
    averaged = frame.groupby(keys, as_index=False)[columns].mean()
    baseline = averaged[averaged.condition == 'neutral']
    summary = {'model': design['model'], 'exploratory': True,
               'probe_sha256': hashlib.sha256(args.probe.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'hf_layer': layer, 'calls': len(frame), 'unique_items': len(items),
               'interpretation': 'Context-conditioned linear readout changes, not demonstrated moral alignment change or contagion. Transfer to these agent prompts is not established.',
               'units': 'Frozen Jeff selection-score SD; not a probability or a standardized effect size for this new battery.',
               'aggregation': 'Exact repeats collapsed; both answer mappings averaged with the semantic probe direction unchanged. Persona/item pairs are descriptive, not independent simulations.',
               'baseline_transfer_check': {}, 'changes': {}, 'mapping_sensitivity': {}}
    for name in scorers:
        raw = baseline[f'{name}_raw']
        summary['baseline_transfer_check'][name] = {
            'persona_item_pairs': len(baseline), 'auroc': float(roc_auc_score(baseline.label, raw)),
            'raw_zero_threshold_accuracy': float(((raw > 0).astype(int) == baseline.label).mean())}
        mapped = frame.groupby(keys + ['mapping'])[f'{name}_z'].mean().unstack('mapping')
        gap = (mapped['standard'] - mapped['reversed']).abs()
        summary['mapping_sensitivity'][name] = {'mean_absolute_z_gap': float(gap.mean()),
                                                'max_absolute_z_gap': float(gap.max())}
    for condition, reference in [('repeat', 'neutral'), ('seeded', 'neutral'),
                                 ('mentioned', 'neutral'), ('seeded', 'mentioned')]:
        paired = averaged[averaged.condition == condition].merge(
            averaged[averaged.condition == reference], on=['agent', 'probe_id', 'on_target', 'label'],
            suffixes=('_current', '_reference'), validate='one_to_one')
        label = f'{condition}_minus_{reference}'
        summary['changes'][label] = {}
        for group, target in [('target', True), ('control', False)]:
            part = paired[paired.on_target == target]
            metrics = {'persona_item_pairs': len(part)}
            for name in scorers:
                delta = part[f'{name}_z_current'] - part[f'{name}_z_reference']
                metrics[name] = {'mean_signed_z_change': float(delta.mean()),
                                 'mean_absolute_z_change': float(delta.abs().mean()),
                                 'max_absolute_z_change': float(delta.abs().max()),
                                 'mean_reference_signed_z_change': float((delta * (2*part.label - 1)).mean())}
            summary['changes'][label][group] = metrics
    comparison = json.loads(args.comparison.read_text())
    check_root = args.comparison.parent
    serial = np.stack([vector(check_root / p['serial_activation']) for p in comparison['pairs']])
    batch = np.stack([vector(check_root / p['batch_activation']) for p in comparison['pairs']])
    summary['batch_numerical_reference'] = {'comparison_passed': comparison['passed'],
        'warning': 'Different prompts/conditions and single serial/batch passes; this is a numerical reference, not a statistical noise distribution.'}
    for name, scorer in scorers.items():
        delta = scalers[name](scorer.score(batch)) - scalers[name](scorer.score(serial))
        summary['batch_numerical_reference'][name] = {
            'mean_absolute_z_difference': float(np.abs(delta).mean()),
            'max_absolute_z_difference': float(np.abs(delta).max())}
    args.output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output / 'per_call_scores.csv', index=False)
    averaged.to_csv(args.output / 'averaged_scores.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
