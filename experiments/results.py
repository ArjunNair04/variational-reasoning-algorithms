#!/usr/bin/env python3
"""Check frozen numeric evidence and redraw small result plots, without models or datasets."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
DATA = HERE / 'results'


def rows(name):
    with (DATA / name).open() as handle:
        return list(csv.DictReader(handle))


def close(left, right, tolerance=1e-6):
    if not math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=tolerance):
        raise AssertionError(f'{left} != {right}')


def verify():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        content = (DATA / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record['sha256'], name
        assert len(rows(name)) == record['rows'], name
    allocations = rows('chapter4_development_allocations.csv')
    for record in allocations:
        weights = json.loads(record['weights_descending_json'])
        close(sum(weights), 1, 2e-6)
        close(len(weights), record['trace_count'])
        close(1 / sum(w*w for w in weights), record['effective_count'])
        entropy = -sum(w*math.log(w) for w in weights if w > 0)
        close(entropy, record['entropy'])
        assert float(record['effective_count']) <= math.exp(entropy) + 1e-5
        assert math.exp(entropy) <= len(weights) + 1e-5
    transitions = rows('diagnostics/vout_fixed_q_transitions.csv')
    grouped = defaultdict(list)
    for record in transitions:
        assert record['status'] == 'eligible'
        close(float(record['Q_after_observed_before_next_update'])-float(record['Q_before']), record['Q_change'])
        grouped[record['seed'], record['round_zero_based']].append(record)
    assert len(grouped) == 96 and len(transitions) == 288
    for group in grouped.values():
        group.sort(key=lambda r: int(r['observed_update_one_based']))
        assert [int(r['observed_update_one_based']) for r in group] == [1, 2, 3]
        for before, after in zip(group, group[1:]):
            close(before['Q_after_observed_before_next_update'], after['Q_before'])
    rounds = rows('diagnostics/vout_fixed_q_rounds.csv')
    for record in rounds:
        group = grouped[record['seed'], record['round_zero_based']]
        close(sum(float(r['Q_change']) for r in group), record['Q_change_through_three_updates'])
        assert record['fourth_update_observed'] == '0'
    positive = sum(float(r['Q_change']) > 1e-8 for r in transitions)
    round_positive = sum(float(r['Q_change_through_three_updates']) > 1e-8 for r in rounds)
    assert positive == 247 and round_positive == 94

    groups = defaultdict(list)
    for record in rows('diagnostics/transfer_category_cross_tabs.csv'):
        groups[record['dataset'], record['role'], record['seed']].append(record)
    assert len(groups) == 56
    total = 0
    for record in rows('diagnostics/transfer_seed_metrics.csv'):
        group = groups[record['dataset'], record['role'], record['seed']]
        n = int(record['questions']); total += n
        assert sum(int(r['count']) for r in group) == n
        for output, predicate in [
            ('final_extracted_answer_accuracy', lambda r: r['extracted_correct']=='1'),
            ('final_strict_terminal_accuracy', lambda r: r['category']=='strict_correct'),
            ('natural_eos_rate', lambda r: r['natural_eos']=='1'),
            ('token_limit_rate', lambda r: r['token_limit']=='1'),
        ]:
            close(sum(int(r['count']) for r in group if predicate(r))/n, record[output])
        gap = sum(int(r['count']) for r in group if r['extracted_correct']=='1' and r['category']!='strict_correct')
        close(gap, record['gap_count'])
    assert total == 42000
    for summary in rows('new_dataset_method_summary.csv'):
        seed_rows = [r for r in rows('diagnostics/transfer_seed_metrics.csv')
                     if r['dataset']==summary['dataset'] and r['role']==summary['role']]
        assert len(seed_rows) == 7
        for metric in ['final_extracted_answer_accuracy','final_strict_terminal_accuracy','natural_eos_rate','token_limit_rate']:
            close(mean(float(r[metric]) for r in seed_rows), summary[metric])
    selected = {r['method']:r for r in rows('selected_schedule_method_summary.csv')}
    delta_gain = float(selected['answer_guided_finite_support']['final_parsed'])-float(selected['paired_zero_update_base']['final_parsed'])
    close(delta_gain, 4.39)
    return {'numeric_files':len(manifest['files']), 'allocation_records':len(allocations),
            'fixed_Q_transitions':len(transitions), 'positive_fixed_Q_transitions':positive,
            'rounds_above_initial_Q_after_three_updates':round_positive,
            'transfer_seed_dataset_cells':56, 'transfer_responses_accounted_for':total,
            'selected_delta_gain_pp':round(delta_gain,2)}


def plot(destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    destination.mkdir(parents=True, exist_ok=True)
    names = {'answer_guided_finite_support':'Delta', 'answer_weighted_fresh':'Importance',
             'paired_zero_update_base':'Frozen base','gold_trace_SFT':'Gold SFT (extra solutions)'}
    data = rows('selected_schedule_method_summary.csv')
    fig, ax = plt.subplots(figsize=(8,4.8))
    positions = np.arange(len(data))
    ax.barh(positions-.18,[float(r['final_parsed']) for r in data],height=.36,label='Answer accuracy')
    ax.barh(positions+.18,[float(r['final_terminal_valid']) for r in data],height=.36,label='Terminal-format accuracy')
    ax.set_yticks(positions,[names.get(r['method'],r['method']) for r in data]);ax.invert_yaxis()
    ax.set_xlabel('Mean accuracy (%)');ax.set_xlim(0,100)
    ax.set_title('Selected schedules: seven paired training seeds');ax.legend(loc='upper center', bbox_to_anchor=(.5,-.16), ncol=2)
    fig.tight_layout();fig.savefig(destination/'selected_methods.pdf');plt.close(fig)

    data=rows('chapter4_efficiency_checkpoint_summary.csv');fig,ax=plt.subplots(figsize=(7,4))
    for cell in dict.fromkeys(r['cell'] for r in data):
        group=sorted([r for r in data if r['cell']==cell],key=lambda r:float(r['training_completions']))
        label=cell.replace('PIS','Importance').replace('VIN','Prior delta')
        ax.plot([float(r['training_completions']) for r in group],[float(r['mean_answer_accuracy_change_pp']) for r in group],marker='o',label=label)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xlabel('Training completions');ax.set_ylabel('Answer-accuracy gain (percentage points)')
    ax.set_title('Common-grid checkpoint means: three paired seeds');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(destination/'sample_budget.pdf');plt.close(fig)

    data=rows('diagnostics/vout_fixed_q_transitions.csv');fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist([float(r['Q_change']) for r in data],bins=25);axes[0].axvline(0,color='black',linewidth=.7)
    axes[0].set_xlabel('Observed within-round change in fixed-weight Q');axes[0].set_ylabel('Update count')
    trajectory=rows('diagnostics/vout_validation_trajectories.csv')
    for seed in ['31','47','73']:
        group=[r for r in trajectory if r['seed']==seed]
        axes[1].plot([int(r['completed_rounds']) for r in group],[float(r['answer_gain_from_base_pp']) for r in group],label=f'Seed {seed}')
    axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_xlabel('Completed rounds');axes[1].set_ylabel('Answer-accuracy gain (points)');axes[1].legend()
    fig.suptitle('Fixed-weight delta diagnostic: first three updates observed; fourth unobserved')
    fig.tight_layout();fig.savefig(destination/'fixed_objective.pdf');plt.close(fig)
    print(f'Wrote three plots to {destination}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--plot',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(),indent=2))
    if args.plot:plot(args.plot)


if __name__=='__main__':main()
