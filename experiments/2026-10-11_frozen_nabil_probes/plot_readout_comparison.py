"""Plot frozen readout arms, with the trained probe's different population separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'endpoint', 'mean', 'trained', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    arms = [(json.loads(a.original.read_text()), 'Old call_end', '#9a9a9a'),
            (json.loads(a.endpoint.read_text()), 'Command endpoint', '#4285b4'),
            (json.loads(a.mean.read_text()), 'Command-token mean', '#de8b39')]
    trained = json.loads(a.trained.read_text())
    if any((d['positives'], d['tasks_pos'], d['tasks_neg']) != (72, 33, 40) for d, *_ in arms):
        raise ValueError('Figure requires the original development population')
    if (trained['checks']['heldout']['pos'], trained['checks']['heldout']['pos_tasks'],
            trained['checks']['heldout']['neg_tasks']) != (31, 12, 16):
        raise ValueError('Figure requires the original held-out trained-probe result')
    max_ci = max(v['catch_ci95'][1] * 100 for d, *_ in arms for v in d['members'].values())
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), height_ratios=[7, 2], sharex=True)
    members = [('q2_apollo_l31_at_block31', 'Apollo, block 31'),
               ('aisi_l32_at_block31', 'AISI, block 31'), ('aisi_l32_at_block32', 'AISI, block 32')]
    labels = []
    for group, (member, title) in enumerate(members):
        for arm, (data, readout, color) in enumerate(arms):
            r = data['members'][member]
            y = group * 4 + arm
            value, (lo, hi) = r['catch_at_1pct'] * 100, np.array(r['catch_ci95']) * 100
            axes[0].barh(y, value, color=color, height=.65)
            axes[0].errorbar(value, y, xerr=[[value-lo], [hi-value]], color='black', capsize=3, fmt='none')
            axes[0].text(hi + .7, y, f'{value:.1f}%', va='center', fontsize=10)
            labels.append((y, f'{title} · {readout}'))
    axes[0].set_yticks([x[0] for x in labels], [x[1] for x in labels])
    axes[0].invert_yaxis()
    axes[0].set_title('Frozen probes: same 40 development tasks, 72 positives across 33 tasks', loc='left', fontsize=12)
    r = trained['real']
    value, (lo, hi) = r['catch'] * 100, np.array(r['catch_95']) * 100
    axes[1].barh(0, value, color='#9a9a9a', height=.45)
    axes[1].errorbar(value, 0, xerr=[[value-lo], [hi-value]], color='black', capsize=3, fmt='none')
    axes[1].text(hi + .7, 0, f'{value:.1f}%', va='center', fontsize=10)
    axes[1].set_yticks([0], ['Trained on our labels · call_end'])
    axes[1].set_ylim(-.8, .8)
    axes[1].set_title('Separate reference: 16 held-out tasks, 31 positives across 12 tasks\nDifferent readout/population; not a controlled comparison', loc='left', fontsize=11)
    for ax in axes:
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.2)
        ax.tick_params(axis='y', length=0)
    axes[1].set_xlim(0, max(60, hi + 8, max_ci + 8))
    axes[1].set_xlabel('Weighted sabotage catch (%) at nominal 1% FPR on honest A — higher is better')
    fig.suptitle('Separate the endpoint correction from command-token averaging', fontsize=15, x=.03, ha='left')
    fig.text(.03, .025, 'Input: archived Qwen3.5-27B tool calls; exact served IDs, no new prompt.\n'
             'Error bars: 95% task-bootstrap intervals. Frozen arms are post hoc on the same development data;\n'
             'these marginal intervals do not establish an improvement. Honest-B FPR is reported in the results table.', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .94), h_pad=2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        target = a.out.with_suffix(suffix)
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
