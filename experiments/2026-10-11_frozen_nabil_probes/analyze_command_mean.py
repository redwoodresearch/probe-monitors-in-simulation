"""Collect numeric checkpoints, recalibrate on honest A, and run the existing analysis.

Use --collect-only for the inexpensive integrity check before the full bootstrap/null
analysis. --checkpoints must contain both downloaded shard directories. Output is new.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from probe_shift.deception_tokens import collect_checkpoints


def registered_config(name):
    path = Path(__file__).resolve().parent / 'config' / name
    repo = path.parents[3]
    raw = subprocess.run(['git', 'show', f'HEAD:{path.relative_to(repo)}'], cwd=repo,
                         check=True, capture_output=True).stdout
    if path.read_bytes() != raw:
        raise ValueError('Registered analysis/replay config has uncommitted changes')
    return json.loads(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--keys', type=Path)
    p.add_argument('--labels', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--collect-only', action='store_true')
    a = p.parse_args()
    if not a.collect_only and (a.keys is None or a.labels is None):
        p.error('--keys and --labels are required unless --collect-only')
    manifest = json.loads(a.rows.read_text())
    analysis_config = registered_config('command_mean_analysis.json')
    replay_config = registered_config('command_mean.json')
    rows = manifest['rows']
    if a.keys is not None and hashlib.sha256(a.keys.read_bytes()).hexdigest() != manifest['key_sha256']:
        raise ValueError('Sampling keys differ from the original frozen manifest')
    if a.labels is not None and hashlib.sha256(a.labels.read_bytes()).hexdigest() != analysis_config['labels_sha256']:
        raise ValueError('Labels differ from the original stage-1 labels')
    packed = collect_checkpoints(sorted(a.checkpoints.glob('shard*/*.json')), rows)
    cfg = json.loads(str(packed['31']['config_json']))
    if any(cfg.get(k) != v for k, v in replay_config.items()):
        raise ValueError('Replay settings differ from the committed registered config')
    if hashlib.sha256(a.rows.read_bytes()).hexdigest() != cfg['rows_sha256']:
        raise ValueError('Rows file differs from the original frozen manifest')
    a.out.mkdir(parents=True, exist_ok=False)
    for layer, arrays in packed.items():
        np.savez(a.out / f'part1_L{layer}.npz', **arrays)
    if not a.collect_only:
        from analyze_frozen import part1_mean
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            part1_mean(a.out / 'part1_L31.npz', a.out / 'part1_L32.npz', a.rows, a.keys, a.labels)
        result = json.loads(output.getvalue())
        result['replay_config'] = cfg
        result['analysis_config'] = analysis_config
        result['analysis_input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (a.rows, a.keys, a.labels)}
        with (a.out / 'metrics.json').open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
