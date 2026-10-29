"""CPU recovery of archived command endpoints; no model or Modal container is launched."""
import argparse
import asyncio
import contextlib
import hashlib
import io
import json
from pathlib import Path

from analyze_command_mean import registered_config
from analyze_frozen import part1
from probe_shift.deception_endpoint import READOUT, recover
import score_frozen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('rows', 'keys', 'labels', 'checkpoints', 'source-cache', 'original-scores',
                 'aisi-checkpoint', 'q2-checkpoint', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    cfg = registered_config('command_mean.json')
    analysis = registered_config('command_mean_analysis.json')
    manifest = json.loads(a.rows.read_text())
    for path, digest in ((a.rows, cfg['rows_sha256']), (a.keys, manifest['key_sha256']),
                         (a.labels, analysis['labels_sha256'])):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen input hash mismatch: {path.name}')
    replay_cfg = asyncio.run(recover(manifest['rows'], sorted(a.checkpoints.glob('shard*/*.json')),
        a.source_cache, a.out, a.original_scores, a.aisi_checkpoint, a.q2_checkpoint, score_frozen, cfg))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        part1(a.out / 'part1_L31.npz', a.out / 'part1_L32.npz', a.rows, a.keys, a.labels,
              label_note=f'post-hoc corrected command-content endpoint; {READOUT}; same development cohort')
    result = json.loads(buf.getvalue())
    result.update(readout=READOUT, replay_config=replay_cfg, analysis_config=analysis,
        analysis_input_sha256={x.name: hashlib.sha256(x.read_bytes()).hexdigest()
                               for x in (a.rows, a.keys, a.labels)},
        code_sha256={str(x.relative_to(Path.cwd())): hashlib.sha256(x.read_bytes()).hexdigest()
                     for x in (Path(__file__).resolve(), Path('src/probe_shift/deception_endpoint.py').resolve(),
                               Path(__file__).resolve().with_name('analyze_frozen.py'))})
    (a.out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'metrics': str(a.out / 'metrics.json')}))


if __name__ == '__main__':
    main()
