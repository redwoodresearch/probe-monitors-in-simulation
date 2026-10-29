"""Recover the corrected command endpoint from archived activations, without inference."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
import zipfile

import modal
import numpy as np

from probe_shift.deception_tokens import collect_checkpoints, match_rows

READOUT = 'command_content_endpoint_raw_score_v1'
ACTS = 'probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def probe_directions(aisi_path, q2_path, frozen):
    """Read hash-pinned arrays only; never unpickle the public torch checkpoint.

    load_aisi in the original scorer checked data/0 bit-for-bit against the tensor.
    Requiring that exact file hash makes its raw float32 storage safe to reuse here.
    """
    a, q = Path(aisi_path).read_bytes(), Path(q2_path).read_bytes()
    if sha(a) != frozen.AISI['sha256'] or sha(q) != frozen.Q2['sha256']:
        raise ValueError('Probe checkpoint hash mismatch')
    with zipfile.ZipFile(io.BytesIO(a)) as z:
        names = [n for n in z.namelist() if n.endswith('/data/0')]
        if len(names) != 1:
            raise ValueError('Ambiguous AISI direction storage')
        aisi = np.frombuffer(z.read(names[0]), dtype='<f4')
    with np.load(io.BytesIO(q), allow_pickle=False) as z:
        coef, mean, scale = (z[k][frozen.Q2['index']].astype(np.float64)
                             for k in ('coef', 'scaler_mean', 'scaler_scale'))
    if any(x.shape != (frozen.D_MODEL,) or not np.isfinite(x).all()
           for x in (aisi, coef, mean, scale)) or np.any(scale <= 0):
        raise ValueError('Invalid probe arrays')
    # Match the original scorer's float32 direction, including its rounding.
    return frozen.directions(aisi, (coef / scale).astype(np.float32)), -float((mean / scale) @ coef)


def bind_commands(rows, metadata, checkpoints):
    """Prove each stored command row is the last token of the validated mean mask."""
    groups = {}
    records = match_rows(rows, metadata)
    for r, m, c, e in records:
        if c.get('row') is None or not 0 <= c['row'] < m['n_rows'] or c['n'] != e['n']:
            raise ValueError('Missing command activation or inconsistent prefix')
        k = (r['loc'][0], r['loc'][1], m['key'])
        groups.setdefault(k, []).append((r, m, c, e))
    if set(groups) != set(checkpoints):
        raise ValueError('Checkpoint epochs differ from frozen cohort')
    for k, group in groups.items():
        d = checkpoints[k]
        if sha(json.dumps(group, sort_keys=True).encode()) != d['source']['metadata_sha256']:
            raise ValueError('Source metadata differs from verified replay')
        if [r['id'] for r, *_ in group] != d['ids']:
            raise ValueError('Checkpoint row order differs')
        for i, (_, _, c, e) in enumerate(group):
            if (c['readout'] != d['token_indices'][i][-1]
                    or e['readout'] != d['source']['endpoint_indices'][i]):
                raise ValueError('Stored endpoint differs from verified token span')
    return records


async def recover(rows, checkpoint_paths, cache, out, original, aisi_path, q2_path, frozen, replay_config):
    packed = collect_checkpoints(checkpoint_paths, rows)
    cfg = json.loads(str(packed['31']['config_json']))
    if any(cfg.get(k) != v for k, v in replay_config.items()):
        raise ValueError('Replay differs from registered configuration')
    checkpoints = {}
    for p in checkpoint_paths:
        d = json.loads(p.read_text())
        if tuple(d['epoch']) in checkpoints:
            raise ValueError('Duplicate checkpoint epoch')
        checkpoints[tuple(d['epoch'])] = d
    directions, bias = probe_directions(aisi_path, q2_path, frozen)
    vol = modal.Volume.from_name(ACTS, version=2)
    sem = asyncio.Semaphore(8)

    async def read(path):
        async with sem:
            return b''.join([b async for b in vol.read_file.aio(path)])

    async def meta(path):
        dest = cache / path
        if not dest.exists():
            raw = await read(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        return json.loads(dest.read_text())

    async def location(loc):
        x, arm, chunk = loc
        ix = await meta(f'{x}/chunks/{chunk}.json')
        eps = []
        for e in ix['epochs']:
            if e['arm'] != arm:
                continue
            m = await meta(f'{x}/meta/{arm}/{hashlib.sha1(e["key"].encode()).hexdigest()[:20]}.json')
            if m['chunk'] != chunk or any(m[k] != e[k] for k in ('row_start', 'n_rows')):
                raise ValueError('Chunk/metadata ownership differs')
            eps.append(m)
        return loc, eps

    metadata = dict(await asyncio.gather(*(location(loc) for loc in sorted({tuple(r['loc'][:3]) for r in rows}))))
    records = bind_commands(rows, metadata, checkpoints)
    for (x, arm, _), d in checkpoints.items():
        ecfg = await meta(f'{x}/config.json')
        if ecfg != d['source']['extraction_config']:
            raise ValueError('Extraction configuration differs from verified replay')
        if (ecfg['model'], ecfg['revision']) != (cfg['model'], cfg['revision']):
            raise ValueError('Wrong extraction model/revision')
    out.mkdir(parents=True, exist_ok=False)
    groups = {}
    for i, (r, m, c, e) in enumerate(records):
        x, _, chunk, old_row = r['loc']
        groups.setdefault((x, chunk), []).append((i, m['row_start'] + c['row'], old_row))
    proof = {'readout': READOUT, 'volume': ACTS, 'replay_fingerprint': str(packed['31']['fingerprint']),
             'rows': len(rows), 'epochs': len(checkpoints), 'files': [], 'layers': {}}
    same = np.array([c['readout'] == e['readout'] for _, _, c, e in records])
    proof['changed_endpoints'] = int((~same).sum())
    for layer, (names, w) in directions.items():
        # Only selected rows are retained locally; whole source files are hashed.
        X = np.empty((len(rows), frozen.D_MODEL), np.float16)
        old = np.empty_like(X)

        async def chunk_at(key, items):
            path = f'{key[0]}/by_layer/L{layer}/{key[1]}.npy'
            raw = await read(path)
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            if arr.ndim != 2 or arr.shape[1] != frozen.D_MODEL or arr.dtype != np.float16:
                raise ValueError('Unexpected source activation format')
            ii, cc, ee = np.array(items).T
            X[ii], old[ii] = arr[cc], arr[ee]
            proof['files'].append({'path': path, 'sha256': sha(raw), 'bytes': len(raw)})
        await asyncio.gather(*(chunk_at(k, items) for k, items in sorted(groups.items())))
        if not np.isfinite(X).all() or not np.isfinite(old).all():
            raise ValueError('Non-finite archived activation')
        b = np.array([bias if n == 'q2_apollo_l31_at_block31' else 0. for n in names])
        scores = X.astype(np.float64) @ w.astype(np.float64) + b
        legacy = old.astype(np.float64) @ w.astype(np.float64) + b
        with np.load(original / f'part1_L{layer}.npz', allow_pickle=False) as z:
            if list(z['ids']) != [r['id'] for r in rows] or list(z['columns']) != names:
                raise ValueError('Original score identity differs')
            error = float(np.max(np.abs(legacy - z['scores'])))
        if error > cfg['endpoint_score_atol']:
            raise ValueError('Original released score parity failed')
        np.testing.assert_array_equal(X[same], old[same])
        one = packed[str(layer)]['token_counts'] == 1
        single_error = float(np.max(np.abs(scores[one] - packed[str(layer)]['scores'][one])))
        if single_error > cfg['endpoint_score_atol']:
            raise ValueError('Single-token endpoint/mean parity failed')
        np.savez(out / f'part1_L{layer}.npz', ids=packed[str(layer)]['ids'], columns=names,
                 scores=scores, readout=READOUT, fingerprint=packed[str(layer)]['fingerprint'])
        np.savez(out / f'selected_activations_L{layer}.npz', command=X, call_end=old)
        proof['layers'][str(layer)] = {'released_score_max_abs_error': error,
            'unchanged_endpoint_rows': int(same.sum()), 'unchanged_activations_exact': True,
            'single_token_rows': int(one.sum()), 'single_token_max_abs_score_error': single_error}
        print(json.dumps({'scored_layer': layer, **proof['layers'][str(layer)]}), flush=True)
    proof['files'].sort(key=lambda r: r['path'])
    (out / 'verification.json').write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    return cfg
