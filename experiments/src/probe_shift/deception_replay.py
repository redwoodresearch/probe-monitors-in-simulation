"""Replay the frozen sabotage cohort to measure command-token means, without refitting.

Only the thin entry point is a CLI. Source Volumes are read-only. Each shard writes
one atomic numeric checkpoint per epoch on a separate output Volume.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import modal

from probe_shift.deception_tokens import READOUT, replay_fingerprint

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/2026-10-11_frozen_nabil_probes'
ACTS = 'probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08'
Q2 = 'probe-shift_aswin-surya_sub-q2-deception-qwen35_kill-2026-11-08'
HF = 'probe-shift_aswin-surya_hf-cache'
OUTPUT = 'probe-shift_aswin-surya_sub-decmean_scores_kill-2026-11-12'
app = modal.App('probe-shift_aswin-surya_sub-decmean_replay_kill-2026-11-12')
leases = modal.Dict.from_name('probe-shift_aswin-surya_sub-decmean_leases_kill-2026-11-12', create_if_missing=True)
out_vol = modal.Volume.from_name(OUTPUT, create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version='3.12')
         .uv_pip_install('torch==2.8.0', 'transformers==5.19.0', 'accelerate>=1.1.0', 'numpy',
                         'huggingface_hub', 'flash-linear-attention==0.5.2')
         .env({'HF_HOME': '/hf'})
         .add_local_python_source('probe_shift')
         .add_local_file(EXPERIMENT / 'score_frozen.py', '/root/score_frozen.py')
         .add_local_file(EXPERIMENT / 'score_part2_generic.py', '/root/score_part2_generic.py')
         .add_local_file(ROOT / 'experiments/2026-10-08_q3_onpolicy_figures/run_replay_readouts.py', '/root/run_replay_readouts.py'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.function(image=image, gpu='H200', memory=65536, timeout=3600, retries=0, max_containers=2,
              volumes={'/hf': modal.Volume.from_name(HF).read_only(),
                       '/acts': modal.Volume.from_name(ACTS, version=2).read_only(),
                       '/q2': modal.Volume.from_name(Q2).read_only(), '/out': out_vol})
def replay(cfg, rows, shard):
    import sys
    from collections import defaultdict

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.deception_tokens import affine_mean, command_indices, ids_sha, match_rows

    sys.path.insert(0, '/root')
    import score_frozen as frozen
    from run_replay_readouts import bytes_to_unicode

    writer = f'{os.environ.get("MODAL_TASK_ID", "unknown")}@{time.time_ns()}'
    lease_key = f'{cfg["run"]}/shard{shard:02d}'
    attempt = cfg['attempt']
    if not leases.put(f'{lease_key}/a{attempt}', writer, skip_if_exists=True):
        raise RuntimeError('This shard attempt already has a writer; use a later attempt')
    held = leases.get(lease_key)
    if held is not None and held['attempt'] >= attempt:
        raise RuntimeError('This shard is owned by an equal/later attempt')
    leases[lease_key] = {'writer': writer, 'attempt': attempt}
    def check_writer():
        if (leases.get(lease_key) or {}).get('writer') != writer:
            raise RuntimeError('A later shard attempt superseded this writer')
    out_vol.reload()
    run = Path('/out') / cfg['run'] / f'shard{shard:02d}'
    run.mkdir(parents=True, exist_ok=True)
    identity = replay_fingerprint(cfg, rows)
    # Discover ownership from metadata, then load only epochs containing frozen rows.
    metadata, configs, groups = {}, {}, defaultdict(list)
    locations = {tuple(r['loc'][:3]) for r in rows}
    for x, arm, chunk in sorted(locations):
        root = Path('/acts') / x
        ecfg = json.loads((root / 'config.json').read_text())
        if (ecfg['model'], ecfg['revision']) != (cfg['model'], cfg['revision']):
            raise ValueError('Extraction model/revision does not match the frozen probe')
        configs[x] = ecfg
        index = json.loads((root / 'chunks' / f'{chunk}.json').read_text())
        eps = []
        for e in index['epochs']:
            if e['arm'] != arm:
                continue
            name = hashlib.sha1(e['key'].encode()).hexdigest()[:20]
            m = json.loads((root / 'meta' / arm / f'{name}.json').read_text())
            if m['chunk'] != chunk or m['row_start'] != e['row_start'] or m['n_rows'] != e['n_rows']:
                raise ValueError('Metadata/chunk ownership mismatch')
            if m['inputs_sha256'] != ecfg['inputs_sha256'][f'{arm}.jsonl']:
                raise ValueError('Metadata staged-input hash differs from extraction config')
            eps.append(m)
        metadata[(x, arm, chunk)] = eps
    for r, m, command, endpoint in match_rows(rows, metadata):
        groups[(r['loc'][0], r['loc'][1], m['key'])].append((r, m, command, endpoint))
    mine = sorted(groups)[shard::cfg['shards']]
    if cfg['pilot']:
        # Deterministic first epoch per shard; full run uses a different output name.
        mine = mine[:1]
    staged = {}
    for x, arm in sorted({(x, a) for x, a, _ in mine}):
        ecfg = configs[x]
        p = Path('/acts') / ecfg['inputs'] / 'inputs' / f'{arm}.jsonl'
        if digest(p) != ecfg['inputs_sha256'][p.name]:
            raise ValueError('Served token IDs differ from the archived extraction inputs')
        for line in p.read_text().splitlines():
            e = json.loads(line)
            k = (x, arm, e['key'])
            if k in staged:
                raise ValueError('Duplicate staged epoch')
            staged[k] = e
    tok = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'], local_files_only=True)
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    special_ids = set(tok.all_special_ids)
    im_start = tok.convert_tokens_to_ids('<|im_start|>')
    def token_bytes(ids):
        return [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in ids]

    # Validate every requested mask and prefix before loading the model.
    jobs = []
    for k in mine:
        ep, masks, old_ends = staged[k], [], []
        pieces = token_bytes(ep['ids'])
        specials = [i for i, t in enumerate(ep['ids']) if t in special_ids]
        for r, m, c, end in groups[k]:
            matches = [a for a in ep['calls'] if a.get('kind') == 'command'
                       and (a['irid'], a['call_index']) == (c['irid'], c['call_index'])]
            if len(matches) != 1:
                raise ValueError('Command identity is not unique in its served epoch')
            a = matches[0]
            if any(a[f] != c[f] for f in ('readout', 'n', 'param', 'name')):
                raise ValueError('Command metadata differs from served-epoch metadata')
            if a['n'] > len(ep['ids']) or ids_sha(ep['ids'][:a['n']]) != a['sha256']:
                raise ValueError('Served prefix hash mismatch')
            if not 0 <= end['readout'] < a['n'] or end['n'] != a['n']:
                raise ValueError('Old call_end is outside this request')
            headers = [i for i in range(a['readout']) if ep['ids'][i] == im_start]
            if not headers:
                raise ValueError('No assistant message boundary before the command')
            start = headers[-1] + 1
            if not b''.join(pieces[start:a['readout']+1]).startswith(b'assistant\n'):
                raise ValueError('Command endpoint is not in an assistant message')
            masks.append(command_indices(pieces, a, specials, request_start=start))
            old_ends.append(end['readout'])
        checkpoint = run / (hashlib.sha256('|'.join(k).encode()).hexdigest() + '.json')
        source = {'extraction_config': configs[k[0]], 'input_ids_sha256': ids_sha(ep['ids']),
                  'metadata_sha256': hashlib.sha256(json.dumps(groups[k], sort_keys=True).encode()).hexdigest(),
                  'token_indices': [m.tolist() for m in masks], 'endpoint_indices': old_ends}
        if checkpoint.exists():
            previous = json.loads(checkpoint.read_text())
            if (previous['fingerprint'] != identity or previous['ids'] != [r['id'] for r, *_ in groups[k]]
                    or previous['source'] != source):
                raise ValueError('Resume checkpoint has other code/config/rows')
            continue
        jobs.append((k, ep, masks, old_ends, checkpoint, source))
    if not jobs:
        return {'shard': shard, 'epochs': len(mine), 'new_epochs': 0, 'fingerprint': identity}
    aisi, aisi_info = frozen.load_aisi()
    q2w, q2b, q2_info = frozen.load_q2()
    directions = frozen.directions(aisi, q2w)
    lm = AutoModelForCausalLM.from_pretrained(cfg['model'], revision=cfg['revision'], local_files_only=True,
                                            dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa').eval()
    base = lm.model
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    started = time.monotonic()
    for k, ep, masks, old_ends, checkpoint, source in jobs:
        check_writer()
        values, parity, hooks = {}, {}, []
        # Read only the selected command tokens and the legacy endpoints, not every activation.
        indices = np.unique(np.concatenate([*masks, old_ends]))
        idx = torch.tensor(indices, device='cuda', dtype=torch.long)
        local_masks = [np.searchsorted(indices, m) for m in masks]
        local_ends = np.searchsorted(indices, old_ends)
        for layer, (names, w) in directions.items():
            def capture(module, args, output, layer=layer, names=names, w=w):
                h = (output[0] if isinstance(output, tuple) else output)[0].index_select(0, idx).float().cpu().numpy()
                b = np.array([q2b if n == 'q2_apollo_l31_at_block31' else 0.0 for n in names])
                values[str(layer)] = {'columns': names,
                    'scores': [affine_mean(h[m], w, b).tolist() for m in local_masks]}
                # The old dump quantized bf16 outputs to fp16 before scoring; match it exactly here.
                endpoint_h = h[local_ends].astype(np.float16).astype(np.float32)
                old = []
                for r, meta, _c, _end in groups[k]:
                    x, _arm, chunk, row = r['loc']
                    path = Path('/acts') / x / 'by_layer' / f'L{layer}' / f'{chunk}.npy'
                    old.append(np.load(path, mmap_mode='r', allow_pickle=False)[row])
                old = np.asarray(old, dtype=np.float32)
                if not np.isfinite(endpoint_h).all() or not np.isfinite(old).all():
                    raise ValueError('Non-finite legacy endpoint activations')
                old_scores, new_scores = old @ w + b, endpoint_h @ w + b
                error = float(np.max(np.abs(old_scores - new_scores)))
                parity[str(layer)] = {'max_abs_score_error': error,
                                      'max_abs_activation_error': float(np.max(np.abs(old - endpoint_h)))}
                if error > cfg['endpoint_score_atol']:
                    raise ValueError(f'Legacy endpoint parity failed at block {layer}: {error}')
                values[str(layer)]['endpoint_scores'] = new_scores.tolist()
            hooks.append(base.layers[layer].register_forward_hook(capture))
        ids = ep['ids'] + [pad_id] * (-len(ep['ids']) % configs[k[0]]['pad_to'])
        try:
            with torch.inference_mode():
                base(input_ids=torch.tensor([ids], device='cuda'), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        record = {'fingerprint': identity, 'config': cfg, 'readout': READOUT, 'epoch': list(k),
                  'ids': [r['id'] for r, *_ in groups[k]], 'token_counts': [len(m) for m in masks],
                  'token_indices': [m.tolist() for m in masks], 'input_ids_sha256': ids_sha(ep['ids']), 'source': source,
                  'aisi': aisi_info, 'q2': q2_info, 'layers': values, 'parity': parity}
        tmp = checkpoint.with_suffix(f'.{attempt}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(record, allow_nan=False))
        check_writer()
        os.replace(tmp, checkpoint)
        out_vol.commit()
        print(json.dumps({'shard': shard, 'completed_epoch': list(k), 'calls': len(masks),
                          'elapsed_s': round(time.monotonic()-started), 'parity': parity}), flush=True)
    return {'shard': shard, 'epochs': len(mine), 'new_epochs': len(jobs), 'fingerprint': identity}


@app.local_entrypoint()
def main(config: str, rows: str, pilot: bool = True, attempt: int = 1):
    cfg = json.loads(Path(config).read_text())
    raw = Path(rows).read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg['rows_sha256']:
        raise ValueError('Frozen cohort rows do not match the registered manifest')
    selected = json.loads(raw)['rows']
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate frozen row IDs')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                      check=True, capture_output=True, text=True).stdout.strip():
        raise ValueError('Commit scoring changes before launching')
    if attempt < 1 or cfg['shards'] not in (1, 2):
        raise ValueError('Require a positive attempt and one or two shards')
    sources = (Path(__file__), ROOT / 'src/probe_shift/deception_tokens.py', EXPERIMENT / 'score_frozen.py')
    for p in sources:
        subprocess.run(['git', 'ls-files', '--error-unmatch', str(p.relative_to(ROOT))], cwd=ROOT,
                       check=True, capture_output=True)
    cfg |= {'attempt': attempt, 'pilot': pilot, 'git_commit': commit, 'readout': READOUT,
            'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                sources}}
    if pilot:
        cfg['run'] += '_pilot'
    answers = list(replay.starmap([(cfg, selected, i) for i in range(cfg['shards'])], return_exceptions=True))
    failed = [repr(a) for a in answers if isinstance(a, BaseException)]
    print(json.dumps({'volume': OUTPUT, 'run': cfg['run'], 'failed_shards': failed,
                      'completed_shards': [a for a in answers if not isinstance(a, BaseException)]}))
    if failed:
        raise RuntimeError(f'{len(failed)} shards failed; incomplete scores cannot be analyzed')
