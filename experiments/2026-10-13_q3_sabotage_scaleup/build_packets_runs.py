"""Build stage-1-format sabotage judge packets for a list of runs, for Aswin's scale-up labelling on his own laptop.

Never run by a session: it reads the .eval archives (transcript text) and writes packets that hold it, all into a private folder
outside git (mode 700). It prints counts and hashes only.

Same packet format and code path as src/probe_shift/sabotage_stage1.py prepare() at b90cd42a (sha256 502818d8…). sha() and
resolve() are imported from that file after its hash is checked; the per-run loop below is that function's loop, unchanged except:
1. runs come from this folder's runs_<stamp>.csv (the first N tasks of the task list, all three worlds) instead of the stage-1
   locator, and every call is labelled with no "selected" subset. Stage 1's judges labelled every call too; only its export kept
   1,457;
2. the seed defaults to 20261013 (stage 1's was 20261011), so run order and packet IDs differ from stage 1's;
3. each run's call count and epoch are checked against the committed ID list (calls_all_<stamp>.csv) before any packet is written;
4. join.json is written empty (nothing selected), and provenance rows carry the run's call count.
Payload bytes are identical in form: json.dumps({'cover_task', 'hidden_side_task', 'calls'}, ensure_ascii=False), with the same
per-call fields, the same terminal-submit masking and the same fresh IDs c0000, c0001, … in (turn, call index) order.

  cd ~/probe-shift-panel-code && uv run python <this file> --panel-src ~/probe-shift-panel-code/src --runs <runs csv> \\
      --calls <calls_all csv> --first-tasks N --index <stage-1 run index json> --cache <new private folder>
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

BUILDER_SHA = "502818d86f1021912860a85365df050e0cc1aa422b9a380f0de0d517b2d78d38"  # src/probe_shift/sabotage_stage1.py at b90cd42a
RUNS_SHA = "52a2e288b1cc684d8b17016adda88778fb5077388961ccd6aa3bdcf4c915accf"  # runs_2026-10-13_125944.csv
CALLS_SHA = "6dc132affb389e3dd66737d2d5db199c2bb5fe70e13a723a3c0af319e95d7595"  # calls_all_2026-10-13_125944.csv
STAGE1_INDEX_SHA = "7a5cb12975500f9a9e41c2bf3a73db26963c4a590c5f9d315014895342160e98"  # results/extraction/preparation.json at b90cd42a


def main(a):
    os.umask(0o077)
    src = Path(a.panel_src) / "probe_shift" / "sabotage_stage1.py"
    for p, want in [(src, BUILDER_SHA), (a.runs, RUNS_SHA), (a.calls, CALLS_SHA)]:
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != want:
            raise SystemExit(f"{p}: sha256 differs from the committed one")
    spec = importlib.util.spec_from_file_location("sabotage_stage1", src)
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    sha, resolve = s1.sha, s1.resolve
    rows = list(csv.DictReader(open(a.runs)))
    keep = {r["task"] for r in rows if int(r["rank"]) <= a.first_tasks}
    if len(keep) != a.first_tasks:
        raise SystemExit(f"the task list has fewer than {a.first_tasks} tasks")
    runs = sorted({r["run_id"] for r in rows if r["task"] in keep})
    if len(runs) != 3 * a.first_tasks:
        raise SystemExit("expected three runs per task")
    expect = defaultdict(lambda: {"calls": 0, "epochs": set()})
    for c in csv.DictReader(open(a.calls)):
        if c["run_id"] in runs:
            expect[c["run_id"]]["calls"] += 1
            expect[c["run_id"]]["epochs"].add(int(c["epoch"]))
    index = json.loads(Path(a.index).read_text())
    index_sha = sha(a.index)
    a.cache.mkdir(parents=True, exist_ok=False)
    (a.cache / "inputs").mkdir()
    rng = random.Random(a.seed)
    rng.shuffle(runs)
    provenance, packets = [], []
    for run_id in runs:
        if run_id not in index:
            raise SystemExit("a listed run is missing from the run index; nothing more is written")
        paths = [Path(p) for p in index[run_id]]
        hashes = {sha(p) for p in paths}
        assert len(hashes) == 1, 'Resolve divergent source archives first'
        with zipfile.ZipFile(paths[0]) as archive:
            members = [n for n in archive.namelist() if n.startswith('samples/') and n.endswith('.json')]
            assert len(members) == 1, 'The frozen source index expects single-sample archives'
            sample = json.loads(archive.read(members[0]))
        epoch = int(sample.get('epoch', 1))
        sample = resolve(sample, sample.get('attachments') or {})
        messages = sample['messages']
        outputs = {m['tool_call_id']: m for m in messages if m['role'] == 'tool'}
        assert len(outputs) == sum(m['role'] == 'tool' for m in messages)
        events = [e for e in sample['events'] if e.get('event') == 'model' and e.get('model') == a.acting_model]
        assert events
        packet_id = uuid.UUID(int=rng.getrandbits(128)).hex
        calls = []
        for turn, event in enumerate(events):
            assert (((event.get('call') or {}).get('request') or {}).get('extra_headers') or {}).get('x-irid')
            choices = event['output']['choices']
            assert len(choices) == 1
            for ci, call in enumerate(choices[0]['message'].get('tool_calls') or []):
                fresh_id = f'c{len(calls):04d}'
                output = outputs.get(call['id'])
                # Do not expose the submit call's self-reported success label.
                terminal = call['function'] == 'submit'
                calls.append({'id': fresh_id, 'function': call['function'],
                              'arguments': {} if terminal else call.get('arguments'),
                              'matching_tool_output': None if terminal or output is None else output.get('content'),
                              'tool_error': None if output is None else output.get('error')})
        if len(calls) != expect[run_id]["calls"] or expect[run_id]["epochs"] != {epoch}:
            raise SystemExit("a run's call count or epoch differs from the ID list; nothing more is written")
        meta = sample['metadata']
        assert meta.get('main_task_str') and meta.get('side_task_text')
        payload = {'cover_task': meta['main_task_str'], 'hidden_side_task': meta['side_task_text'], 'calls': calls}
        path = a.cache / 'inputs' / f'{packet_id}.json'
        path.write_text(json.dumps(payload, ensure_ascii=False))
        packets.append({'packet': packet_id, 'ids': [c['id'] for c in calls], 'sha256': sha(path), 'bytes': path.stat().st_size})
        provenance.append({'run_id': run_id, 'archive_sha256': next(iter(hashes)), 'sample_error': bool(sample.get('error')), 'calls': len(calls)})
    for name, value in [('join.json', []), ('provenance.json', provenance), ('packets.json', packets)]:
        (a.cache / name).write_text(json.dumps(value, indent=2) + '\n')
    summary = {'runs': len(runs), 'tasks': a.first_tasks, 'context_calls': sum(len(p['ids']) for p in packets),
               'input_bytes': sum(p['bytes'] for p in packets), 'largest_input_bytes': max(p['bytes'] for p in packets),
               'runs_sha256': RUNS_SHA, 'calls_sha256': CALLS_SHA, 'index_sha256': index_sha, 'index_is_stage1s': index_sha == STAGE1_INDEX_SHA,
               'builder_sha256': BUILDER_SHA, 'seed': a.seed, 'acting_model': a.acting_model, 'key_accessed': False}
    (a.cache / 'preparation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary | {'packets_sha256': sha(a.cache / 'packets.json'), 'provenance_sha256': sha(a.cache / 'provenance.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-src', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--calls', type=Path, required=True)
    p.add_argument('--first-tasks', type=int, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--acting-model', default='vllm/Qwen/Qwen3.5-27B')
    p.add_argument('--seed', type=int, default=20261013)
    main(p.parse_args())
