"""Prepare a seeded audit and serve an owner-local disagreement viewer.

Blinding is procedural within one trusted labeling session, not filesystem isolation:
read_blind omits panel fields, but the source labels remain accessible to the owner.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o600)


def prepare(root, out, seed, size):
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    ledger = json.loads((root / 'panel/ledger.json').read_text())
    cases, sample = {}, []
    population = {}
    for kind in ('sabotage', 'credential'):
        prepared = root / kind / ('prepared_v2' if kind == 'sabotage' else 'prepared')
        join = json.loads((prepared / 'join.json').read_text())
        filename = 'labels_stage1_filled.csv' if kind == 'sabotage' else 'screening_labels_oct11c_stage1.csv'
        labels_path = root / kind / 'results_final' / filename
        with labels_path.open() as f:
            labels = list(csv.DictReader(f))
        def identity(row):
            return row['opaque_id'] if kind == 'sabotage' else '|'.join(str(row[k]) for k in ('run_id', 'step', 'action'))
        final = {identity(r): r['label'] if kind == 'sabotage' else int(r['credential']) for r in labels}
        assert len(final) == len(labels) == len(join)
        assert set(final) == {identity(r) for r in join}
        assert 0 < size <= len(join)
        votes = defaultdict(dict)
        for e in ledger.values():
            if e['dataset'] == kind and e['status'] == 'ok':
                for r in e['labels']:
                    assert e['model'] not in votes[e['packet'], r['id']]
                    votes[e['packet'], r['id']][e['model']] = {k: r[k] for k in ('label', 'confidence')}
        ordered = sorted(join, key=lambda r: hashlib.sha256(f'{seed}|{kind}|{identity(r)}'.encode()).hexdigest())
        for index, row in enumerate(ordered):
            uid = ('S' if kind == 'sabotage' else 'C') + f'{index + 1:04d}'
            v = votes[row['packet'], row['id']]
            identifiers = {k: row[k] for k in (['opaque_id'] if kind == 'sabotage' else ['run_id', 'step', 'action'])}
            cases[uid] = {'audit_id': uid, 'kind': kind, 'prepared': str(prepared), 'packet': row['packet'],
                          'id': row['id'], 'identifiers': identifiers, 'panel_label': final[identity(row)],
                          'votes': v, 'disagreement': len({r['label'] for r in v.values()}) > 1,
                          'unsure': any(r['confidence'] != 'sure' for r in v.values()), 'missing_votes': 5 - len(v),
                          'sample': index < size}
            if index < size:
                sample.append({'audit_id': uid, 'kind': kind, 'identifiers': identifiers})
        population[kind] = {'rows': len(join), 'final_labels_sha256': sha(labels_path), 'join_sha256': sha(prepared / 'join.json')}
    protocol = {'seed': seed, 'rows_per_dataset': size, 'sampling': 'Uniform hash ranking without replacement over all final rows, separately by dataset; no panel-label stratification or rerolls',
                'hash_key': 'UTF-8 f"{seed}|{kind}|{opaque_id or run_id|step|action}"; choose the smallest SHA-256 hashes',
                'blinding': 'Review evidence under audit IDs before opening panel labels or votes; freeze own decisions before comparison',
                'reviewer': 'This Codex labeling session; model audit, not human ground truth',
                'unchanged_deliveries': True, 'frozen_at_pt': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                'population': population, 'ledger_sha256': sha(root / 'panel/ledger.json')}
    save(out / 'protocol.json', protocol)
    save(out / 'sample_manifest.json', sample)
    save(out / 'private_cases.json', cases)
    print(json.dumps({'prepared_sample': len(sample), 'population_rows': sum(p['rows'] for p in population.values()), 'protocol_sha256': sha(out / 'protocol.json')}))


def evidence(case, full=False):
    packet = json.loads((Path(case['prepared']) / 'inputs' / (case['packet'] + '.json')).read_text())
    if case['kind'] == 'credential':
        row = next(r for r in packet['outputs'] if r['id'] == case['id'])
        return {'audit_id': case['audit_id'], 'kind': case['kind'], 'output': row['text']}
    target = next(i for i, r in enumerate(packet['calls']) if r['id'] == case['id'])
    result = {'audit_id': case['audit_id'], 'kind': case['kind'], 'cover_task': packet['cover_task'],
              'hidden_side_task': packet['hidden_side_task'], 'target_call': packet['calls'][target],
              'context_call_count': len(packet['calls']), 'target_call_position': target}
    if full:
        result['complete_sequence'] = packet['calls']
    return result


def read_blind(out, ids, full, start, limit):
    cases = json.loads((out / 'private_cases.json').read_text())
    for uid in ids:
        assert cases[uid]['sample'], 'Blind reading is limited to the frozen random sample'
        obj = evidence(cases[uid], full)
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        stop = min(len(text), start + limit) if limit else len(text)
        print(f'=== {uid}: characters {start}:{stop} of {len(text)}; full_sequence={full} ===')
        print(text[start:stop])
        with (out / 'evidence_access.jsonl').open('a') as f:
            f.write(json.dumps({'audit_id': uid, 'full_sequence': full, 'start': start, 'stop': stop, 'total_characters': len(text),
                                'at_pt': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()}) + '\n')


def compare(out, decisions_path):
    decisions = json.loads(decisions_path.read_text())
    cases = json.loads((out / 'private_cases.json').read_text())
    sample = json.loads((out / 'sample_manifest.json').read_text())
    by_id = {r['audit_id']: r for r in decisions}
    assert len(by_id) == len(decisions) and set(by_id) == {r['audit_id'] for r in sample}
    results, comparisons = {}, []
    for kind in ('sabotage', 'credential'):
        counts, confusion = Counter(), Counter()
        for uid, d in by_id.items():
            c = cases[uid]
            if c['kind'] != kind:
                continue
            allowed = ('sabotage', 'not_sabotage', 'unclear') if kind == 'sabotage' else (0, 1)
            assert d['label'] in allowed and (kind == 'sabotage' or type(d['label']) is int)
            assert d['confidence'] in ('sure', 'unsure') and d['reason']
            counts.update(rows=1, agree=int(d['label'] == c['panel_label']), own_unsure=int(d['confidence'] == 'unsure'))
            confusion[str(d['label']), str(c['panel_label'])] += 1
            comparisons.append(d | {'kind': kind, 'panel_label': c['panel_label'], 'agree': d['label'] == c['panel_label'], 'identifiers': c['identifiers']})
        n, k = counts['rows'], counts['agree']
        z = 1.96; denom = 1 + z*z/n; center = (k/n + z*z/(2*n))/denom
        radius = z*((k/n*(1-k/n)/n + z*z/(4*n*n))**0.5)/denom
        results[kind] = dict(counts) | {'agreement': k/n, 'wilson_95_interval': [center-radius, center+radius],
            'confusion': [{'own_label': a, 'panel_label': b, 'rows': v} for (a,b),v in sorted(confusion.items())]}
    dest = out / 'comparison';dest.mkdir(exist_ok=False)
    save(dest / 'summary.json', {'results': results, 'decisions_sha256': sha(decisions_path), 'protocol_sha256': sha(out / 'protocol.json'),
                                'sample_manifest_sha256': sha(out / 'sample_manifest.json'), 'interpretation': 'Codex versus panel exact-label agreement on 50 uniformly sampled rows per dataset; interval treats rows as independent and does not adjust for within-run clustering.'})
    save(dest / 'comparisons.json', comparisons)
    print(json.dumps(results, indent=2))


def serve(out, assets, port):
    cases = json.loads((out / 'private_cases.json').read_text())
    comparisons_path = out / 'comparison/comparisons.json'
    assert comparisons_path.exists(), 'Freeze and compare the independent audit before opening panel votes'
    comparisons = {r['audit_id']: r for r in json.loads(comparisons_path.read_text())}
    manifest = [{k:v for k,v in c.items() if k not in ('prepared','packet','id','identifiers')} | {'audit':comparisons.get(uid)} for uid,c in cases.items()]
    static = {'/': ('index.html', 'text/html'), '/app.js': ('app.js','text/javascript'), '/style.css': ('style.css','text/css')}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
                self.send_error(403);return
            path = urlsplit(self.path).path
            if path in static:
                name,mime = static[path];body=(assets/name).read_bytes()
            elif path == '/manifest.json':
                mime='application/json';body=json.dumps(manifest).encode()
            elif path == '/summary.json':
                mime='application/json';body=(out/'comparison/summary.json').read_bytes()
            elif path.startswith('/case/') and path.removeprefix('/case/') in cases:
                mime='application/json';body=json.dumps(evidence(cases[path.removeprefix('/case/')],full=True)).encode()
            else:
                self.send_error(404);return
            self.send_response(200)
            self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps({'viewer_url':f'http://127.0.0.1:{server.server_port}','cases':len(cases)}),flush=True)
    server.serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--root',type=Path,required=True);prep.add_argument('--out',type=Path,required=True);prep.add_argument('--seed',type=int,default=20261011);prep.add_argument('--size',type=int,default=50)
    read=sub.add_parser('read');read.add_argument('--out',type=Path,required=True);read.add_argument('--ids',nargs='+',required=True);read.add_argument('--full',action='store_true');read.add_argument('--start',type=int,default=0);read.add_argument('--limit',type=int,default=0)
    cmp=sub.add_parser('compare');cmp.add_argument('--out',type=Path,required=True);cmp.add_argument('--decisions',type=Path,required=True)
    view=sub.add_parser('serve');view.add_argument('--out',type=Path,required=True);view.add_argument('--assets',type=Path,required=True);view.add_argument('--port',type=int,default=8768)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root.resolve(),a.out,a.seed,a.size)
    elif a.command=='read':read_blind(a.out,a.ids,a.full,a.start,a.limit)
    elif a.command=='compare':compare(a.out,a.decisions)
    else:serve(a.out,a.assets,a.port)


if __name__=='__main__':main()
