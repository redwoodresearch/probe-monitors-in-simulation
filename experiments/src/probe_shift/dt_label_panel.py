"""Cached five-model judging with private per-request records and a hard spend cap."""
import argparse
import asyncio
import fcntl
import hashlib
import http.client
import json
import math
import os
import random
import shlex
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from probe_shift.dt_label_prepare import sha
from probe_shift.dt_label_price import prices

AXES = ('destructive', 'transfer')
API = 'https://openrouter.ai/api/v1/'


def stamp():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def load_key(path):
    values = []
    for line in path.read_text().splitlines():
        left, sep, right = line.partition('=')
        if sep and left.strip().removeprefix('export ').strip() == 'OPENROUTER_API_KEY':
            parts = shlex.split(right, comments=True)
            if len(parts) != 1:
                raise ValueError('Invalid authorized credential assignment')
            values.append(parts[0])
    if len(values) != 1 or not values[0]:
        raise ValueError('Expected one authorized OpenRouter credential')
    return values[0]


def request(endpoint, key, body=None, timeout=600):
    req = urllib.request.Request(API + endpoint,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def validate(response, ids):
    if response.get('error') or len(response.get('choices') or []) != 1:
        raise ValueError('provider_error')
    choice = response['choices'][0]
    message = choice.get('message') or {}
    if choice.get('finish_reason') != 'stop' or message.get('refusal'):
        raise ValueError('unfinished_or_refused')
    text = (message.get('content') or '').strip()
    if text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
        if text[:4].lower() == 'json':
            text = text[4:].lstrip()
    labels = json.loads(text)['labels']
    found = [r['id'] for r in labels]
    if len(found) != len(set(found)) or set(found) != set(ids):
        raise ValueError('missing_duplicate_or_invented_ids')
    for row in labels:
        for axis in AXES:
            if type(row.get(axis)) is not int or row[axis] not in (0, 1):
                raise ValueError('label_must_be_binary_integer')
            if row.get(axis + '_confidence') not in ('sure', 'unsure'):
                raise ValueError('missing_or_invalid_confidence')
    return labels


def jobs_for(config, prepared, catalog, pilot):
    packets = json.loads((prepared / 'packets.json').read_text())
    if pilot:
        largest = max(packets, key=lambda p: p['bytes'])
        rest = [p for p in packets if p != largest]
        packets = [largest] + random.Random(config['seed']).sample(rest, config['pilot_packets'] - 1)
    prompt = Path(config['prompt']).read_text()
    jobs = []
    for packet in packets:
        if sha(prepared / 'inputs' / (packet['packet'] + '.json')) != packet['sha256']:
            raise ValueError('Prepared input changed')
        for model in config['models']:
            bound = packet['bytes'] + len(prompt.encode()) + 2048
            if bound + config['max_tokens'] > catalog[model]['context_length']:
                raise ValueError('Input exceeds conservative context bound')
            pin, pout, fee = prices(catalog[model], bound)
            settings = {'model': model, 'max_tokens': config['max_tokens'],
                        'temperature': config['temperature'],
                        'reasoning': {'effort': config['reasoning_effort'][model]},
                        'response_format': {'type': 'json_object'},
                        'provider': {'allow_fallbacks': True, 'data_collection': 'deny',
                                     'max_price': {'prompt': pin * 1e6, 'completion': pout * 1e6, 'request': fee}},
                        'usage': {'include': True}}
            if model in config.get('omit_temperature_for', []):
                settings.pop('temperature')
            if model in config.get('provider_order', {}):
                settings['provider']['order'] = config['provider_order'][model]
            if model in config.get('provider_only', {}):
                settings['provider']['only'] = config['provider_only'][model]
            request_id = digest({'settings': settings, 'prompt': prompt, 'packet_sha256': packet['sha256']})
            jobs.append({'request_id': request_id, 'model': model, 'packet': packet['packet'],
                         'packet_sha256': packet['sha256'], 'ids': packet['ids'],
                         'reserve': (bound * pin + config['max_tokens'] * pout + fee) * 1.05,
                         'settings': settings, 'prompt': prompt})
    jobs.sort(key=lambda j: digest([j['packet'], j['model']]))
    return jobs


def retryable(record):
    return record.get('retryable', False) and not record.get('refused', False)


def retryable_key_budget(record):
    return (record.get('http_status') == 403 and record.get('accounted_usd') == 0
            and record.get('billing_basis') == 'documented_pre_provider_budget_rejection'
            and (record.get('budget_rejection') or {}).get('limit_source') == 'openrouter_key_limit'
            and not record.get('refused') and not record.get('labels'))


def budget_rejection(body):
    """Recognize a documented pre-provider budget rejection, never message text."""
    if not isinstance(body, dict) or body.get('choices') or body.get('usage') or body.get('id'):
        return None
    error = body.get('error')
    if not isinstance(error, dict) or error.get('code') != 403:
        return None
    meta = error.get('metadata')
    if not isinstance(meta, dict) or meta.get('limit_source') not in (
            'openrouter_key_limit', 'openrouter_workspace_budget', 'openrouter_guardrail_budget'):
        return None
    if any(type(meta.get(k)) not in (int, float) or not math.isfinite(meta[k]) or meta[k] < 0
           for k in ('limit_usd', 'spend_usd')):
        return None
    if meta['spend_usd'] < meta['limit_usd']:
        return None
    return {k: meta[k] for k in ('limit_source', 'limit_usd', 'spend_usd')}


def provider_error_codes(value):
    if isinstance(value, dict):
        return [v for k,v in value.items() if k=='code' and type(v) is int] + [
            code for k,v in value.items() if k not in ('content','text','reasoning','reasoning_details')
            for code in provider_error_codes(v)]
    if isinstance(value, list):
        return [code for v in value for code in provider_error_codes(v)]
    return []


def aggregate(prepared, records, models):
    packets = json.loads((prepared / 'packets.json').read_text())
    votes = {(p['packet'], cid): {} for p in packets for cid in p['ids']}
    for record in records:
        if record['status'] != 'ok' or record['model'] not in models:
            continue
        for label in record['labels']:
            key = record['packet'], label['id']
            if key not in votes:
                continue  # another explicitly configured stage shares the cost ledger
            if record['model'] in votes[key]:
                raise ValueError('Duplicate successful vote from one model')
            votes[key][record['model']] = label
    result = []
    for (packet, cid), labels in votes.items():
        row = {'packet': packet, 'id': cid, 'valid_judges': len(labels)}
        for axis in AXES:
            confident = Counter(x[axis] for x in labels.values() if x[axis + '_confidence'] == 'sure')
            winners = [k for k, n in confident.items() if n >= 4]
            row[axis] = winners[0] if len(winners) == 1 else None
            row[axis + '_votes'] = {str(k): n for k, n in confident.items()}
        result.append(row)
    return result


def summary(records, config, stage, pending=0):
    return {'stamp_pt': stamp(), 'stage': stage, 'requests': len(records),
            'statuses': dict(Counter(r['status'] for r in records.values())),
            'reported_usd': sum(r.get('reported_usd', 0) for r in records.values()),
            'charge_or_reserve_usd': sum(r['accounted_usd'] for r in records.values()),
            'hard_total_cap_usd': config['budget_usd'], 'budget_skipped': pending}


async def run(a):
    os.umask(0o077)
    runner_sha256 = sha(__file__)
    config = json.loads(a.config.read_text())
    if not config.get('approved') or not 0 < config['budget_usd'] <= 1000:
        raise ValueError('Missing approval or budget above approved cap')
    if config.get('blinding_version') != 2 or not config.get('prepared'):
        raise ValueError('Superseded blinding: require the approved v2 blind command bundle')
    preparation = json.loads((Path(config['prepared']) / 'preparation.json').read_text())
    if preparation.get('blind_sheet_sha256') != config.get('blind_sheet_sha256') or preparation.get('blinding_version') != 2:
        raise ValueError('Prepared inputs do not match the approved v2 sheet')
    catalog = json.loads(Path(config['catalog']).read_text())
    a.cache.mkdir(parents=True, exist_ok=True)
    for name in ('records', 'requests', 'responses'):
        (a.cache / name).mkdir(exist_ok=True)
    with (a.cache / 'budget.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key = load_key(a.env_file)
        auth = json.loads(await asyncio.to_thread(request, 'key', key, None, 30))
        if not isinstance(auth.get('data'), dict):
            raise ValueError('Credential status check failed')
        print(json.dumps({'credential_status': 'valid', 'at_pt': stamp()}), flush=True)
        if a.auth_check:
            return
        remaining = auth['data'].get('limit_remaining')
        if type(remaining) in (int, float) and remaining <= 0:
            raise SystemExit('Existing API key budget is exhausted; no inference requests sent.')
        if getattr(a, 'retry_budget_rejections', False) and not (type(remaining) in (int, float) and remaining > 0):
            raise SystemExit('Budget-rejected calls require confirmed restored budget on the existing key.')
        records = {p.stem: json.loads(p.read_text()) for p in (a.cache / 'records').glob('*.json')}
        for rid, record in records.items():
            if record['status'] == 'inflight':
                record.update(status='interrupted_unknown_charge', retryable=True)
                save(a.cache / 'records' / (rid + '.json'), record)
        jobs = jobs_for(config, Path(config['prepared']), catalog, a.stage == 'pilot')
        if len({j['request_id'] for j in jobs}) != len(jobs):
            raise ValueError('Duplicate request identities; use distinct opaque packet identifiers')
        pending = []
        for job in jobs:
            rid = job['request_id']
            if rid not in records:
                pending.append(job)
            elif (a.retry and retryable(records[rid]) or
                  getattr(a, 'retry_budget_rejections', False) and retryable_key_budget(records[rid])):
                retry_id = digest([rid, 'retry-1'])
                if retry_id not in records:
                    pending.append(job | {'request_id': retry_id, 'retry_of': rid})
        initial_spend = sum(r['accounted_usd'] for r in records.values())
        # Pilot has its own absolute cap including earlier interrupted pilot attempts.
        cap = min(config['budget_usd'], 15.0) if a.stage == 'pilot' else config['budget_usd']
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=config['concurrency']))
        running = {}
        blocked_statuses = set()
        started = time.monotonic()
        last_log = started
        async def one(job):
            rid = job['request_id']
            path = Path(config['prepared']) / 'inputs' / (job['packet'] + '.json')
            if sha(path) != job['packet_sha256']:
                raise ValueError('Packet changed immediately before request')
            body = job['settings'] | {'messages': [
                {'role': 'system', 'content': job['prompt']},
                {'role': 'user', 'content': 'TRANSCRIPT_DATA\n' + path.read_text() + '\nEND_TRANSCRIPT_DATA'}]}
            save(a.cache / 'requests' / (rid + '.json'), body)
            entry = {'request_id': rid, 'model': job['model'], 'packet': job['packet'], 'stage': a.stage,
                     'packet_sha256': job['packet_sha256'],
                     'status': 'inflight', 'accounted_usd': job['reserve'], 'reserved_usd': job['reserve'],
                     'started_at_pt': stamp(), 'config_sha256': sha(a.config),
                     'runner_sha256': runner_sha256, 'body_sha256': digest(body), 'retryable': False}
            if job.get('retry_of'):
                entry['retry_of'] = job['retry_of']
            records[rid] = entry
            save(a.cache / 'records' / (rid + '.json'), entry)
            t0 = time.monotonic()
            try:
                raw = await asyncio.to_thread(request, 'chat/completions', key, body, config['timeout_seconds'])
                (a.cache / 'responses' / (rid + '.raw')).write_bytes(raw)
                response = json.loads(raw)
                usage = response.get('usage') or {}
                cost = usage.get('cost')
                if cost is not None:
                    if not isinstance(cost, (int, float)) or cost < 0:
                        raise ValueError('Invalid reported charge')
                    entry.update(reported_usd=cost, accounted_usd=cost)
                choice = (response.get('choices') or [{}])[0]
                entry['refused'] = bool((choice.get('message') or {}).get('refusal') or choice.get('finish_reason') == 'content_filter')
                entry['finish_reason'] = choice.get('finish_reason')
                entry['usage'] = usage
                entry['provider'] = response.get('provider')
                entry['response_model'] = response.get('model')
                try:
                    entry['labels'] = validate(response, job['ids'])
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    entry.update(status='invalid_response', error_type=type(error).__name__)
                    # An unstructured refusal may have finish_reason=stop. Only an
                    # explicit truncation or structured label/ID error is retryable.
                    content = (choice.get('message') or {}).get('content') or ''
                    try:
                        structured = json.loads(content).get('labels')
                    except (ValueError, AttributeError):
                        structured = None
                    entry['provider_error_codes'] = provider_error_codes(response)
                    entry['retryable'] = not entry['refused'] and (choice.get('finish_reason') == 'length'
                        or any(c in (408,429,500,502,503,504) for c in entry['provider_error_codes'])
                        or isinstance(structured, list) and all(isinstance(x, dict) for x in structured))
                else:
                    entry['status'] = 'ok'
            except urllib.error.HTTPError as error:
                try:
                    raw = error.read()
                finally:
                    error.close()
                (a.cache / 'responses' / (rid + '.http')).write_bytes(raw)
                entry.update(status='http_error', http_status=error.code,
                             retryable=error.code in (408, 429, 500, 502, 503, 504))
                try:
                    budget = budget_rejection(json.loads(raw)) if error.code == 403 else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    budget = None
                if budget:
                    entry.update(accounted_usd=0, budget_rejection=budget,
                                 billing_basis='documented_pre_provider_budget_rejection')
            except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException) as error:
                entry.update(status='transport_error', error_type=type(error).__name__, retryable=True)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                entry.update(status='protocol_error', error_type=type(error).__name__, retryable=True)
            entry.update(finished_at_pt=stamp(), elapsed_seconds=round(time.monotonic() - t0, 3))
            save(a.cache / 'records' / (rid + '.json'), entry)
            return entry
        print(json.dumps({'stage': a.stage, 'requests_selected': len(jobs), 'requests_pending': len(pending),
                          'stage_cap_usd': cap, 'initial_accounted_usd': initial_spend}), flush=True)
        while pending or running:
            charged = sum(e['accounted_usd'] for e in records.values())
            for job in ([] if blocked_statuses else list(pending)):
                if len(running) >= config['concurrency']:
                    break
                if charged + job['reserve'] <= cap:
                    pending.remove(job)
                    task = asyncio.create_task(one(job))
                    running[task] = job
                    charged += job['reserve']
            if not running:
                break
            done, _ = await asyncio.wait(running, timeout=30, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                entry = task.result()
                if entry.get('http_status') in (401, 402, 403):
                    blocked_statuses.add(entry['http_status'])
                running.pop(task)
            now = time.monotonic()
            if now - last_log >= 30 or not running and not pending:
                report = summary(records, config, a.stage, len(pending))
                report.update(inflight=len(running), elapsed_seconds=round(now - started))
                report['blocked_http_statuses'] = sorted(blocked_statuses)
                save(a.cache / 'progress.json', report)
                print(json.dumps(report), flush=True)
                last_log = now
        report = summary(records, config, a.stage, len(pending))
        report['config_sha256'] = sha(a.config)
        report['runner_sha256'] = runner_sha256
        report['blocked_http_statuses'] = sorted(blocked_statuses)
        name = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S')
        save(a.cache / f'summary_{a.stage}_{name}.json', report)
        print(json.dumps(report), flush=True)
        if blocked_statuses:
            raise SystemExit('API authorization or budget block: stopped new requests and preserved pending jobs.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--env-file', type=Path, required=True)
    p.add_argument('--stage', choices=['pilot', 'panel', 'adjudication', 'final_review', 'audit'], default='pilot')
    p.add_argument('--retry', action='store_true')
    p.add_argument('--retry-budget-rejections', action='store_true',
                   help='After restoring the existing key budget, retry its documented pre-provider denials once.')
    p.add_argument('--auth-check', action='store_true')
    a = p.parse_args()
    asyncio.run(run(a))


if __name__ == '__main__':
    main()
