"""Cached OpenRouter labeling with a shared, crash-safe spending ledger.

No transcript command is executed. Only IDs and numeric accounting appear in logs.
The cache holds sensitive requests/responses and must stay outside Git.
"""
import argparse
import asyncio
import fcntl
import hashlib
import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def stamp():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    tmp.chmod(0o600)
    tmp.replace(path)


class NonJSONResponse(ValueError):
    """Malformed HTTP payload, distinct from a model's structured reply or refusal."""
    def __init__(self, body):
        super().__init__('non_json_http_payload')
        self.body = body


def request(url, body=None, key=None, timeout=600):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise NonJSONResponse(raw) from error


def label_text(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
        if text[:4].lower() == 'json':
            text = text[4:].lstrip()
    return text


def validate(response, ids, kind):
    if response.get('error') or not response.get('choices'):
        raise ValueError('provider_error_or_missing_choices')
    choice = response['choices'][0]
    if choice.get('finish_reason') in ('length', 'content_filter') or choice['message'].get('refusal'):
        raise ValueError('truncated_or_refused')
    text = label_text(choice['message'].get('content') or '')
    labels = json.loads(text)['labels']
    got = [r['id'] for r in labels]
    if len(got) != len(set(got)) or set(got) != set(ids):
        raise ValueError('missing_duplicate_or_invented_ids')
    allowed = {'sabotage', 'not_sabotage', 'unclear'} if kind == 'sabotage' else {0, 1}
    for row in labels:
        if row['label'] not in allowed or row['confidence'] not in ('sure', 'unsure'):
            raise ValueError('invalid_label_or_confidence')
        if kind != 'sabotage' and type(row['label']) is not int:
            raise ValueError('credential_must_be_integer')
    return labels


def prices_for(model, input_bound):
    prices = model['pricing']
    pin, pout = float(prices['prompt']), float(prices['completion'])
    for override in prices.get('overrides', []):
        if input_bound >= int(override['min_prompt_tokens']):
            pin, pout = float(override['prompt']), float(override['completion'])
    return pin, pout


def retryable(entry, cache):
    if entry['status'] in ('transport_error', 'protocol_error', 'interrupted_unknown_charge'):
        return True
    if entry['status'] == 'http_error':
        return entry.get('http_status') in (408, 429, 500, 502, 503, 504)
    if entry['status'] == 'invalid_response':
        path = cache / 'responses' / (entry['request_id'] + '.json')
        if not path.exists():
            return False
        response = json.loads(path.read_text())
        choice = (response.get('choices') or [{}])[0]
        # Never retry or rephrase a refusal. Retry truncation or a structured label ID error only.
        message = choice.get('message') or {}
        if message.get('refusal') or choice.get('finish_reason') == 'content_filter':
            return False
        provider_error = response.get('error') or choice.get('error')
        if provider_error:
            return isinstance(provider_error, dict) and provider_error.get('code') in (408, 429, 500, 502, 503, 504)
        if choice.get('finish_reason') == 'length':
            return True
        if choice.get('finish_reason') != 'stop':
            return False
        text = label_text(message.get('content') or '')
        try:
            labels = json.loads(text)['labels']
        except (ValueError, KeyError, TypeError):
            return False
        allowed = {'sabotage', 'not_sabotage', 'unclear'} if entry['dataset'] == 'sabotage' else {0, 1}
        return isinstance(labels, list) and all(isinstance(row, dict) and row.get('label') in allowed for row in labels)
    return False


def make_jobs(config, catalog, pilot):
    jobs = []
    for dataset in config['datasets']:
        cache = Path(dataset['prepared'])
        packets = json.loads((cache / 'packets.json').read_text())
        # Pilot uses the median and largest packet, exposing truncation/context problems early.
        if pilot:
            ordered = sorted(packets, key=lambda p: p['bytes'])
            packets = [ordered[len(ordered) // 2], ordered[-1]]
        prompt = Path(dataset['prompt']).read_text()
        for packet in packets:
            path = cache / 'inputs' / (packet['packet'] + '.json')
            assert digest(path.read_bytes()) == packet['sha256'], 'Prepared input changed'
            content = 'TRANSCRIPT_DATA\n' + path.read_text() + '\nEND_TRANSCRIPT_DATA'
            for model in config['models']:
                body = {'model': model, 'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}],
                        'max_tokens': dataset['max_tokens'], 'reasoning': {'effort': config['reasoning_effort']},
                        'temperature': 0, 'response_format': {'type': 'json_object'},
                        'provider': {'allow_fallbacks': True, 'data_collection': 'deny'}, 'usage': {'include': True}}
                # One token per UTF-8 byte is conservative; reserve output including reasoning at the cap.
                bound = len(prompt.encode()) + len(content.encode()) + 2048
                assert bound + dataset['max_tokens'] <= catalog[model]['context_length'], 'Input too large; never truncate'
                pin, pout = prices_for(catalog[model], bound)
                reserve = (bound * pin + dataset['max_tokens'] * pout + float(catalog[model]['pricing'].get('request', 0))) * 1.05
                request_id = digest(json.dumps(body, sort_keys=True).encode())
                jobs.append({'request_id': request_id, 'dataset': dataset['kind'], 'packet': packet['packet'],
                             'model': model, 'ids': packet['ids'], 'body': body, 'reserve': reserve,
                             'estimated': (bound / 3.5 * pin + (len(packet['ids']) * 35 + 1000) * pout) * 1.05})
    # Interleave the datasets so neither monopolizes the budget.
    jobs.sort(key=lambda j: digest((j['dataset'] + j['packet'] + j['model']).encode()))
    return jobs


async def run(a):
    os.umask(0o077)
    a.cache.mkdir(parents=True, exist_ok=True)
    with (a.cache / 'budget.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await run_locked(a)


async def run_locked(a):
    runner_sha256 = digest(Path(__file__).read_bytes())
    config_sha256 = digest(a.config.read_bytes())
    config = json.loads(a.config.read_text())
    catalog = {m['id']: m for m in (await asyncio.to_thread(request, 'https://openrouter.ai/api/v1/models'))['data']}
    assert len(config['models']) == len(set(config['models']))
    selected_catalog = {m: catalog[m] for m in config['models']}
    catalog_path = a.cache / ('catalog_' + datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S') + '.json')
    save(catalog_path, selected_catalog)
    jobs = make_jobs(config, catalog, a.pilot)
    all_jobs = make_jobs(config, catalog, False)
    estimate = {'requests': len(jobs), 'full_requests': len(all_jobs), 'estimated_full_usd': round(sum(j['estimated'] for j in all_jobs), 2),
                'conservative_full_reserve_usd': round(sum(j['reserve'] for j in all_jobs), 2), 'budget_usd': config['budget_usd']}
    print(json.dumps(estimate), flush=True)
    save(a.cache / ('estimate_pilot.json' if a.pilot else 'estimate_full.json'), estimate)
    if a.dry_run:
        return
    from dotenv import dotenv_values
    key = dotenv_values(a.env_file).get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('OPENROUTER_API_KEY absent from the authorized env file')
    ledger_path = a.cache / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    for item in ledger.values():
        if item['status'] == 'inflight':
            item['status'] = 'interrupted_unknown_charge'
    save(ledger_path, ledger)
    pending = []
    for job in jobs:
        rid = job['request_id']
        if rid not in ledger:
            pending.append(job)
        elif getattr(a, 'retry_transient', False) and retryable(ledger[rid], a.cache):
            retry_id = digest((rid + ':retry-1').encode())
            if retry_id not in ledger:
                pending.append(job | {'request_id': retry_id, 'retry_of': rid})
    running = {}
    started = time.monotonic()

    async def one(job):
        rid = job['request_id']
        save(a.cache / 'requests' / (rid + '.json'), job['body'])
        entry = {'request_id': rid, 'dataset': job['dataset'], 'packet': job['packet'], 'model': job['model'],
                 'status': 'inflight', 'charge_or_reserve_usd': job['reserve'], 'started_at_pt': stamp()}
        entry.update(body_sha256=digest(json.dumps(job['body'], sort_keys=True).encode()), retry_of=job.get('retry_of'),
                     runner_sha256=runner_sha256, config_sha256=config_sha256)
        ledger[rid] = entry
        save(ledger_path, ledger)
        response = None
        try:
            response = await asyncio.to_thread(request, 'https://openrouter.ai/api/v1/chat/completions', job['body'], key, config['timeout_seconds'])
            save(a.cache / 'responses' / (rid + '.json'), response)
            cost = (response.get('usage') or {}).get('cost')
            if cost is not None:
                entry['reported_cost_usd'] = float(cost)
                entry['charge_or_reserve_usd'] = float(cost)
            labels = validate(response, job['ids'], job['dataset'])
            entry.update(status='ok', labels=labels, provider=response.get('provider'), response_model=response.get('model'), usage=response.get('usage'))
        except urllib.error.HTTPError as error:
            entry.update(status='http_error', http_status=error.code)
            # Full provider diagnostics remain private; no error-body text is printed.
            (a.cache / 'responses').mkdir(exist_ok=True)
            (a.cache / 'responses' / (rid + '.http.txt')).write_bytes(error.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException) as error:
            entry.update(status='transport_error', error_type=type(error).__name__)
        except NonJSONResponse as error:
            entry.update(status='protocol_error', error_type=type(error).__name__)
            (a.cache / 'responses').mkdir(exist_ok=True)
            (a.cache / 'responses' / (rid + '.http.bin')).write_bytes(error.body)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            entry.update(status='invalid_response', error_type=type(error).__name__)
        entry['finished_at_pt'] = stamp()
        save(ledger_path, ledger)
        return entry

    while pending or running:
        charged = sum(e['charge_or_reserve_usd'] for e in ledger.values())
        for job in list(pending):
            if len(running) >= config['concurrency']:
                break
            if charged + job['reserve'] <= config['budget_usd']:
                pending.remove(job)
                task = asyncio.create_task(one(job))
                running[task] = job
                charged += job['reserve']
        if not running:
            break
        done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            entry = task.result()
            running.pop(task)
            statuses = Counter(e['status'] for e in ledger.values())
            print(json.dumps({'completed': sum(v for k, v in statuses.items() if k != 'inflight'), 'statuses': statuses,
                              'charge_or_reserve_usd': round(sum(e['charge_or_reserve_usd'] for e in ledger.values()), 4),
                              'last_dataset': entry['dataset'], 'last_model': entry['model'], 'last_status': entry['status'],
                              'elapsed_seconds': round(time.monotonic() - started)}), flush=True)
    summary = {'finished_at_pt': stamp(), 'statuses': dict(Counter(e['status'] for e in ledger.values())),
               'charge_or_reserve_usd': sum(e['charge_or_reserve_usd'] for e in ledger.values()),
               'budget_skipped': len(pending), 'pilot': a.pilot, 'config_sha256': config_sha256,
               'runner_sha256': runner_sha256, 'retry_transient': getattr(a, 'retry_transient', False)}
    suffix = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d_%H%M%S')
    save(a.cache / ('summary_' + ('pilot_' if a.pilot else 'full_') + suffix + '.json'), summary)
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--env-file', type=Path)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--retry-transient', action='store_true', help='Retry network/format failures once; never retry refusals')
    a = p.parse_args()
    assert a.dry_run or a.env_file, 'Specify the user-authorized env file'
    asyncio.run(run(a))


if __name__ == '__main__':
    main()
