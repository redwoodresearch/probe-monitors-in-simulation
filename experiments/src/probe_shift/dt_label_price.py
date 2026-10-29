"""Price prepared command packets from a saved public catalog, without model calls."""
import argparse
import json
from pathlib import Path
from probe_shift.dt_label_prepare import sha, write_json


def prices(model, tokens):
    p = model['pricing']
    pin, pout = float(p['prompt']), float(p['completion'])
    for override in p.get('overrides', []):
        if tokens >= override['min_prompt_tokens']:
            pin, pout = float(override['prompt']), float(override['completion'])
    return pin, pout, float(p.get('request', 0))


def main():
    p = argparse.ArgumentParser()
    for name in ('prepared', 'config', 'prompt', 'catalog', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    catalog = json.loads(a.catalog.read_text())
    packets = json.loads((a.prepared / 'packets.json').read_text())
    prompt_bytes = len(a.prompt.read_bytes()) + 200
    estimates, reserves = {}, {}
    for model in cfg['models']:
        m = catalog[model]
        if cfg['reasoning_effort'][model] not in m['reasoning']['supported_efforts']:
            raise ValueError('Unsupported reasoning effort')
        estimated, reserve = 0, 0
        for packet in packets:
            if sha(a.prepared / 'inputs' / (packet['packet'] + '.json')) != packet['sha256']:
                raise ValueError('Prepared input changed')
            token_bound = packet['bytes'] + prompt_bytes + 2048
            if token_bound + cfg['max_tokens'] > m['context_length']:
                raise ValueError('Packet exceeds conservative context bound')
            estimated_input = (packet['bytes'] + prompt_bytes) / cfg['estimated_input_bytes_per_token']
            estimated_output = len(packet['ids']) * cfg['estimated_output_tokens_per_call'] + cfg['estimated_reasoning_tokens_per_request']
            if estimated_output > cfg['max_tokens']:
                raise ValueError('Estimated output exceeds configured maximum')
            pin, pout, request = prices(m, estimated_input)
            estimated += estimated_input * pin + estimated_output * pout + request
            pin, pout, request = prices(m, token_bound)
            reserve += (token_bound * pin + cfg['max_tokens'] * pout + request) * 1.05
        estimates[model] = round(estimated, 2)
        reserves[model] = round(reserve, 2)
    rows = sum(len(p['ids']) for p in packets)
    escalation_rows = rows * cfg['estimated_escalation_fraction']
    escalation = 0
    for model in cfg['adjudication_reviewers']:
        pin, pout, request = prices(catalog[model], cfg['estimated_adjudication_input_tokens_per_call'])
        escalation += escalation_rows * (cfg['estimated_adjudication_input_tokens_per_call'] * pin + cfg['estimated_adjudication_output_tokens_per_call'] * pout + request)
    base = sum(estimates.values())
    result = {'rows': rows, 'packets': len(packets), 'panel_requests': len(packets) * len(cfg['models']),
              'estimated_panel_usd': round(base, 2), 'panel_per_model_usd': estimates,
              'estimated_two_review_escalation_usd': round(escalation, 2),
              'estimated_panel_plus_escalation_usd': round(base + escalation, 2),
              'with_buffer_usd': round((base + escalation) * (1 + cfg['cost_buffer_fraction']), 2),
              'all_requests_at_maximum_reserve_usd': round(sum(reserves.values()), 2),
              'catalog_sha256': sha(a.catalog), 'config_sha256': sha(a.config), 'prompt_sha256': sha(a.prompt),
              'packets_sha256': sha(a.prepared / 'packets.json'),
              'assumptions': {k: v for k, v in cfg.items() if k.startswith('estimated_') or k == 'cost_buffer_fraction'},
              'caveat': 'Not measured inference cost. Tokenization, reasoning, failures, extra context and final adjudication may change spend. A confirmed hard cap must be enforced.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
