"""Create the hashed numeric-input bundle consumed by run_pilot.py; no compute or upload."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main(root, baseline, run, pad_token_id):
    destination = root / 'input_manifest.json'
    if destination.exists():
        raise FileExistsError('Use a fresh input directory; do not replace a frozen manifest')
    config = json.loads((root / 'manifest.json').read_text())['config']
    config['pad_token_id'] = pad_token_id
    target = root / 'baseline.json'
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(baseline, target)
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in ('real_inputs.jsonl', 'synthetic_inputs.jsonl', 'baseline.json')}
    destination.write_text(json.dumps({'config': config, 'files': files, 'run': run}, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--pad-token-id', type=int, required=True)
    args = parser.parse_args()
    main(args.input, args.baseline, args.run, args.pad_token_id)
