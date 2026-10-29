"""Free checks for the reproducibility fixes; never initialize Modal or run a model."""
import ast
import datetime
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def function(name, source, namespace):
    tree = ast.parse((HERE / source).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), source, 'exec'), namespace)
    return namespace[name]


def main():
    window = function('billing_window', 'run_guarded.py', {'datetime': datetime})
    assert window({'stages': [{'start_utc': '2026-01-01T00:00:00+00:00'}]}, datetime.date(2026, 10, 10))[0] == '2025-12-31'
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run = root / 'run'
        run.mkdir()
        (run / 'input_manifest.json').write_text('{"version":1}')
        namespace = {'Path': Path, 'json': json, 'hashlib': hashlib, 'ACTS': root,
                     'IMPLEMENTATION_SHA': 'frozen', 'vol': SimpleNamespace(commit=lambda: None)}
        bind = function('bind_checkpoint', 'run_pilot.py', namespace)
        expected = bind('run', 'score', [], probe_sha256='original')
        assert bind('run', 'score', [run / 'saved.json'], probe_sha256='original') == expected
        for stage, existing, probe in [('score', [], 'changed'), ('legacy', [run / 'saved.json'], 'original')]:
            try:
                bind('run', stage, existing, probe_sha256=probe)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed or unbound checkpoint accepted')
        (run / 'input_manifest.json').write_text('{"version":2}')
        try:
            bind('run', 'score', [], probe_sha256='original')
        except ValueError:
            pass
        else:
            raise AssertionError('Changed inputs accepted')
        (root / 'manifest.json').write_text('{"config":{"model":"test"}}')
        for name in ('real_inputs.jsonl', 'synthetic_inputs.jsonl'):
            (root / name).write_text('{"ids":[1,2]}\n')
        baseline = root / 'source.json'
        baseline.write_text('{"probes":{}}')
        stage = function('main', 'stage_inputs.py', {'Path': Path, 'json': json, 'hashlib': hashlib, 'shutil': __import__('shutil')})
        stage(root, baseline, 'fresh-run', 123)
        manifest = json.loads((root / 'input_manifest.json').read_text())
        assert manifest['config']['pad_token_id'] == 123
        assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in manifest['files'].items())
    print('PASS: historical billing, immutable checkpoint provenance, and numeric input staging')


if __name__ == '__main__':
    main()
