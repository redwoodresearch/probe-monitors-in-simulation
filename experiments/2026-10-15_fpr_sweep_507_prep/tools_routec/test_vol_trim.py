"""vol_trim.py: a positive control (content + exact trailer passes, byte for byte) and three negatives (short stream, wrong trailer, size 0).
Synthetic bytes only; no Modal call, no repo state.

  uv run --offline --no-project --python 3.12 python -I experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_vol_trim.py
"""

import subprocess
import sys
from pathlib import Path

TRIM = Path(__file__).resolve().parent / "vol_trim.py"
TRAILER = "✓ Finished downloading files to local!\n".encode()


def run(size, data):
    return subprocess.run([sys.executable, "-I", str(TRIM), str(size)], input=data, capture_output=True)


def main():
    body = bytes(range(256)) * 3
    ok = run(len(body), body + TRAILER)
    assert ok.returncode == 0 and ok.stdout == body, "positive control"
    assert run(len(body) + 1, body + TRAILER).returncode != 0, "size off by one"
    bad = run(len(body), body + TRAILER[:-1] + b"?")
    assert bad.returncode != 0 and bad.stdout == b"", "wrong trailer writes nothing"
    assert run(0, TRAILER).returncode != 0, "size 0"
    print("ALL PASS (4 cases)")


if __name__ == "__main__":
    main()
