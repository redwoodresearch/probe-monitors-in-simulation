"""The fill step for the CAT/PS NOTE's VARSCP (a8953dad's R-1, 10-18; the reviewed fill-tool pattern of tools_routec/fill_run_tablesfigs.py
on aswin/routec-tables-figs): every value is regex-checked FIRST (no quote, newline or control character can pass a regex below), each
placeholder line `    NAME='<…>'` must occur exactly once and is replaced by `    NAME='<value>'`, the filled block differs from the
reviewed block (read from the NOTE at a commit, its sha256 checked) in exactly those lines, the output is written to <out>.tmp,
`zsh -f -n` must pass on it, and only then is it renamed to <out> (never over an existing file). Placeholders not named stay as they
are (each later block that reads one fails closed on its own check).

  uv run --offline --no-project --python 3.12 python -I tools_routec/fill_varscp.py --commit <H> --block-sha <reviewed VARSCP sha256> \
      --out <file> NAME=VALUE [NAME=VALUE ...]                                  (from the 507-prep folder)
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

NOTE = "experiments/2026-10-15_fpr_sweep_507_prep/NOTE_routec_cat_auth_ps_2026-10-17.md"
BLOCK = "VARSCP"
STAMP = r"20[0-9]{2}-[01][0-9]-[0-3][0-9]_[0-2][0-9][0-5][0-9][0-5][0-9]"
SIGNOFF = "experiments/2026-10-14_priority1_spec_09/signoff/"
RX = {"STAMP_L": STAMP, "STAMP_A": STAMP, "TAG": r"PS|CAT",
      "S1L_C": r"[0-9a-f]{40}", "S1L_S": r"[0-9a-f]{64}", "S1L_P": re.escape(SIGNOFF + "S1a-launch_") + STAMP + r"\.json",
      "S1A3_C": r"[0-9a-f]{40}", "S1A3_S": r"[0-9a-f]{64}", "S1A3_P": re.escape(SIGNOFF + "S1a-consume_") + STAMP + r"\.json",
      "S1C_C": r"[0-9a-f]{40}", "S1C_S": r"[0-9a-f]{64}", "S1C_P": re.escape(SIGNOFF + "S1a-consume-CAT_") + STAMP + r"\.json"}  # S1A3MC (10-18)


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def extract(text, block=BLOCK):
    """The lines strictly between `    # BEGIN <block>` and `    # END <block>` (the NOTE's awk rule), each with its newline."""
    lines = text.splitlines(keepends=True)
    b = [i for i, s in enumerate(lines) if s == f"    # BEGIN {block}\n"]
    e = [i for i, s in enumerate(lines) if s == f"    # END {block}\n"]
    if len(b) != 1 or len(e) != 1 or e[0] <= b[0]:
        stop(f"{block}: not exactly one BEGIN / END pair")
    return "".join(lines[b[0] + 1:e[0]])


def fill(block_text, values):
    """-> the filled block text. values: {NAME: value}; every check before any substitution."""
    for k, v in values.items():
        if k not in RX:
            stop(f"{k}: not a VARSCP placeholder")
        if not isinstance(v, str) or not re.fullmatch(RX[k], v):
            stop(f"{k}: the value fails its regex")
    lines = block_text.splitlines(keepends=True)
    hit = {}
    for k in values:
        idx = [i for i, s in enumerate(lines) if re.fullmatch(rf"    {k}='<[^'\n]*>'\n", s)]
        if len(idx) != 1:
            stop(f"{k}: {len(idx)} placeholder lines, not 1")
        hit[idx[0]] = k
    out = [f"    {hit[i]}='{values[hit[i]]}'\n" if i in hit else s for i, s in enumerate(lines)]
    changed = [i for i, (a, b) in enumerate(zip(lines, out)) if a != b]
    if len(out) != len(lines) or sorted(changed) != sorted(hit):
        stop("the filled block differs from the reviewed one outside the placeholder lines")
    return "".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--block-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit) or not re.fullmatch(r"[0-9a-f]{64}", a.block_sha):
        stop("--commit / --block-sha")
    values = {}
    for p in a.pairs:
        k, sep, v = p.partition("=")
        if not sep or k in values:
            stop(f"{k}: not NAME=VALUE, or named twice")
        values[k] = v
    text = subprocess.run(["git", "show", f"{a.commit}:{NOTE}"], capture_output=True, check=True).stdout.decode()
    blk = extract(text)
    if hashlib.sha256(blk.encode()).hexdigest() != a.block_sha:
        stop(f"{BLOCK} at {a.commit} is not the reviewed block (sha256)")
    filled = fill(blk, values)
    out, tmp = Path(a.out), Path(a.out + ".tmp")
    if out.exists():
        stop(f"{out} exists")
    with tmp.open("x") as fh:  # a8953dad's O-9: the output appears only after zsh -n passes
        fh.write(filled)
    if subprocess.run(["zsh", "-f", "-n", str(tmp)], capture_output=True).returncode != 0:
        stop(f"zsh -f -n fails on {tmp}")
    if out.exists():
        stop(f"{out} exists")
    tmp.rename(out)
    print(f"FILL_VARSCP_OK {out} sha256 {hashlib.sha256(filled.encode()).hexdigest()} names {' '.join(sorted(values))} "
          f"reviewed {a.block_sha} at {a.commit}")


if __name__ == "__main__":
    sys.exit(main())
