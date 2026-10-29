"""Planted check 2's mistake, put back into one real str_replace output: R2 undone. R2 (fixes.py) turns the simulator's plain
diff (`{sign}{code}` lines, ending in a newline; 5,644 of the 6,146 sim outputs R2 changed) into the real gutter
`{n:>4}|{sign} {code}` with no trailing newline. undo_r2 maps a real gutter diff back to the plain form plus one trailing
newline; a blank context line in R2's own form (`   2|  `) becomes " " (the stripped real forms `   2|` and `   2| ` fail
the round-trip gate and are refused). It returns None for anything that is not a numbered str_replace diff, or whose
result R2 would not map back to the same text (the round-trip gate).
"""

import re
import sys
from pathlib import Path

SWAP = Path(__file__).resolve().parents[1] / "2026-10-07_q2_rendering_swap"
sys.path.insert(0, str(SWAP if SWAP.exists() else Path(__file__).parent / "swap"))  # on Modal: /root/swap
import str_replace_tells  # noqa: E402
from rerender import DIFF  # noqa: E402

GUTTER = re.compile(r"^ *\d+\|(?:([ +\-])(?: (.*))?)?$")
SEP = "+++ edited_version\n"


def undo_r2(out: str) -> str | None:
    if not isinstance(out, str) or DIFF not in out[:300] or SEP not in out or out.endswith("\n"):
        return None
    head, _, body = out.partition(SEP)
    lines = body.split("\n")
    if not lines or not str_replace_tells.HUNK.match(lines[0]):
        return None
    new = []
    for line in lines:
        if str_replace_tells.HUNK.match(line):
            new.append(line)
            continue
        m = GUTTER.match(line)
        if not m:
            return None
        new.append((m.group(1) or " ") + (m.group(2) or ""))
    plain = head + SEP + "\n".join(new) + "\n"
    return plain if str_replace_tells.variant_output(plain, "sr_gutter_no_newline")[0] == out else None
