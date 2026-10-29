"""Synthetic checks of fill_varscp (a8953dad's R-1, 10-18): the NOTE's working-tree VARSCP text is read as data; no block is run, no
marker or record is read. zsh -f -n only."""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fill_varscp as F  # noqa: E402


def stops(fn, *a):
    try:
        fn(*a)
    except SystemExit as e:
        return str(e).startswith("STOP")
    return False


def main():
    blk = F.extract((HERE.parents[2] / F.NOTE).read_text())
    syn = {"STAMP_L": "2026-10-18_120000", "STAMP_A": "2026-10-18_130000", "TAG": "PS", "S1L_C": "1" * 40, "S1L_S": "2" * 64,
           "S1L_P": F.SIGNOFF + "S1a-launch_2026-10-18_110000.json", "S1A3_C": "3" * 40, "S1A3_S": "4" * 64,
           "S1A3_P": F.SIGNOFF + "S1a-consume_2026-10-18_140000.json", "S1C_C": "5" * 40, "S1C_S": "6" * 64,
           "S1C_P": F.SIGNOFF + "S1a-consume-CAT_2026-10-18_150000.json"}
    full = F.fill(blk, syn)
    with tempfile.NamedTemporaryFile("w", suffix=".zsh", delete=False) as fh:
        fh.write(full)
    cases = {
        "every placeholder in the block has a regex, and the synthetic values pass": set(syn) == set(F.RX)
        and all(f"    {k}='<" in blk for k in syn),
        "filled: no placeholder of a named value left; exactly those lines changed": all(f"    {k}='<" not in full for k in syn)
        and sum(a != b for a, b in zip(blk.splitlines(), full.splitlines())) == len(syn),
        "the filled block passes zsh -f -n": subprocess.run(["zsh", "-f", "-n", fh.name], capture_output=True).returncode == 0,
        "a partial fill leaves the other placeholders as they were": "    S1A3_C='<" in F.fill(blk, {"TAG": "CAT"}),
        "a quote injection STOPs before any substitution": stops(F.fill, blk, syn | {"S1L_C": "x'; true; '"}),
        "a newline in a value STOPs": stops(F.fill, blk, syn | {"STAMP_L": "2026-10-18_120000\ntrue"}),
        "a consume path given as the launch marker STOPs": stops(F.fill, blk, syn | {"S1L_P": syn["S1A3_P"]}),
        "a full consume path given as the CAT-only marker STOPs": stops(F.fill, blk, syn | {"S1C_P": syn["S1A3_P"]}),
        "a CAT-only path given as the full consume marker STOPs": stops(F.fill, blk, syn | {"S1A3_P": syn["S1C_P"]}),
        "an unknown name STOPs": stops(F.fill, blk, syn | {"H": "5" * 40}),
        "a missing placeholder line STOPs": stops(F.fill, blk.replace("    TAG='<", "    TAG0='<", 1), syn),
        "a second BEGIN STOPs extraction": stops(F.extract, "    # BEGIN VARSCP\n    # BEGIN VARSCP\nx\n    # END VARSCP\n"),
    }
    for name, ok in cases.items():
        print(("PASS " if ok else "FAIL ") + name)
    assert all(cases.values()), "fill_varscp tests failed"
    print(f"{len(cases)} checks pass")


if __name__ == "__main__":
    main()
