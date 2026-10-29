"""Derive the new-mains copies of the scale-up's box-side scripts by a FIXED substitution table (RULE.md, Isolation):
every path, tmux socket, session name and config name of the scale-up becomes the new-mains one, so a new-mains split
can share a box with a scale-up follow-up without touching its files, sessions or pause files. Nothing else changes;
the diff of each output against its source is committed next to it (results/rename_diffs/).

  python nm_rename.py <src> <dst> [<src> <dst> ...]
"""

import difflib
import sys
from pathlib import Path

# Order matters: longer, more specific patterns first.
SUBS = [
    ("/home/aswin/phase2/attack_check_311", "/home/aswin/phase2/nm/attack_check_nm"),
    ("~/phase2/attack_check_311", "~/phase2/nm/attack_check_nm"),
    ("phase2/attack_check_311", "phase2/nm/attack_check_nm"),
    ("/home/aswin/phase2/s311_checks", "/home/aswin/phase2/nm/checks"),
    ("~/phase2/s311_checks", "~/phase2/nm/checks"),
    ("phase2/s311_checks", "phase2/nm/checks"),
    ("/home/aswin/phase2/approaches", "/home/aswin/phase2/nm/approaches"),
    ("/home/aswin/phase2/runs", "/home/aswin/phase2/nm/runs"),
    ('"-L", "phase2"', '"-L", "nm"'),
    ('"-L", "guards"', '"-L", "nmguards"'),
    ('"-L", "gates"', '"-L", "nmgates"'),
    ("-L phase2", "-L nm"),
    ("-L guards", "-L nmguards"),
    ("-L gates", "-L nmgates"),
    ("live(\"phase2\"", "live(\"nm\""),
    ("live(\"guards\"", "live(\"nmguards\""),
    ("live('phase2'", "live('nm'"),
    ("live('guards'", "live('nmguards'"),
    ("tasks_311_q", "tasks_nm_s"),
    ("pilot_311_fu_q", "pilot_nm_fu_s"),
    ("pilot_311_q", "pilot_nm_s"),
    ("servers_311_fu", "servers_nm_fu"),
    ("servers_311", "servers_nm"),
    ("fu_bases_q", "fu_bases_s"),
    ("end_main_q", "end_main_s"),
    ("s311q", "snms"), ("f311q", "fnms"), ("em311q", "emnms"), ("rp311q", "rpnms"), ("gs311q", "gsnms"),
    ("sc311q", "scnms"), ("bg311q", "bgnms"), ("gate311q", "gatenms"), ("gb311q", "gbnms"), ("gf311q", "gfnms"),
    ("ub311q", "ubnms"), ("gp311q", "gpnms"),
    ("config/tasks_nm_s[1-4]_launch", "config/tasks_nm_s[1-5]_launch"),
]


def rename(text: str) -> tuple[str, dict]:
    counts = {}
    for a, b in SUBS:
        n = text.count(a)
        if n:
            counts[a] = n
            text = text.replace(a, b)
    return text, counts


def main() -> None:
    args = sys.argv[1:]
    assert args and len(args) % 2 == 0, __doc__
    out_dir = Path(__file__).parent / "results" / "rename_diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in zip(args[::2], args[1::2]):
        s, d = Path(src), Path(dst)
        new, counts = rename(s.read_text())
        for leftover in ("attack_check_311", "s311_checks", "-L phase2", '"phase2"', "'phase2'", "311q"):
            assert leftover not in new, f"{s.name}: '{leftover}' survives the rename"
        d.write_text(new)
        diff = "".join(difflib.unified_diff(s.read_text().splitlines(True), new.splitlines(True), str(s.name), str(d.name)))
        (out_dir / f"{d.name}.diff").write_text(diff)
        print(f"{s.name} -> {d}: {sum(counts.values())} substitutions {counts}")


if __name__ == "__main__":
    main()
