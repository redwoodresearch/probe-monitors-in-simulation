"""Route C XC (09's fill rule, 10-18): the commits that the COMMITTED AUTH file's rule references cite and that are NOT ancestors of RH.
They are parsed with rp's own pattern (run_peraction311.py L5522 RX_RULE_; the fields L5528-5534 read: group 2 or 4 is the commit,
group 3 picks READ_LOG.md or RULE.md, group 5 is the line). Every such commit must be on a pushed branch (a remote-tracking ref contains
it), and its cited file must exist at it; a cited line must lie within the file's line count, which is counted and never printed
(0d's 10:03 ruling). With --expect, the derived list must equal it exactly, or the script STOPs: a cited commit missing from it, or an
entry no rule cites.

  uv run --offline --no-project --python 3.12 python -I tools_routec/derive_xc.py --commit <RH, 40 hex> --auth <repo path of the AUTH> \
      [--expect '<XC as a JSON list>']

Prints the derived XC as compact JSON, then "XC_OK <n>"; or "STOP <reason>" lines and exit 1. Reads the AUTH file at RH by git show,
git metadata, and READ_LOG.md / RULE.md only as a line count. The AUTH file is K by its writer: its PATHGATE K-allowlist entry names the
writer, and PATHGATE3A/3P enforce that entry (path and sha256) before S3DCS calls this script. For the 10-16 AUTH (PROV) the writer is
tools_pooledworld/make_authorized.py @ 28661737: entries at L68 (key, roles, rule, and a note built from fixed text and citations), the
write at L69-72 (what, made_pt, manifest@sha, catalogue@sha, authorized_overlaps); no measured value.
"""
import argparse
import json
import re
import subprocess
import sys

RX_RULE = r"RULE item (\d+) @ ([0-9a-f]{7,40})|(RULE|READ_LOG) @ ([0-9a-f]{7,40}):(\d+)"  # = rp L5522 RX_RULE_ (the test asserts it)
SPEC09 = "experiments/2026-10-14_priority1_spec_09/"


def derive(auth, resolve, is_ancestor, on_pushed, n_lines):
    """Pure. auth: the parsed AUTH json; resolve(c) -> full sha or None; is_ancestor(full) -> bool (an ancestor of RH);
    on_pushed(full) -> bool; n_lines(full, path) -> line count, or None when the file is absent. -> (sorted XC, problems)."""
    ents = auth.get("authorized_overlaps") if isinstance(auth, dict) else None
    if not isinstance(ents, list) or not ents:
        return [], ["the AUTH file has no authorized_overlaps list"]
    xc, probs = set(), []
    for e in ents:
        r = e.get("rule") if isinstance(e, dict) else None
        m = re.fullmatch(RX_RULE, r) if isinstance(r, str) else None
        if not m:
            probs.append(f"a rule not in rp's form: {r!r}")
            continue
        c = m.group(2) or m.group(4)
        full = resolve(c)
        if not full or not re.fullmatch(r"[0-9a-f]{40}", full) or not full.startswith(c):
            probs.append(f"a rule commit that does not resolve to exactly one commit: {c}")
            continue
        if is_ancestor(full):
            continue
        path = SPEC09 + ("READ_LOG.md" if m.group(3) == "READ_LOG" else "RULE.md")
        n = n_lines(full, path)
        if not on_pushed(full):
            probs.append(f"a rule commit on no pushed branch: {full}")
        elif n is None:
            probs.append(f"the cited file is absent at {full}: {path}")
        elif m.group(5) and not 1 <= int(m.group(5)) <= n:
            probs.append(f"a cited line outside the file at {full}: line {m.group(5)} of {n}")
        else:
            xc.add(full)
    return sorted(xc), sorted(set(probs))


def compare(xc, expect):
    """-> problems: the expected XC must equal the derived one exactly (no missing commit, no extra entry, full 40 hex)."""
    if not isinstance(expect, list) or not all(isinstance(c, str) and re.fullmatch(r"[0-9a-f]{40}", c) for c in expect):
        return ["the expected XC is not a JSON list of full 40-hex commits"]
    miss, extra = sorted(set(xc) - set(expect)), sorted(set(expect) - set(xc))
    return ([f"a cited commit missing from XC: {c}" for c in miss] + [f"an XC entry no rule cites: {c}" for c in extra]
            + (["XC is not sorted and unique"] if not miss and not extra and expect != sorted(set(expect)) else []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--expect", default=None)
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        print("STOP --commit: 40 hex")
        sys.exit(1)

    def git(*x):
        return subprocess.run(["git", *x], capture_output=True)

    r = git("show", f"{a.commit}:{a.auth}")
    if r.returncode != 0:
        print("STOP the AUTH file is not committed at --commit")
        sys.exit(1)
    auth = json.loads(r.stdout)
    memo = {}

    def once(key, f):
        if key not in memo:
            memo[key] = f()
        return memo[key]

    def resolve(c):  # a03f5d93 (10-18): an abbreviation must name exactly ONE commit; ambiguous or unknown -> None (STOP)
        def one():
            objs = git("rev-parse", f"--disambiguate={c}").stdout.decode().split()
            cs = [o for o in objs if git("cat-file", "-t", o).stdout.decode().strip() == "commit"]
            return cs[0] if len(cs) == 1 else None
        return once(("r", c), one)

    def is_ancestor(full):
        return once(("a", full), lambda: git("merge-base", "--is-ancestor", full, a.commit).returncode == 0)

    def on_pushed(full):
        return once(("p", full), lambda: bool(git("for-each-ref", "--contains", full, "--format=%(refname)", "refs/remotes").stdout.strip()))

    def n_lines(full, path):
        def count():
            if git("cat-file", "-e", f"{full}:{path}").returncode != 0:
                return None
            return len(git("show", f"{full}:{path}").stdout.decode().splitlines())  # counted as rp counts (L5533); no line printed
        return once(("n", full, path), count)

    xc, probs = derive(auth, resolve, is_ancestor, on_pushed, n_lines)
    if a.expect is not None:
        try:
            exp = json.loads(a.expect)
        except json.JSONDecodeError:
            exp = None
        probs += compare(xc, exp)
    print(json.dumps(xc, separators=(",", ":")))
    for p in probs:
        print("STOP", p)
    if probs:
        sys.exit(1)
    print("XC_OK", len(xc))


if __name__ == "__main__":
    main()
