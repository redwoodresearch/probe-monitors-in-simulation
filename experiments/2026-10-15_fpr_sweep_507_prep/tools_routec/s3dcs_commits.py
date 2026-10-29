"""Route C S3DCS, the commit set at RH (09's NCS ruling (b) and a03f5d93's REQUIRED-B, 10-18). S3DCS calls this file by path and blob,
so its logic is tested directly (test_s3dcs_commits.py), never through NOTE text (rule 15).

1. The commit set. U = D's commits at RH | F3 | XC, as a SET: a commit the AUTH cites that is already in D or F3 counts once, so NCS does
   not grow (09's ruling (b)). It requires:
   - |D | F3| == 384;
   - the spec's `commits` equal U as a set, with no duplicate;
   - --ncs == |U|.
2. The tokens. Every 7-40 hex word in the given configs at RH (the FROZEN INPUTS_FULL and the 223409 fold drops; each passed as
   path@sha256 and hash-checked) must:
   - name an object, as `<token>^{object}`. A bare full 40-hex string passes `rev-parse --verify` even when no such object exists
     (REQUIRED-B);
   - be a commit (NOT_A_COMMIT otherwise; no token is skipped);
   - be an ancestor of RH, or be on a pushed branch (a remote-tracking ref contains it) and in U.

  uv run --offline --no-project --python 3.12 python -I tools_routec/s3dcs_commits.py --commit <RH> --dspec <path> --spec <path> \
      --xc '<XC as a JSON list>' --ncs <N> --config <path@sha256> [--config <path@sha256> ...]

Prints "S3DCS_COMMITS_OK ncs <N> tokens <n>", or "STOP <reason>" lines and exit 1. It reads the two step specs and the configs at RH by
git show, and git metadata. The configs are K by their PATHGATE K-allowlist entries (pkg3: INPUTS_FULL and the fold drops), which
PATHGATE3A/3P enforce before S3DCS. A STOP line names a token or a commit id, never any other config text.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys

F3 = ["b024422e0abf901f3041092df3538d32fb0b882e", "5473efdb8b019ae814cbc70c5b450434b9ae82ca", "79f07a49c7b7e7a93e5b00ea66b8482b9e78718e"]
N_D_F3 = 384
RX_TOKEN = r"\b[0-9a-f]{7,40}\b"
HEX40 = re.compile(r"[0-9a-f]{40}")


def commit_set(d_commits, spec_commits, xc, ncs):
    """Pure. -> (U as a sorted list, problems)."""
    probs = []
    lists = {"D's commits": d_commits, "the spec's commits": spec_commits, "XC": xc}
    for name, v in lists.items():
        if not isinstance(v, list) or not all(isinstance(c, str) and HEX40.fullmatch(c) for c in v):
            probs.append(f"{name} is not a list of full 40-hex commits")
    if probs:
        return [], probs
    if len(set(d_commits) | set(F3)) != N_D_F3:
        probs.append(f"|D | F3| is {len(set(d_commits) | set(F3))}, not {N_D_F3}")
    u = set(d_commits) | set(F3) | set(xc)
    if len(spec_commits) != len(set(spec_commits)):
        probs.append("the spec's commits repeat")
    if set(spec_commits) != u:
        probs.append(f"the spec's commits differ from D | F3 | XC: {len(u - set(spec_commits))} missing, {len(set(spec_commits) - u)} extra")
    if ncs != len(u):
        probs.append(f"NCS is {ncs}, but |D | F3 | XC| is {len(u)}")
    return sorted(u), probs


def tokens_ok(tokens, u, resolve_obj, peel_commit, is_ancestor, on_pushed):
    """Pure. tokens: the hex words; resolve_obj(t) -> full sha or None; peel_commit(t) -> full commit sha or None. -> problems."""
    probs = []
    for t in sorted(set(tokens)):
        if not resolve_obj(t):
            probs.append(f"UNRESOLVABLE {t}")
            continue
        c = peel_commit(t)
        if not c:
            probs.append(f"NOT_A_COMMIT {t}")
        elif is_ancestor(c):
            continue
        elif not on_pushed(c):
            probs.append(f"UNPUSHED {c}")
        elif c not in u:
            probs.append(f"NOT_IN_BUNDLE {c}")
    return probs


def git(*x):
    return subprocess.run(["git", *x], capture_output=True)


def resolve_obj(t):
    """REQUIRED-B: `<t>^{object}` makes rev-parse check that the object exists (a bare 40-hex string is accepted without the check)."""
    r = git("rev-parse", "--verify", "-q", f"{t}^{{object}}")
    out = r.stdout.decode().strip()
    return out if r.returncode == 0 and HEX40.fullmatch(out) else None


def peel_commit(t):
    r = git("rev-parse", "--verify", "-q", f"{t}^{{commit}}")
    out = r.stdout.decode().strip()
    return out if r.returncode == 0 and HEX40.fullmatch(out) else None


def main():
    ap = argparse.ArgumentParser()
    for f in ("--commit", "--dspec", "--spec", "--xc", "--ncs"):
        ap.add_argument(f, required=True)
    ap.add_argument("--config", action="append", required=True)
    a = ap.parse_args()
    if not HEX40.fullmatch(a.commit) or not re.fullmatch(r"[0-9]+", a.ncs):
        print("STOP --commit must be 40 hex and --ncs an integer")
        sys.exit(1)

    def show(path):
        r = git("show", f"{a.commit}:{path}")
        if r.returncode != 0:
            print(f"STOP not committed at --commit: {path}")
            sys.exit(1)
        return r.stdout

    try:
        xc = json.loads(a.xc)
    except json.JSONDecodeError:
        xc = None
    d = json.loads(show(a.dspec)).get("commits")
    s = json.loads(show(a.spec)).get("commits")
    u, probs = commit_set(d, s, xc, int(a.ncs))
    toks = []
    for ps in a.config:
        path, _, want = ps.rpartition("@")
        b = show(path)
        if not re.fullmatch(r"[0-9a-f]{64}", want) or hashlib.sha256(b).hexdigest() != want:
            probs.append(f"a config's sha256 differs from its pin: {path}")
            continue
        toks += re.findall(RX_TOKEN, b.decode())
    if not probs:
        probs = tokens_ok(toks, set(u), resolve_obj, peel_commit,
                          lambda c: git("merge-base", "--is-ancestor", c, a.commit).returncode == 0,
                          lambda c: bool(git("for-each-ref", "--contains", c, "--format=%(refname)", "refs/remotes").stdout.strip()))
    for p in probs:
        print("STOP", p)
    if probs:
        sys.exit(1)
    print(f"S3DCS_COMMITS_OK ncs {len(u)} tokens {len(set(toks))}")


if __name__ == "__main__":
    main()
