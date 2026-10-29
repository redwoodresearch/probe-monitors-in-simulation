"""Tests of s3dcs_commits.py (09's NCS ruling (b), a03f5d93's REQUIRED-B; 10-18). The pure checks run on synthetic commit lists with fake
git functions. Three cases call the real git adapters read-only on this checkout (no git init, no write): HEAD resolves, a full 40-hex
string naming no object is UNRESOLVABLE (the REQUIRED-B bug: plain `rev-parse --verify` accepts it), and HEAD's tree is NOT_A_COMMIT.
Run it from the repo root (the git calls use the working directory).

  uv run --offline --no-project --python 3.12 python -I experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_s3dcs_commits.py
"""
import importlib.util
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("s3dcs_commits_under_test", HERE / "s3dcs_commits.py")
S = importlib.util.module_from_spec(s)
s.loader.exec_module(S)

F3 = S.F3
D = [f"{i:040x}" for i in range(1, 382)] + [F3[0]]  # 382 commits, one of them in F3: |D | F3| = 384, as at RH
NEW = "a" * 40
U = sorted(set(D) | set(F3))
ANC, PUSHED = {D[0]}, {NEW, F3[1]}


def cs(spec, xc, ncs):
    return S.commit_set(D, spec, xc, ncs)


def tk(tokens, u, objs=None):
    objs = objs if objs is not None else {D[0]: "commit", NEW: "commit", F3[1]: "commit", "c" * 40: "commit", "b" * 40: "tree"}
    full = {t[:7]: t for t in objs} | {t: t for t in objs}
    return S.tokens_ok(tokens, set(u), lambda t: full.get(t), lambda t: full.get(t) if objs.get(full.get(t)) == "commit" else None,
                       lambda c: c in ANC, lambda c: c in PUSHED | {"c" * 40})


res = {}
u, pr = cs(U, [], 384)
res["positive control: the spec is D | F3, XC empty, NCS 384: PASS"] = pr == [] and len(u) == 384
u, pr = cs(sorted(U + [NEW]), [NEW], 385)
res["XC adds a new commit: NCS 385, PASS"] = pr == [] and len(u) == 385
u, pr = cs(U, [F3[0]], 384)
res["09's test: an AUTH citing a commit already in F3 (b024422e): PASS with NCS unchanged (384)"] = pr == [] and len(u) == 384
u, pr = cs(U, [F3[0]], 385)
res["the same with NCS grown to 385 (the old 384 + len(XC) rule): STOP"] = pr == ["NCS is 385, but |D | F3 | XC| is 384"]
u, pr = cs(U, [NEW], 385)
res["XC adds a commit the spec lacks: STOP"] = pr == ["the spec's commits differ from D | F3 | XC: 1 missing, 0 extra"]
u, pr = cs(U + [U[0]], [], 384)
res["a repeated spec commit: STOP"] = pr == ["the spec's commits repeat"]
u, pr = S.commit_set(D[1:], U, [], 384)
res["|D | F3| not 384: STOP"] = pr[0] == "|D | F3| is 383, not 384"
u, pr = cs(U, None, 384)
res["XC = null: STOP"] = pr == ["XC is not a list of full 40-hex commits"]
res["tokens: an ancestor, a pushed bundle commit and an abbreviation of one: PASS"] = tk([D[0], F3[1], F3[1][:7]], U) == []
res["tokens: a full 40-hex naming no object: UNRESOLVABLE"] = tk(["d" * 40], U) == [f"UNRESOLVABLE {'d' * 40}"]
res["tokens: a tree (not a commit): NOT_A_COMMIT, not skipped"] = tk(["b" * 40], U) == [f"NOT_A_COMMIT {'b' * 40}"]
res["tokens: a commit on no pushed branch: UNPUSHED"] = tk([D[1]], U, objs={D[1]: "commit"}) == [f"UNPUSHED {D[1]}"]
res["tokens: a pushed commit outside the bundle: NOT_IN_BUNDLE"] = tk(["c" * 40], U) == [f"NOT_IN_BUNDLE {'c' * 40}"]
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.decode().strip()
tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], capture_output=True).stdout.decode().strip()
absent = "1" * 40
res["real git: HEAD resolves and peels to itself"] = S.resolve_obj(head) == head and S.peel_commit(head) == head
res["real git (REQUIRED-B): a full 40-hex naming no object does not resolve"] = \
    subprocess.run(["git", "cat-file", "-e", absent], capture_output=True).returncode != 0 and S.resolve_obj(absent) is None
res["real git: HEAD's tree resolves but is not a commit"] = S.resolve_obj(tree) == tree and S.peel_commit(tree) is None
for k, v in res.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("ALL PASS" if all(res.values()) else "FAILED")
if not all(res.values()):
    raise SystemExit(1)
