"""G7b on route C, option B (09's ruling, GATES Amendment 20): derive the runner-folder pw.tree_sha maps from git at S1's capture commit
(4feb6694b) and A's (f2d47f19f), print the differing files with their full sha256 from git, check that they equal the checker's TREE_SHA_DIFF,
and run the checker's tree_sha_diff_viol on the real pair with A's pinned modules (its record's meta.pooled_world.modules_sha256 keys).
Keys and shas only. Run from the repo (or worktree) root:
    PYTHONPATH=src uv run python experiments/2026-10-15_e2e_sim_pipeline/check_tree_sha_routec.py"""
import gzip
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, "experiments/2026-10-15_e2e_sim_pipeline")
import check_stagea_alltasks as M  # noqa: E402

D507 = "experiments/2026-10-15_fpr_sweep_507_prep"
S1C, AC = "4feb6694b24cc82a0bca27df033241530c3ae877", "f2d47f19fd873aa44b5267d88bf7cb3795f875b7"
A_REC = ("a936a5d9762763f9feed351aebdeeec2d8114f21", f"{D507}/results/routec_records/routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_"
         "2026-10-19_180327.json.gz", "54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a")
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True).stdout  # noqa: E731


def tree_map(c):  # as the checker's capture computes pw.tree_sha: every top-level *.py of the runner folder -> sha256 of its bytes
    ls_ = [ln.split("\t", 1) for ln in git("ls-tree", "--full-tree", c, f"{D507}/").decode().splitlines()]
    return {p: hashlib.sha256(git("show", f"{c}:{p}")).hexdigest() for m, p in ls_ if m.split()[1] == "blob" and p.endswith(".py")}


rb_ = gzip.decompress(git("show", f"{A_REC[0]}:{A_REC[1]}"))
assert hashlib.sha256(rb_).hexdigest() == A_REC[2], "A's record sha256"
pinned = set(json.loads(rb_)["meta"]["pooled_world"]["modules_sha256"])
t1, t2 = tree_map(S1C), tree_map(AC)
diff = {k: [t1[k], t2[k]] for k in sorted(set(t1) & set(t2)) if t1[k] != t2[k]}
out = {"s1_commit": S1C, "a_commit": AC, "files": [len(t1), len(t2)], "same_names": sorted(t1) == sorted(t2), "pinned_modules": len(pinned),
       "pinned_equal": all(t1.get(k) == t2.get(k) is not None for k in pinned), "differing_from_git": diff,
       "equals_TREE_SHA_DIFF": diff == {k: list(v) for k, v in M.TREE_SHA_DIFF.items()},
       "viol_on_real_pair": M.tree_sha_diff_viol(t1, t2, pinned, M.TREE_SHA_DIFF)}
print(json.dumps(out, indent=1))
assert out["same_names"] and out["pinned_equal"] and out["equals_TREE_SHA_DIFF"] and out["viol_on_real_pair"] == [], "option B premise (STOP)"
