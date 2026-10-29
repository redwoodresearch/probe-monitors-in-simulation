"""Test of PATHGATE3's config list (pathgate_code.py cfgs; 2e, 10-17), on a throwaway git repo with fabricated files (no real config
or record is read). Cases, each with its exact outcome:
- positive control (the bug): PATHGATE2's rule resolves a git-form flag "<commit>:<path>:<line>" to "$E/<commit>:<path>:<line>", which
  is no file at H, so PATHGATE2 STOPs on every package 3 spec;
- a package-3-shaped spec: the list is exactly the 4 fixed configs plus the relative-path flags (a list flag's elements included);
  2 git-form values (":<line>" and "@<sha256>") and 1 xfer/ value are checked and not listed; "@out:", the staging prefix as
  copies_root and a "TBD: … (xfer/…)" placeholder are not listed: OK;
- git-form: a commit neither an ancestor of H nor in the spec's commits: BAD, and OK once the spec lists it; an unknown commit, a path
  missing at its commit, an "@sha256" that differs, a ".." path: BAD each;
- xfer/: no "@sha256", another prefix, a ".." segment: BAD each;
- a laptop path flag (/Users/…, /private/tmp/…) outside the staging prefix: BAD each.

  TMPDIR=<a scratchpad dir> uv run --no-project --python 3.12 python -I experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_pathgate_cfgs.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = "experiments/2026-10-15_fpr_sweep_507_prep"
PFX = "/Users/test/scratch/fpr507"
FIXED = [f"{E}/config/common455.json", "experiments/2026-10-14_fpr_sweep_unit/config/inputs_b.json",
         f"{E}/config/inputs_final_v2_pins.json", f"{E}/config/membership710.json"]


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True,
                          check=True).stdout.strip()


def put(repo, path, data):
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data)


def commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def main():
    res = {}
    tmp = Path(tempfile.mkdtemp(prefix="pathgate_cfgs_test_"))
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    put(repo, "data/log.md", "line\n")
    put(repo, "data/rec.json", "{}\n")
    c0 = commit(repo, "c0")
    rec_sha = hashlib.sha256(b"{}\n").hexdigest()
    git(repo, "checkout", "-q", "-b", "side")
    put(repo, "data/side.json", "{}\n")
    s = commit(repo, "side")
    git(repo, "checkout", "-q", "main")
    xf = "xfer/fpr507prep/records/results/x_catalogue_manifest.json@" + "a" * 64
    base = {"script": f"{E}/run_peraction311.py", "commits": [c0], "staging": {"prefix": PFX},
            "entry": {"attr": "main", "flags": {
                "inputs": "config/a.json@" + "b" * 64, "e12": ["config/b.json", "config/c.json@x"], "alltasks": True,
                "after_32k_log": f"{c0[:9]}:data/log.md:1240", "rec32_final": f"{c0[:9]}:data/rec.json@{rec_sha}", "pw_manifest": xf,
                "pw_authorized": "TBD: route C authorized (xfer/fpr507prep/x)", "copies_root": PFX,
                "spec_only": f"@out:{E}/results/spec_x_2026-10-17_000000.pkl"}}}
    variants = {"ok": base}

    def var(name, **flags):
        v = json.loads(json.dumps(base))
        v["entry"]["flags"] |= flags
        variants[name] = v
    var("side", rec32_final=f"{s}:data/side.json")
    variants["side_listed"] = json.loads(json.dumps(variants["side"])) | {"commits": [c0, s]}
    var("unknown", after_32k_log="deadbeef1:data/log.md:3")
    var("missing", after_32k_log=f"{c0}:data/nope.md:3")
    var("badsha", rec32_final=f"{c0}:data/rec.json@" + "c" * 64)
    var("dotdot", rec32_final=f"{c0}:data/../data/rec.json")
    var("xf_nosha", pw_manifest="xfer/fpr507prep/records/results/x.json")
    var("xf_prefix", pw_manifest="xfer/other/records/x.json@" + "a" * 64)
    var("xf_dotdot", pw_manifest="xfer/fpr507prep/../x.json@" + "a" * 64)
    var("laptop_users", e12="/Users/someone/config/b.json")
    var("laptop_tmp", e12=["config/b.json", "/private/tmp/x/c.json"])
    for n, v in variants.items():
        put(repo, f"steps/{n}.json", json.dumps(v, indent=1))
    h = commit(repo, "H")

    def cfgs(name):
        r = subprocess.run([sys.executable, "-I", str(HERE / "pathgate_code.py"), "cfgs", "--commit", h, "--spec", f"steps/{name}.json"],
                           cwd=repo, capture_output=True, text=True)
        return r.returncode, r.stdout.splitlines()

    pg2 = f"{E}/{c0[:9]}:data/log.md:1240"  # PATHGATE2's resolution of after_32k_log
    res["positive control (the bug): PATHGATE2's rule makes the git-form flag a path that is no file at H"] = \
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{h}:{pg2}"], capture_output=True).returncode != 0
    rc, lines = cfgs("ok")
    want = sorted(FIXED + [f"{E}/config/a.json", f"{E}/config/b.json", f"{E}/config/c.json"])
    res["a package-3-shaped spec: exactly the 4 fixed + 3 relative configs; 2 git-form and 1 xfer checked, not listed: OK"] = \
        rc == 0 and lines[:-1] == want and lines[-1] == "PATHGATE_CFGS_OK steps/ok.json configs 7 gitform 2 xfer 1"

    def one_bad(name, needle):
        rc, lines = cfgs(name)
        bads = [x for x in lines if x.startswith("BAD ")]
        return rc == 1 and len(bads) == 1 and needle in bads[0] and lines[-1] == "STOP PATHGATE_CFGS 1 problems"
    res["NEGATIVE git-form on a commit neither an ancestor of H nor in the spec's commits: BAD"] = one_bad("side", "not in the bundle")
    rc, lines = cfgs("side_listed")
    res["the same with that commit in the spec's commits: OK"] = rc == 0 and lines[-1].startswith("PATHGATE_CFGS_OK steps/side_listed.json configs 7 gitform 2")
    res["NEGATIVE git-form on an unknown commit: BAD"] = one_bad("unknown", "not in the repository")
    res["NEGATIVE git-form path missing at its commit: BAD"] = one_bad("missing", "not a file at its commit")
    res["NEGATIVE git-form @sha256 that differs from the file's: BAD"] = one_bad("badsha", "differs")
    res["NEGATIVE git-form path with a '..' segment: BAD"] = one_bad("dotdot", "not a file at its commit")
    res["NEGATIVE xfer/ without @sha256: BAD"] = one_bad("xf_nosha", "an xfer value must be")
    res["NEGATIVE xfer/ under another prefix: BAD"] = one_bad("xf_prefix", "an xfer value must be")
    res["NEGATIVE xfer/ with a '..' segment: BAD"] = one_bad("xf_dotdot", "an xfer value must be")
    res["NEGATIVE a /Users/... flag outside the staging prefix: BAD"] = one_bad("laptop_users", "a laptop path")
    res["NEGATIVE a /private/tmp/... list element: BAD"] = one_bad("laptop_tmp", "a laptop path")
    print("synthetic repo kept (no deletions):", tmp)
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
