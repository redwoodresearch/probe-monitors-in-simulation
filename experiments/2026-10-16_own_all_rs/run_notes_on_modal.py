"""Re-run own_all NOTE / check scripts as ONE Modal CPU function (0d / Aswin's absolute rule, 10-16: no laptop compute; 4f 09:1x): the laptop only
writes a git bundle of the repo (every ref) and uploads it with the small input copies the scripts read (score-manifest and meta JSON copies, run
lists); the container rebuilds the repo from the bundle at the given HEAD, runs each script AT THE COMMIT THAT PRODUCED ITS NOTE (git show into the
script's own directory) on the same arguments, and returns the NOTE it writes; the entrypoint asserts by key that each equals its committed NOTE
(or, for a NOTE never committed, the given laptop copy) and writes an append-only equality record. Scripts run as child processes, where run_own_all
behaves as on the laptop (modal.is_local() is true in a Function's child process). No scores, no activations; JSON, git and sha256 only.
  modal run --detach experiments/2026-10-16_own_all_rs/run_notes_on_modal.py --jobs <jobs.json> --bundle <path> (bundle + inputs uploaded first)"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_notes_kill-2026-11-14"
VOL_NAME = "probe-shift_aswin-surya_sub-p2probes_notes_kill-2026-11-14"
image = modal.Image.debian_slim(python_version="3.12").apt_install("git").pip_install("modal==1.6.1")
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True, version=2)


@app.function(volumes={"/vol": vol}, cpu=4, memory=8192, timeout=3600, retries=0)
def run(bundle: str, bundle_sha: str, head: str, jobs: list) -> dict:
    import os

    vol.reload()
    bp = Path("/vol") / bundle
    h = hashlib.sha256()
    with bp.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    if h.hexdigest() != bundle_sha:
        raise SystemExit("the bundle differs from its sha256")
    repo = Path("/tmp/repo")
    sh = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    sh("-c", "advice.detachedHead=false", "fetch", "-q", str(bp), "refs/*:refs/*")
    sh("-c", "advice.detachedHead=false", "checkout", "-q", head)
    if sh("rev-parse", "HEAD").strip() != head:
        raise SystemExit("the rebuilt repo is not at the given HEAD")
    out = {}
    for j in jobs:
        sp = repo / j["script"]
        tmp = sp.parent / f"_modal_job_{j['name']}.py"  # the script at its NOTE's commit, in its own directory (its REPO = parents[2])
        tmp.write_bytes(subprocess.run(["git", "-C", str(repo), "show", f"{j['script_commit']}:{j['script']}"], capture_output=True, check=True).stdout)
        args = [a.replace("{inputs}", "/vol/inputs").replace("{out}", j["out"]) for a in j["args"]]
        args += [] if any("{out}" in a for a in j["args"]) else [j["out"]]  # the NOTE path goes where the script's signature puts it
        env = {k: v for k, v in os.environ.items()} | {"PYTHONPATH": str(repo / "src")}
        p = subprocess.run(["python", str(tmp), *args], cwd=str(repo), capture_output=True, text=True, env=env)
        res = {"returncode": p.returncode, "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:]}
        op = repo / j["out"]
        if op.exists():
            res["note"] = op.read_text()
        out[j["name"]] = res
    return {"jobs": out, "head": sh("rev-parse", "HEAD").strip()}


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in (_diff(a.get(k), b.get(k), f"{path}.{k}") if k in a and k in b else [f"{path}.{k}"])]
    return [] if a == b else [path or "."]


@app.local_entrypoint()
def main(jobs: str, bundle: str):
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no") or git("rev-parse", "HEAD") != git("rev-parse", "origin/aswin/own-all-loo"):
        raise SystemExit("uncommitted tracked changes, or HEAD is not on origin")
    head = git("rev-parse", "HEAD")
    jb = Path(jobs).read_bytes()
    js = json.loads(jb)
    bsha = Path(bundle + ".sha256").read_text().split()[0]
    res = run.remote(f"bundles/{Path(bundle).name}", bsha, head, js)  # uploaded beforehand: modal volume put <VOL> <bundle> bundles/; inputs under inputs/
    rec = {"label": "own_all NOTE scripts re-run on Modal CPU (0d / Aswin's no-laptop rule; 4f 09:1x, 10-16): by-key equality of each Modal-written NOTE with its committed NOTE "
                    "(or laptop copy); JSON / git / sha256 only; APPEND-ONLY",
           "head": head, "bundle_sha256": bsha, "jobs_sha256": hashlib.sha256(jb).hexdigest(), "app": APP, "results": {}}
    ok = True
    for j in js:
        r = res["jobs"][j["name"]]
        ref_b = (REPO / j["compare_to"]).read_bytes() if not j.get("compare_to_local") else Path(j["compare_to"]).read_bytes()
        if not j.get("compare_to_local") and git("show", f"HEAD:{j['compare_to']}") != ref_b.decode():
            raise SystemExit(f"{j['compare_to']}: not committed as read")
        diffs = _diff(json.loads(r["note"]), json.loads(ref_b)) if "note" in r else ["(no NOTE written)"]
        rec["results"][j["name"]] = {"script": j["script"], "script_commit": j["script_commit"], "returncode": r["returncode"], "compare_to": j["compare_to"],
                                     "compare_to_sha256": hashlib.sha256(ref_b).hexdigest(), "modal_note_sha256": hashlib.sha256(r["note"].encode()).hexdigest() if "note" in r else None,
                                     "byte_equal": "note" in r and r["note"].encode() == ref_b, "key_diffs": diffs[:20], "equal_by_key": not diffs and r["returncode"] == 0,
                                     "stdout_tail": r["stdout_tail"][-400:]}
        good = rec["results"][j["name"]]["equal_by_key"] and rec["results"][j["name"]]["byte_equal"]  # both, or the job fails (Codex r27)
        rec["results"][j["name"]]["pass"] = good
        ok &= good
        if j.get("write_as") and good:  # a NOTE produced only on Modal (never committed from the laptop): keep the Modal copy, append-only, ONLY when it equals its reference
            (REPO / j["write_as"]).open("x").write(r["note"])
            rec["results"][j["name"]]["written_as"] = j["write_as"]
    rec["pass"] = ok
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    outp = REPO / "experiments" / "2026-10-14_p2_family_expansion" / "results" / f"own_all_notes_modal_rerun_{stamp}.json"
    outp.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({"pass": ok, "results": {k: (v["returncode"], v["equal_by_key"], v["byte_equal"], v["key_diffs"][:3]) for k, v in rec["results"].items()},
                      "path": str(outp.relative_to(REPO))}))
    if not ok:
        raise SystemExit("STOP: a re-run NOTE differs from its reference, or a script failed (see the record)")
