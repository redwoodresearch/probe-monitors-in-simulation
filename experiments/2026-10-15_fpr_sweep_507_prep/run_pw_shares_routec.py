"""Route C pooled-world honest alarm shares for Figures 1 and 6 (2e's GO, 10-18). RUN-LENGTH's analyze_pw_shares.py runs UNCHANGED (git blob
7f291f03 @ 1b65339a9, sha256 64c0297a...; read from git and pinned by sha256) in this script's own Modal container, on ONE pooled-world Stage A
record, exactly as run_analyze_modal.py ran it on 10-16 for the 10-17 figures (app ap-UUrA257LQXfb9BdEaNc5qn): the record's /sol path, its
sha256, --combo pooled_all_origins. It reads A only (no npz, no other record). This wrapper adds only what 09's rulings ask of a route C
derived record:
  - the label is composed at run time from A's pinned, committed label sidecar: "<the sidecar's label bytes> | analyze_pw_shares on route C
    pooled A @ <A sha>" (09, 10-18). STOP unless the sidecar's bytes hash to their pin, it is exactly {record, record_sha256, label},
    record_sha256 is A's sha, the label holds PRELIMINARY, and the label's UTF-8 bytes hash to --a-label-sha (09's registered A label,
    6100557b...; 2e 10-18). The analyzer's own description moves from "label" to "what";
  - the output's own label sidecar {record, record_sha256, label} is written next to it, exclusive create (09: one sidecar per record);
  - CONTROL mode (--control; 2e 10-18): a pool-neutral CONTROL_ONLY label, no sidecar, and the analyzer's output is compared in the
    container with a sha-pinned reference record (the 10-16 output) by key, every key but "provenance": MATCH or MISMATCH, keys only.
The laptop never parses a record: it sends paths, pins, and the analyzer's and the sidecar's bytes unparsed, writes the returned bytes as
returned (byte-identical committed copies under results/pw_shares_routec/), and prints keys, shas and verdicts only.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach run_pw_shares_routec.py --record-src <path under
      xfer/fpr507prep/records/> --record-sha <sha256> --record-source <text> --label-sidecar git:<commit>:<path>.json@<sha256>
      --a-label-sha <sha256>
  (a CONTROL run: --control --ref-src <path of the reference pw-shares record> --ref-sha <sha256>, and no sidecar or label sha)
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-routec-grp_pwshares_kill-2026-11-14")  # route C (10-18): 63 characters
image = modal.Image.debian_slim(python_version="3.12")  # the analyzer is standard library only
SOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # A and the reference record (read only)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-render_final-figs_kill-2026-11-13", create_if_missing=True)  # as the group steps
ANALYZER = {"commit": "1b65339a924394fbd5ce70068ba80e93df39a885", "path": "experiments/2026-10-15_fpr_sweep_507_prep/analyze_pw_shares.py",
            "sha256": "64c0297aa534a234301fdb30140b5ebc24d4967557db0de0d52ace9d8b534c7b"}  # RUN-LENGTH's, unchanged (git blob 7f291f03)
COMBO = "pooled_all_origins"
RECORDS = "xfer/fpr507prep/records/"
RESULTS = "experiments/2026-10-15_fpr_sweep_507_prep/results/pw_shares_routec"  # the committed copies (the sidecar's "record")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rel_ok(p, prefix):  # as the group steps (a37cc): a Volume path is relative, under prefix, with no ".."
    return isinstance(p, str) and not p.startswith("/") and p.startswith(prefix) and ".." not in Path(p).parts


def a_label(record_sha, side_bytes, side_sha, label_sha):
    """run_group_catch.a_label's checks (09, 10-17), plus the label bytes' sha256 against --a-label-sha (2e 10-18: 09's registered A label).
    -> "<label> | analyze_pw_shares on route C pooled A @ <A sha>" (09 10-18: derived labels start with A's registered label bytes)."""
    if not isinstance(side_bytes, bytes) or sha(side_bytes) != side_sha:
        raise SystemExit("STOP: the label sidecar's sha256 differs from its pin")
    sc = json.loads(side_bytes)
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"} or not isinstance(sc["record"], str) or not sc["record"]:
        raise SystemExit("STOP: the label sidecar is not {record, record_sha256, label}")
    if sc["record_sha256"] != record_sha:
        raise SystemExit("STOP: the label sidecar names another A record than the one this step consumes")
    lab = sc["label"]
    if not isinstance(lab, str) or not lab.strip():
        raise SystemExit("STOP: the label sidecar's label is missing or empty")
    if "PRELIMINARY" not in lab:
        raise SystemExit("STOP: the label is not PRELIMINARY and no FINAL-release record is pinned (09)")
    if sha(lab.encode()) != label_sha:
        raise SystemExit("STOP: the sidecar's label is not 09's registered A label (--a-label-sha)")
    return f"{lab} | analyze_pw_shares on route C pooled A @ {record_sha}"


def control_label(record_sha, ref_sha):  # 2e 10-18: pool-neutral (names neither pool), CONTROL_ONLY
    return (f"CONTROL_ONLY: analyze_pw_shares (unchanged) on the pooled-world record @ {record_sha}, compared by key with the reference "
            f"pw-shares record @ {ref_sha}; never a figure input")


def compare(out, ref):
    """Keys only: every key of the analyzer's output but "provenance" against the reference record's (2e 10-18)."""
    ko, kr = set(out) - {"provenance"}, set(ref) - {"provenance"}
    diff = sorted(k for k in ko & kr if json.dumps(out[k], sort_keys=True) != json.dumps(ref[k], sort_keys=True))
    return {"match": ko == kr and not diff, "keys_compared": sorted(ko & kr), "keys_differing": diff, "keys_only_in_output": sorted(ko - kr),
            "keys_only_in_reference": sorted(kr - ko), "excluded": ["provenance"]}


def read_pinned(root, rel, want, prefix):
    """A Volume file, sha-checked BEFORE it is parsed (a .json.gz is checked decompressed, as the analyzer does). -> bytes."""
    if not rel_ok(rel, prefix):
        raise SystemExit(f"STOP: {rel!r} is not a relative path under {prefix}")
    b = (root / rel).read_bytes()
    if rel.endswith(".json.gz"):
        import gzip

        b = gzip.decompress(b)
    if sha(b) != want:
        raise SystemExit(f"STOP: {Path(rel).name}: sha256 differs from its pin")
    return b


def make(record_src, record_sha, stamp, analyzer_bytes, launch, me_bytes, root, label_bytes=b"", label_sha="", a_label_sha="", control=False,
         ref_src="", ref_sha="", python="python3"):
    """Pure but for the analyzer's subprocess: checks, runs the unchanged analyzer, composes the record. -> (files {name: bytes}, summary)."""
    import tempfile

    if sha(me_bytes) != launch.get("script_sha256"):  # rule 27: the code that runs is the code the laptop pinned
        raise SystemExit("STOP: the container's copy of this script is not the pinned one")
    if sha(analyzer_bytes) != ANALYZER["sha256"]:
        raise SystemExit("STOP: the analyzer is not RUN-LENGTH's unchanged analyze_pw_shares.py")
    read_pinned(root, record_src, record_sha, RECORDS)  # A, before any compute (the analyzer checks it again)
    if control:
        if label_bytes or label_sha or a_label_sha:
            raise SystemExit("STOP: a CONTROL run takes no label sidecar")
        ref = json.loads(read_pinned(root, ref_src, ref_sha, RECORDS))
        lab = control_label(record_sha, ref_sha)
    else:
        if ref_src or ref_sha:
            raise SystemExit("STOP: a route C run takes no reference record")
        lab = a_label(record_sha, label_bytes, label_sha, a_label_sha)
    work = Path(tempfile.mkdtemp())
    (work / "results").mkdir()
    (work / "analyze_pw_shares.py").write_bytes(analyzer_bytes)
    prov = {"label": "route C pw-shares wrapper on Modal CPU (run_pw_shares_routec.py)", "stamp": stamp, "git_head": launch.get("git"),
            "script": {"name": "analyze_pw_shares.py", "sha256": ANALYZER["sha256"], "git": f"{ANALYZER['commit']}:{ANALYZER['path']}"},
            "wrapper_sha256": launch.get("script_sha256"), "inputs": [{"name": Path(record_src).name, "sha256": record_sha}]}
    pr = subprocess.run([python, "-I", str(work / "analyze_pw_shares.py"), str(root / record_src), record_sha, "--combo", COMBO], cwd=str(work),
                        capture_output=True, text=True, timeout=840, env=dict(os.environ) | {"RUN_ANALYZE_PROVENANCE": json.dumps(prov)})
    print(pr.stdout, pr.stderr, sep="\n", flush=True)  # the analyzer prints keys, counts and its problems' text only (no share value)
    made = sorted((work / "results").iterdir())
    if pr.returncode != 0 or len(made) != 1:
        raise SystemExit(f"STOP: analyze_pw_shares exited {pr.returncode} with {len(made)} output files")
    ab = made[0].read_bytes()
    out = json.loads(ab)
    doc = {"label": lab, "what": out["label"]} | {k: v for k, v in out.items() if k != "label"} | {"wrapper": {
        "script": "run_pw_shares_routec.py", "analyzer": ANALYZER, "analyzer_output": {"name": made[0].name, "sha256": sha(ab)},
        "record": {"src": record_src, "sha256": record_sha}, "mode": "CONTROL_ONLY" if control else "route C", "stamp": stamp,
        "created_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")} | launch}
    if control:
        doc["control"] = compare(out, ref) | {"reference": {"src": ref_src, "sha256": ref_sha}}
    stem = f"pw_shares_{COMBO}_{Path(record_src).name.removesuffix('.gz').removesuffix('.json')}_{'CONTROL_ONLY' if control else 'routeC'}{stamp}"
    b = json.dumps(doc, indent=1).encode()
    files = {f"{stem}.json": b}
    if not control:  # 09: the output's own sidecar, the anchor's _LABEL.json shape
        files[f"{stem}_LABEL.json"] = json.dumps({"record": f"{RESULTS}/{stem}.json", "record_sha256": sha(b), "label": lab}, indent=1,
                                                 ensure_ascii=False).encode()
    summ = {"analyzer_pass": out.get("pass") is True, "mode": doc["wrapper"]["mode"],
            "control": {k: doc["control"][k] for k in ("match", "keys_differing", "keys_only_in_output", "keys_only_in_reference")} if control else None}
    return files, summ


@app.function(image=image, cpu=2, memory=(4096, 16384), timeout=900, retries=0, volumes={"/sol": SOL, "/out": OUT})
def build(record_src: str, record_sha: str, stamp: str, analyzer_bytes: bytes, launch: dict, label_bytes: bytes = b"", label_sha: str = "",
          a_label_sha: str = "", control: bool = False, ref_src: str = "", ref_sha: str = "") -> dict:
    files, summ = make(record_src, record_sha, stamp, analyzer_bytes, launch, Path(__file__).read_bytes(), Path("/sol"), label_bytes, label_sha,
                       a_label_sha, control, ref_src, ref_sha)
    d_ = Path("/out/pw_shares_routec")
    d_.mkdir(parents=True, exist_ok=True)
    for n_, b_ in files.items():
        with open(d_ / n_, "xb") as f:
            f.write(b_)
    OUT.commit()
    print(json.dumps({n_: {"volume_path": f"pw_shares_routec/{n_}", "sha256": sha(b_)} for n_, b_ in files.items()}), flush=True)
    return {"files": {n_: {"bytes": b_, "sha256": sha(b_), "volume_path": f"pw_shares_routec/{n_}"} for n_, b_ in files.items()}} | summ


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def sidecar_bytes(pointer):
    """Laptop side: git:<commit>:<path>@<sha256> -> (bytes, sha256), the blob read unparsed (the container checks and parses it)."""
    spec, _, h = pointer.rpartition("@")
    src, _, rest = spec.partition(":")
    c_, _, p_ = rest.partition(":")
    if src != "git" or len(c_) != 40 or not p_.endswith(".json") or len(h) != 64:
        sys.exit("STOP: --label-sidecar must be git:<40-hex commit>:<path>.json@<sha256>")
    return git("show", f"{c_}:{p_}"), h


def check_args(record_src, record_sha, label_sidecar, a_label_sha, control, ref_src, ref_sha):
    """Laptop side, before the paid call: paths under xfer/fpr507prep/records/, 64-hex pins, and the two modes kept apart."""
    def hx(s):
        return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)

    if not rel_ok(record_src, RECORDS) or not hx(record_sha):
        sys.exit(f"STOP: --record-src must be a relative path under {RECORDS} and --record-sha 64 hex")
    if control:
        if label_sidecar or a_label_sha or not rel_ok(ref_src, RECORDS) or not hx(ref_sha):
            sys.exit("STOP: a CONTROL run takes --ref-src and --ref-sha, and no --label-sidecar or --a-label-sha")
    elif ref_src or ref_sha or not label_sidecar or not hx(a_label_sha):
        sys.exit("STOP: a route C run takes --label-sidecar and --a-label-sha, and no --ref-src or --ref-sha")


@app.local_entrypoint()
def main(record_src: str, record_sha: str, record_source: str, label_sidecar: str = "", a_label_sha: str = "", control: bool = False,
         ref_src: str = "", ref_sha: str = ""):
    if git("status", "--porcelain", "--untracked-files=no").strip():
        sys.exit("STOP: uncommitted tracked changes")
    me = Path(__file__).name
    if git("hash-object", str(HERE / me)).strip() != git("rev-parse", f"HEAD:./{me}").strip():  # the code sent is the committed code
        sys.exit(f"STOP: {me} differs from HEAD")
    check_args(record_src, record_sha, label_sidecar, a_label_sha, control, ref_src, ref_sha)
    ab = git("show", f"{ANALYZER['commit']}:{ANALYZER['path']}")
    if sha(ab) != ANALYZER["sha256"]:
        sys.exit("STOP: the analyzer blob is not the pinned one")
    lb_, lh_ = (b"", "") if control else sidecar_bytes(label_sidecar)  # 09: the pinned label sidecar, as bytes (parsed in the container)
    if not record_source.strip():
        sys.exit("STOP: --record-source is empty")
    stamp = "_" + datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    launch = {"record_source": record_source, "label_sidecar": label_sidecar, "a_label_sha256": a_label_sha,
              "git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": sha((HERE / me).read_bytes())}
    outd = HERE / "results" / "pw_shares_routec"
    outd.mkdir(parents=True, exist_ok=True)  # before the paid call (a0b05 R-2)
    res = build.remote(record_src, record_sha, stamp, ab, launch, lb_, lh_, a_label_sha, control, ref_src, ref_sha)
    for n_, f_ in res["files"].items():  # written as returned: byte-identical to the Volume copies, never parsed here
        if sha(f_["bytes"]) != f_["sha256"]:
            sys.exit("STOP: the returned bytes differ from the container's sha256")
        with open(outd / n_, "xb") as fh:  # exclusive create; results/ is append-only
            fh.write(f_["bytes"])
    print(json.dumps({"out": {n_: {"path": str(outd / n_), "sha256": f_["sha256"], "volume_path": f_["volume_path"]} for n_, f_ in res["files"].items()},
                      "mode": res["mode"], "analyzer_pass": res["analyzer_pass"], "control": res["control"],
                      "inputs": {"record_sha256": record_sha, "ref_sha256": ref_sha or None, "a_label_sha256": a_label_sha or None}}))
    if not res["analyzer_pass"]:
        sys.exit("STOP: the analyzer did not pass")
    if control and not res["control"]["match"]:
        sys.exit("STOP: CONTROL MISMATCH (the differing keys are listed above)")
