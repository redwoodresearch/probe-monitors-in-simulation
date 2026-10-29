"""POST HOC (Codex on #879): completeness of the reference audit's inputs. Reads S3 listings, launch records and the real
honest cohort's manifest and log hashes only; no transcript text; the sealed zoo C runs are not read. Local, $0.

1. Every run folder under phase2/runs/ on S3: its launch.json is fetched (zoo C's are sealed: their committed launch
   config is checked instead), hashed and compared with the local cache copy the audit read, and its references tallied
   by source arm. The audit's 193 Qwen simulated-attack references are complete if no other launch has any.
2. The real honest manifest the audit was given: its sha256 against S3's manifest.json, and its per-file sha256s against
   every local cohort log the audit read sample IDs from.

  uv run --no-project python experiments/2026-10-11_q2_reference_audit/check_complete_POSTHOC.py <real honest manifest.json>
Writes its result, then exits non-zero unless every check passes.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
C = Path.home() / ".cache/probe-shift/s3"
BUCKET, PREFIX = "individual-aswin-surya-probe-shift-kill-2027-04-06", "phase2/runs/"
COHORT = "rr-sce-sim-transcripts", "transcripts/2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/"
ZOO_C_CONFIG = "origin/main:experiments/2026-10-07_phase2_attack_check/config/pilot_zoo_c_oct11c.yaml"  # not on this branch
AUDIT = HERE / "results" / "reference_audit_2026-10-11_194008.json"  # the audit whose inputs are checked
AWS = ["aws", "--profile", "redwood-sandbox"]


def s3(*args) -> bytes:
    return subprocess.run(AWS + list(args), capture_output=True, check=True, timeout=300).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(real_manifest: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    pre = json.loads(s3("s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", PREFIX, "--delimiter", "/", "--output", "json"))
    if pre.get("IsTruncated"):
        raise SystemExit("the S3 listing is truncated")
    folders = sorted(p["Prefix"][len(PREFIX):].rstrip("/") for p in pre.get("CommonPrefixes", []))
    zc = subprocess.run(["git", "show", ZOO_C_CONFIG], capture_output=True, text=True, check=True, cwd=HERE).stdout
    zoo_c_sim = [l.split("#")[0].strip() for l in zc.splitlines() if l.startswith("simulators:")]
    audit = json.loads(AUDIT.read_text())
    expected = {k: v["references"] for k, v in audit["Qwen Phase 2 simulated runs"].items()}
    launches, total = {}, Counter()
    for f in folders:
        if f.endswith(".request_logs") or f.endswith("_report"):
            launches[f] = "not a launch (request logs or a report)"; continue
        if "zooc" in f:
            if zoo_c_sim != ["simulators: []"]:
                raise SystemExit("zoo C's launch config is not simulator-free")
            launches[f] = "sealed zoo C stage, not read: its launch config (pilot_zoo_c_oct11c.yaml) has simulators: [] (real honest_b only)"
            continue
        keys = json.loads(s3("s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", f"{PREFIX}{f}/launch.json", "--output", "json")).get("Contents", [])
        if not any(k["Key"] == f"{PREFIX}{f}/launch.json" for k in keys):
            launches[f] = "no launch.json"; continue
        b = s3("s3", "cp", f"s3://{BUCKET}/{PREFIX}{f}/launch.json", "-")  # any other AWS error raises
        d = json.loads(b); r = d.get("references") or {}
        arms = Counter(v["source"].split("/runs/")[1].split("/")[1] for v in r.values())
        local = C / BUCKET / PREFIX / f / "launch.json"
        launches[f] = {"launch.json sha256": sha(b), "local cache copy (read by the audit)": "absent" if not local.exists() else
                       ("identical" if sha(local.read_bytes()) == sha(b) else "DIFFERS"), "simulated arms": (d.get("pilot") or {}).get("sim_arms"),
                       "references": len(r), "reference source arm": dict(arms)}
        total.update(arms)
    audited = {f for f, v in launches.items() if isinstance(v, dict) and v["local cache copy (read by the audit)"] == "identical"}
    missed = {f: v["references"] for f, v in launches.items() if isinstance(v, dict) and f not in audited and v["references"]}
    # the real honest manifest
    m_s3 = s3("s3", "cp", f"s3://{COHORT[0]}/{COHORT[1]}manifest.json", "-")
    man = json.loads(real_manifest.read_text()); listed = Counter(e["sha256"] for e in man["files"])
    logs = sorted((C / COHORT[0] / COHORT[1] / "logs").glob("*.eval"))
    log_sha = Counter(sha(p.read_bytes()) for p in logs)
    checks = {"the audit's launches are exactly the S3 launches whose cache copy it read (identical bytes)": audited == set(expected),
              "each audited launch's reference count equals the audit's": all(launches[k]["references"] == n for k, n in expected.items() if k in launches and isinstance(launches[k], dict)),
              "no other launch has references": not missed,
              "given manifest equals S3's": sha(real_manifest.read_bytes()) == sha(m_s3),
              "manifest's file hashes equal the local logs' (multisets)": log_sha == listed,
              "manifest's distinct hashes equal the audit's recorded count": len(listed) == audit["Sol simulated honest n437"]["real honest cohort"]["log hashes in its manifest"],
              "every unsealed run folder has a launch.json": not [f for f, v in launches.items() if v == "no launch.json"]}
    out = {"label": "POST HOC (Codex on #879)", "git": head, "audit checked": {AUDIT.name: sha(AUDIT.read_bytes())},
           "zoo C launch config": {ZOO_C_CONFIG: sha(zc.encode())}, "checks": checks, "all checks pass": all(checks.values()),
           "1. Qwen launches on S3": {"run folders": len(folders), "launches": launches, "references by source arm, all read launches": dict(total),
                                      "launches with references the audit did not read": missed},
           "2. real honest manifest": {"given file sha256": sha(real_manifest.read_bytes()), "S3 manifest.json sha256": sha(m_s3),
                                       "given file equals S3's": sha(real_manifest.read_bytes()) == sha(m_s3),
                                       "files listed": sum(listed.values()), "local logs read by the audit": len(logs),
                                       "local log sha256s equal the listed ones (as multisets)": log_sha == listed}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"complete_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)
    if not out["all checks pass"]:
        raise SystemExit("a completeness check failed: see the result file")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
