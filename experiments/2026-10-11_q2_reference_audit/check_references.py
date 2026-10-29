"""Reference audit (Aswin's row 1, relayed by the orchestrator): which real run each simulated run was built from, so
no real-vs-sim result compares a simulated run with its own source. Reads manifests, launch records and honest-run
sample IDs only; no transcript text. Local, $0.

1. Sol simulated honest n437: each task's reference (manifest_v2 `tasks[*].reference`) is checked against every real
   honest n1000 log, by sample ID and by file sha256 (the cohort manifest's per-file hashes).
2. Qwen Phase 2: each batch's launch.json `references`. The script tallies which arm each simulated run's reference came
   from.

  uv run --no-project --python 3.14 python experiments/2026-10-11_q2_reference_audit/check_references.py \
      <sim honest manifest_v2.json> <real honest manifest.json>
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"sim honest manifest_v2": "62085c4335e2ba66", "real honest manifest": None}
C = Path.home() / ".cache/probe-shift/s3"
REAL_HONEST = C / "rr-sce-sim-transcripts/transcripts/2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000/logs"
QWEN = C / "individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/runs"


def main(sim_manifest: Path, real_manifest: Path):
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__).resolve())], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    if hashlib.sha256(sim_manifest.read_bytes()).hexdigest()[:16] != SHA["sim honest manifest_v2"]:
        raise SystemExit("sim honest manifest differs")
    sim, real_m = json.loads(sim_manifest.read_text()), json.loads(real_manifest.read_text())
    real_ids, real_batch = set(), Counter()
    for f in sorted(REAL_HONEST.glob("*.eval")):
        with zipfile.ZipFile(f) as z:
            for n in (n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")):
                real_ids.add(str(json.loads(z.read(n))["id"])); real_batch[f.name.split("_")[0]] += 1
    real_sha = {e["sha256"] for e in real_m["files"]}
    refs = [t["reference"] for t in sim["tasks"]]
    sim_out = {"tasks": len(refs), "reference fork runs": dict(Counter(r["fork_run"].split("-")[0] + "-" + r["fork_run"].split("-")[1] + "-" + r["fork_run"].split("-")[2] for r in refs)),
               "reference source sample ID found among the real honest cohort's samples": sum(r["source_sample"] in real_ids for r in refs),
               "reference file sha256 equal to a real honest cohort log": sum(r["sha256"] in real_sha for r in refs),
               "real honest cohort": {"samples": len(real_ids), "by batch prefix": dict(real_batch), "log hashes in its manifest": len(real_sha)},
               "selection rule (manifest)": sim.get("reference_selection")}
    qwen = {}
    for lj in sorted(QWEN.glob("*/launch.json")):
        d = json.loads(lj.read_text()); r = d.get("references") or {}
        qwen[lj.parent.name] = {"simulated arms": (d.get("pilot") or {}).get("sim_arms"), "references": len(r),
                                "reference source arm": dict(Counter(v["source"].split("/runs/")[1].split("/")[1] for v in r.values()))}
    out = {"git": head, "Sol simulated honest n437": sim_out, "Qwen Phase 2 simulated runs": qwen,
           "not checked here": "Sol simulated attack n1000: its references are honest runs; the 200 base200 ones are the base200 real honest transcripts (read-only provenance check, 2026-10-06). Reading attack-run metadata is out of scope here."}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"reference_audit_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
