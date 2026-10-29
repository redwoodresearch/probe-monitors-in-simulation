"""Placeholder census, part 3 (review threads on #758/#759). Counts only; no text is printed or written. Attack logs are
counted as in part 2, under Aswin's OK just before 16:57 PT ("Yes, counts only is fine").

1. Inventory. Every local log of each cohort is compared with the S3 listing (key and bytes). Where the cohort's S3
   manifest records them, the script also checks sha256, manifest coverage and the manifest's log and trajectory counts.
   A partial cache would otherwise give a false zero. Each cohort gets a verdict; the script exits non-zero after writing
   if any cohort fails.
2. Model-event inputs. Current Inspect logs store each model event's input as `input_refs` ranges into
   `events_data.messages`, and leave `input` empty. Parts 1 and 2 read only `input`. This counts references in the
   expanded inputs, and in the whole `events_data.messages` pool.
3. Qwen honest A. It splits into the base task folders and the `_rerunN` / `_redrawN` folders.

  nice -n 19 uv run --no-project --python 3.14 python experiments/2026-10-11_q2_placeholder_census/count_placeholders_part3.py
"""
import hashlib
import json
import os
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REF = re.compile(r"attachment://[0-9a-f]+")
RETRY = re.compile(r"_(rerun|redraw)\d+$")
C = Path(os.path.expanduser("~/.cache/probe-shift/s3"))
SOL, QB = "rr-sce-sim-transcripts", "individual-aswin-surya-probe-shift-kill-2027-04-06"
QBATCHES = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup", "2026-10-10_001314_scaleup_topup"]


def sol(cohort, expected, lists_all, fleet_name=""):
    return {"bucket": SOL, "prefixes": [f"transcripts/{cohort}/logs/"], "manifests": [f"transcripts/{cohort}/manifest.json"],
            "expected": expected, "lists_all": lists_all, "fleet_name": fleet_name}


def qwen(arm):
    return {"bucket": QB, "prefixes": [f"phase2/runs/{b}/{arm}/" for b in QBATCHES],
            "manifests": [f"phase2/runs/{b}.manifest_v1.json" for b in QBATCHES], "expected": None, "lists_all": True, "fleet_name": ""}


def by_task(d):
    return len({t["log"] for t in d["tasks"]}), len(d["tasks"])


# expected (logs, trajectories) as each S3 manifest states them; lists_all: the manifest names every log of the cohort (the
# simulated attack's lists only its 800 fleet logs, not the 200 base samples; the real attack manifests list no files)
COHORTS = {
    "Sol real honest n1000": sol("2026-09-18_la-public10_real-honest-solmedium-step150-overlapaware-n1000",
                                 lambda d: (d["eval_files"], d["trajectories"]), True),
    "Sol simulated honest n437": sol("2026-10-06_la-public10_sim-astra-medium-honest-origtraj-read-nocritic-fork0", by_task, True),
    "Sol real attack epoch 1 n1000": sol("2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch1-n1000",
                                         lambda d: (d["epochs"]["1"]["eval_files"], d["epochs"]["1"]["trajectories"]), False),
    "Sol real attack epoch 2 n1000": sol("2026-09-18_la-public10_real-attack-solmedium-step150-guidancematched-epoch2-n1000",
                                         lambda d: (d["epochs"]["2"]["eval_files"], d["epochs"]["2"]["trajectories"]), False),
    "Sol simulated attack n1000": sol("2026-09-18_la-public10_sim-astra-medium-attack-origtraj-read-nocritic-fork0-overlapaware-n1000",
                                      lambda d: (None, d["pooled_samples"]), False, fleet_name="overlap800_"),  # logs/overlap800_<file>
    "Qwen scale-up honest A": qwen("honest_a"),
    "Qwen scale-up honest B": qwen("honest_b"),
}


def aws(*args):
    return subprocess.run(["aws", *args, "--profile", "redwood-sandbox"], capture_output=True, text=True, check=True).stdout


def manifest_index(bucket, key, fleet_name):
    """The S3 keys a manifest lists (Sol real honest's `files`, the simulated attack's `fleet_files`, the simulated
    honest's `tasks[*].log`, the Qwen batches' `files`), {key: sha256} for those it hashes, and the manifest itself."""
    d = json.loads(aws("s3", "cp", f"s3://{bucket}/{key}", "-"))
    base = key.rsplit("/", 1)[0] + "/"
    entries = (d.get("files") or [{**v, "file": fleet_name + v["file"]} for v in (d.get("fleet_files") or {}).values()]
               or [{"file": t["log"]} for t in d.get("tasks") or []])
    keys, hashes = set(), {}
    for e in entries if isinstance(entries, list) else []:
        k = e.get("key") or (base + e["file"] if "/" in e["file"] else base + "logs/" + e["file"])
        keys.add(k)
        if "sha256" in e:
            hashes[k] = e["sha256"]
    return keys, hashes, d


def inventory(name, spec):
    listed = {}
    for p in spec["prefixes"]:
        page = json.loads(aws("s3api", "list-objects-v2", "--bucket", spec["bucket"], "--prefix", p, "--output", "json") or "{}")
        listed |= {o["Key"]: o["Size"] for o in page.get("Contents") or [] if o["Key"].endswith(".eval")}
    local = {}
    for p in spec["prefixes"]:
        for f in (C / spec["bucket"] / p).rglob("*.eval"):
            local[str(f.relative_to(C / spec["bucket"]))] = f
    keys, hashes, stated = set(), {}, None
    for m in spec["manifests"]:
        k, h, d = manifest_index(spec["bucket"], m, spec["fleet_name"]); keys |= k; hashes |= h
        stated = spec["expected"](d) if spec["expected"] else None
    listed_in_manifest = {k for k in keys if k.endswith(".eval") and any(k.startswith(p) for p in spec["prefixes"])}
    checked = [k for k in local if k in hashes]
    bad_sha = [k for k in checked if hashlib.sha256(local[k].read_bytes()).hexdigest() != hashes[k]]
    inv = {"S3 logs": len(listed), "local logs": len(local), "on S3, not local": len(set(listed) - set(local)),
           "local, not on S3": len(set(local) - set(listed)),
           "byte mismatches": sum(local[k].stat().st_size != listed[k] for k in set(local) & set(listed)),
           "manifest-listed logs": len(listed_in_manifest), "manifest-listed, not local": len(listed_in_manifest - set(local)),
           "local, not in the manifest": len(set(local) - listed_in_manifest) if spec["lists_all"] else "manifest lists only part",
           "sha256 checked against the manifest": len(checked), "sha256 mismatches": len(bad_sha),
           "manifest-stated (logs, trajectories)": stated,
           "local file list sha256": hashlib.sha256("\n".join(f"{k} {local[k].stat().st_size}" for k in sorted(local)).encode()).hexdigest()}
    return inv, local


def count(files):
    c = Counter({"logs": len(files)})
    for f in files:
        with zipfile.ZipFile(f) as z:
            for n in (n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")):
                s = json.loads(z.read(n)); c["samples"] += 1
                pool = (s.get("events_data") or {}).get("messages") or []
                c["events_data.messages entries"] += len(pool); c["refs in events_data.messages"] += len(REF.findall(json.dumps(pool)))
                for e in (e for e in s.get("events") or [] if e.get("event") == "model"):
                    c["model events"] += 1
                    c["model events with input_refs"] += bool(e.get("input_refs")); c["model events with a non-empty input"] += bool(e.get("input"))
                    expanded = [pool[i] for lo, hi in e.get("input_refs") or [] for i in range(lo, hi)] + list(e.get("input") or [])
                    c["expanded model-event input messages"] += len(expanded)
                    c["refs in expanded model-event inputs"] += len(REF.findall(json.dumps(expanded)))
                    c["model events with any ref in the expanded input"] += bool(REF.search(json.dumps(expanded)))
    return dict(c)


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    out = {"git": head, "cohorts": {}}
    for name, spec in COHORTS.items():
        inv, local = inventory(name, spec)
        row = {"inventory": inv}
        if name == "Qwen scale-up honest A":
            folder = {k: k.split("/")[4] for k in local}  # phase2/runs/<batch>/honest_a/<task folder>/<log>
            base = sorted(f for k, f in local.items() if not RETRY.search(folder[k]))
            retry = sorted(f for k, f in local.items() if RETRY.search(folder[k]))
            row |= {"base task folders": count(base), "rerun/redraw folders": count(retry) | {"folders": sorted(folder[str(f.relative_to(C / QB))] for f in retry)}}
        else:
            row |= {"all logs": count(sorted(local.values()))}
        samples = sum(v.get("samples", 0) for k, v in row.items() if k != "inventory")
        fails = [k for k in ("on S3, not local", "local, not on S3", "byte mismatches", "manifest-listed, not local", "sha256 mismatches") if inv[k]]
        if spec["lists_all"] and inv["local, not in the manifest"]:
            fails.append("local, not in the manifest")
        stated = inv["manifest-stated (logs, trajectories)"]
        if stated and stated[0] is not None and stated[0] != inv["local logs"]:
            fails.append(f"{inv['local logs']} local logs vs {stated[0]} in the manifest")
        if stated and stated[1] != samples:
            fails.append(f"{samples} samples vs {stated[1]} trajectories in the manifest")
        row["inventory verdict"] = fails or "complete"
        out["cohorts"][name] = row
        print(name, json.dumps(row), flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"placeholders_q2_part3_{stamp}.json", "x") as fh:  # one line per cohort
        fh.write('{"git": %s, "cohorts": {\n' % json.dumps(head) + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out["cohorts"].items()) + "\n}}\n")
    print(dest)
    if failed := [k for k, v in out["cohorts"].items() if v["inventory verdict"] != "complete"]:
        raise SystemExit(f"inventory incomplete or inconsistent for: {failed} (verdicts written to {dest.name})")


if __name__ == "__main__":
    main()
