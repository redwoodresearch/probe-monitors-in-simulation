"""0d for Aswin (10-16 ~18:1x): attribute October's OpenAI API bill. For every .eval log reachable on the project's Modal Volumes
(chain evals_v1, q4own evals_v1, the pilots Volume, the q2 fresh-honest Volume), read the HEADER ONLY (header_only=True: no samples) and
return keys and numbers: the run's path parts (launch / arm / task folder names), status, start time, sample count, and Inspect's
stats.model_usage per model (input, output, cache-read, cache-write, reasoning tokens). Model names must be model-id shaped. Priced
inside the same Modal job by analyze_openai_usage.analyze() at the sims harness's own table (src/sim/prices.py @ ba990399); the local
entrypoint only reads git (the INPUTS_FULL inventories' routed keys) and writes the two records. Single remote driver.
  modal run run_openai_usage_scan.py --inputs-full <path> --commit <sha> --out results/openai_usage_<ts>.json
"""
import json
import pathlib
import re
import subprocess

import modal

APP = "probe-shift_aswin-surya_sub-sp493_openaiusage_kill-2026-10-23"
VOLS = {"chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "q4own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
        "pilots": "probe-shift_aswin-surya_sub-q3pilots_runs_kill-2026-11-13", "q2fresh": "probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13"}
ROOTS = {"chain": "evals_v1", "q4own": "evals_v1", "pilots": "", "q2fresh": ""}  # where each Volume keeps its .eval logs
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]{1,40}/[A-Za-z0-9_.:/-]{1,80}")
PART = re.compile(r"[A-Za-z0-9_.@:+=-]{1,160}")
ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
SIDECARS = (".request_logs", ".docker", ".launcher", ".resource")  # launcher sidecar folders next to a launch (pruned by substring)
FIELDS = ("input_tokens", "output_tokens", "input_tokens_cache_read", "input_tokens_cache_write", "reasoning_tokens", "total_tokens")
HERE = pathlib.Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.12").pip_install("inspect_ai==0.3.251", "zstandard==0.25.0")
         .add_local_file(HERE / "analyze_openai_usage.py", "/root/analyze_openai_usage.py"))
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v) for k, v in VOLS.items()}  # no version: the pilots Volume is v1, the others v2 (first launch stopped on it)


@app.function(volumes={f"/{k}": v.read_only() for k, v in vols.items()}, cpu=2, memory=4096, timeout=3600, retries=0)
def headers(chunk: list) -> list:
    from inspect_ai.log import read_eval_log

    out = []
    for vol, rel in chunk:
        parts = rel.split("/")
        rec = {"vol": vol, "parts": [p if PART.fullmatch(p) else "<non-name>" for p in parts]}
        try:
            log = read_eval_log(f"/{vol}/{ROOTS[vol]}/{rel}".replace("//", "/"), header_only=True)
            mu = {}
            for m, u in ((log.stats.model_usage or {}) if log.stats else {}).items():
                name = m if isinstance(m, str) and MODEL_ID.fullmatch(m) else "<non-model-id>"
                prev = mu.setdefault(name, {f: 0 for f in FIELDS})  # += so two non-model-id names never overwrite (review #7)
                for f in FIELDS:
                    prev[f] += int(getattr(u, f, None) or 0)
            res = log.results
            ids = {k: (v if isinstance(v, str) and ID.fullmatch(v) else (None if v is None else "<non-id>"))
                   for k, v in (("eval_id", log.eval.eval_id), ("run_id", log.eval.run_id), ("task_id", log.eval.task_id),
                                ("eval_set_id", getattr(log.eval, "eval_set_id", None)))}
            out.append(rec | ids | {"status": log.status, "started_at": log.stats.started_at if log.stats else None,
                                    "completed_at": log.stats.completed_at if log.stats else None,
                              "n_samples": getattr(res, "total_samples", None) if res else None, "model_usage": mu})
        except Exception as e:
            out.append(rec | {"read_error": type(e).__name__})
    return out


@app.function(volumes={f"/{k}": v.read_only() for k, v in vols.items()}, cpu=4, memory=16384, timeout=3600, retries=0)
def drive(routed: list, label: str) -> dict:
    import os
    import sys

    sys.path.insert(0, "/root")
    from analyze_openai_usage import analyze

    files, roots_seen = [], {}
    for vol, sub in ROOTS.items():
        base = f"/{vol}/{sub}".rstrip("/")
        n = 0
        if os.path.isdir(base):
            def boom(e):
                raise e

            for dp, dns, fs in os.walk(base, onerror=boom):  # an unreadable folder fails loudly (review #10)
                dns[:] = [d for d in dns if not any(t in d for t in SIDECARS)]
                for f in fs:
                    if f.endswith(".eval"):
                        files.append((vol, os.path.relpath(os.path.join(dp, f), base)))
                        n += 1
        roots_seen[vol] = {"root": base, "exists": os.path.isdir(base), "n_eval": n}
    chunks = [files[i:i + 150] for i in range(0, len(files), 150)]
    rows = [x for part in headers.map(chunks) for x in part]
    scan = {"roots": roots_seen, "n_files": len(files), "rows": rows}
    try:  # the analysis runs here, on Modal (Aswin's 10-13 rule); a failure keeps the scan rows (review r3 #6)
        return {"scan": scan, "analysis": analyze(scan, routed, label)}
    except Exception as e:
        return {"scan": scan, "analysis_error": type(e).__name__}


@app.function(cpu=4, memory=16384, timeout=1800, retries=0)
def reanalyze_remote(scan: dict, routed: list, label: str) -> dict:
    import sys

    sys.path.insert(0, "/root")
    from analyze_openai_usage import analyze
    return analyze(scan, routed, label)


@app.local_entrypoint()
def reanalyze(scan_rows: str, inputs_full: str, commit: str, out: str):
    """re-run the analysis ON MODAL from a saved scan-rows file (no rescan): modal run run_openai_usage_scan.py::reanalyze ..."""
    f = open(out, "x")
    scan = json.load(open(scan_rows))
    full = json.loads(subprocess.check_output(["git", "show", f"{commit}:{inputs_full}"]))
    routed = set()
    for p in full["inventories"].values():
        routed |= {re.sub(r"#e\d+$", "", r["key"]) for r in json.loads(subprocess.check_output(["git", "show", f"{p['commit']}:{p['path']}"]))["runs"]}
    a = reanalyze_remote.remote(scan, sorted(routed), f"{inputs_full} @ {commit}")
    json.dump({"what": __doc__.strip().splitlines()[0], "app": APP, "volumes": VOLS, "scan_rows": scan_rows, "n_routed_keys": len(routed)} | a, f, indent=1)
    f.close()
    print("runs", a["n_runs"], "total $", round(a["total_usd"]), "by family", a["by_family"], "repeats", a["repeats"])


@app.local_entrypoint()
def main(inputs_full: str, commit: str, out: str):
    f = open(out, "x")  # before any remote work
    g = open(out.replace(".json", "_scan_rows.json"), "x")
    full = json.loads(subprocess.check_output(["git", "show", f"{commit}:{inputs_full}"]))
    routed = set()
    for p in full["inventories"].values():  # git reads only: the routed run keys
        routed |= {re.sub(r"#e\d+$", "", r["key"]) for r in json.loads(subprocess.check_output(["git", "show", f"{p['commit']}:{p['path']}"]))["runs"]}
    res = drive.remote(sorted(routed), f"{inputs_full} @ {commit}")  # the one remote call; listing, fan-out and analysis happen inside Modal
    json.dump({"app": APP, "roots": res["scan"]["roots"], "n_files": res["scan"]["n_files"], "rows": res["scan"]["rows"]}, g)
    g.close()
    if "analysis_error" in res:
        f.close()
        raise SystemExit(f"analysis failed in Modal: {res['analysis_error']} (scan rows saved)")
    json.dump({"what": __doc__.strip().splitlines()[0], "app": APP, "volumes": VOLS, "n_routed_keys": len(routed)} | res["analysis"], f, indent=1)
    f.close()
    a = res["analysis"]
    print("files", res["scan"]["n_files"], "runs", a["n_runs"], "read errors", a["n_read_errors"], "headerless", a["n_headerless_usage_unknown"],
          "total $", round(a["total_usd"]), "by family", a["by_family"])
