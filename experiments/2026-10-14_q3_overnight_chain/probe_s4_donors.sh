# Read-only key check (09/d9 10-15 ~06:58): the S4 main's honest_a/honest_b counted attempts for the 19 4d cells on this box:
# launcher state, counted attempt's .eval count and sha256. Keys, states and hashes only; never opens a transcript.
date '+probe at %Y-%m-%d %H:%M:%S %Z'; hostname
python3 - "nm264_tiktok,nm149_model_registry,nm307_user_tracking,nm110_media_processing,nm258_tiktok,nm100_file_indexing,nm107_media_processing,nm187_shared_dotfiles,nm201_slack_clone,nm179_shared_dotfiles,nm020_auto_workflow,nm157_model_registry,nm057_clinical_trial,nm332_web_scraping,nm337_web_scraping,nm082_file_indexing,nm228_slack_clone,nm037_auto_workflow,nm356_web_scraping" <<'PY'
import json, os, sys, hashlib, glob
cells = sys.argv[1].split(",")
R = os.path.expanduser("~/phase2/nm/runs"); run = "2026-10-14_newmains_s4"; D = f"{R}/{run}"
out = {"run_dir_exists": os.path.isdir(D), "launch_json": os.path.isfile(f"{D}/launch.json"),
       "top": sorted(os.listdir(D))[:20] if os.path.isdir(D) else None,
       "refs": len(glob.glob(f"{D}/references/*.eval")) if os.path.isdir(D) else None}
if out["launch_json"]:
    d = json.load(open(f"{D}/launch.json")); out["launch_sha256"] = hashlib.sha256(open(f"{D}/launch.json","rb").read()).hexdigest()
    rows = {}
    for r in d["runs"]:
        if r["task"] not in cells: continue
        for a in r.get("attempts") or []:
            if a.get("redrawn_because") or a.get("kind") == "rerun": continue
            rel = (a.get("log_dir") or "").split(f"{run}/", 1)[-1]; arm = rel.split("/", 1)[0]
            if arm not in ("honest_a", "honest_b"): break
            evs = sorted(glob.glob(f"{D}/{rel}/*.eval"))
            rows[f"{arm}/{r['task']}"] = {"state": r.get("state"), "evals": len(evs),
                "sha256": hashlib.sha256(open(evs[0],"rb").read()).hexdigest() if len(evs) == 1 else None}
            break
    out["rows"] = rows
print(json.dumps(out))
PY
