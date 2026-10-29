"""Compose-file race check (d9, 2026-10-15 ~00:10 PT), ON ONE BOX, read-only: every attempt of this fork's launches
(scaleup311 mains, follow-ups and the Q2 rerun; the canary launches) against every OTHER launcher's attempts on the same box
and settings folder. control-tower writes one generated compose file per environment and tag
(`.compose-<tag>-<md5(compose path)>.yml`, control_tower/sandbox/docker.py:143-152, a plain write_text), and one launcher's
90 s stagger only orders its own launches. The written content is a function of the run's recorded inputs: the settings
folder, PHASE2_DIGEST_SET (the replayed digest set; [] = tag form), the ENV_IMAGE_TAG_* pins, INTERNET_SIMULATOR_IMAGE and
the launcher's IMAGES_JSON/digest-set file. So:
  affected   = an attempt of another launcher in the same environment and settings folder ran at an overlapping time
               (each attempt's [start - 90 s, end]);
  clean      = affected, every overlapping attempt's recorded inputs equal this attempt's (both launchers wrote the same
               bytes), and this attempt has no compose/YAML-related pre-start failure or error;
  mismatched = affected and some overlapping attempt's recorded inputs differ (the file it read may hold the other's bytes);
  failed_loud= affected, inputs equal, but a compose/YAML-related pre-start failure or error is recorded (a torn read).
Keys, times and input hashes only; error strings are only searched by script for the words compose/yaml.
  python3 compose_race_check.py <box label>   (prints one JSON line)"""
import glob, hashlib, json, sys
from datetime import datetime, timedelta
from pathlib import Path

box = sys.argv[1]
INPUT_KEYS = ("PHASE2_DIGEST_SET", "INTERNET_SIMULATOR_IMAGE", "CONTROL_TOWER_SETTINGS_DIR")
MINE = ("newmains",)  # new-mains copy of the scale-up fork's check @ eac98f598: only this line changed


def t(s):
    return datetime.fromisoformat(s) if s else None


attempts = []
for f in sorted(glob.glob(str(Path.home() / "phase2/runs/*/launch.json")) + glob.glob(str(Path.home() / "phase2/nm/runs/*/launch.json"))):
    launch = f.split("/runs/")[1].split("/")[0]
    d = json.loads(Path(f).read_text())
    cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
    settings = cfg.get("settings_dir")
    pilot = d.get("pilot", {})
    for r in d.get("runs") or []:  # new-mains copy: a launch record without "runs" (another schema) contributes nothing
        if "task" not in r:  # new-mains copy: launch records of another schema (sims-astra sim-honest) carry no task; skipped
            continue
        task = r["task"] if isinstance(r.get("task"), str) else r["task"]["key"]
        key_env = task.split("_", 1)[1].removesuffix("_r2") if "_" in task else None  # t487_auto_workflow -> auto_workflow
        for n, a in enumerate(r["attempts"]):
            ev = a.get("env_vars") or {}
            env = key_env  # the environment the run started (its compose file); LA_ENV is cross-checked below
            inputs = {k: ev.get(k) for k in INPUT_KEYS}
            inputs |= {k: v for k, v in ev.items() if k.startswith("ENV_IMAGE_TAG_")}
            inputs |= {"settings_dir": settings, "digest_sets": pilot.get("digest_sets"), "images_json": ev.get("IMAGES_JSON")}
            text = json.dumps([a.get("prestart"), (a.get("outcome") or {}).get("errors"), (a.get("outcome") or {}).get("sample_errors")]).lower()
            attempts.append({"launch": launch, "mine": any(m in launch for m in MINE), "key": r["key"], "n": n + 1, "env": env,
                             "real": r["real"], "start": t(a.get("start")), "end": t(a.get("end")) or t(d.get("finished")),
                             "inputs": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12],
                             "settings": settings, "la_env_differs": bool(ev.get("LA_ENV")) and ev.get("LA_ENV") != key_env, "loud": ("compose" in text) or ("yaml" in text)})
LIVE_END = datetime.fromisoformat("2026-10-15T23:59:59-07:00")
assert all(a["env"] for a in attempts), "an attempt without an environment"
out = {"box": box, "launches": {}, "affected": [], "clean": 0, "mismatched": [], "failed_loud": [], "unaffected": 0, "no_start": 0,
       "la_env_differs_all_attempts": sum(a["la_env_differs"] for a in attempts), "attempts_all_launchers": len(attempts)}
for a in attempts:
    if not a["mine"]:
        continue
    L = out["launches"].setdefault(a["launch"], {"attempts": 0, "affected": 0, "clean": 0, "mismatched": 0, "failed_loud": 0})
    L["attempts"] += 1
    if a["start"] is None:
        out["no_start"] += 1
        continue
    lo, hi = a["start"] - timedelta(seconds=90), a["end"] or LIVE_END
    others = [b for b in attempts if b["launch"] != a["launch"] and b["env"] == a["env"] and b["settings"] == a["settings"]
              and b["start"] is not None and b["start"] - timedelta(seconds=90) <= hi and (b["end"] or LIVE_END) >= lo]
    if not others:
        out["unaffected"] += 1
        continue
    L["affected"] += 1
    rec = {"launch": a["launch"], "key": a["key"], "attempt": a["n"], "env": a["env"], "real": a["real"],
           "overlaps": sorted({f"{b['launch']}:{'real' if b['real'] else 'sim'}" for b in others})}
    out["affected"].append(rec)
    if any(b["inputs"] != a["inputs"] for b in others):
        L["mismatched"] += 1
        out["mismatched"].append(rec | {"inputs": a["inputs"], "other_inputs": sorted({b["inputs"] for b in others})})
    elif a["loud"]:
        L["failed_loud"] += 1
        out["failed_loud"].append(rec)
    else:
        L["clean"] += 1
        out["clean"] += 1
print(json.dumps(out, default=str))
