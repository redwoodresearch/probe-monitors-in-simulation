"""The 196-task impactful-call monitor's calls, with each call's command text, for the command-text classifier. Local, $0.

The call set is the probe's, rebuilt exactly as run_score_onpolicy.load builds it (config/score_scaleup196.yaml: the five
segment-end extractions, worlds real = arm `attack`, sim = `sim_attack@september`, run2 = `attack` of the second attack
run; counted attempts only; rule-1 command rows with a readout labelled impactful or read_only; the real-and-sim
cohort), and it must equal the probe's observed-draw file (meta and folds) row for row, or this raises. Each row's text
is the value of its rule-1 command parameter, read from the eval's model event (x-irid = the row's irid, tool call
number `call_index`), with attachments resolved; the tool name, parameter and rule-1 label recomputed from that call
must equal the row's, or this raises. Every eval file is checked against its run's S3 manifest sha256.

Output: the text table in the gitignored data/ folder (command text never enters git or results) and a summary of counts
and hashes in results/. No text is printed.

  PYTHONPATH=<repo>/src uv run --no-project --python 3.14 python build_text_calls.py --metas <acts_meta root> \
      --evals <phase2 runs cache> --draw0 <onpolicy_2x2_scaleup196_draws0-0_*.json>
"""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.calls import fingerprint, label_call
from probe_shift.reader import command_param

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
PILOT = ROOT / "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
EXTRACTIONS = {"2026-10-09_rerun_strict_segends_v1": "2026-10-09_000523_rerun_strict",
               "2026-10-10_scaleup_canary_segends_v1": "2026-10-09_173815_scaleup_canary",
               "2026-10-10_scaleup_l2_segends_v1": "2026-10-09_182505_scaleup",
               "2026-10-10_scaleup_topup_segends_v1": "2026-10-10_001314_scaleup_topup",
               "2026-10-10_second_attack_segends_v1": "2026-10-10_001314_second_attack"}
WORLDS = {"real": ("attack", None), "sim": ("sim_attack@september", None), "run2": ("attack", "2026-10-10_second_attack_segends_v1")}
REF = re.compile(r"attachment://[0-9a-f]{32}")


def frozen_folds(pilot_file: Path) -> dict:
    """Amendment 25's folds, as run_score_onpolicy.frozen_folds: (rank within the environment + environment index) mod 5."""
    d = json.loads(pilot_file.read_text())
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs = sorted(d["eligible_mains_by_env"], key=h)
    by_env = {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    if [c["env"] for c in d["cells"][: len(envs)]] != envs or any(v != sorted(v, key=h) for v in by_env.values()):
        raise ValueError("the pilot file's order is not the sha256 order")
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


def world_of(arm, run):
    hits = [w for w, (a, x) in WORLDS.items() if a == arm and x == run] or [w for w, (a, x) in WORLDS.items() if a == arm and x is None]
    return hits[0] if hits else None


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest_shas(evals: Path, run_dir: str) -> dict:
    m = json.loads((evals / f"{run_dir}.manifest_v1.json").read_text())
    files = m.get("files") or m.get("objects") or []
    out = {}
    for f in files:
        key = f.get("key") or f.get("path") or f.get("file")
        if key and f.get("sha256"):
            out[key.split(f"{run_dir}/", 1)[-1]] = f["sha256"]
    return out


def main(metas: str, evals: str, draw0: str):
    mroot, eroot = Path(metas), Path(evals)
    folds = frozen_folds(PILOT)
    rows, counted = [], {}
    for x in EXTRACTIONS:  # the probe's loader, step for step
        done = json.loads((mroot / x / "complete.json").read_text())
        files = sorted((mroot / x / "meta").glob("*/*.json"))
        if len(files) != done["epochs"]:
            raise ValueError(f"{x}: {len(files)} metas vs {done['epochs']} epochs")
        for f in files:
            m = json.loads(f.read_text())
            world = world_of(m["arm"], x)
            if world is None or m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            counted.setdefault((world, m["task"]), set()).add(m["run"])
            for c in m["calls"]:
                if c.get("kind", "command") == "command" and c["row"] is not None and c["label"] in ("impactful", "read_only"):
                    rows.append({"world": world, "task": m["task"], "irid": c["irid"], "call": c["call"], "y": int(c["label"] == "impactful"),
                                 "call_index": c["call_index"], "name": c["name"], "param": c["param"], "label": c["label"],
                                 "extraction": x, "run": m["run"]})
    dup = {k: v for k, v in counted.items() if len(v) > 1}
    if dup:
        raise ValueError(f"{len(dup)} world/task pairs have several counted runs")
    with_calls = {w: {r["task"] for r in rows if r["world"] == w} for w in WORLDS}
    cohort = with_calls["real"] & with_calls["sim"]
    if cohort != with_calls["real"] & with_calls["sim"] & with_calls["run2"]:
        raise ValueError("run2 changes the real-and-sim cohort")
    rows = [r for r in rows if r["task"] in cohort]
    D0 = json.loads(Path(draw0).read_text())
    mine = [{k: r[k] for k in ("world", "task", "irid", "call", "y")} for r in rows]
    if mine != D0["meta"] or folds != D0["folds"]:
        raise ValueError("the rebuilt call set or folds differ from the probe's observed-draw file")
    # the command text, from the evals; every file checked against its run's S3 manifest
    shas, ev_cache, checked = {}, {}, Counter()
    texts = []
    for r in rows:
        rd = EXTRACTIONS[r["extraction"]]
        if rd not in shas:
            shas[rd] = manifest_shas(eroot, rd)
        p = eroot / rd / r["run"]
        if p not in ev_cache:
            if shas[rd].get(r["run"]) != sha_file(p):
                raise ValueError(f"{rd}/{r['run']}: sha256 differs from the S3 manifest")
            checked["eval files sha-checked"] += 1
            with zipfile.ZipFile(p) as z:
                (n,) = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
                s = json.loads(z.read(n))
            by = {}
            for e in s.get("events") or []:
                if e.get("event") != "model":
                    continue
                irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
                tcs = ((((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls")) or []
                if irid is not None:
                    if irid in by and by[irid] != tcs:
                        raise ValueError(f"{p}: request {irid} logged twice with different calls")
                    by[irid] = tcs
            ev_cache = {p: (by, s.get("attachments") or {})}  # one eval at a time: rows are grouped by run
        by, att = ev_cache[p]
        tc = by[r["irid"]][r["call_index"]]
        args = {k: (att[v.removeprefix("attachment://")] if isinstance(v, str) and REF.fullmatch(v) else v)
                for k, v in (tc.get("arguments") or {}).items()}
        name = tc.get("function")
        if name != r["name"] or command_param(name, args) != r["param"] or label_call(name, args)["label"] != r["label"]:
            raise ValueError(f"{r['run']} {r['irid']}#{r['call_index']}: tool, parameter or label disagrees with the extraction")
        v = args[r["param"]]
        if not isinstance(v, str):
            raise ValueError(f"{r['irid']}#{r['call_index']}: command value is not a string")
        texts.append(v)
        checked["rows: tool, parameter and label recomputed and equal"] += 1
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    table = {"meta": mine, "folds": folds, "text": texts}
    data_path = HERE / "data" / f"text_calls_{stamp}.json"
    data_path.parent.mkdir(exist_ok=True)
    with data_path.open("x") as fh:
        fh.write(json.dumps(table))
    git = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=HERE).stdout.strip())}
    summary = {"git": git, "labeler": fingerprint(), "written": stamp, "draw0_file": Path(draw0).name, "draw0_sha256": sha_file(Path(draw0)),
               "calls": len(rows), "tasks": len(cohort), "matches_probe_call_set": True, "checks": dict(checked),
               "by_world": {w: {"impactful": sum(r["world"] == w and r["y"] == 1 for r in rows),
                                "read_only": sum(r["world"] == w and r["y"] == 0 for r in rows)} for w in WORLDS},
               "text_sha256": hashlib.sha256(json.dumps(texts).encode()).hexdigest(),
               "table_file": str(data_path.relative_to(HERE)), "table_sha256": sha_file(data_path)}
    out = HERE / "results" / f"text_calls_summary_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("x") as fh:
        fh.write(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("calls", "tasks", "checks", "by_world", "text_sha256")}))
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metas", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--draw0", required=True)
    main(**vars(ap.parse_args()))
