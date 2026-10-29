"""Item 3 follow-up (0d, 10-14): where "side task" sits in the la18 weak-simulator folder's system parts. POST HOC.
Counts, keys and hashes only: no prompt or transcript text is returned, printed or saved. Modal CPU.

Item 3's v5 scan (aswin/q2-item3-probe-sees @ e57a51ac) found the generic word "side task" in 5 system parts, all in
2026-10-11_la18_weaksim_probe-inputs-l1_v1 (b7's claim B weak-simulator control). Its staging (run_weaksim_stage_modal.py
@ 918593817) gives every row the HONEST system prompt of its main task, from the la18 honest cohort. This checks, by hash:
(1) which rows carry it (arm, run key, cell) and how many per arm (the folder has no honest arm);
(2) whether it is one prompt template: each flagged row's normalized system sha256, the sha256 of each normalized line that
    holds the phrase, the system sha of the same cell in every arm, the distinct system shas of each flagged main task, real
    vs R0 system equality over all cells, and how many main tasks' system parts hold each flagged line;
(3) every other non-sealed staged folder: rows of the flagged main tasks (system sha equal or not, phrase present or not),
    rows anywhere whose system part holds a flagged line, and the folders with "la18" in the name.
The system part and its normalization are v5's (`parts_of`, `norm`): system messages joined by newlines, whitespace runs to
one space, lowercased. Sealed folders (oct11c, lean230) are skipped unread, as in v5.

usage: uv run --no-project --with modal==1.6.1 modal run --detach run_sidetask_check.py --git <sha>
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q2-item3_sidetask-check_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "probe-shift_aswin-surya_sub-q2-item3_work_kill-2026-11-13"
FOLDER = "2026-10-11_la18_weaksim_probe-inputs-l1_v1"
SEALED = ("oct11c", "lean230")
PHRASE = "side task"

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS)
work = modal.Volume.from_name(WORK)
img = modal.Image.debian_slim(python_version="3.12")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def h(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def system_raw(messages: list[dict]) -> str:
    out = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            out.append(c if isinstance(c, str) else "".join(x.get("text") or "" for x in c or [] if isinstance(x, dict)))
    return "\n".join(out)


def main_task(r: dict) -> str:
    return r.get("main_task") or (r.get("cell") or "").split(":")[0]


def facts(r: dict) -> dict:
    raw = system_raw(r["messages"])
    s = norm(raw)
    lines = [norm(x) for x in raw.splitlines() if norm(x)]
    hit = [x for x in lines if PHRASE in x]
    pos = s.find(PHRASE)
    return {"arm": r.get("arm"), "key": r.get("key"), "cell": r.get("cell"), "main_task": main_task(r),
            "environment": r.get("environment"), "system_sha": h(s), "system_chars": len(s), "occurrences": s.count(PHRASE),
            "occurrences_inside_lines": sum(x.count(PHRASE) for x in hit), "phrase_line_shas": sorted({h(x) for x in hit}),
            "first_position_fraction": round(pos / len(s), 3) if pos >= 0 else None,
            "line_shas": sorted({h(x) for x in lines})}


@app.function(image=img, volumes={"/acts": acts}, cpu=2, memory=8192, timeout=900)
def scan_folder(path: str) -> list[dict]:
    with open(f"/acts/{path}") as fh:
        return [facts(r) | {"file": path} for r in map(json.loads, fh) if isinstance(r.get("messages"), list)]


@app.function(image=img, volumes={"/acts": acts}, cpu=2, memory=8192, timeout=900)
def scan_other(path: str, mains: list[str], line_shas: list[str]) -> dict:
    mains, line_shas = set(mains), set(line_shas)
    rows, n, nomsg = [], 0, 0
    with open(f"/acts/{path}") as fh:
        for r in map(json.loads, fh):
            if not isinstance(r.get("messages"), list):
                nomsg += 1
                continue
            n += 1
            f = facts(r)
            if f["main_task"] in mains or line_shas & set(f["line_shas"]) or f["occurrences"]:
                rows.append({k: f[k] for k in ("arm", "key", "main_task", "system_sha", "occurrences", "phrase_line_shas")}
                            | {"holds a flagged line": bool(line_shas & set(f["line_shas"]))})
    return {"file": path, "rows scanned": n, "rows without messages": nomsg, "matches": rows}


@app.function(image=img, volumes={"/acts": acts, "/work": work}, cpu=1, memory=4096, timeout=1800)
def run_all(git: str) -> str:
    acts.reload()
    folders = sorted(p.parent.name for p in Path("/acts").glob("*/inputs") if any(p.glob("*.jsonl")))
    sealed = [f for f in folders if any(s in f for s in SEALED)]
    folders = [f for f in folders if f not in sealed]
    files = {f: sorted(str(x.relative_to("/acts")) for x in Path(f"/acts/{f}/inputs").glob("*.jsonl")) for f in folders}
    W = [x for res in scan_folder.map(files[FOLDER]) for x in res]
    flagged = [r for r in W if r["occurrences"]]
    mains = sorted({r["main_task"] for r in flagged})
    flines = sorted({x for r in flagged for x in r["phrase_line_shas"]})
    arms = sorted({r["arm"] for r in W})
    by_cell = defaultdict(dict)
    for r in W:
        by_cell[r["cell"]][r["arm"]] = r["system_sha"]
    real_r0 = Counter("equal" if a.get("real-attack-la18") == a.get("sim-attack-la18-r0") else "differ"
                      for a in by_cell.values() if "real-attack-la18" in a and "sim-attack-la18-r0" in a)
    sys_by_main = defaultdict(set)
    for r in W:
        sys_by_main[r["main_task"]].add(r["system_sha"])
    mains_with_line = {s: sorted({r["main_task"] for r in W if s in r["line_shas"]}) for s in flines}
    others = [x for f in folders if f != FOLDER for x in files[f]]
    O = list(scan_other.map(others, kwargs={"mains": mains, "line_shas": flines}))
    om = [m | {"folder": o["file"].split("/")[0]} for o in O for m in o["matches"]]
    wsys = {m: sorted(sys_by_main[m]) for m in mains}
    out = {
        "POST HOC": True, "git": git, "folder": FOLDER, "volume": ACTS, "files": files[FOLDER],
        "sealed folders skipped (not read)": sealed, "phrase": PHRASE,
        "normalization": "v5: system messages joined by newlines, whitespace runs to one space, lowercased; shas are sha256[:16]",
        "(1) rows by arm": {a: {"rows": sum(r["arm"] == a for r in W), "rows with the phrase in system": sum(r["arm"] == a for r in flagged)}
                            for a in arms},
        "(1) flagged rows": [{k: r[k] for k in ("arm", "key", "cell", "main_task", "environment", "file", "system_sha", "system_chars",
                                               "occurrences", "occurrences_inside_lines", "phrase_line_shas", "first_position_fraction")}
                             for r in sorted(flagged, key=lambda r: (r["cell"], r["arm"]))],
        "(2) flagged cells: system sha in every arm": {c: by_cell[c] for c in sorted({r["cell"] for r in flagged})},
        "(2) flagged main tasks: distinct system shas in the folder": {m: len(sys_by_main[m]) for m in mains},
        "(2) real vs R0 system sha over all cells with both": dict(real_r0),
        "(2) cells and main tasks in the folder": {"cells": len(by_cell), "main tasks": len(sys_by_main),
                                                   "distinct system shas": len({r["system_sha"] for r in W})},
        "(2) flagged phrase lines: main tasks in the folder whose system holds each": mains_with_line,
        "(3) other folders scanned": len({o["file"].split("/")[0] for o in O}),
        "(3) other rows scanned": sum(o["rows scanned"] for o in O),
        "(3) folders with la18 in the name": [f for f in folders + sealed if "la18" in f],
        "(3) rows elsewhere of the flagged main tasks": [
            {k: m[k] for k in ("folder", "arm", "key", "main_task", "occurrences", "holds a flagged line")}
            | {"system sha equals the weaksim folder's": m["system_sha"] in wsys[m["main_task"]]}
            for m in om if m["main_task"] in wsys],
        "(3) rows elsewhere holding a flagged line or the phrase, any main task": [
            {k: m[k] for k in ("folder", "arm", "key", "main_task", "occurrences", "holds a flagged line")}
            for m in om if m["holds a flagged line"] or m["occurrences"]],
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    name = f"sidetask_check_POSTHOC_{stamp}.json"
    Path("/work/results").mkdir(parents=True, exist_ok=True)
    with open(f"/work/results/{name}", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    work.commit()
    return json.dumps({"name": name, "result": out})


@app.local_entrypoint()
def main(git: str):
    res = json.loads(run_all.remote(git))
    d = Path(__file__).parent / "results"
    d.mkdir(exist_ok=True)
    with open(d / res["name"], "x") as fh:
        fh.write(json.dumps(res["result"], indent=1) + "\n")
    print(res["name"])
