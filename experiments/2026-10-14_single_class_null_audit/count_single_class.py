"""Counts only (RULE.md): for each audited job, the units (main tasks, or runs for within-run shuffles) whose labelled rows
all carry one label, so no within-unit shuffle can move them, and the share of labelled rows they hold.
usage (inside the Modal container): python count_single_class.py <jobs.json> <input dir> <out json> <git sha>"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

JOBS, IN, OUT, GIT = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]


def rows_of(src):
    """One input -> list of dicts. JSON lists of dicts; JSONL; or .npz with the named arrays (spec 'npz_fields')."""
    p = IN / src["file"]
    if "table_after" in src:  # a printed table: the rows after the line starting with this marker, until a blank line
        lines = p.read_text().splitlines()
        i = next(k for k, x in enumerate(lines) if x.startswith(src["table_after"])) + 1
        out = []
        for x in lines[i:]:
            if not x.strip():
                break
            v = x.split()
            assert len(v) == len(src["columns"]), f"{p}: bad table row {x!r}"
            out.append(dict(zip(src["columns"], v)))
        return out
    if p.suffix == ".npz":
        import numpy as np
        z = np.load(p, allow_pickle=False)
        cols = {k: (z[k][:, 0] if k in src.get("col0", []) else z[k]).tolist() for k in src["npz_fields"]}  # col0: column 0 = the real labels
        n = len(next(iter(cols.values())))
        assert all(len(v) == n for v in cols.values()), f"{p}: npz fields differ in length"
        return [{k: cols[k][i] for k in cols} for i in range(n)]
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    d = json.loads(p.read_text())
    for k in src.get("path", []):
        d = d[k]
    assert isinstance(d, list), f"{p}: not a list after path {src.get('path')}"
    return d


def count(job):
    rows = []
    for src in job["inputs"]:
        rs = rows_of(src)
        if "mask" in src:  # keep rows whose boolean field is true (e.g. the training rows a shuffle permutes)
            rs = [r for r in rs if bool(r[src["mask"]])]
        for k, v in src.get("where", {}).items():  # exact-match filters, e.g. {"arm": ["sim-attack", "sim-honest"]}
            rs = [r for r in rs if r.get(k) in v]
        rows += rs
    lab, unit = job["label_field"], job["unit_field"]
    labmap = job.get("label_map")  # e.g. {"sim-attack": 1, "sim-honest": 0}; rows whose value is unmapped are dropped
    c = defaultdict(Counter)
    dropped = 0
    for r in rows:
        y = r.get(lab)
        y = labmap.get(str(y)) if labmap is not None else y
        if y is None:
            dropped += 1
            continue
        u = r[unit] if "unit_split" not in job else str(r[unit]).split(job["unit_split"][0])[job["unit_split"][1]]  # e.g. "world|task" -> task
        c[u][str(y)] += 1
    labels = sorted({y for u in c.values() for y in u})
    single = {y: [u for u in c if set(c[u]) == {y}] for y in labels}
    mixed = [u for u in c if len(c[u]) > 1]
    n_rows = sum(sum(v.values()) for v in c.values())
    rows_single = {y: sum(c[u][y] for u in single[y]) for y in labels}
    out = {"job": job["name"], "unit": unit, "label_field": lab, "labels": labels, "rows_labelled": n_rows, "rows_dropped_unmapped": dropped,
           "units": len(c), "units_mixed": len(mixed), "units_single_class": {y: len(single[y]) for y in labels},
           "rows_in_single_class_units": rows_single, "share_rows_unmovable": round(sum(rows_single.values()) / n_rows, 4) if n_rows else None,
           "rows_by_label": {y: sum(c[u][y] for u in c) for y in labels}}
    if "gate" in job:  # exact reproduction of a committed count
        g = job["gate"]
        out["gate_passes"] = all(out[k] == v for k, v in g.items())
    return out


jobs = json.loads(Path(JOBS).read_text())
gate_jobs = [j for j in jobs if "gate" in j]
assert gate_jobs, "no gate job"
res = [count(j) for j in gate_jobs]  # (Codex, post-launch) the gate runs and is checked before any other input is read
if not all(r["gate_passes"] for r in res):
    raise SystemExit(f"gate failed, nothing else counted: {res}")
res += [count(j) for j in jobs if "gate" not in j]
for r in res:
    print(json.dumps(r), flush=True)
Path(OUT).write_text(json.dumps({"label": "counts only (RULE.md)", "git": GIT, "jobs": res}, indent=1))
