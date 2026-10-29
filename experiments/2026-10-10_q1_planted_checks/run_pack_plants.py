"""James's checks (c) and (d), part 2 of 4: per-layer packs of the three readouts (PLAN.md). Modal CPU, one container per
(level, layer); chunk files read whole.

Readouts per tool output: output_end (the last content token) and the moving averages output_end_ema0.95 and
output_end_ema0.8, joined by the output's identity (step, action), never by storage offset.
- Level 0: every run of `2026-10-10_q1plants_l0_qwen3-32b_l8_v1`.
- Level 7: the planted copies of `2026-10-10_q1plants_l7_qwen3-32b_l8_v1`, plus the 75 cells' unplanted real and
  simulated runs from the old ladder. Plain rows come from `2026-10-06_la_segment-end-scrub7_qwen3-32b_alllayers_v1`,
  moving averages from `2026-10-06_la_segment-end-scrub7-ema_qwen3-32b_alllayers_v1`.
  - Reproduction gate first: each of the 10 `-repro` runs must match its existing run. Every output_end and
    moving-average position (kind, step, action, token) must be identical, and every row must have cosine >= 0.9999 and
    relative L2 error <= 0.01 against the existing row. Otherwise nothing is written.
- Families share outputs: per cell, the simulated run and its tell copies are scored on the outputs present in all of
  them, and the real run and its year copies likewise. Both copies of a pair therefore read the same outputs; the
  outputs a planted copy lost to the cut are counted.
Writes `q1plants/<level>/packed_<digest>/L<l>_<readout>.npy` (float16, rows in q1_eval.step_table order) and `runs.L<l>.json`
(arm, key, cell, main_task, environment, fold, positions; `plant` = the planted output's [step, action] on tell copies).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \
      experiments/2026-10-10_q1_planted_checks/run_pack_plants.py
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-q1plants_pack_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
READOUTS = ["output_end", "output_end_ema0.95", "output_end_ema0.8"]
NEW = {"level0": "2026-10-10_q1plants_l0_qwen3-32b_l8_v1", "level7": "2026-10-10_q1plants_l7_qwen3-32b_l8_v1"}
STAGED = {"level0": "2026-10-10_q1plants_l0_inputs_v1", "level7": "2026-10-10_q1plants_l7_inputs_v1"}
# PLAN.md Amendment 1: level 7's neutral-line copies, from their own staging and extraction
NEUTRAL_RUN, NEUTRAL_STAGED = "2026-10-10_q1plants_l7n_qwen3-32b_l8_v1", "2026-10-10_q1plants_l7n_inputs_v1"
OLD7 = {"plain": "2026-10-06_la_segment-end-scrub7_qwen3-32b_alllayers_v1", "ema": "2026-10-06_la_segment-end-scrub7-ema_qwen3-32b_alllayers_v1"}
FAMILY = {"sim-attack": "sim", "sim-attack-tell-early": "sim", "sim-attack-tell-mid": "sim",
          "sim-attack-neutral-early": "sim", "sim-attack-neutral-mid": "sim",
          "real-attack-e1": "real", "real-attack-e1-year2023": "real", "real-attack-e1-year2027": "real"}
COS_MIN, REL_MAX = 0.9999, 0.01

app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_python_source("probe_shift")


def metas(run_dir: Path, arms: set, cells: set | None) -> list[dict]:
    out = []
    for arm in sorted(arms):
        for f in sorted((run_dir / "meta" / arm).glob("*.json")):
            m = json.loads(f.read_text())
            if cells is None or m["cell"] in cells:
                out.append(m)
    return out


def rows_of(m: dict) -> dict:
    """{readout: {(step, action): row}} over kept output positions with a row (plain and moving-average kinds)."""
    out = {}
    for p in m["positions"]:
        if p["kind"] in READOUTS and p.get("kept", True) and p.get("row") is not None:
            out.setdefault(p["kind"], {})[(p["step"], p["action"])] = p["row"]
    return out


def tokens_of(m: dict) -> dict:
    return {(p["kind"], p["step"], p["action"]): p["token"] for p in m["positions"] if p["kind"] in READOUTS and p.get("kept", True)}


class Chunks:
    """Whole-file chunk reads (row-indexed memory maps fetch page by page from the Volume), one file held at a time."""

    def __init__(self, run_dir: Path, layer: int):
        import numpy as np

        self.np, self.d, self.name, self.a = np, run_dir / "by_layer" / f"L{layer}", None, None

    def rows(self, m: dict, rs: list[int]):
        if m["chunk"] != self.name:
            self.a, self.name = self.np.load(self.d / f"{m['chunk']}.npy"), m["chunk"]
        return self.a[m["row_start"] + self.np.array(rs, int)]


@app.function(image=image, cpu=4, memory=16384, volumes={"/acts": vol}, timeout=3600, retries=0)
def pack(level: str, layer: int, cells: list[str], fold_of: dict, digest: str) -> dict:
    import numpy as np

    acts = Path("/acts")
    staged = json.loads((acts / STAGED[level] / "stage_report.json").read_text())
    plants = {(f"sim-attack-tell-{t['position']}", t["cell"]): t["step_action"] for t in staged["tell"]}
    now = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((acts / staged["source"] / "inputs").glob("shard_*.jsonl"))}
    if now != staged["source_shard_sha256"]:
        raise ValueError(f"{staged['source']} changed since staging")
    new = metas(acts / NEW[level], set(FAMILY) | {"sim-attack-repro", "real-attack-e1-repro"}, None)
    by = {(m["arm"], m["cell"]): m for m in new}
    # the complete manifest staging wrote, and nothing else
    expect = set(plants) | {(f"real-attack-e1-year{y}", t["cell"]) for t in staged["years"] for y in (2023, 2027)}
    expect |= ({(a, c) for c in cells for a in ("real-attack-e1", "sim-attack")} if level == "level0" else
               {(f"{a}-repro", c) for c in cells[:5] for a in ("real-attack-e1", "sim-attack")})
    if set(by) != expect:
        raise ValueError(f"{NEW[level]}: {len(expect - set(by))} staged runs not extracted, {len(set(by) - expect)} unexpected")
    src = {(m["arm"], m["cell"]): ("new", m) for m in new if m["arm"] in FAMILY}
    if level == "level7":  # the neutral line's copies, checked the same way
        staged_n = json.loads((acts / NEUTRAL_STAGED / "stage_report.json").read_text())
        plants_n = {(f"sim-attack-neutral-{t['position']}", t["cell"]): t["step_action"] for t in staged_n["tell"]}
        at = lambda d: {(a.rsplit("-", 1)[1], c): tuple(sa) for (a, c), sa in d.items()}  # (early|mid, cell) -> (step, action)
        if at(plants_n) != at(plants):
            raise ValueError("the neutral copies' positions differ from the tell's")
        new_n = metas(acts / NEUTRAL_RUN, {"sim-attack-neutral-early", "sim-attack-neutral-mid"}, None)
        if {(m["arm"], m["cell"]) for m in new_n} != set(plants_n):
            raise ValueError(f"{NEUTRAL_RUN}: the extracted neutral copies differ from the staged ones")
        plants |= plants_n
        src |= {(m["arm"], m["cell"]): ("newn", m) for m in new_n}
    report = Counter()
    if level == "level7":  # the unplanted runs from the old ladder, after the reproduction gate
        old = {k: {(m["arm"], m["cell"]): m for m in metas(acts / OLD7[k], {"real-attack-e1", "sim-attack"}, set(cells))} for k in OLD7}
        for k in OLD7:  # same staged inputs as this staging read (per-file hashes were not recorded in 2026-10-06's runs)
            if json.loads((acts / OLD7[k] / "config.json").read_text())["inputs_from"] != staged["source"]:
                raise ValueError(f"{OLD7[k]} read other inputs than {staged['source']}")
            if set(old[k]) != {(a, c) for c in cells for a in ("real-attack-e1", "sim-attack")}:
                raise ValueError(f"{OLD7[k]} lacks some of the 75 cells' unplanted runs")
        gate = {"runs": 0, "outputs": 0, "min_cos": 1.0, "max_rel": 0.0}
        chunks = {k: Chunks(acts / OLD7[k], layer) for k in OLD7} | {"new": Chunks(acts / NEW[level], layer)}
        for (arm, cell), m in sorted((k, v) for k, v in by.items() if k[0].endswith("-repro")):
            base = arm.removesuffix("-repro")
            o_plain, o_ema = old["plain"][(base, cell)], old["ema"][(base, cell)]
            want = tokens_of(m)
            have = tokens_of(o_plain) | tokens_of(o_ema)
            if want != {k: v for k, v in have.items() if k[0] in READOUTS}:
                raise ValueError(f"reproduction gate: {arm} {cell} positions differ from the existing extraction")
            for rd in READOUTS:
                o = o_plain if rd == "output_end" else o_ema
                keys = sorted(rows_of(m)[rd])
                a = chunks["new"].rows(m, [rows_of(m)[rd][k] for k in keys]).astype(np.float32)
                b = chunks["plain" if rd == "output_end" else "ema"].rows(o, [rows_of(o)[rd][k] for k in keys]).astype(np.float32)
                cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
                rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
                if not (len(keys) and np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(cos).all() and np.isfinite(rel).all()):
                    raise ValueError(f"reproduction gate: {arm} {cell} {rd}: no rows or non-finite values at L{layer}")
                gate["outputs"] += len(keys)
                gate["min_cos"], gate["max_rel"] = float(np.minimum(gate["min_cos"], cos.min())), float(np.maximum(gate["max_rel"], rel.max()))
            gate["runs"] += 1
        if gate["runs"] != 10 or gate["min_cos"] < COS_MIN or gate["max_rel"] > REL_MAX:
            raise ValueError(f"reproduction gate failed at L{layer}: {gate}")
        report["gate"] = gate
        for k in ("real-attack-e1", "sim-attack"):
            for c in cells:
                src[(k, c)] = ("old", old["plain"][(k, c)], old["ema"][(k, c)])
    # Families share outputs: the identities kept with all three readouts in every member present.
    outs = {}
    for (arm, cell), s in src.items():
        r = rows_of(s[1]) if s[0] in ("new", "newn") else rows_of(s[1]) | rows_of(s[2])
        outs[(arm, cell)] = set.intersection(*(set(r.get(rd, {})) for rd in READOUTS))
    common = {}
    for (arm, cell), o in outs.items():
        key = (FAMILY[arm], cell)
        common[key] = o if key not in common else common[key] & o
    runs, X = [], {rd: [] for rd in READOUTS}
    chunks = {"new": Chunks(acts / NEW[level], layer)} | ({k: Chunks(acts / OLD7[k], layer) for k in OLD7} | {"newn": Chunks(acts / NEUTRAL_RUN, layer)}
                                                          if level == "level7" else {})
    for (arm, cell), s in sorted(src.items(), key=lambda kv: (kv[1][1]["chunk"], kv[0])):
        keys = sorted(common[(FAMILY[arm], cell)])
        report[f"{arm}: outputs dropped to the family's common set"] += len(outs[(arm, cell)]) - len(keys)
        if not keys:
            report[f"{arm}: runs without a common output"] += 1
            continue
        for rd in READOUTS:
            if s[0] in ("new", "newn"):
                m, ch = s[1], chunks[s[0]]
            else:
                m, ch = (s[1], chunks["plain"]) if rd == "output_end" else (s[2], chunks["ema"])
            X[rd].append(ch.rows(m, [rows_of(m)[rd][k] for k in keys]).astype(np.float16))
        m = s[1]
        tok = tokens_of(m)
        pos = [{"kind": "output_end", "step": k[0], "action": k[1], "tool": None, "token": tok.get(("output_end",) + k), "row": j, "kept": True}
               for j, k in enumerate(keys)]
        runs.append({"arm": arm, "key": m["key"], "cell": cell, "main_task": m["main_task"], "environment": m["environment"],
                     "fold": fold_of[m["main_task"]], "positions": pos} | ({"plant": plants[(arm, cell)]} if (arm, cell) in plants else {}))
    from probe_shift.q1_eval import step_table

    t = step_table(runs, "output_end")
    out = Path("/acts") / "q1plants" / level / f"packed_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    for rd in READOUTS:
        A = np.concatenate(X[rd])
        if len(A) != len(t["run"]) or not np.isfinite(A).all():
            raise ValueError(f"{rd}: {len(A)} rows vs step table {len(t['run'])}, or non-finite values")
        with open(out / f".L{layer}_{rd}.tmp", "wb") as fh:
            np.save(fh, A)
        (out / f".L{layer}_{rd}.tmp").replace(out / f"L{layer}_{rd}.npy")
    (out / f".runs.L{layer}.tmp").write_text(json.dumps(runs))
    (out / f".runs.L{layer}.tmp").replace(out / f"runs.L{layer}.json")
    vol.commit()
    return {"level": level, "layer": layer, "rows": len(t["run"]), "runs": dict(Counter(r["arm"] for r in runs)), "report": dict(report)}


@app.local_entrypoint()
def main(levels: str = "level0,level7"):
    cells = json.loads((HERE / "config" / "sample_cells.json").read_text())["cells"]
    fold_of = json.loads((ROOT / "splits" / "folds_v2.json").read_text())["task_folds"]["main_task_to_fold"]
    for lv in levels.split(","):
        spec = {"level": lv, "new": NEW[lv], "old": OLD7 if lv == "level7" else None, "cells": cells, "readouts": READOUTS}
        if lv == "level7":  # Amendment 1: level 7's pack adds the neutral copies (level 0's digest is unchanged)
            spec["neutral"] = NEUTRAL_RUN
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
        print(f"{lv}: pack digest {digest}", flush=True)
        for r in pack.starmap([(lv, l, cells, fold_of, digest) for l in LAYERS]):
            print(json.dumps(r), flush=True)
