"""POST HOC (RULE.md): streaks of empty tool outputs per run, from extraction metadata only, and the share of the
monitors' false alarms inside runs with long streaks. $0, local. Never reads `token_text` or any transcript text.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-12_q3_empty_streaks/analyze_empty_streaks.py \
      --acting-metas <dir: <extraction>/<arm>/*.json> --draws0 <onpolicy_2x2_scaleup196_draws0-0_2026-10-10_141824.json> \
      --replay <dir with the five replay_readouts_*segends*.json> \
      --reading-metas <dir: <extraction>/<arm>/*.json> --refit <dir with the refit files named in RULE.md>
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
DRAWS0_SHA = "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf"
AGG_SHA = "31f88cd5da9594a18235ca43abb5088360f8fb082ed8bef2a398b2672cd02cfb"
ACT_EXTR = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1",
            "2026-10-10_scaleup_topup_segends_v1"]
RUN2_EXTR = "2026-10-10_second_attack_segends_v1"
ACT_WORLDS = {"real": [(e, "attack") for e in ACT_EXTR], "sim": [(e, "sim_attack@september") for e in ACT_EXTR],
              "run2": [(RUN2_EXTR, "attack")], "honest_a": [(e, "honest_a") for e in ACT_EXTR],
              "honest_b": [(e, "honest_b") for e in ACT_EXTR], "sim_honest": [(ACT_EXTR[0], "sim_honest@september")]}
CELL = {"real": "real->real", "sim": "real->sim", "run2": "real->run2"}
READ_EXTR = "2026-10-06_la_segment-end-v4_qwen35-27b_alllayers_v1"
READ_ARMS = ["real-attack-e1", "real-attack-e2", "sim-attack"]
READERS = {"qwen3.5": ("refit_qwen35_rows.json", "refit_qwen35_scores_boot_0000.npz", "refit_qwen35_probes_boot_0000.npz"),
           "qwen3-32b": ("q32_rows.json", "q32_scores_boot_0000.npz", "q32_probes_boot_0000.npz")}
THRESHOLDS = (20, 50)
REPLAY = {"2026-10-09_rerun_strict_segends_v1": "replay_readouts_rerun_strict_segends_2026-10-09_135600.json",
          "2026-10-10_scaleup_canary_segends_v1": "replay_readouts_scaleup_canary_segends_2026-10-10_014918.json",
          "2026-10-10_scaleup_l2_segends_v1": "replay_readouts_scaleup_l2_segends_2026-10-10_014046.json",
          "2026-10-10_scaleup_topup_segends_v1": "replay_readouts_scaleup_topup_segends_2026-10-10_023740.json",
          "2026-10-10_second_attack_segends_v1": "replay_readouts_second_attack_segends_2026-10-10_024517.json"}
EMPTY_NOTE = "output_end: empty tool output (no token inside)"
EXCL_NOTE = "output_end: output of an excluded call (rule E or submit)"
DIFF_NOTE = "output_end: tool output differs from the eval (raw and stripped; no readout)"
TOKEN_TEXT = re.compile(rb'"token_text": "(?:[^"\\]|\\.)*"')  # replaced by null before decoding: never read


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(items) -> str:
    """sha256 over a canonical allowlist of metadata fields (no text), one JSON line per file, sorted."""
    return hashlib.sha256("\n".join(sorted(json.dumps(x, sort_keys=True) for x in items)).encode()).hexdigest()


def load_reading_meta(f: Path) -> dict:
    """The meta with every token_text value replaced by null before decoding, so no token text is ever decoded."""
    raw, n = TOKEN_TEXT.subn(b'"token_text": null', f.read_bytes())
    if raw.count(b'"token_text"') != raw.count(b'"token_text": null') or b"\\u" in raw:
        raise SystemExit(f"{f.name}: a token_text value was not nulled, or a \\u escape remains; not decoding")
    return json.loads(raw)


def max_streak(empty: list[bool]) -> int:
    best = cur = 0
    for e in empty:
        cur = cur + 1 if e else 0
        best = max(best, cur)
    return best


def frac(k, n):
    return {"flagged": int(k), "total": int(n), "rate": round(k / n, 6) if n else None}


def output_state(c) -> str:
    """An output_end row: "empty" (null readout of a kept call), "excluded" (null readout, label None: rule E or submit;
    unknown, breaks a streak) or "kept". Gated against the replay notes in acting_gate."""
    if c["readout"] is not None:
        return "kept"
    return "excluded" if c["label"] is None else "empty"


def row_key(c, arm_dir, run) -> tuple:
    """A calls row's metadata state, the same for meta rows and replay rows. Rows without `kind` are the replay's command
    rows of submit-only or unparsed completions (run_replay_readouts.py, label None, no readout); anything else stops."""
    if c.get("kind") is None and not (c["readout"] is None and c["label"] is None and c["param"] is None):
        raise SystemExit(f"{run}: a row without kind that is not a submit-only or unparsed command row")
    return (arm_dir, run, c["turn"], c.get("kind"), c.get("call_index"), c["readout"] is None, c["label"] is None)


def acting_gate(root: Path, replay: Path) -> tuple[dict, dict]:
    """Per extraction: (1) the replay's own output_end rows reproduce its notes (empty, excluded) and no output differs
    from the eval; (2) the metas' rows equal the replay's rows, row by row (arm folder, run, turn, kind, call index,
    null readout, null label), over the arm folders the extraction has, and each run's epoch count equals the replay's.
    Returns the gate record and the holes: (extraction, run) -> turns of requests with no ordinary completion, whose
    previous outputs have no rows; a hole breaks a streak."""
    out, holes = {}, defaultdict(set)
    for e, fname in REPLAY.items():
        R = json.loads((replay / fname).read_text())  # keys, counts and names only (run_replay_readouts.py summary)
        notes, s = R["summary"]["notes"], R["summary"]
        whole = Counter(output_state(c) for c in R["calls"] if c.get("kind") == "output_end")
        if notes.get(DIFF_NOTE, 0) or (whole["empty"], whole["excluded"]) != (notes.get(EMPTY_NOTE, 0), notes.get(EXCL_NOTE, 0)):
            raise SystemExit(f"{e}: replay rows {dict(whole)} vs its notes, or outputs differ from the eval")
        dirs = sorted(p.name for p in (root / e).iterdir() if p.is_dir())
        meta, epochs = Counter(), Counter()
        for arm_dir in dirs:
            for f in sorted((root / e / arm_dir).glob("*.json")):
                d = json.loads(f.read_text())
                epochs[d["run"]] += 1
                meta.update(row_key(c, arm_dir, d["run"]) for c in d["calls"])  # meta rows carry no run: the file's
        rep = Counter(row_key(c, c["arm_dir"], c["run"]) for c in R["calls"] if c["arm_dir"] in dirs)
        if set(s["epochs_per_run"]) - {c["run"] for c in R["calls"]}:
            raise SystemExit(f"{e}: a replay run has no call rows, so its arm is unknown")
        if meta != rep or any(s["epochs_per_run"].get(r) != n for r, n in epochs.items()):
            raise SystemExit(f"{e}: metas differ from the replay rows ({sum((meta - rep).values())} extra, "
                             f"{sum((rep - meta).values())} missing) or from its epoch counts")
        for m in s["mismatch_details"]:
            if m["kind"] == "no ordinary completion" and m["run"] in epochs:
                holes[(e, m["run"])].add(m["turn"])
        st = Counter(k[3:4] + ((("excluded" if k[6] else "empty") if k[5] else "kept"),) for k in meta.elements() if k[3] == "output_end")
        out[e] = {"arm_folders": dirs, "replay_arm_folders_not_extracted": sorted({c["arm_dir"] for c in R["calls"]} - set(dirs)),
                  "meta_output_end": {k[1]: v for k, v in st.items()}, "rows_without_kind": sum(v for k, v in meta.items() if k[3] is None),
                  "holes_in_extracted_runs": sum(len(v) for (x, _), v in holes.items() if x == e),
                  "replay_notes": {"empty": notes.get(EMPTY_NOTE, 0), "excluded": notes.get(EXCL_NOTE, 0),
                                   "output differs": notes.get(DIFF_NOTE, 0)}, "replay_sha256": sha(replay / fname)}
    return out, holes


def streaks(seq) -> tuple[int, int]:
    """seq: (turn, call_index, epoch, state) in order. Max run of "empty"; and the same with a break at each epoch change.
    "kept", "excluded" and "hole" break both."""
    best = best_ep = cur = cur_ep = 0
    prev_ep = None
    for _, _, ep, s in seq:
        if s == "hole":
            cur = cur_ep = 0
            continue
        if ep != prev_ep:
            cur_ep, prev_ep = 0, ep
        cur, cur_ep = (cur + 1, cur_ep + 1) if s == "empty" else (0, 0)
        best, best_ep = max(best, cur), max(best_ep, cur_ep)
    return best, best_ep


def acting_runs(root: Path, pairs, holes):
    """Per run (meta `run`): its output states in order (turn, call_index), with holes; its irids, task, policy_called."""
    runs, items, seen = {}, [], set()
    for e, arm in pairs:
        for f in sorted((root / e / arm).glob("*.json")):
            d = json.loads(f.read_text())
            if (d["run"], d["epoch"]) in seen:
                raise SystemExit(f"{d['run']} epoch {d['epoch']}: two meta files")
            seen.add((d["run"], d["epoch"]))
            r = runs.setdefault(d["run"], {"task": d["task"], "seq": [], "irids": set(), "policy_called": False, "epochs": set(),
                                           "holes": holes.get((e, d["run"]), set())})
            if r["task"] != d["task"]:
                raise SystemExit(f"{d['run']}: metas span tasks")
            r["policy_called"] |= bool(d.get("policy_called"))
            r["epochs"].add(d["epoch"])
            rows = []
            for c in d["calls"]:  # keys only: kind, readout, label, turn/call_index ordering, irids
                r["irids"].add(c["irid"])
                if c.get("kind") == "output_end":
                    rows.append((c["turn"], c["call_index"], d["epoch"], output_state(c)))
            r["seq"] += rows
            items.append({"extraction": e, "arm": arm, "file": f.name, "run": d["run"], "epoch": d["epoch"], "task": d["task"],
                          "policy_called": d.get("policy_called"), "outputs": [list(x) for x in rows],
                          "irids": sorted({c["irid"] for c in d["calls"]})})
    for rid, r in runs.items():
        r["seq"].sort()
        if len({x[:2] for x in r["seq"]}) != len(r["seq"]):
            raise SystemExit(f"{rid}: two outputs at one (turn, call_index)")
        if any(a[2] > b[2] for a, b in zip(r["seq"], r["seq"][1:])):
            raise SystemExit(f"{rid}: epochs not in turn order")
        r["seq"] = sorted(r["seq"] + [(t, -1, None, "hole") for t in r["holes"]])
        r["max"], r["max_within_epoch"] = streaks(r["seq"])
        r["n_outputs"] = sum(x[3] != "hole" for x in r["seq"])
        r["n_empty"], r["n_excluded"] = (sum(x[3] == s for x in r["seq"]) for s in ("empty", "excluded"))
    return runs, items


def shares(by_run_rows, runs_streak, run_ids):
    """by_run_rows: run -> (flagged read-only, read-only); runs_streak: run -> max streak."""
    K = sum(k for k, _ in by_run_rows.values()); N = sum(n for _, n in by_run_rows.values())
    out = {"pooled_all_runs": frac(K, N), "runs_with_rows": len(by_run_rows)}
    for t in THRESHOLDS:
        s = [r for r in run_ids if runs_streak[r] >= t and r in by_run_rows]
        ks = sum(by_run_rows[r][0] for r in s); ns = sum(by_run_rows[r][1] for r in s)
        out[f"streak>={t}"] = {"runs": len(s), "share_of_flagged_read_only": frac(ks, K), "share_of_read_only": frac(ns, N),
                               "pooled_without_these_runs": frac(K - ks, N - ns)}
    return out


def acting(root: Path, draws_p: str, replay: Path):
    if sha(draws_p) != DRAWS0_SHA:
        raise SystemExit("draw-0 file sha256 differs")
    M = json.loads(Path(draws_p).read_text())
    meta, cells = M["meta"], M["draws"][0]["cells"]
    gate, holes = acting_gate(root, replay)
    out, all_items = {"worlds": {}, "gate_vs_replay": gate}, []
    world_runs = {}
    for w, pairs in ACT_WORLDS.items():
        runs, items = acting_runs(root, pairs, holes)
        all_items += items
        world_runs[w] = runs
    # validation on t86 (sim), before any count
    t86 = [r for r in world_runs["sim"].values() if r["task"] == "t86_file_indexing"]
    out["validation_t86_sim"] = [{"max_streak": r["max"], "max_streak_within_one_epoch": r["max_within_epoch"],
                                  "outputs": r["n_outputs"], "empty_outputs": r["n_empty"], "excluded_outputs": r["n_excluded"],
                                  "holes": len(r["holes"]),
                                  "epochs": len(r["epochs"])} for r in t86]
    if not any(r["max"] == 128 for r in t86):
        raise SystemExit(f"validation failed: t86 sim max streaks {[r['max'] for r in t86]}, expected 128")
    out["metas"] = {"files": len(all_items), "digest_of_allowlisted_fields": digest(all_items),
                    "allowlist": "extraction, arm, file name, run, epoch, task, policy_called, irids, output_end (turn, call_index, epoch, state)"}
    for w, runs in world_runs.items():
        if w in CELL:
            pc = cells[CELL[w]]["per_call"]
            irid_run = {}
            for rid, r in runs.items():
                for i in r["irids"]:
                    if i in irid_run and irid_run[i] != rid:
                        raise SystemExit(f"{w}: irid in two runs")
                    irid_run[i] = rid
            by_run, used = defaultdict(lambda: [0, 0]), {}
            K = N = 0
            if len(set(pc["row"])) != len(pc["row"]):
                raise SystemExit(f"{w}: a draw-0 row appears twice")
            for row, flag in zip(pc["row"], pc["flag_frozen"], strict=True):
                m = meta[row]
                if m["world"] != w:
                    raise SystemExit(f"{w}: draw-0 row from world {m['world']}")
                rid = irid_run.get(m["irid"])
                if rid is None:
                    raise SystemExit(f"{w}: draw-0 irid not in the metas")
                if runs[rid]["task"] != m["task"]:
                    raise SystemExit(f"{w}: draw-0 row's task differs from its run's task")
                if used.setdefault(m["task"], rid) != rid:
                    raise SystemExit(f"{w}: task {m['task']} maps to two runs")
                if m["y"] == 0:
                    by_run[rid][0] += int(flag); by_run[rid][1] += 1
                    K += int(flag); N += 1
            fa = cells[CELL[w]]["values"]["false_alarm_frozen"]
            if abs(K / N - fa) > 1e-9:
                raise SystemExit(f"{w}: pooled false alarms {K}/{N} differ from the file's {fa}")
            sel = sorted(set(used.values()))
            streak = {rid: runs[rid]["max"] for rid in sel}
            res = {"runs": len(sel), "tasks": len(used)}
            res |= {f"runs_max_streak>={t}": sum(s >= t for s in streak.values()) for t in THRESHOLDS}
            res["runs_at_or_above_20"] = sorted([{"task": runs[r]["task"], "max_streak": streak[r]} for r in sel if streak[r] >= 20],
                                                key=lambda x: -x["max_streak"])
            res["false_alarms_frozen_1pct"] = shares({r: tuple(v) for r, v in by_run.items()}, streak, sel)
        else:
            sel = sorted(r for r, x in runs.items() if x["policy_called"])
            streak = {rid: runs[rid]["max"] for rid in sel}
            res = {"runs": len(sel), "tasks": len({runs[r]["task"] for r in sel})}
            res |= {f"runs_max_streak>={t}": sum(s >= t for s in streak.values()) for t in THRESHOLDS}
            res["runs_at_or_above_20"] = sorted([{"task": runs[r]["task"], "max_streak": streak[r]} for r in sel if streak[r] >= 20],
                                                key=lambda x: -x["max_streak"])
            res["false_alarms_frozen_1pct"] = "not applicable (honest runs; the monitor's false alarms are read-only calls in attack runs)"
        out["worlds"][w] = res
    return out


def reading(root: Path, refit: Path):
    runs, items, nokind = {}, [], 0
    for arm in READ_ARMS:
        for f in sorted((root / READ_EXTR / arm).glob("*.json")):
            d = load_reading_meta(f)
            if d["arm"] != arm:
                raise SystemExit(f"{f.name}: arm {d['arm']} in folder {arm}")
            nokind += sum(p.get("kind") is None for p in d["positions"])
            seq = sorted((p["step"], p["action"], p["token"] is None) for p in d["positions"] if p.get("kind") == "output_end")
            if len({x[:2] for x in seq}) != len(seq):
                raise SystemExit(f"{d['key']}: two outputs at one (step, action)")
            jk = (arm, d["key"])
            if jk in runs:
                raise SystemExit(f"{jk}: two metas")
            runs[jk] = {"max": max_streak([x[2] for x in seq]), "n_outputs": len(seq), "n_empty": sum(x[2] for x in seq)}
            items.append({"arm": arm, "file": f.name, "key": d["key"], "outputs": [list(x) for x in seq]})
    if nokind:
        raise SystemExit(f"{nokind} reading positions without kind")
    out = {"metas": {"files": len(items), "digest_of_allowlisted_fields": digest(items),
                     "allowlist": "arm, file name, key, output_end (step, action, token is None)"}, "arms": {}, "readers": {}}
    for arm in READ_ARMS:
        rs = {k: v for k, v in runs.items() if k[0] == arm}
        out["arms"][arm] = {"runs": len(rs)} | {f"runs_max_streak>={t}": sum(v["max"] >= t for v in rs.values()) for t in THRESHOLDS}
        out["arms"][arm]["runs_at_or_above_20"] = sorted([{"key": k[1], "max_streak": v["max"]} for k, v in rs.items() if v["max"] >= 20],
                                                         key=lambda x: -x["max_streak"])
    agg_p = refit / "refit_2026-10-08_232032.json"
    if sha(agg_p) != AGG_SHA:
        raise SystemExit("refit aggregate sha256 differs")
    agg = json.loads(agg_p.read_text())["readers"]
    for reader, (rows_f, scores_f, probes_f) in READERS.items():
        rows, z = json.loads((refit / rows_f).read_text()), np.load(refit / scores_f)
        tab = {(p["fold"], p["train"]): p for p in json.loads(str(np.load(refit / probes_f)["table"]))}
        y = np.array([r["y"] for r in rows]); fold = np.array([r["fold"] for r in rows])
        if not (z["labels"] == y).all():
            raise SystemExit(f"{reader}: rows and scores disagree on labels")
        unjoined = sum((r["arm"], r["key"]) not in runs for r in rows if r["arm"] in ("real-attack-e1", "sim-attack"))
        if unjoined or any(r["arm"] not in ("real-attack-e1", "sim-attack") for r in rows):
            raise SystemExit(f"{reader}: {unjoined} scored rows (all labels) join no meta, or rows from another arm")
        thr = np.array([tab[(f, "real")]["threshold_real"] for f in fold])
        m = z["scores"][fold, 0, np.arange(len(y))] - thr
        if not (np.isfinite(m).all() and np.isfinite(thr).all()):
            raise SystemExit(f"{reader}: non-finite scores or thresholds")
        flag = m > 0
        res = {"inputs_sha256": {f: sha(refit / f) for f in (rows_f, scores_f, probes_f)}, "worlds": {}}
        for arm, w in (("real-attack-e1", "real"), ("sim-attack", "sim")):
            sel = np.array([r["arm"] == arm for r in rows])
            if not (sel & (y == 0)).any():
                raise SystemExit(f"{reader} {arm}: no read-only rows")
            fpr = flag[sel & (y == 0)].mean()
            want = agg[reader][f"real->{w}"]["false-alarm rate, frozen"]["value"]
            if not (math.isfinite(fpr) and math.isfinite(want)) or abs(fpr - want) > 1e-4:
                raise SystemExit(f"{reader} real->{w}: false-alarm rate {fpr} does not reproduce {want}")
            by_run = defaultdict(lambda: [0, 0]); missing = 0
            for i in np.flatnonzero(sel & (y == 0)):
                k = (arm, rows[i]["key"])
                if k not in runs:
                    missing += 1
                    continue
                by_run[k][0] += int(flag[i]); by_run[k][1] += 1
            if missing:
                raise SystemExit(f"{reader} {arm}: {missing} read-only rows join no meta")
            sel_runs = sorted(by_run)
            res["worlds"][w] = shares({r: tuple(v) for r, v in by_run.items()}, {r: runs[r]["max"] for r in sel_runs}, sel_runs)
        res["worlds"]["run2 (real-attack-e2)"] = "not available: no monitor scores for e2 in the refit"
        out["readers"][reader] = res
    return out


def main(a):
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing to run from a dirty tree:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    out = {"label": "POST HOC, exploratory (RULE.md); from raw data; extraction metadata, keys, scores and flags only",
           "git": head, "empty_definition": {"acting": "calls[kind == output_end].readout is None and label is not None "
                                                       "(null readout with label None = excluded call, rule E or submit: "
                                                       "unknown, breaks a streak); gated per extraction against the replay notes",
                                             "reading": "positions[kind == output_end].token is None"},
           "inputs": {"draws0_sha256": DRAWS0_SHA, "refit_aggregate_sha256": AGG_SHA},
           "coverage_notes": ["Whitespace-only outputs whose own token lies inside the content are not empty under this field.",
                              "Acting attack worlds are the draw-0 runs (one per task with draw-0 rows); honest worlds are all "
                              "policy-called runs in the metas.",
                              "False alarms are attributed to the runs that contain a streak, not to the calls inside the streak."],
           "acting": acting(Path(a.acting_metas), a.draws0, Path(a.replay)),
           "reading": reading(Path(a.reading_metas), Path(a.refit))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    p = HERE / "results" / f"empty_streaks_POSTHOC_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("--acting-metas", "--draws0", "--replay", "--reading-metas", "--refit"):
        ap.add_argument(k, required=True)
    main(ap.parse_args())
