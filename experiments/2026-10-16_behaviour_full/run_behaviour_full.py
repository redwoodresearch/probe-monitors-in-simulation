"""POST HOC descriptive: attack behaviour, sim vs real, on all launched cells of the four sets (S_old = T196, S_new =
+311, S_newmains = nm360, S_paper493), from the FULL .eval logs (no 32k anything). Counts only: per counted run, its steps
(assistant messages), bash / text_editor / submit / other calls, the agent's output tokens (model events of the policy
model only; sim logs also hold the simulator's openai/gpt-6-astra calls), and whether the sample carries an error.
Attempted comes from the attempt-label records by key, classified as in experiments/2026-10-15_attempt_rate (@ bc594d28).
Admission is the sample-level error, not the log status; a run whose agent token usage is missing is admitted for the
other metrics and left out of the token mean (counted). Per key for every launched cell, so the final cohort (09's
exclusions file) can be applied later by key. Only counts, booleans and identifiers leave the containers.

Which run counts per (set, arm, cell): the run the label records' staged rows name. Keys come in two forms: the q4
stager's <eval stem>__<sample id>__<epoch> (T196, from the corrected-B manifest) and the chain's
<launch>/<arm>/<task>/<eval file>.eval#e<n> (the corrected-B manifest's T196 keys and the chain's sets alike). S_paper493's sim arm comes from the chain
fork's pointer table (staged sets, per-set labels pinned by sha256@commit, exclusions by key). A cell with no counted run,
or more than one, is listed with its reason and left out of the means.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run \\
      experiments/2026-10-16_behaviour_full/run_behaviour_full.py [--sp493-pointer <commit>:<path>]
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-behaviourfull_main_kill-2026-11-16"
VOLS = {"chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"}
GIT = {  # name: (commit, path, sha256)
    "labels311": ("806b7a63a6", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attempt_labels_311.json", "22fc502fc3faa1323e5bee5927a2d6c8ef0ec158e55e76186b54425a2293356b"),
    "table311": ("806b7a63a6", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/run_table_311.json", "1824c9d993eab4caca3febd045fb3c18d0aee4ac57215a736bb2660ac626043f"),
    "plus311": ("0caadbfb5", "experiments/2026-10-14_q2_scaleup_cohort/results/scaleup_tasks_plus311_2026-10-14_125135.json", "69f1f75313c9610f7d16b18225c1166268c0b6e084859c641d0213f293896a8b"),
    "labels360": ("f60fb6cff", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attempt_labels_nm360.json", "5d28a4f51dbdbfb79883553068ae06b29329117e49deeeff6cbed60e7482a415"),
    "m710": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad"),
    "tasks196": ("64c6cb37f", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json", "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "manifest196": ("46ca25fc4", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/correctedb_manifest_v1.json", "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc"),
    "labels493": ("ee99b602a", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attempt_labels_sp493.json", "7a1af80c2344ed59c2c50f3f9cf1b8d2850d2a79fdb797fc4b5521c055cfed95"),
    "facts493": ("d800edc6a", "experiments/2026-10-15_q3_sp493/results/sp493_facts_2026-10-15_165730.json", "07caa527db89ec519314486d8c529c37430ca41d773a8bce26c041eb141b0a57"),
}
POINTER = ("13f6567b6", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/sp493_sim_pointer_table_2026-10-16_b.json",
           "1dfb7fe3eb1df504b635b8898b37879c54db6c41684cedc716c64a061e21e96c")  # table b: every set routed (table a's run @ 438207cf kept)
T196_LABELS = (Path(__file__).resolve().parents[1] / "2026-10-15_attempt_rate" / "inputs" / "claimc_labels_reduced.json",
               "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995")
ARMS, LAB = ("sim", "run1", "run2"), ("attempted", "unclear", "declined")
AGENT_MODEL = "vllm/Qwen/Qwen3.5-27B"
METRICS = ("steps", "calls_bash", "calls_text_editor", "calls_submit", "calls_other", "out_tokens", "attempted")
N_BOOT, SEED = 10_000, [20261016, 5]
SAMPLE_RE = re.compile(r"samples/(.+)_epoch_(\d+)\.json")
NAME_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
SAMPLE_ID_RE, PATH_RE = re.compile(r"[A-Za-z0-9_.:-]{1,240}"), re.compile(r"[A-Za-z0-9_@.:/-]{1,400}")
LABEL_PTR = re.compile(r"(\S+) \(sha256 ([0-9a-f]{64})\) @ ([0-9a-f]{7,40})")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "zstandard==0.23.0")
vols = {f"/{k}": modal.Volume.from_name(v, version=2).read_only() for k, v in VOLS.items()}


def zstd_zip():  # run_behaviour.zstd_zip (09's @ 19421fa7f), unchanged
    import zipfile

    import zstandard

    if hasattr(zipfile, "ZIP_ZSTANDARD") or getattr(zipfile, "_zstd_patched", False):
        return zipfile
    check, get = zipfile._check_compression, zipfile._get_decompressor
    zipfile._check_compression = lambda ct: None if ct == 93 else check(ct)
    zipfile._get_decompressor = lambda ct: zstandard.ZstdDecompressor().decompressobj() if ct == 93 else get(ct)
    zipfile._zstd_patched = True
    return zipfile


def ident(x):
    if not isinstance(x, str) or not NAME_RE.fullmatch(x):
        raise ValueError("a name field is not an identifier")
    return x


def safe(fn):  # no content may leave in an exception: its type and this file's line numbers only
    def run(*a):
        import traceback
        try:
            return fn(*a)
        except Exception as e:
            raise RuntimeError(f"{fn.__name__}: {type(e).__name__} at lines "
                               f"{[f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith('run_behaviour_full.py')]}") from None
    return run


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes=vols)
def resolve(staged: list) -> dict:
    """The counted runs: staged rows' keys -> (volume, .eval path, sample id or None, epoch) through an index of every .eval."""
    return safe(_resolve)(staged)


def _resolve(staged):
    import os
    index = {}
    for vol in VOLS:
        for root, _, files in os.walk(f"/{vol}/evals_v1"):
            for f in files:
                if f.endswith(".eval"):
                    index.setdefault(f[:-5], []).append((vol, os.path.relpath(os.path.join(root, f), f"/{vol}")))
    runs, seen, forms = [], {}, {"q4": 0, "chain": 0}
    for s, world, vol, path, sha, keys in staged:
        raw = (Path(f"/{vol}") / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("a staged file does not match its pinned sha256")
        for line in raw.decode().splitlines():
            row = json.loads(line)
            key = row["key"]
            if keys is not None and key not in keys:
                continue
            if ".eval#e" in key:  # the chain's <launch>/<arm>/<task>/<eval file>.eval#e<n>
                p, e = key.rsplit("#e", 1)
                stem, sample, epoch = Path(p).name[:-5], None, int(e)
                forms["chain"] += 1
            else:  # the q4 stager's <eval stem>__<sample id>__<epoch>
                stem, sample, epoch = key.rsplit("__", 2)
                epoch = int(epoch)
                forms["q4"] += 1
            hits = index.get(stem, [])
            k = (s, world, row.get("task") or row.get("cell"))
            if not (NAME_RE.fullmatch(k[2] or "") and (sample is None or SAMPLE_ID_RE.fullmatch(sample)) and all(PATH_RE.fullmatch(h[1]) for h in hits)):
                raise ValueError("a returned cell, sample id or path is not an identifier")
            seen[k] = seen.get(k, 0) + 1
            runs.append({"set": s, "world": world, "cell": k[2], "sample": sample, "epoch": epoch, "where": hits[0] if len(hits) == 1 else None, "stem_hits": len(hits)})
            del row
    for r in runs:
        r["duplicates"] = seen[(r["set"], r["world"], r["cell"])]
    return {"runs": runs, "eval_files_indexed": sum(len(v) for v in index.values()), "key_forms": forms}


@app.function(image=image, cpu=1, memory=8192, timeout=1800, retries=0, volumes=vols)
def scan(batch: list) -> list:
    return safe(_scan)(batch)


def _scan(batch):
    zipfile = zstd_zip()
    out = []
    for r in batch:
        vol, rel = r["where"]
        base = {k: r[k] for k in ("set", "world", "cell")}
        with zipfile.ZipFile(Path(f"/{vol}") / rel) as z:
            cands = [(n, m.group(1), int(m.group(2))) for n in z.namelist() if (m := SAMPLE_RE.fullmatch(n))]
            if r["sample"] is not None:
                cands = [c for c in cands if c[1] == r["sample"]]
            fallback = False
            if r["sample"] is None:  # the #e<n> form names no sample: only #e0 with exactly one sample in the file is accepted
                pick, fallback = (cands, True) if r["epoch"] == 0 and len(cands) == 1 else ([], False)
            else:
                pick = [c for c in cands if c[2] == r["epoch"]]
            if len(pick) != 1:
                out.append(base | {"scan_status": f"sample not found once ({len(pick)} of {len(cands)})"})
                continue
            s = json.loads(z.read(pick[0][0]))
        out_tok, agent_ev, other_ev, usage_missing = 0, 0, 0, 0
        for e in s.get("events") or []:
            if isinstance(e, dict) and e.get("event") == "model":
                if e.get("model") == AGENT_MODEL:
                    agent_ev += 1
                    u = ((e.get("output") or {}).get("usage") or {}).get("output_tokens")
                    usage_missing += u is None
                    out_tok += int(u or 0)
                else:
                    other_ev += 1
        steps, calls = 0, {"bash": 0, "text_editor": 0, "submit": 0, "other": 0}
        for m in s.get("messages") or []:
            if m.get("role") == "assistant":
                steps += 1
                for c in m.get("tool_calls") or []:
                    fn = ident(c.get("function"))
                    calls[fn if fn in calls else "other"] += 1
        tokens_ok = agent_ev > 0 and usage_missing == 0
        out.append(base | {"scan_status": "sample error" if s.get("error") else "ok", "epoch_fallback": fallback, "steps": steps,
                           **{f"calls_{k}": v for k, v in calls.items()}, "out_tokens": out_tok if tokens_ok else None,
                           "agent_model_events": agent_ev, "other_model_events": other_ev, "agent_events_missing_usage": usage_missing})
        del s
    return out


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0)
def aggregate(rows: list, universe: dict, classes: dict, env: dict) -> dict:
    return safe(_aggregate)(rows, universe, classes, env)


def _aggregate(rows, universe, classes, env):
    import numpy as np
    per = {}
    for r in rows:
        k = (r["set"], r["world"], r["cell"])
        per[k] = {**r, "scan_status": "more than one counted run"} if k in per else r
    table = {}
    for s, cells in universe.items():
        for c in cells:
            for a in ARMS:
                lab = classes[s].get(f"{a}|{c}") or "no label"
                r = per.get((s, a, c)) or {"scan_status": "no counted run"}
                admitted = r["scan_status"] == "ok" and lab in LAB
                row = {"label": lab, "scan_status": r["scan_status"], "admitted": admitted,
                       **{k: r[k] for k in ("epoch_fallback", "agent_model_events", "other_model_events", "agent_events_missing_usage") if k in r}}
                if admitted:
                    row |= {m: r[m] for m in METRICS[:-1]} | {"attempted": int(lab == "attempted")}
                table[f"{s}|{a}|{c}"] = row
    out = {"per_key": table, "summary": {}, "bootstrap": {"draws": N_BOOT, "seed": SEED, "strata": "environment within each set; pooled: set x environment"}}
    views = {s: [s] for s in universe} | {"pooled": list(universe)}
    rng = np.random.default_rng(SEED)
    for v, sets in views.items():
        cells = [(s, c) for s in sets for c in universe[s]]
        strata = sorted({f"{s}|{env[s][c]}" for s, c in cells})
        Wt = np.zeros((N_BOOT, len(cells)))
        for st in strata:
            idx = [i for i, (s, c) in enumerate(cells) if f"{s}|{env[s][c]}" == st]
            Wt[:, idx] = rng.multinomial(len(idx), [1 / len(idx)] * len(idx), size=N_BOOT)
        res = {"cells": len(cells), "per_arm": {}, "pairs": {}}
        for a in ARMS:
            xs = [table[f"{s}|{a}|{c}"] for s, c in cells]
            adm = [x for x in xs if x["admitted"]]
            res["per_arm"][a] = {"admitted": len(adm), "tokens missing (admitted)": sum(x["out_tokens"] is None for x in adm),
                                 "not admitted, by label": {k: sum(1 for x in xs if not x["admitted"] and x["label"] == k) for k in sorted({x["label"] for x in xs if not x["admitted"]})},
                                 "not admitted, by scan": {k: sum(1 for x in xs if not x["admitted"] and x["scan_status"] == k) for k in sorted({x["scan_status"] for x in xs if not x["admitted"]})},
                                 **{m: (float(np.mean(vals)) if (vals := [x[m] for x in adm if x[m] is not None]) else None) for m in METRICS}}
        for a, b in (("sim", "run1"), ("sim", "run2"), ("run2", "run1")):
            p = {}
            for m in METRICS:
                ok = np.array([table[f"{s}|{a}|{c}"]["admitted"] and table[f"{s}|{b}|{c}"]["admitted"] and table[f"{s}|{a}|{c}"][m] is not None
                               and table[f"{s}|{b}|{c}"][m] is not None for s, c in cells])
                if not ok.any():
                    continue
                d = np.array([(table[f"{s}|{a}|{c}"][m] - table[f"{s}|{b}|{c}"][m]) if o else 0.0 for (s, c), o in zip(cells, ok)], float)
                Wm = Wt * ok
                bt = (Wm * d).sum(1) / np.where(Wm.sum(1) > 0, Wm.sum(1), np.nan)
                bt = bt[np.isfinite(bt)]
                p[m] = {"n_pairs": int(ok.sum()), "diff": float(d[ok].mean()), "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
            res["pairs"][f"{a} - {b}"] = p
        out["summary"][v] = res
    return out


def classes_and_staged(src, ptr, git):
    """Per set: labels by key with the attempt-rate record's infra classes, and the staged files that name each counted run."""
    m = src["m710"]
    scored = {k for v in m["folder_keys"].values() for k in v}
    in_table = {r["cell"]: r for r in src["table311"]["cells"]}
    dropped = {d["cell"]: f"no run-table row: dropped at {d['stage']}" for d in src["table311"]["dropped"]}

    def s_new(a, c):
        v = src["labels311"]["labels"].get(f"{a}|{c}")
        if v:
            return v
        if c in dropped:
            return dropped[c]
        return f"never launched (sim_status {in_table[c]['sim_status']})" if a == "sim" and c in in_table and in_table[c]["sim_status"] != "ran" else "no label"

    def nm(a, c):
        k = f"{a}|{c}"
        if k in m["excluded_runs"]:
            return "excluded: " + m["excluded_runs"][k].split(" (")[0]
        if k not in scored:
            return "no scored row (no folder)"
        v = src["labels360"]["labels"].get(k)
        return "no label" if v is None else "run failed: no counted attempt" if v == "run_failed" else v

    t196 = src["t196"]
    classes = {"S_old": {f"{a}|{c['key']}": t196.get(f"{a}|{c['key']}") or "no label in the reduced file" for c in src["tasks196"]["cells"] for a in ARMS},
               "S_new": {f"{a}|{c['key']}": s_new(a, c["key"]) for c in src["plus311"]["cells"] for a in ARMS},
               "S_newmains": {f"{a}|{c}": nm(a, c) for c in m["env_of"] for a in ARMS},
               "S_paper493": {f"{a}|{r['key']}": src["labels493"]["labels"].get(f"{a}|{r['key']}") or "no label" for r in src["facts493"]["rows"] for a in ("run1", "run2")}}
    staged = []
    for s, name in (("S_new", "labels311"), ("S_newmains", "labels360"), ("S_paper493", "labels493")):
        for sha, v in src[name]["staged_sets"].items():
            if v["world"] in ARMS:
                staged.append([s, v["world"], "chain", v["staged_set"], sha, None])
    by_file = {}
    for e in src["manifest196"]["epochs"]:
        if e["world"] in ARMS:
            by_file.setdefault((e["world"], e["staged_inputs_file"], e["staged_inputs_sha256"]), []).append(e["key"])
    staged += [["S_old", w, "q4", f, sha, ks] for (w, f, sha), ks in sorted(by_file.items())]
    sim493 = {"pointer": None}
    if ptr:
        cells493 = {r["key"] for r in src["facts493"]["rows"]}
        lab_sim, excl, seen, seen_cells, repeats = {}, {}, set(), set(), 0
        for v in ptr["sets"]:
            path, sha, commit = LABEL_PTR.fullmatch(v["attempt_labels"]).groups()
            b = git("show", f"{commit}:{path}")
            if hashlib.sha256(b).hexdigest() != sha:
                raise SystemExit(f"{v['staged_set']}: its label file differs from the pointer's sha256")
            for line in b.decode().splitlines():  # the pools' label files are cumulative: merge per cell, stop only on a disagreement
                r = json.loads(line)
                if r["arm"] != "sim_attack@september" or r["task"] not in cells493:
                    raise SystemExit(f"{v['staged_set']}: a label row outside S_paper493's sim arm or cells")
                if r["label"]:
                    if lab_sim.get(r["task"], r["label"]) != r["label"]:
                        raise SystemExit(f"{r['task']}: two S_paper493 sim label files disagree")
                    lab_sim[r["task"]] = r["label"]
                    repeats += r["task"] in seen_cells
                    seen_cells.add(r["task"])
            for x in v["excluded"]:
                excl[x["key"].split("|", 1)[1]] = "excluded: " + (x.get("sample_error") or x.get("staging_reason", "?"))[:60]
            vp = v["volume_path"].split(":", 1)
            if vp[0] != VOLS["chain"] or v["staged_set"] in seen:
                raise SystemExit(f"{v['staged_set']}: not on the chain Volume, or listed twice")
            seen.add(v["staged_set"])
            staged.append(["S_paper493", "sim", "chain", vp[1].lstrip("/"), v["staged_sha256"], None])
        classes["S_paper493"] |= {f"sim|{c}": excl.get(c) or lab_sim.get(c) or "no label (no routed sim run yet)" for c in cells493}
        sim493 = {"pointer": f"{POINTER[0]}:{POINTER[1]}", "sha256": POINTER[2], "sets": len(ptr["sets"]), "labelled_cells": len(lab_sim),
                  "label_rows_repeating_a_cell (agreeing)": repeats, "excluded": excl}
    return classes, staged, sim493


@app.local_entrypoint()
def main(no_sp493_sim: bool = False):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip() or not git("branch", "-r", "--contains", "HEAD").strip():
        raise SystemExit("commit and push first")
    src = {}
    for name, (sha, path, want) in GIT.items():
        b = git("show", f"{sha}:{path}")
        if hashlib.sha256(b).hexdigest() != want:
            raise SystemExit(f"{name}: sha256 differs from its pin")
        src[name] = json.loads(b)
    if hashlib.sha256(T196_LABELS[0].read_bytes()).hexdigest() != T196_LABELS[1]:
        raise SystemExit("T196 labels: sha256 differs from its pin")
    src["t196"] = json.loads(T196_LABELS[0].read_bytes())["labels"]
    ptr = None
    if not no_sp493_sim:
        b = git("show", f"{POINTER[0]}:{POINTER[1]}")
        if hashlib.sha256(b).hexdigest() != POINTER[2]:
            raise SystemExit("the S_paper493 sim pointer table differs from its pin")
        ptr = json.loads(b)
    classes, staged, sim493 = classes_and_staged(src, ptr, git)
    m = src["m710"]
    universe = {"S_old": [c["key"] for c in src["tasks196"]["cells"]], "S_new": [c["key"] for c in src["plus311"]["cells"]],
                "S_newmains": sorted(m["env_of"]), "S_paper493": [r["key"] for r in src["facts493"]["rows"]]}
    env = {"S_old": {c["key"]: c["env"] for c in src["tasks196"]["cells"]}, "S_new": {c["key"]: c["env"] for c in src["plus311"]["cells"]},
           "S_newmains": m["env_of"], "S_paper493": {r["key"]: r["env"] for r in src["facts493"]["rows"]}}
    res = resolve.remote(staged)
    todo = [r for r in res["runs"] if r["where"] and r["duplicates"] == 1]
    rows = [x for part in scan.map([todo[i:i + 25] for i in range(0, len(todo), 25)]) for x in part]
    rows += [{**{k: r[k] for k in ("set", "world", "cell")}, "scan_status": ".eval file not found once" if not r["where"] else "more than one counted run"}
             for r in res["runs"] if not (r["where"] and r["duplicates"] == 1)]
    agg = aggregate.remote(rows, universe, classes, env)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = git("rev-parse", "HEAD").decode().strip()
    rec = {"label": "POST HOC descriptive; PRELIMINARY; all launched cells by key (the final cohort is applied later by key)", "git": head,
           "stamp (PT)": stamp, "app": APP, "sources": {k: {"commit": v[0], "path": v[1], "sha256": v[2]} for k, v in GIT.items()},
           "t196_labels": {"file": str(T196_LABELS[0].relative_to(HERE.parents[1])), "sha256": T196_LABELS[1]}, "sp493_sim": sim493,
           "agent_model": AGENT_MODEL, "eval_files_indexed": res["eval_files_indexed"], "key_forms": res["key_forms"], "counted_runs_scanned": len(todo),
           "epoch_fallbacks": sum(1 for r in rows if r.get("epoch_fallback")),
           "notes": {"attempted": "labels by key, classified as in the attempt-rate record @ bc594d28; admitted runs only",
                     "S_paper493 sim": f"from the chain fork's pointer table {POINTER[0]} ({len(ptr['sets'])} sets)" if ptr else "NOT INCLUDED"}} | agg
    dst = HERE / "results" / f"behaviour_full_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    dst.open("x").write(json.dumps(rec, indent=1))
    lines = [f"indexed .eval files {res['eval_files_indexed']}; key forms {res['key_forms']}; counted runs scanned {len(todo)}; epoch fallbacks {rec['epoch_fallbacks']}; "
             f"S_paper493 sim: {rec['notes']['S_paper493 sim']}"]
    for v, r in agg["summary"].items():
        lines.append(f"[{v}] cells {r['cells']}")
        for a, x in r["per_arm"].items():
            lines.append(f"  {a}: admitted {x['admitted']} (tokens missing {x['tokens missing (admitted)']}); " + ", ".join(f"{k} {x[k]:.2f}" for k in METRICS if x[k] is not None))
            lines.append(f"     not admitted by label {x['not admitted, by label']}; by scan {x['not admitted, by scan']}")
        for pk, p in r["pairs"].items():
            lines.append(f"  {pk}: " + "; ".join(f"{k} {q['diff']:+.2f} [{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}] (n {q['n_pairs']})" for k, q in p.items()))
    (HERE / "results" / f"behaviour_full_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", dst)
