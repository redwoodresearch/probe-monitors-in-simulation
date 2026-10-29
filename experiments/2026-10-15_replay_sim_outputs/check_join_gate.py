"""Gate 7, the RULE's join gate ("Staging and extraction gates", item 7), on extraction METADATA only. Modal CPU. No score,
activation or text leaves the container: of each meta call only kind, irid, src_irid, call_index, name (the tool-function name),
readout, row and row_mean are used (never `param` or `label`); staged token ids are read only by the scorer's own skip_rows
(run_scaleup_score.py @ 758381d3e, byte-checked) to find fixed-token call_end rows.

Per admitted run (Phase 1: 90 replay_attack epochs vs corrected B's run1; placebo: 45 replay_honest_b vs corrected B's honest_b;
the source epoch through phase1_manifest.json / placebo_manifest.json's corrected_map_key, which staging keeps as the replay key):
  (a) identity: replay meta key, arm, task, run and attempt fields match the source epoch; task -> main matches (T196); the
      replay meta folders hold exactly the admitted keys (no extras, no duplicates);
  (b) the action identity and order map: the ordered list of every staged unit (irid, src_irid, kind, call_index, tool-function
      name) equals the original's (the replay changes tool-output content only; stage_q4b makes a unit for every turn);
  (c) per readout, R = output_end rows with a span-mean row and C = call_end rows minus the scorer's fixed-token rows (the rows
      run_replay_score.py scores): the rows both sides retain appear in the same order within the shared prefix (before the
      earlier cut); each row only one side retains is explained by the other side's state of that unit: no readout after its last
      readout on a cut render (cut, or empty content there: the meta cannot separate them; both are allowed exclusions) | no
      readout before its last readout or on an uncut render (empty content: the reader's "no token entirely inside") | a
      fixed-token exclusion; anything else (a missing unit, or a unit the other side reads but does not retain) fails.
Each side is also checked for consistency (row iff readout; span-mean rows only on kept output_end units and on all of them;
readouts below the cut and strictly increasing over output_end / call_end; the scorer's (src_irid, call_index) unique per readout).
A render counts as cut when its staged ids reach 32,768 tokens; its shared-prefix bound is then the position after its last readout.
Cross-checks: the replay side's retained rows per arm equal the scorer records' row counts (run_replay_score.py score mode,
records @ 5f36f373b); the fixed tokens and dropped-row counts equal theirs. Control: the 10 gate-6 control epochs (staged with no
swap) must match corrected B's run1 exactly (identical maps and retained rows, no one-sided rows).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run \\
      experiments/2026-10-15_replay_sim_outputs/check_join_gate.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-replay_kill-2026-11-14"
BVOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"  # corrected B: staged inputs and extractions
QVOL = "probe-shift_aswin-surya_sub-q2-onpolicysim_data_kill-2026-11-14"  # the replay's staged set and extraction
HFVOL = "probe-shift_aswin-surya_hf-cache"
XR, X2E, XQ = "experiments/2026-10-15_replay_sim_outputs", "experiments/2026-10-14_q35_sol_avg_POSTHOC", "experiments/2026-10-15_q2_replay/results"
PINS = {  # name -> (commit, repo path, sha256)
    "manifest": ("46ca25fc4", f"{X2E}/results/correctedb_manifest_v1.json", "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc"),
    "t196": ("4d1eec572", "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json",
             "55d9a8ec68f4685410d65c97bb84f470efb7cd8f05287384f92465ed5eadedbb"),
    "phase1": ("604bc7c60", f"{XR}/phase1_manifest.json", "872ddc7c68bd856f1b21d3effb5f368068d058372cedc9d749532c57bc9e2ce7"),
    "placebo": ("604bc7c60", f"{XR}/placebo_manifest.json", "8eb57dae18ff40fd1c6da7fdc178868d8773415252a57fe4b83510c7ea58cf94"),
    "stage_record": ("20aa4ebb", f"{XQ}/stage_t196_v1_stage_2026-10-15_182816.json", "d977ffa0c5f082e7cbc8181790ba9a79462fb3470199d09bce40c716403cd78b"),
    "scorer": ("758381d3e", f"{X2E}/run_scaleup_score.py", "97870e3b304130603c92c027b5f5144880f32f1dfde6505f3efb76f88994dd47"),
    "score_attack": ("5f36f373b", f"{XR}/results/replay_score_score_2026-10-15_191919.json", "ede53bc017487f2e7d68123f52a8276abf5188c1207bf6b227bb75c63232eb16"),
    "score_honest_b": ("5f36f373b", f"{XR}/results/replay_score_score_2026-10-15_192115.json", "72b344747915e34bd3c2a0ccc19ca4d09adc7b423e44f879c82f2af104f4e4e1"),
    "score_control": ("5f36f373b", f"{XR}/results/replay_score_score_2026-10-15_191238.json", "4c8de0104632acdc3b3643db8b098baefd5a8ae1e0f1a9d19b2f929324fbbbc3"),
}
EXTRACT, STAGED = "replay/t196_v1_extract_v1", "replay/t196_v1/staged_v1"  # on QVOL (extract_replay_t196.yaml @ 3cce87d85)
CUT = 32768
CONTROL = ["t46_file_indexing", "t64_tiktok", "t178_clinical_trial", "t92_clinical_trial", "t38_model_registry", "t164_media_processing",
           "t117_media_processing", "t192_clinical_trial", "t190_clinical_trial", "t42_clinical_trial"]  # RULE gate 6 (11:4x note)
ARMS = {"attack": ("replay_attack", "run1", "phase1", "score_attack"), "honest_b": ("replay_honest_b", "honest_b", "placebo", "score_honest_b"),
        "control": ("replay_control_run1", "run1", None, "score_control")}

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class GateError(Exception):
    """This script's own failures: messages built only from keys, names and counts."""


def _safe(fn, what: str):
    """run_replay_score.py's pattern, stricter: any failure is re-raised OUTSIDE the except block (no chained context) as a fresh
    error. A GateError keeps its own message; a decoding or parsing failure keeps only its category and numeric position
    (JSONDecodeError holds the whole document and UnicodeDecodeError its bytes; Modal pickles exceptions out of the container);
    anything else keeps only its type name."""
    err = None
    try:
        return fn()
    except GateError as e:
        err = f"{what}: {e}"
    except json.JSONDecodeError as e:
        err = f"{what}: JSON decode error at line {e.lineno} column {e.colno}"
    except UnicodeDecodeError as e:
        err = f"{what}: UTF-8 decode error at byte {e.start}"
    except Exception as e:  # noqa: BLE001  (re-raised below, sanitized: type name only)
        err = f"{what}: {type(e).__name__}"
    raise GateError(err)  # a sanitized message, so an outer _safe keeps it


def _fname(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def _calls(m: dict) -> list:
    """Only the fields this gate may use: identity, position and row fields (never param or label)."""
    return [{"id": (str(c["irid"]), str(c["src_irid"]), c["kind"], int(c["call_index"]), c.get("name")), "readout": c.get("readout"),
             "row": c.get("row"), "row_mean": c.get("row_mean")} for c in m["calls"]]


def _ntok(m: dict) -> int:
    """The staged ids' length: the meta's n_tokens, else every call's n (stage_q4b gives each call n = len(ids))."""
    if not m["calls"]:
        return m.get("n_tokens", 0)
    ns = {c["n"] for c in m["calls"]}
    if len(ns) > 1 or ("n_tokens" in m and ns and ns != {m["n_tokens"]}):
        raise GateError(f"{m['key']}: calls disagree on the staged length")
    return m["n_tokens"] if "n_tokens" in m else ns.pop()


def _last_readout(calls: list) -> int:
    """Index after the side's last unit with a readout (0 if none)."""
    kept = [i for i, c in enumerate(calls) if c["readout"] is not None]
    return kept[-1] + 1 if kept else 0


def _retained(calls: list, ro: str, skip: set) -> list:
    if ro == "R":
        return [i for i, c in enumerate(calls) if c["id"][2] == "output_end" and c["row_mean"] is not None]
    return [i for i, c in enumerate(calls) if c["id"][2] == "call_end" and c["row"] is not None and (c["id"][0], c["id"][3]) not in skip]


def _validate(m: dict, calls: list, skip: set) -> list:
    """The side's own consistency, as the extractor and scorer require: row iff readout; a span-mean row only on an output_end
    with a readout, and every kept output_end has one; readouts inside the staged ids and below the cut; output_end / call_end
    readouts strictly increasing in staged order (the order the cut and empty-content inferences rely on); the scorer's
    (src_irid, call_index) unique among each readout's retained rows."""
    n, bad = _ntok(m), []
    for c in calls:
        r = c["readout"]
        if (c["row"] is None) != (r is None):
            bad.append("row without readout or readout without row")
        if c["row_mean"] is not None and (c["id"][2] != "output_end" or r is None):
            bad.append("span-mean row on a non-output_end or unread unit")
        if c["id"][2] == "output_end" and r is not None and c["row_mean"] is None:
            bad.append("kept output_end without a span-mean row")
        if r is not None and not (type(r) is int and 0 <= r < min(n, CUT)):
            bad.append("readout outside the staged ids or at/after the cut")
    seq = [c["readout"] for c in calls if c["id"][2] in ("output_end", "call_end") and type(c["readout"]) is int]
    if any(b <= a for a, b in zip(seq, seq[1:], strict=False)):
        bad.append("output_end / call_end readouts not strictly increasing")
    for ro in ("R", "C"):
        u = [(calls[i]["id"][1], calls[i]["id"][3]) for i in _retained(calls, ro, skip)]
        if len(set(u)) != len(u):
            bad.append(f"{ro}: duplicate (src_irid, call_index) among retained rows")
    return sorted(set(bad))


def compare(mo: dict, mr: dict, skip_o: set, skip_r: set) -> dict:
    """One run: identity, side consistency, the full identity map, then per readout the shared-prefix order and the reason for
    every one-sided row. A readout of None means, by the reader (kept = token is not None and token < cut), either no content
    token (empty content) or a token at/after the cut: before the side's last readout, or on a render that was not cut, it is
    empty content; after the last readout of a cut render the meta cannot separate the two (both are allowed exclusions)."""
    co, cr = _calls(mo), _calls(mr)
    io, ir = [c["id"] for c in co], [c["id"] for c in cr]
    if len(set(io)) != len(io) or len(set(ir)) != len(ir):
        return {"pass": False, "fail": ["duplicate unit identity"]}
    bad = [f"original: {x}" for x in _validate(mo, co, skip_o)] + [f"replay: {x}" for x in _validate(mr, cr, skip_r)]
    trunc = {"o": _ntok(mo) >= CUT, "r": _ntok(mr) >= CUT}
    last = {"o": _last_readout(co), "r": _last_readout(cr)}
    cut = {s: last[s] if trunc[s] else len(io if s == "o" else ir) for s in ("o", "r")}  # shared-prefix bound per side
    out = {"map_equal": io == ir, "n_units": [len(io), len(ir)], "name_in_meta": all(x[4] is not None for x in io + ir),
           "truncated (staged length >= 32,768)": [trunc["o"], trunc["r"]], "prefix_bound": [cut["o"], cut["r"]], "consistency_failures": bad}
    pos = {"o": {u: i for i, u in enumerate(io)}, "r": {u: i for i, u in enumerate(ir)}}
    side, skip = {"o": co, "r": cr}, {"o": skip_o, "r": skip_r}
    ok = out["map_equal"] and not bad
    for ro in ("R", "C"):
        ret = {s: [side[s][i]["id"] for i in _retained(side[s], ro, skip[s])] for s in ("o", "r")}
        both = set(ret["o"]) & set(ret["r"])
        pre = {s: [u for u in ret[s] if u in both and pos["o"][u] < cut["o"] and pos["r"][u] < cut["r"]] for s in ("o", "r")}
        reasons = {"o_only": {}, "r_only": {}}
        for s, t in (("o", "r"), ("r", "o")):
            for u in ret[s]:
                if u in both:
                    continue
                j, c = pos[t].get(u), None
                if j is not None:
                    c = side[t][j]
                if c is None:
                    why = "UNEXPLAINED: unit missing on the other side"
                elif c["readout"] is None and trunc[t] and j >= last[t]:
                    why = "cut: after the other side's last readout on a cut render (or empty content there; the meta cannot separate them)"
                elif c["readout"] is None:
                    why = "empty content on the other side (no readout inside its retained span)"
                elif ro == "C" and c["row"] is not None and (u[0], u[3]) in skip[t]:
                    why = "fixed-token exclusion on the other side"
                else:
                    why = "UNEXPLAINED: the other side reads this unit but does not retain it"
                reasons[f"{s}_only"][why] = reasons[f"{s}_only"].get(why, 0) + 1
        order_ok = pre["o"] == pre["r"]
        unexplained = sum(v for d in reasons.values() for k, v in d.items() if k.startswith("UNEXPLAINED"))
        out[ro] = {"retained": [len(ret["o"]), len(ret["r"])], "both": len(both), "shared_prefix_rows": len(pre["o"]),
                   "shared_prefix_order_equal": order_ok, **reasons, "unexplained": unexplained}
        ok = ok and order_ok and not unexplained
    out["pass"] = bool(ok)
    return out


@app.function(image=image, cpu=(4, 4), memory=(32768, 32768), timeout=1800, retries=0, scaledown_window=2,
              volumes={"/b": modal.Volume.from_name(BVOL, version=2).read_only(), "/q": modal.Volume.from_name(QVOL, version=2).read_only(),
                       "/hf": modal.Volume.from_name(HFVOL).read_only()})
def gate(blobs: dict) -> dict:
    return _safe(lambda: _gate(blobs), "join gate")


def _gate(blobs: dict) -> dict:  # noqa: C901
    for n, (_, _, h) in PINS.items():
        if _sha(blobs[n]) != h:
            raise GateError(f"in-job: {n} sha256 differs")
    Path("/tmp/x2e").mkdir(exist_ok=True)
    (Path("/tmp/x2e") / "run_scaleup_score.py").write_bytes(blobs["scorer"])
    sys.path.insert(0, "/tmp/x2e")
    import run_scaleup_score as S

    t196 = json.loads(blobs["t196"])
    main_of = {c["key"]: c["main"] for c in t196["cells"]}
    mf = {e["key"]: e for e in json.loads(blobs["manifest"])["epochs"]}
    stg = json.loads(blobs["stage_record"])["inputs_sha256"]
    adm = {"attack": json.loads(blobs["phase1"])["runs"], "honest_b": json.loads(blobs["placebo"])["runs"]}
    fails, per_run = [], {}
    # (a) the replay meta folders hold exactly the admitted keys; the control folder the 10 pinned control run1 epochs
    rep, want_keys = {}, {}
    for role, (arm, world, _, _) in ARMS.items():
        d = Path("/q") / EXTRACT / "meta" / arm
        ms = {}
        for f in sorted(d.glob("*.json")):
            m = _safe(lambda f=f: json.loads(f.read_text()), f"{arm}/{f.name}")
            if m.get("arm") != arm or f.stem != _fname(m["key"]) or m.get("inputs_sha256") != stg[f"{arm}.jsonl"]:
                fails.append(f"{arm}/{f.name}: arm, file name or inputs_sha256 is not this arm's staged file")
            if m["key"] in ms:
                fails.append(f"{arm}: key twice")
            ms[m["key"]] = m
        rep[role] = ms
        if role == "control":
            p1 = {r["task"]: r["corrected_map_key"] for r in adm["attack"]}
            want_keys[role] = {p1[t]: t for t in CONTROL}
        else:
            want_keys[role] = {r["corrected_map_key"]: r["task"] for r in adm[role]}
            if any(r["world"] != world for r in adm[role]):
                fails.append(f"{role}: a manifest row is not world {world}")
        if set(ms) != set(want_keys[role]):
            fails.append(f"{arm}: meta keys != admitted keys (extra {len(set(ms) - set(want_keys[role]))}, missing {len(set(want_keys[role]) - set(ms))})")
    key_sets = {role: {"meta_files": len(rep[role]), "admitted": len(want_keys[role]), "equal": set(rep[role]) == set(want_keys[role])} for role in ARMS}
    # original epochs and metas (corrected B), with their identity checks
    orig, want_o, shas_o = {}, {}, {}
    for role, (_, world, _, _) in ARMS.items():
        for k, task in want_keys[role].items():
            e = mf.get(k)
            if e is None or e["world"] != world or e["task"] != task:
                fails.append(f"{role}: {k}: not a corrected-B {world} epoch of {task}")
                continue
            mfile = Path("/b") / e["extraction_run"] / "meta" / e["arm"] / f"{_fname(k)}.json"
            if f"{e['extraction_run']}/meta/{e['arm']}/{mfile.name}" != e["meta_file"]:
                fails.append(f"{k}: meta path differs from the manifest's")
            m = _safe(lambda mfile=mfile: json.loads(mfile.read_text()), f"{e['meta_file']}")
            if m["key"] != k or m["arm"] != e["arm"] or m["inputs_sha256"] != e["staged_inputs_sha256"] or m["task"] != task:
                fails.append(f"{k}: original meta key, arm, inputs or task is not the manifest's")
            orig[(role, k)] = m
            want_o.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(k)
            shas_o[e["staged_inputs_file"]] = e["staged_inputs_sha256"]
            r = rep[role].get(k)
            if r is not None:
                same = all(r.get(f) == m.get(f) for f in ("run", "task", "epoch", "attempt_index", "attempt_kind", "redrawn_because", "policy_called"))
                if not same or main_of[r["task"]] != main_of[task]:
                    fails.append(f"{role}: {k}: run, task, epoch, attempt or main differs between the replay and its source")
    print(f"metas read: original {len(orig)}, replay {sum(len(v) for v in rep.values())}", flush=True)
    # fixed-token rows, by the scorer's own rule, on both staged sets
    so = _safe(lambda: S.skip_rows("/b", {r: {a: sorted(set(ks)) for a, ks in d.items()} for r, d in want_o.items()}, shas_o), "skip_rows on corrected B's staged inputs")
    sr = {role: _safe(lambda role=role: S.skip_rows("/q", {STAGED: {ARMS[role][0]: sorted(want_keys[role])}},  # one call per arm (shared keys)
                                                    {f"{STAGED}/inputs/{ARMS[role][0]}.jsonl": stg[f"{ARMS[role][0]}.jsonl"]}),
                      f"skip_rows on {ARMS[role][0]}.jsonl") for role in ARMS}
    rec = {role: json.loads(blobs[ARMS[role][3]]) for role in ARMS}
    xc = {"fixed_tokens_equal_score_records": all(so["fixed_tokens"] == sr[role]["fixed_tokens"] == rec[role]["skip"]["fixed_tokens"] for role in ARMS),
          "fixed_tokens": so["fixed_tokens"],
          "replay_dropped_rows_equal_score_records": all(sr[role]["dropped_rows"][f"{STAGED}/inputs/{ARMS[role][0]}.jsonl"] ==
                                                         rec[role]["skip"]["dropped_rows"][f"{STAGED}/inputs/{ARMS[role][0]}.jsonl"] for role in ARMS)}
    print("skip rows done", flush=True)
    # per run: (b) and (c)
    arms_out = {}
    for role in ARMS:
        tot = {"runs": 0, "runs_pass": 0, "map_equal": 0, "truncated_original": 0, "truncated_replay": 0, "consistency_failure_runs": 0, "name_in_meta_all": True}
        for ro in ("R", "C"):
            tot[ro] = {"retained_original": 0, "retained_replay": 0, "overlap": 0, "shared_prefix_rows": 0, "order_mismatch_runs": 0,
                       "original_only": {}, "replay_only": {}, "unexplained": 0}
        for k in sorted(want_keys[role]):
            if (role, k) not in orig or k not in rep[role]:
                continue
            norm = lambda xs: {(str(i), int(j)) for i, j in xs}  # noqa: E731  (the identity tuple's types)
            c = compare(orig[(role, k)], rep[role][k], norm(so["skip"][k]), norm(sr[role]["skip"][k]))
            per_run[f"{role}|{want_keys[role][k]}"] = c
            tot["runs"] += 1
            tot["runs_pass"] += c["pass"]
            if "fail" in c:
                continue
            tot["map_equal"] += c["map_equal"]
            tot["truncated_original"] += c["truncated (staged length >= 32,768)"][0]
            tot["truncated_replay"] += c["truncated (staged length >= 32,768)"][1]
            tot["consistency_failure_runs"] += bool(c["consistency_failures"])
            tot["name_in_meta_all"] &= c["name_in_meta"]
            for ro in ("R", "C"):
                x, t = c[ro], tot[ro]
                t["retained_original"] += x["retained"][0]
                t["retained_replay"] += x["retained"][1]
                t["overlap"] += x["both"]
                t["shared_prefix_rows"] += x["shared_prefix_rows"]
                t["order_mismatch_runs"] += not x["shared_prefix_order_equal"]
                t["unexplained"] += x["unexplained"]
                for a, b in (("o_only", "original_only"), ("r_only", "replay_only")):
                    for why, n in x[a].items():
                        t[b][why] = t[b].get(why, 0) + n
        for ro in ("R", "C"):
            sc = rec[role]["probes"]["own_frac100"]["readouts"][ro]["file"]["rows"]
            tot[ro]["replay_retained_equals_score_record_rows"] = tot[ro]["retained_replay"] == sc
            tot[ro]["score_record_rows"] = sc
        tot["pass"] = (tot["runs"] == len(want_keys[role]) == tot["runs_pass"] and all(tot[ro]["replay_retained_equals_score_record_rows"] for ro in ("R", "C")))
        if role == "control":  # no swap: everything must be identical
            tot["control_identical"] = all(f"control|{t}" in per_run for t in CONTROL) and tot["map_equal"] == len(CONTROL) and all(
                not per_run[f"control|{t}"].get(ro, {}).get(s) for t in CONTROL for ro in ("R", "C") for s in ("o_only", "r_only"))
            tot["pass"] = tot["pass"] and tot["control_identical"]
        arms_out[role] = tot
    ok = not fails and all(v["equal"] for v in key_sets.values()) and all(v["pass"] for v in arms_out.values()) and all(
        v for k, v in xc.items() if k != "fixed_tokens")
    return {"pass": ok, "fails": fails[:50], "n_fails": len(fails), "key_sets": key_sets, "cross_checks": xc, "arms": arms_out,
            "skip_dropped_rows": {"original": so["dropped_rows"], "replay": {role: sr[role]["dropped_rows"] for role in ARMS}}, "per_run": per_run,
            "tool_function_name": "present in the meta (calls[].name, the staged call's function name); part of the identity tuple"
            if all(v["name_in_meta_all"] for v in arms_out.values()) else "NOT present for every unit: identity tuple without it"}


def _lines(s: dict) -> list:
    out = [f"join gate (RULE staging/extraction gate 7)  git {s['git'][:9]}  stamp {s['stamp']}  PASS {s['pass']}  fails {s['n_fails']} {s['fails'][:5]}",
           f"key sets: {json.dumps(s['key_sets'])}", f"cross-checks: {json.dumps(s['cross_checks'])}", f"tool-function name: {s['tool_function_name']}"]
    for role, t in s["arms"].items():
        out.append(f"{role}: runs {t['runs']} pass {t['runs_pass']} map_equal {t['map_equal']} consistency-failure runs {t['consistency_failure_runs']} "
                   f"truncated original {t['truncated_original']} "
                   f"replay {t['truncated_replay']} -> arm pass {t['pass']}" + (f" control_identical {t['control_identical']}" if "control_identical" in t else ""))
        for ro in ("R", "C"):
            x = t[ro]
            out.append(f"  {ro}: retained original {x['retained_original']} replay {x['retained_replay']} (score record {x['score_record_rows']}, equal "
                       f"{x['replay_retained_equals_score_record_rows']}) overlap {x['overlap']} shared-prefix rows {x['shared_prefix_rows']} "
                       f"order-mismatch runs {x['order_mismatch_runs']} | original-only {json.dumps(x['original_only'])} | replay-only "
                       f"{json.dumps(x['replay_only'])} | unexplained {x['unexplained']}")
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes: commit (and push) this script first")
    git("ls-files", "--error-unmatch", str(Path(__file__).resolve().relative_to(REPO)))
    blobs = {}
    for n, (c, p, h) in PINS.items():
        blobs[n] = subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True, cwd=REPO).stdout
        if _sha(blobs[n]) != h:
            raise SystemExit(f"{n}: pinned bytes differ ({c}:{p})")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    s = {"label": "RULE gate 7 (join gate), extraction metadata only; PRELIMINARY until an independent recheck", "git": git("rev-parse", "HEAD"),
         "stamp": stamp, "pins": {n: {"commit": c, "path": p, "sha256": h} for n, (c, p, h) in PINS.items()},
         "volumes": {"original": BVOL, "replay": f"{QVOL}:/{EXTRACT}", "replay_staged": f"{QVOL}:/{STAGED}"}} | gate.remote(blobs)
    dest = HERE / "results" / f"join_gate_{stamp}.json"
    with dest.open("x") as fh:
        json.dump(s, fh, indent=1)
    lines = _lines(s)
    with dest.with_suffix(".printout.txt").open("x") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"{dest.name} sha256 {_sha(dest.read_bytes())}")
    if not s["pass"]:
        raise SystemExit("gate 7 did NOT pass (record written above)")
