"""09's container check on CAT's 10-16 diff (10-18, S1a item 3, clauses 1-4; clause 5, M1, is accepted). Side a is the 10-16 catalogue spec
on the solavg Volume, side b the launched CAT pickle; both are unpickled only in a Modal CPU container. Keys, shas, booleans and counts out.

Clause 1, full_allowed per job: A_j / R_j = the (vol, path, sha256) triples added / removed (a sha change is a remove plus an add). The
EXPECTED delta uses full_override's own rule: the triple of every entry _walk_entries collects (L375-376, L380-392, imported from the
pinned full_override.py), over the S_old branch's uncut entries of apply_full (real arms with vol conv, L330 / L334; relaunch arms, L339 /
L346; every H_sim entry of the label's file, L352-358, since s_old_simh keeps them all, L245-246), applied to each side's combined S_old
override (combine, L80) and roles_old. Every key of `overrides` and of every roles_* block must be equal on both sides except the pins of
S_old|C and S_old|R and roles_old gaining the pilot folder as simh (inputs_delta; any other difference STOPs, on the laptop before any
Modal call and again in the container; 09, 10-18), so the other sets' entries cancel; rp L6352 / L6403 add nothing else that differs. expected_added = T(FROZEN) - T(side a), expected_removed the reverse; PASS iff
A_j == expected_added and R_j == expected_removed for every job.
Clause 2: each label's tagged (s_old_only) simh_old entries on side b's S_old jobs equal FROZEN's extra-root entries for that label's file
(the S_old override's H_sim entries with root == the pilot root; written by apply_full L352-358, given to every job by full_jobs
L474-475, rp L6352 / L6667); per label, the jobs carrying them and the entries per job are reported (the 16-20 per-label path counts).
Clause 3: side b's meta pins equal FROZEN (inputs_full, every override pin) and the three named pins. Clause 4: every differing path of the
committed diff record falls in clauses 1-3 or the listed allowed meta paths; anything else is reported and fails.

  modal run -n <description> tools_routec/full_allowed_check.py --a vol:<path> --b <local .pkl> --diff-gz <path> --diff-raw-sha <sha256>
      --frozen <path> --inputs-a <path> --out results/spec_diff/<name>.json        (paths relative to the 507-prep folder; --out exclusive)
"""

import gzip
import hashlib
import json
import pickle
import re
import subprocess
from pathlib import Path

import modal

VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # = spec_diff.VOL
PILOT = "simh_q4f_v1_pilot__sim_honest"  # FROZEN's s_old_simh_extra_roots root (= full_override.S_OLD_EXTRA)
TAG = "s_old_only"  # = full_override.S_OLD_ONLY
M = "meta/pooled_world/modules_sha256/experiments/2026-10-15_fpr_sweep_507_prep/"
EXPECT3 = {"fold_minus1_drops": ("fold_minus1_drops_TEST_2026-10-17_223409.json", None, "d898a5762984dbf7030ac965bb72870f41f31fee6bc0c1ce548da307776a68bb"),
           "census_uncut": ("nofilter_keys_FULL_TEST_2026-10-17_220545.json", "c979a1ce2", "2b3380ae40dd3bd81934006d36e27e9253fda153171a18326edad736249d6cc6"),
           "waitlist": ("waitlist_full_2026-10-17_223345.json", "e26ddf213", "6529298069c686c062961a6c1a763a8f28def3cfd55f188e728aac0756ce47b2")}
P3 = ({f"meta/full_length/inputs_full/{k}" for k in ("path", "sha256")} | {f"meta/full_length/overrides/S_old|{r}/{k}" for r in "CR" for k in ("commit", "path", "sha256")}
      | {f"meta/{n}/{k}" for n, (_, c, _) in EXPECT3.items() for k in (("path", "sha256") if c is None else ("commit", "path", "sha256"))})
P4 = ({M + f for f in ("run_common710.py", "run_common455.py", "inputs_full.py", "inputs_final.py", "inputs_final_v2.py", "full_override.py", "copies.py")}
      | {"meta/full_length/allowed_files", "meta/full_length/inputs_full/status", "meta/full_length/s_old_simh_extra_roots"})
RX1 = re.compile(r"jobs/\d+/1/full_allowed/\d+(/\d+)?")
RX2 = re.compile(r"jobs/\d+/1/simh_old/[^/]+/\d+")

_here = Path(__file__).resolve().parent
_prep = _here.parent if modal.is_local() else Path("/root")  # the 507-prep folder locally; /root (one parent) in the container
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_facheck_kill-2026-11-14")  # a8953dad / a0b05 O-1: 62 characters
vol = modal.Volume.from_name(VOL, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
         .add_local_file(str(_prep / "full_override.py"), "/root/full_override.py")
         .add_local_file(str(_prep / "cutfull_decisive.py"), "/root/cutfull_decisive.py"))  # full_override's module-top import


def triple(x):
    return (x.get("vol", "conv"), x["path"], x["sha256"])  # full_override L375's tuple


ALLOWED_OVR = {"S_old|C", "S_old|R"}


def inputs_delta(fa, fb):
    """09 (10-18): every key of `overrides` and of every roles_* block, side a's INPUTS_FULL against FROZEN, by key and value (each override
    pin is {commit, path, sha256}). Allowed differences, exactly: the pins of S_old|C and S_old|R, and roles_old gaining the pilot folder
    with role simh. -> a list of the other differences by key (empty = as allowed); any entry STOPs the check."""
    bad = []
    oa, ob = fa.get("overrides"), fb.get("overrides")
    if not isinstance(oa, dict) or not isinstance(ob, dict):
        return ["overrides: not an object on both sides"]
    if set(oa) != set(ob):
        bad.append(f"overrides keys: only_a {sorted(set(oa) - set(ob))} only_b {sorted(set(ob) - set(oa))}")
    bad += [f"overrides {k}" for k in sorted(set(oa) & set(ob)) if k not in ALLOWED_OVR and oa[k] != ob[k]]
    for blk in sorted({k for k in set(fa) | set(fb) if k.startswith("roles")}):
        ra, rb = fa.get(blk), fb.get(blk)
        if blk != "roles_old":
            if ra != rb:
                bad.append(f"{blk}")
            continue
        if not isinstance(ra, dict) or not isinstance(rb, dict):
            bad.append("roles_old: not an object on both sides")
            continue
        if set(ra) - set(rb) or set(rb) - set(ra) != {PILOT} or rb.get(PILOT) != "simh":
            bad.append(f"roles_old keys: only_a {sorted(set(ra) - set(rb))} only_b {sorted(set(rb) - set(ra))} (allowed: only_b [{PILOT}] as simh)")
        bad += [f"roles_old {k}" for k in sorted(set(ra) & set(rb)) if ra[k] != rb[k]]
    return bad


def stop_if_inputs_differ(fa, fb):
    """The ONE stop for 09's condition, used by the laptop path (main, before any Modal call) and the container path (check_remote):
    any difference inputs_delta reports raises SystemExit("STOP: …"); it never becomes pass false. -> [] when there is none."""
    bad = inputs_delta(fa, fb)
    if bad:
        raise SystemExit("STOP: the two INPUTS_FULLs differ outside S_old|C, S_old|R and the pilot folder of roles_old: " + "; ".join(bad))
    return bad


def s_old_entries(FO, ov, roles):
    """The S_old branch's uncut entries, as apply_full makes them (L330 real with vol conv, L339 / L346 relaunch, L353 every H_sim entry),
    collected by FO._walk_entries and turned into triples by L375's rule."""
    by = {r: sorted(f for f, x in roles.items() if x == r) for r in ("real", "relaunch")}
    applied = [{"uncut": [dict(e, folder=f, vol="conv")]} for f in by["real"] for e in (ov.get("arms") or {}).get(f, {}).values()]
    applied += [{"uncut": [dict(e, folder=f)]} for f in by["relaunch"] for e in (ov.get("arms") or {}).get(f, {}).values()]
    applied += [{"simh_uncut": list(es)} for es in (ov.get("simh") or {}).values()]
    return {triple(x) for x in FO._walk_entries(applied)}


def clause1(a, b, exp_add, exp_rem, pilot_triples):
    ja, jb = a.get("jobs") or [], b.get("jobs") or []
    adds, rems = [], []
    for x, y in zip(ja, jb):
        sa = {tuple(t) for t in (x[1] or {}).get("full_allowed") or []}
        sb = {tuple(t) for t in (y[1] or {}).get("full_allowed") or []}
        adds.append(sb - sa)
        rems.append(sa - sb)
    same = len(ja) == len(jb) and all(s == exp_add for s in adds) and all(s == exp_rem for s in rems)
    return {"n_jobs_a": len(ja), "n_jobs_b": len(jb), "added_sizes": sorted({len(s) for s in adds}), "removed_sizes": sorted({len(s) for s in rems}),
            "added_identical_across_jobs": all(s == adds[0] for s in adds) if adds else True,
            "removed_identical_across_jobs": all(s == rems[0] for s in rems) if rems else True,
            "n_expected_added": len(exp_add), "n_expected_removed": len(exp_rem), "n_added_under_pilot_root": len(adds[0] & pilot_triples) if adds else 0,
            "expected_added": sorted(exp_add), "expected_removed": sorted(exp_rem), "pass": same}


def clause2(b, ov_b):
    """Per (job, label): the tagged entries must be exactly the pilot-root entry of the label's file. A registered job (full_as
    "<family>|uncut <C|R>", out_label L54-57) uses the registered file (the probe's "new"); a retrain job may use its retrain file for its own
    label (apply_full's alias, L267), so there the entry must be ONE pilot-root entry of FROZEN's S_old override (reported by file)."""
    simh = ov_b.get("simh") or {}
    pil = {fn: {(e["path"], e["sha256"]) for e in es if e.get("root") == PILOT} for fn, es in simh.items()}
    any_pilot = {t: fn for fn, ts in pil.items() for t in ts}
    fn_of = {p["label"]: p["new"] for j in b.get("jobs") or [] for p in (j[1] or {}).get("probes") or [] if isinstance(p, dict) and "label" in p}
    per, used, bad = {}, set(), []
    for i, j in enumerate(b.get("jobs") or []):
        c = j[1] or {}
        reg_job = "|uncut " in str(c.get("full_as", ""))
        for lb, es in (c.get("simh_old") or {}).items():
            t = {(e["path"], e["sha256"]) for e in es if isinstance(e, dict) and e.get(TAG)}
            if not t:
                continue
            used |= t
            d = per.setdefault(lb, {"n_jobs": 0, "entries_per_job": set(), "origins": set(), "files": set()})
            d["n_jobs"] += 1
            d["entries_per_job"].add(len(t))
            d["origins"].add(str(c.get("origin")))
            d["files"] |= {any_pilot.get(x, "NOT_A_PILOT_ENTRY") for x in t}
            ok = t == pil.get(fn_of.get(lb), set()) and bool(t) or (not reg_job and len(t) == 1 and next(iter(t)) in any_pilot)
            if not ok:
                bad.append(f"jobs/{i} {c.get('full_as')} | {lb}")
    out = {lb: {"n_jobs": d["n_jobs"], "entries_per_job": sorted(d["entries_per_job"]), "origins": sorted(d["origins"]), "files": sorted(d["files"])}
           for lb, d in sorted(per.items())}
    all_p = set(any_pilot)
    return {"labels": out, "n_labels": len(out), "n_pilot_entries_frozen": len(all_p), "n_pilot_entries_used": len(used & all_p),
            "n_bad": len(bad), "bad": bad[:20],
            "pass": bool(out) and not bad and used <= all_p and all(d["origins"] == ["S_old"] and d["entries_per_job"] == [1] for d in out.values())}


def clause3(b, frozen, frozen_sha):
    fl = (b.get("meta") or {}).get("full_length") or {}
    pins = {"inputs_full": {k: (fl.get("inputs_full") or {}).get(k) for k in ("path", "sha256")},
            "overrides": {k: (fl.get("overrides") or {}).get(k) for k in ("S_old|C", "S_old|R")}}
    ok = {"inputs_full": pins["inputs_full"]["sha256"] == frozen_sha and Path(str(pins["inputs_full"]["path"])).name == Path(str(frozen.get("_path"))).name,
          "overrides_all_equal_frozen": fl.get("overrides") == frozen.get("overrides")}
    for n, (base, commit, sha) in EXPECT3.items():
        v = (b.get("meta") or {}).get(n) or {}
        pins[n] = {k: v.get(k) for k in ("commit", "path", "sha256") if k in v}
        ok[n] = (Path(str(v.get("path"))).name == base and v.get("sha256") == sha
                 and (commit is None or str(v.get("commit") or "").startswith(commit)))
    return {"pins": pins, "ok": ok, "pass": all(ok.values())}


def clause4(d):
    paths = [("only_a", p) for p in d.get("only_a") or []] + [("only_b", p) for p in d.get("only_b") or []] + [("changed", p) for p in d.get("changed") or []]
    cls = {"1": 0, "2": 0, "3": 0, "4": 0}
    other = []
    for k, p in paths:
        c = "1" if k != "only_a" and RX1.fullmatch(p) else "2" if k == "only_b" and RX2.fullmatch(p) else "3" if k == "changed" and p in P3 else \
            "4" if k != "only_a" and p in P4 else None
        if c:
            cls[c] += 1
        else:
            other.append(f"{k} {p}")
    return {"n_paths": len(paths), "by_clause": cls, "n_other": len(other), "other": other[:50], "pass": not other}


@app.function(image=image, cpu=(2, 4), memory=(4096, 32768), timeout=1800, retries=0, volumes={"/sol": vol.read_only()})
def check_remote(a_vol_path, b_bytes, diff_gz, frozen_bytes, inputs_a_bytes, ovr) -> dict:
    import full_override as FO  # mounted from the pinned full_override.py (the block checks its blob)

    a = pickle.loads((Path("/sol") / a_vol_path).read_bytes())
    b = pickle.loads(b_bytes)
    frozen, inp_a = json.loads(frozen_bytes), json.loads(inputs_a_bytes)
    frozen["_path"] = ovr["_frozen_path"]
    fla, flb = (a.get("meta") or {}).get("full_length") or {}, (b.get("meta") or {}).get("full_length") or {}
    bad = stop_if_inputs_differ(inp_a, frozen)  # 09 (10-18): again here, on the bytes the container received
    ties = {"a_overrides_equal_inputs_a": fla.get("overrides") == inp_a.get("overrides"),
            "a_inputs_full_sha_equal_inputs_a": (fla.get("inputs_full") or {}).get("sha256") == hashlib.sha256(inputs_a_bytes).hexdigest(),
            "inputs_differ_only_as_allowed": not bad}
    ovs = {}
    for side, fin in (("a", inp_a), ("b", frozen)):
        parts = []
        for k in ("S_old|C", "S_old|R"):
            data = ovr[side][k]
            if hashlib.sha256(data).hexdigest() != fin["overrides"][k]["sha256"]:
                raise SystemExit(f"STOP: {side} {k}: the shipped override's sha256 is not its pin")
            parts.append(json.loads(data))
        ovs[side] = FO.combine(parts)
    ta, tb = s_old_entries(FO, ovs["a"], inp_a["roles_old"]), s_old_entries(FO, ovs["b"], frozen["roles_old"])
    pilot = {triple(e) for es in ovs["b"]["simh"].values() for e in es if e.get("root") == PILOT}
    d = json.loads(gzip.decompress(diff_gz))
    c1, c2, c3, c4 = clause1(a, b, tb - ta, ta - tb, pilot), clause2(b, ovs["b"]), clause3(b, frozen, hashlib.sha256(frozen_bytes).hexdigest()), clause4(d)
    return {"ties": ties, "clause1": c1, "clause2": c2, "clause3": c3, "clause4": c4,
            "pass": all(ties.values()) and c1["pass"] and c2["pass"] and c3["pass"] and c4["pass"]}


@app.local_entrypoint()
def main(a: str, b: str, diff_gz: str, diff_raw_sha: str, frozen: str, inputs_a: str, out: str):
    def git_show(commit, path):
        return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=_prep).stdout

    if not a.startswith("vol:"):
        raise SystemExit("STOP: --a vol:<path>")
    b_bytes, gz = Path(b).read_bytes(), (_prep / diff_gz).read_bytes()
    if hashlib.sha256(gzip.decompress(gz)).hexdigest() != diff_raw_sha:
        raise SystemExit("STOP: the diff record's decompressed sha256 is not --diff-raw-sha")
    fb, ab = (_prep / frozen).read_bytes(), (_prep / inputs_a).read_bytes()
    stop_if_inputs_differ(json.loads(ab), json.loads(fb))  # 09 (10-18): before any Modal call
    ovr = {"_frozen_path": frozen}
    for side, fin in (("a", json.loads(ab)), ("b", json.loads(fb))):  # configs: their override pins only
        ovr[side] = {k: git_show(fin["overrides"][k]["commit"], fin["overrides"][k]["path"]) for k in ("S_old|C", "S_old|R")}
    res = check_remote.remote(a[4:], b_bytes, gz, fb, ab, ovr)
    rec = {"what": "09's full_allowed check, CAT S1a item 3 (clauses 1-4)", "a": {"volume_path": a[4:]},
           "b": {"local": b, "sha256": hashlib.sha256(b_bytes).hexdigest()}, "diff": {"path": diff_gz, "raw_sha256": diff_raw_sha},
           "frozen": {"path": frozen, "sha256": hashlib.sha256(fb).hexdigest()}, "inputs_a": {"path": inputs_a, "sha256": hashlib.sha256(ab).hexdigest()},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=_prep).stdout.decode().strip()} | res
    data = json.dumps(rec, indent=1).encode()
    dest = _prep / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as fh:
        fh.write(data)
    print(json.dumps({"out": out, "sha256": hashlib.sha256(data).hexdigest(), "pass": res["pass"], "ties": res["ties"],
                      "clause_pass": {k: res[k]["pass"] for k in ("clause1", "clause2", "clause3", "clause4")},
                      "clause1": {k: res["clause1"][k] for k in ("added_sizes", "removed_sizes", "added_identical_across_jobs", "n_expected_added",
                                                                  "n_expected_removed", "n_added_under_pilot_root")},
                      "clause4": {k: res["clause4"][k] for k in ("n_paths", "by_clause", "n_other")}}))
