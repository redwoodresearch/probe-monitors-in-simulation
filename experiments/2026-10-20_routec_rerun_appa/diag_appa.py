"""appA diagnostic (09's request, 2026-10-20 ~13:4x PDT): why H_B's counts differ from the record. Keys, counts, booleans
and 0-dp pct strings only. One CPU container; every mount is read-only; nothing is written.

It reuses the committed run_appa.py (mounted beside it, sha-checked in the container) for staged parsing, tallying and the
pct rule, and the main run's own results JSON (sha-pinned) for the counted folders, their chosen staged files and the
record's cells (copied there from the pinned record @ a6d37c024 at run time).

Prints (DIAG_* lines only):
  DIAG_SELF   <group|skip> <check> <True|False>  positive controls (predicted True); NEG_* checks predicted False
  DIAG_KEY    <group> <convention> <key> MATCH|MISMATCH   against the record's cell, keys units, text_editor, bash,
                                                  other_by_name, runs, runs_with_units, unmapped_units
  DIAG_PCT    <group> <convention>=<int|None> ...  0-dp half-up pct (PRESTATED v2 L102-L103)
  DIAG_CAUSE  <group> <cause> <count>             why the copies of each conflicting run differ
  DIAG_PAIRS  <group> <first override>-><other override> <count>
  DIAG_NAPP   <group> <n appearances>=<count> ...
  DIAG_EVAL_COMPRESS {<zip compress_type>: <members>} files=<n>   .eval zip directory metadata only (no member is read)
  DIAG_SKIP   record_skipped=<n> matched_folders=<n> unmatched_keys=<n> matched=[<override>:<folder>, ...]  (amendment 1)
Amendment 1 (09): record_skips_nodedup, record_skips+first_appearance and record_skips+last_appearance for H_B, H_A and
run1 (DIAG_KEY / DIAG_PCT), plus DIAG_SELF run1 first_appearance_equals_stored_main.
Amendment 2 (reviewer and 09): record_skips+conflicts_excluded (DIAG_KEY / DIAG_PCT) and DIAG_CAUSE <g>
record_skips_conflicting_runs <n>; two negative controls, DIAG_SELF skip NEG_perturbed_key_matches_a_folder and DIAG_SELF
H_B NEG_injected_skip_equals_full_first (15 DIAG_SELF lines in all).
Conventions: first_appearance (the run's G3 rule); union and first_eligible (stored by the run; for honest groups they are
the same as first_appearance by construction, checked); conflicts_excluded (09's G3 sensitivity); last_appearance (post
hoc, diagnostic only).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import types
import zipfile
from collections import Counter

import modal

APP_NAME = "probe-shift_aswin-surya_sub-rerun_appadiag_kill-2026-11-19"
assert len(APP_NAME) <= 63
OUT_VOLUME = "probe-shift_aswin-surya_sub-rerun_appa_kill-2026-11-19"
RESULTS_REL = "appa/2026-10-20_132942/20261020T203119Z_c452d576bc50/results_appa_2026-10-20_132942.json"
RESULTS_SHA = "dd433e4aabbe63d15c8f04749a65b7ba2d1ceba01eb78e70e507230df041eace"
RUN_APPA_SHA = "da779ec0cbe072df333dea963d8cbbfa7a4398292154a77e43718bcfa38325d8"
CATALOGUE_REL = ("xfer/fpr507prep/records/results/"
                 "alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-18_112818_catalogue.json")
CATALOGUE_SHA = "72432318a3703d4675a5c4530d77f598a003af351340d895d305b9bd1050adc8"
VOLS = {"/out": OUT_VOLUME,
        "/scores": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14",
        "/staged": "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14",
        "/catalog": "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13",
        "/ev_chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",
        "/ev_q4own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
        "/ev_pilots": "probe-shift_aswin-surya_sub-q3pilots_runs_kill-2026-11-13",
        "/ev_q2fresh": "probe-shift_aswin-surya_sub-q2-fresh_work_kill-2026-11-13"}
EVAL_ROOTS = ("/ev_chain/evals_v1", "/ev_q4own/evals_v1", "/ev_pilots", "/ev_q2fresh")
GROUPS = ("H_B", "H_A")
KEYS = ("units", "text_editor", "bash", "other_by_name", "runs", "runs_with_units", "unmapped_units")
N_EVAL_META = 20
BLOB_NAMES = ("record", "exclusions", "ov_S_new", "ov_S_old", "ov_S_newmains", "ov_S_paper493", "ov_pools_ep2",
              "ov_S_new25f")  # amendment 1: run_appa.PINS names
CAUSES = ("staged_sha_differs", "epoch_launch_path_differs", "epoch_index_differs", "readout_tokens_differ",
          "same_tokens_tools_differ", "same_override_set", "other")

app = modal.App(APP_NAME)
_here = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.1.3")
         .add_local_file(os.path.join(_here, "run_appa.py"), "/root/run_appa.py"))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _cmp(t: dict, run_keys, cell: dict) -> dict:
    """Booleans per key against the record's pooled cell (p2 reader), as run_appa.compare does. run_keys: the
    convention's run keys, or an int count when only the count is stored (then a list-valued record `runs` is ABSENT)."""
    p2 = ((cell or {}).get("readers") or {}).get("p2") or {}  # a missing record field is ABSENT (None), not an error
    rec_runs = (cell or {}).get("runs")
    out = {k: (t[k] == p2[k] if k in p2 else None) for k in ("units", "text_editor", "bash", "runs_with_units", "unmapped_units")}
    out["other_by_name"] = (t["other"] == {n: c for n, c in sorted(p2["other"].items()) if c}
                            if isinstance(p2.get("other"), dict) else None)
    if rec_runs is None:
        out["runs"] = None
    elif isinstance(rec_runs, (list, dict)):
        out["runs"] = None if isinstance(run_keys, int) else set(run_keys) == set(rec_runs)
    else:
        out["runs"] = (run_keys if isinstance(run_keys, int) else len(run_keys)) == rec_runs
    return out


def _record_skips(R, blobs, folders, apps, roles, rec, mine) -> list:
    """Amendment 1 (09): drop exactly the record's skipped_folders, by folder key, then recompute H_B, H_A and run1 as
    record_skips_nodedup (every remaining appearance), record_skips+first_appearance and record_skips+last_appearance.
    A folder's keys: its .arms key or .simh folder name, its p2 (cutfull) folder name, and for .simh its override `root`.
    Membership: honest groups by catalogue role; run1 by world token run1 and the task in the cohort of a remaining
    appearance's set (the run's per-set rule). Prints keys, counts, booleans and 0-dp pct strings only."""
    lines = []
    record = json.loads(blobs["record"])
    skip_keys = []
    for e in record["skipped_folders"]:
        if isinstance(e, str):
            skip_keys.append(e)
        elif isinstance(e, dict) and any(isinstance(e.get(x), str) for x in ("folder", "name", "root", "key", "arm")):
            skip_keys.append(next(e[x] for x in ("folder", "name", "root", "key", "arm") if isinstance(e.get(x), str)))
        else:
            raise ValueError("a skipped_folders element has no folder key")
    ovs = {n: json.loads(blobs[n]) for n in R.OVERRIDES}
    aliases = []
    for f in folders:
        al = {f["name"], f["p2_path"].split("/")[-2]}
        if f["kind"] == "simh":
            al |= {e["root"] for e in ovs[f["override"]]["simh"][R.LEAD3[0]] if e["folder"] == f["name"] and e.get("root")}
        aliases.append(al)
    skip_fi = {fi for fi, al in enumerate(aliases) if al & set(skip_keys)}
    lines.append(f"DIAG_SKIP record_skipped={len(skip_keys)} matched_folders={len(skip_fi)} "
                 f"unmatched_keys={sum(not any(k in al for al in aliases) for k in skip_keys)} "
                 f"matched={sorted(folders[fi]['override'] + ':' + folders[fi]['name'] for fi in skip_fi)}")
    # amendment 2, negative control (predicted False): perturbed skip keys (one character appended; separately, the last
    # character dropped) must match no folder by exact set intersection
    perturbed = {k + "x" for k in skip_keys} | {k[:-1] for k in skip_keys if k}
    lines.append(f"DIAG_SELF skip NEG_perturbed_key_matches_a_folder {any(al & perturbed for al in aliases)}")
    cohort = {s: set(v["cells"]) for s, v in json.loads(blobs["exclusions"])["final_cohort_keys"].items()}

    def member(g, rk, a):  # a: appearances in G1 order
        if g != "run1":
            return bool(a) and g in roles.get(rk, set())
        world, task = rk.split("|", 1)
        return world == "run1" and any(task in cohort.get(folders[x["fi"]]["set"], ()) for x in a)

    tk = ("units", "text_editor", "bash", "other", "runs_with_units", "unmapped_units", "pct")
    for g in ("H_B", "H_A", "run1"):
        full = {rk: a for rk, a in apps.items() if member(g, rk, a)}
        if g == "run1":  # self-check: the main rule reproduces the run's stored run1 totals and run identities
            t0 = R.tally(a[0] for a in full.values())
            ok = all(t0[k] == mine[g][k] for k in tk) and sorted(full) == mine[g]["run_keys"]
            lines.append(f"DIAG_SELF run1 first_appearance_equals_stored_main {ok}")
        if g == "H_B":  # amendment 2, negative control (predicted False): an injected skip must change the totals
            cand = next(rk for rk in sorted(full)
                        if len(full[rk]) == 1 and any(o[0] == "name" for _, o in full[rk][0]["sig"]))
            inj = {full[cand][0]["fi"]}
            kept_inj = {rk: [x for x in a if x["fi"] not in inj] for rk, a in apps.items()}
            kept_inj = {rk: a for rk, a in kept_inj.items() if member(g, rk, a)}
            t_inj, t_full = R.tally(a[0] for a in kept_inj.values()), R.tally(a[0] for a in full.values())
            lines.append(f"DIAG_SELF H_B NEG_injected_skip_equals_full_first "
                         f"{all(t_inj[k] == t_full[k] for k in tk) and sorted(kept_inj) == sorted(full)}")
        kept = {rk: [x for x in a if x["fi"] not in skip_fi] for rk, a in apps.items()}
        kept = {rk: a for rk, a in kept.items() if member(g, rk, a)}
        # amendment 2: G3 under the skips; a run conflicts if its remaining copies differ (G3 signature)
        rs_conflict = {rk for rk, a in kept.items() if any(x["csig"] != a[0]["csig"] for x in a[1:])}
        lines.append(f"DIAG_CAUSE {g} record_skips_conflicting_runs {len(rs_conflict)}")
        conv = {"record_skips_nodedup": (R.tally(x for a in kept.values() for x in a), sorted(kept)),
                "record_skips+first_appearance": (R.tally(a[0] for a in kept.values()), sorted(kept)),
                "record_skips+last_appearance": (R.tally(a[-1] for a in kept.values()), sorted(kept)),
                "record_skips+conflicts_excluded": (R.tally(a[0] for rk, a in kept.items() if rk not in rs_conflict),
                                                    sorted(rk for rk in kept if rk not in rs_conflict))}
        if g == "run1":
            conv = {"first_appearance": (t0, sorted(full))} | conv
        for c, (t, rks) in conv.items():
            for k, ok in _cmp(t, rks, rec.get(g)).items():
                lines.append(f"DIAG_KEY {g} {c} {k} {'ABSENT' if ok is None else 'MATCH' if ok else 'MISMATCH'}")
        lines.append(f"DIAG_PCT {g} " + " ".join(f"{c}={t['pct']}" for c, (t, _) in conv.items()))
    return lines


def _diag(blobs: dict) -> list:
    import numpy as np

    lines = []
    src = open("/root/run_appa.py", "rb").read()
    if _sha(src) != RUN_APPA_SHA:
        raise ValueError("mounted run_appa.py sha differs")
    sys.path.insert(0, "/root")
    import run_appa as R  # noqa: E402  (the committed main-run module)

    if set(blobs) != set(BLOB_NAMES) or any(_sha(blobs[n]) != R.PINS[n][2] for n in BLOB_NAMES):
        raise ValueError("amendment 1 inputs differ from run_appa.PINS")

    raw = open(os.path.join("/out", RESULTS_REL), "rb").read()
    if _sha(raw) != RESULTS_SHA:
        raise ValueError("results JSON sha differs")
    doc = json.loads(raw)
    folders, staged, rec, mine, alt = (doc["detail"]["folders"], doc["detail"]["staged"], doc["record_cells"], doc["mine"],
                                       doc["mine_alternatives"])
    craw = open(os.path.join("/catalog", CATALOGUE_REL), "rb").read()
    if _sha(craw) != CATALOGUE_SHA:
        raise ValueError("catalogue sha differs")
    roles = {rk: {R.ROLE_GROUP[r] for r in c["roles"]} for rk, c in json.loads(craw)["runs"].items()}

    # pass 1: p2 rows of honest A/B runs, per folder (sha-checked against the run's own folder list)
    rows_by_folder = {}
    for fi, f in enumerate(folders):
        b = open(os.path.join("/scores", f["p2_path"]), "rb").read()
        if _sha(b) != f["p2_sha256"]:
            raise ValueError("p2 file sha differs from the run's folder list")
        z = np.load(io.BytesIO(b), allow_pickle=False)
        rks, keys, toks = z["key"].astype(str).tolist(), z["run_key"].astype(str).tolist(), z["token"].astype(np.int64).tolist()
        keep = [(k, rk, t) for k, rk, t in zip(keys, rks, toks)
                if roles.get(rk, set()) & set(GROUPS) or rk.split("|", 1)[0] == "run1"]  # run1 rows: amendment 1
        if keep:
            rows_by_folder[fi] = keep

    # pass 2: staged files of those folders only (run_appa.parse_staged; sha must equal the run's chosen file)
    parsed = {}
    for fi in sorted(rows_by_folder):
        s = folders[fi]["staged_sha256"]
        if s not in parsed:
            p = R.parse_staged(staged[s])
            if p["sha256"] != s:
                raise ValueError("staged file sha differs from the run's chosen file")
            parsed[s] = p["cmd"]
    apps = {}
    for fi in sorted(rows_by_folder):  # folders are in the run's G1 order
        st, per_run = parsed[folders[fi]["staged_sha256"]], {}
        for k, rk, t in rows_by_folder[fi]:
            ep = st.get(k)
            if ep is None:
                out = ("unmapped", "no_epoch")
            else:
                calls = ep.get(t, [])
                out = (("name",) + tuple(calls[0])) if len(calls) == 1 else ("unmapped", "multi_call" if calls else "no_call")
            per_run.setdefault(rk, []).append((k, t, out))
        for rk, rws in per_run.items():
            apps.setdefault(rk, []).append({"fi": fi, "rows": rws, "sig": sorted((t, o) for _, t, o in rws),
                                            "csig": sorted((t, o[:2]) for _, t, o in rws),
                                            "tokens": sorted(t for _, t, _ in rws), "keys": sorted({k for k, _, _ in rws})})

    for g in GROUPS:
        runs = {rk: a for rk, a in apps.items() if g in roles.get(rk, set())}
        conflict = {rk for rk, a in runs.items() if any(x["csig"] != a[0]["csig"] for x in a[1:])}
        conv = {"first_appearance": {rk: a[0] for rk, a in runs.items()},
                "conflicts_excluded": {rk: a[0] for rk, a in runs.items() if rk not in conflict},
                "last_appearance": {rk: a[-1] for rk, a in runs.items()}}
        tal = {c: R.tally(m.values()) for c, m in conv.items()}
        first, stored = tal["first_appearance"], mine[g]
        tk = ("units", "text_editor", "bash", "other", "runs_with_units", "unmapped_units", "pct")
        same = all(first[k] == stored[k] for k in tk) and sorted(conv["first_appearance"]) == stored["run_keys"]
        lines.append(f"DIAG_SELF {g} first_appearance_equals_stored_main {same}")  # totals and run identities
        lines.append(f"DIAG_SELF {g} conflict_runs_equal_stored {sorted(conflict) == stored['conflict_run_keys']}")
        ce = tal["conflicts_excluded"]["pct"]
        lines.append(f"DIAG_SELF {g} conflicts_excluded_pct_equals_stored {ce is not None and ce == stored['pct_excl_conflict_runs']}")
        for v in ("union", "first_eligible"):  # honest groups: identical to first appearance by construction
            a_ = alt[v][g]
            lines.append(f"DIAG_SELF {g} {v}_identical_to_first_appearance "
                         f"{all(a_[k] == first[k] for k in tk) and a_['runs'] == len(conv['first_appearance'])}")
        # negative control (predicted False): drop the first run that has a mapped unit, which must change the totals
        drop = next(rk for rk in sorted(conv["first_appearance"])
                    if any(o[0] == "name" for _, o in conv["first_appearance"][rk]["sig"]))
        drop_one = R.tally(a for rk, a in conv["first_appearance"].items() if rk != drop)
        lines.append(f"DIAG_SELF {g} NEG_first_minus_one_run_equals_stored_main "
                     f"{all(drop_one[k] == stored[k] for k in ('units', 'text_editor', 'bash', 'other', 'runs_with_units'))}")
        results = {c: (tal[c], sorted(conv[c])) for c in conv} | {v: (alt[v][g], alt[v][g]["runs"]) for v in ("union", "first_eligible")}
        for c, (t, rks) in results.items():
            for k, ok in _cmp(t, rks, rec.get(g)).items():
                lines.append(f"DIAG_KEY {g} {c} {k} {'ABSENT' if ok is None else 'MATCH' if ok else 'MISMATCH'}")
        rp2 = (((rec.get(g) or {}).get("readers") or {}).get("p2")) or {}
        rec_pct = R.pct(rp2["text_editor"], rp2["units"]) if "text_editor" in rp2 and "units" in rp2 else "ABSENT"  # Codex d3 #4
        lines.append(f"DIAG_PCT {g} " + " ".join(f"{c}={tal[c]['pct']}" for c in conv)
                     + f" union={alt['union'][g]['pct']} first_eligible={alt['first_eligible'][g]['pct']} record={rec_pct}")
        causes, pairs, spairs = Counter({c: 0 for c in CAUSES}), Counter(), Counter()
        napp = Counter(len(runs[rk]) for rk in conflict)  # conflicting runs only (Codex d1 #2: all runs would sum to `runs`)
        for rk in conflict:
            a = runs[rk]
            f0 = folders[a[0]["fi"]]
            later = [x for x in a[1:] if x["csig"] != a[0]["csig"]]  # only the copies that differ (Codex d1 #5)
            for x in later:
                fx = folders[x["fi"]]
                pairs[f"{f0['override']}->{fx['override']}"] += 1
                spairs[f"{f0['name'].rsplit('__', 1)[-1]}->{fx['name'].rsplit('__', 1)[-1]}"] += 1

            def parts(x, i):
                return sorted({k.rpartition("#e")[i] for k in x["keys"]})
            flags = {"staged_sha_differs": any(folders[x["fi"]]["staged_sha256"] != f0["staged_sha256"] for x in later),
                     "epoch_launch_path_differs": any(parts(x, 0) != parts(a[0], 0) for x in later),
                     "epoch_index_differs": any(parts(x, 2) != parts(a[0], 2) for x in later),
                     "readout_tokens_differ": any(x["tokens"] != a[0]["tokens"] for x in later),
                     "same_tokens_tools_differ": any(x["tokens"] == a[0]["tokens"] for x in later),
                     "same_override_set": any(folders[x["fi"]]["override"] == f0["override"] for x in later)}
            for name, on in flags.items():
                causes[name] += on
            causes["other"] += not any(flags[k] for k in ("staged_sha_differs", "epoch_launch_path_differs",
                                                          "epoch_index_differs", "readout_tokens_differ"))
        lines.append(f"DIAG_CAUSE {g} conflicting_runs {len(conflict)}")
        lines += [f"DIAG_CAUSE {g} {k} {causes[k]}" for k in CAUSES]
        lines += [f"DIAG_PAIRS {g} {k} {v}" for k, v in sorted(pairs.items())]
        lines += [f"DIAG_PAIRS_SUFFIX {g} {k} {v}" for k, v in sorted(spairs.items())]
        lines.append(f"DIAG_NAPP {g} " + " ".join(f"{n}={c}" for n, c in sorted(napp.items())))

    lines += _record_skips(R, blobs, folders, apps, roles, rec, mine)

    # .eval compression methods (zip central directory only; no member is opened); honest runs only, as before amendment 1
    seen, bad, hist, done = 0, 0, Counter(), set()
    for rk in sorted(r for r in apps if roles.get(r, set()) & set(GROUPS)):
        for k in apps[rk][0]["keys"]:
            if seen >= N_EVAL_META:
                break
            rel = k.rpartition("#e")[0]
            hit = next((os.path.join(r, rel) for r in EVAL_ROOTS if os.path.isfile(os.path.join(r, rel))), None)
            if hit and hit not in done:  # distinct files (Codex d1 #6)
                done.add(hit)
                try:
                    with zipfile.ZipFile(hit) as zf:
                        hist.update(str(i.compress_type) for i in zf.infolist())
                except zipfile.BadZipFile:
                    bad += 1
                seen += 1
        if seen >= N_EVAL_META:
            break
    lines.append(f"DIAG_EVAL_COMPRESS {json.dumps(dict(sorted(hist.items())))} files={seen} bad_zip={bad}")
    return lines


@app.function(image=image, cpu=1.0, memory=8192, timeout=600, retries=0,
              volumes={m: modal.Volume.from_name(n).with_mount_options(read_only=True) for m, n in VOLS.items()})
def diag(blobs: dict) -> str:
    """A failure leaves the container only as DIAG_FAILED type=<exception type>, raised outside the except block."""
    err = None
    try:
        return json.dumps(_diag(blobs))
    except Exception as e:  # never swallowed: re-raised below without its message
        err = type(e).__name__
    raise RuntimeError(f"DIAG_FAILED type={err}")


@app.local_entrypoint()
def main():
    """Codex d2 #1: a remote failure is caught here and printed only as its fixed-form DIAG_FAILED line (exit 1). Platform
    failures before the function runs (image build, startup, timeout, out of memory) are reported by the modal client in
    its own words; the launch RUN_ME reads only anchored DIAG_ lines, so a missing DIAG_DONE counts as a failure."""
    here = os.path.abspath(__file__)
    print(f"DIAG_START script_sha256={hashlib.sha256(open(here, 'rb').read()).hexdigest()}", flush=True)
    # amendment 1 inputs: the pinned record, exclusions and six overrides, read from git as bytes (never printed) and
    # sha-checked here against the committed run_appa.PINS, and again in the container. Codex d5: run_appa.py is read
    # once, its sha checked, and exactly those bytes are executed as the module; any local failure prints only its
    # fixed DIAG_FAILED form and exits 1.
    failed, blobs = None, {}
    try:
        rp = os.path.join(os.path.dirname(here), "run_appa.py")
        src = open(rp, "rb").read()
        if hashlib.sha256(src).hexdigest() != RUN_APPA_SHA:
            raise ValueError("local run_appa sha")
        R = types.ModuleType("run_appa")
        R.__file__ = rp
        sys.modules["run_appa"] = R
        exec(compile(src, rp, "exec"), R.__dict__)  # the sha-checked bytes; no numpy needed at import
        for n in BLOB_NAMES:
            commit, path, want = R.PINS[n]
            b = subprocess.run(["git", "-C", os.path.dirname(here), "show", f"{commit}:{path}"], capture_output=True,
                               check=True).stdout
            if hashlib.sha256(b).hexdigest() != want:
                raise ValueError("local pin sha")
            blobs[n] = b
    except Exception as e:  # not swallowed: reported in its fixed form below, then exit 1
        failed = f"DIAG_FAILED type=local_{type(e).__name__}"
    if failed:
        print(failed, flush=True)
        sys.exit(1)
    try:
        out = json.loads(diag.remote(blobs))
    except Exception as e:  # not swallowed: reported in its fixed form below, then exit 1
        m = re.search(r"DIAG_FAILED type=[A-Za-z_0-9]+", str(e))
        failed = m.group(0) if m else f"DIAG_FAILED type=client_{type(e).__name__}"
    if failed:
        print(failed, flush=True)
        sys.exit(1)
    for line in out:
        print(line)
    print("DIAG_DONE", flush=True)
