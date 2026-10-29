"""INPUTS_FINAL: the one manifest that pins every input of the final rebuild (stage A), with a fail-closed verifier.

Schema inputs_final/v1 (every key required; unknown top-level keys fail):
  schema               "inputs_final/v1"
  status               "DRAFT (unfrozen)" | "FROZEN <PT time>"
  cutoff_pt, freeze_pt  09's admission cut-off (20:15) and the manifest freeze (20:30), PT
  label                {"template": 09's text, "filled": null | text filled by fill_label() from the ledger}
  sets.S_new           sol_mappings {batch: {commit, file, sha256, label}}; roots {batch: [root]}; manifest_copies {root: {p2, famexp, own_a}}
                       (sha256 of the local copies; famexp/own_a must equal 4f's file maps); file_maps {commit};
                       honest_a_25f [{cell_prefix, kind p2|famexp|sol, commit, record, sha256, folder, copy_sha256 | null}];
                       dt_sim_honest [{commit, record, sha256}] (their folders must equal the mappings' roots, both ways,
                       except mapped roots listed in dt_roots_pending [root])
  sets.S_newmains      the same keys plus membership710 {path, sha256} and s4_source_map {commit, path, sha256} (d9's S4 map, 4r part): every arm
                       and pool, DT included, resolves through it; no outside_source_map row (key at its orphan root, or its .eval sha256) may
                       appear in the membership or in any pinned input; each 4r cell x world has at most one row, from an s4r folder
  sets.S_old           e2e_config {commit, path}; dt_sim_honest [{commit, record, sha256}]
  ledger               {source, excluded [{set, batch, cell_prefix, class, reason, m_r, ruling}],
                        replacements_admitted [{set, batch, cell_prefix, class, root, ruling}], pending_at_cutoff [{set, batch, cell_prefix, class}],
                        empty_replacement_roots [{set, batch, root, cells, status "empty, excluded by key", evidence [pins]}]}
  epoch2_honest        {status E2_OFF ("epoch-2 not in this build (second refresh)", 0d's wording) | "in", note null | {commit, path}, artifact null | {commit, path, sha256},
                        verified_09 null | {commit}, roots []}; release gate (09 @ d9013b2e7): "in" needs the committed pool-key ARTIFACT and
                        09's by-key verification sha before any epoch-2 file is opened; run_peraction311.py does not read epoch 2 (fail closed)
  nm360_fold_extension {gates {probe: "gate passed …" | "verified @ <sha>" | "cleared @ <sha>" | "held-out gate pending (c6023b805)"}, files null | [...]}
  gapfill_sim2         {status "pending routing" | "unreadable (09's note pending)" | "readable" | "second refresh", note null | {commit, path}, role, files []}
  sel97                {status (as gapfill), note, role, cells {commit, path, sha256} | null, cells_vendored null | {commit, path, sha256} (must equal cells), arms []}
                       ("readable" needs the note, gapfill files + root (role "never a lead"), sel97 cells + arms)
  rms                  stage-B inputs (Z fork's schema): five_arm_cells, arm_pin_sources [], H_B {S_new, S_newmains}, r36, S_old_rms_inputs,
                       s4_source_map (= sets.S_newmains.s4_source_map); H_sim_roots {S_new, S_newmains} (recomputed from the mappings;
                       S_newmains = S1-S3 + S5, 09 17:4x option (a)); scope {S_newmains: RMS_SCOPE_NM}; five_arm_cells.lists {o: {key, n}} (optional);
                       every entry {commit, path, sha256[, key[, n]]}, re-hashed; "key" must resolve to a non-empty value ("n" = its length)
  pools                {stage_a: Stage A's own pools per origin (built by stage_a_pools(); verified equal to a recomputation, and asserted equal to
                        the runner's resolution by run_peraction311.py), rms: pointer to the RMS consumer's rms.* entries}
  extensions           {S_paper493: "not in this build; second refresh", own_all_L48: {status "not in this build"[; second refresh] | "in",
                        pointer null | {score, score_addendum, census, census_addendum, fit, inputs, verify, verification_09: {commit, path, sha256}}, rules}}
                        ('in': own_all_problems re-hashes every pin and checks own_all coverage of every folder Stage A reads)
  (sets.*.sol_mappings[batch].label_override optional {commit, path, sha256}; ledger.notes / ledger.coverage_records [{commit, path, sha256}])

  python inputs_final.py verify <INPUTS_FINAL.json>     # exit 1 on ANY problem; prints every problem
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
E2_OFF = "epoch-2 not in this build (second refresh)"  # 0d's exact wording for tonight's records and captions
RMS_SCOPE_NM = "S_newmains here = S1–S3 + S5 (S4d / S4r not included)"
KEYS = ("schema", "status", "cutoff_pt", "freeze_pt", "label", "sets", "ledger", "epoch2_honest", "nm360_fold_extension", "gapfill_sim2", "sel97", "rms", "extensions",
        "pools")
FINAL_REC = ("3ce6ea55e", "experiments/2026-10-14_xerr_scaleup_09/results/xerr_scaleup_final_2026-10-15_013255.json")  # 09's FINAL (run_common455.FINAL)
POOL_REF = 'result.calibration_records.primary["sol_a|R|0.01|primary|BuC|B"].schemes[<primary>]'  # config/common455.json pool_reference_key
MAPDIR = "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/simh_mapping"
FMAPDIR = "experiments/2026-10-14_p2_family_expansion/results"
import copies as CP  # noqa: E402  (0d's /tmp fix, 10-17: the fpr507 manifest copies through one root and the committed index)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True, cwd=REPO).stdout


def file_maps(commit):
    """4f's pack file maps (base + every addendum at that commit): pack -> {manifest_famexp.json, manifest_own_a.json}."""
    names = subprocess.run(["git", "ls-tree", "--name-only", commit, FMAPDIR + "/"], capture_output=True, text=True, check=True, cwd=REPO).stdout.split()
    packs = {}
    for n in sorted(x for x in names if "simh_pack_file_map" in x):
        for p in json.loads(git_show(commit, n)).get("packs", []):
            packs[p["pack"]] = p["manifest_sha256"]
    return packs


def copy_path(root, kind):
    for d in ("v2man", "nmman"):
        p = f"{d}/{root}__{kind}.json"
        if CP.exists(p):  # by the committed index; CP.read / CP.check then require the file and its sha
            return p
    return None


def verify(m):
    """Every check runs; returns the list of problems (empty = pass). Fail closed: a missing key or file is a problem."""
    bad = []
    if set(m) != set(KEYS):
        bad.append(f"top-level keys: missing {sorted(set(KEYS) - set(m))}, unknown {sorted(set(m) - set(KEYS))}")
        return bad
    if m["schema"] != "inputs_final/v1":
        bad.append("schema is not inputs_final/v1")
    for st in ("S_new", "S_newmains"):
        s = m["sets"].get(st)
        if s is None:
            bad.append(f"sets.{st} missing")
            continue
        try:
            fm = file_maps(s["file_maps"]["commit"])
        except (subprocess.CalledProcessError, KeyError) as e:
            bad.append(f"{st}: file maps unreadable ({e})")
            fm = {}
        for bt, e in s["sol_mappings"].items():
            try:
                b = git_show(e["commit"], f"{MAPDIR}/{e['file']}.json")
            except subprocess.CalledProcessError:
                bad.append(f"{st} {bt}: mapping {e['file']} @ {e['commit']} not in git")
                continue
            if sha(b) != e["sha256"]:
                bad.append(f"{st} {bt}: mapping sha256 differs")
            roots = [r["arm"] for r in json.loads(b)["roots"]]
            if roots != s["roots"].get(bt):
                bad.append(f"{st} {bt}: roots differ from the mapping file")
            for r in roots:
                mc = s["manifest_copies"].get(r)
                if mc is None:
                    bad.append(f"{st} {r}: no manifest copies pinned")
                    continue
                fpend = (s.get("families_pending") or {}).get(r, [])
                if not set(fpend) <= {"famexp", "own_a", "own_all", "dt"}:
                    bad.append(f"{st} {r}: families_pending may only list famexp, own_a, own_all or dt (p2 and Sol are required)")
                for kind in ("p2", "famexp", "own_a"):
                    if kind in fpend:  # pending at cut-off for this scorer only (by key); the runner skips this root for that family
                        if mc.get(kind) is not None:
                            bad.append(f"{st} {r}: {kind} is pending yet a manifest copy is pinned")
                        continue
                    cp = copy_path(r, kind)
                    if cp is None:
                        bad.append(f"{st} {r}: local {kind} manifest copy missing")
                    elif CP.check(cp) or sha(CP.read(cp)) != mc.get(kind):
                        bad.append(f"{st} {r}: {kind} copy sha256 differs from the pin")
                    elif kind != "p2" and fm.get(r, {}).get(f"manifest_{kind}.json") != mc[kind]:
                        bad.append(f"{st} {r}: {kind} manifest not in 4f's file maps @ {s['file_maps']['commit']}")
                if cp is not None and kind == "own_a":
                    p2 = copy_path(r, "p2")
                    if p2 is not None and (CP.check(p2) or json.loads(CP.read(p2)).get("folder", "").split("/")[-1] != r):
                        bad.append(f"{st} {r}: p2 manifest folder is not the root")
        for e in s.get("honest_a_25f", []):
            try:
                if sha(git_show(e["commit"], e["record"])) != e["sha256"]:
                    bad.append(f"{st} 25f {e['cell_prefix']} {e['kind']}: record sha256 differs")
            except subprocess.CalledProcessError:
                bad.append(f"{st} 25f {e['cell_prefix']} {e['kind']}: record not in git")
            if e.get("copy_sha256"):
                cp = f"v2man/{e['copy']}"
                if CP.check(cp) or sha(CP.read(cp)) != e["copy_sha256"]:
                    bad.append(f"{st} 25f {e['cell_prefix']} {e['kind']}: manifest copy missing or differs")
        dt_roots = set()
        for e in s.get("dt_sim_honest", []):
            try:
                b_ = git_show(e["commit"], f"experiments/2026-10-14_p2_dt_scaleup/results/{e['record']}.json")
                if sha(b_) != e["sha256"]:
                    bad.append(f"{st} DT {e['record']}: sha256 differs")
                dt_roots |= set(json.loads(b_)["folders"])
            except subprocess.CalledProcessError:
                bad.append(f"{st} DT {e['record']}: not in git")
        mapped = {r for rs in s["roots"].values() for r in rs}  # DT H_sim must cover exactly the Sol mappings' roots (by root name)
        if dt_roots - mapped:
            bad.append(f"{st} DT: records for roots not in the pinned mappings {sorted(dt_roots - mapped)}")
        if mapped - dt_roots != set(s.get("dt_roots_pending", [])):  # a gap is allowed only when listed (it is printed in the records' DT labels)
            bad.append(f"{st} DT: mapped roots without a pinned DT record {sorted(mapped - dt_roots)} != dt_roots_pending {sorted(s.get('dt_roots_pending', []))}")
    mm = m["sets"]["S_newmains"].get("membership710", {})
    mp = HERE / mm.get("path", "?")
    if not mp.exists() or sha(mp.read_bytes()) != mm.get("sha256"):
        bad.append("S_newmains membership710 missing or sha256 differs")
    ec = (m["sets"].get("S_old") or {}).get("e2e_config") or {}
    try:
        if sha(git_show(ec["commit"], ec["path"])) != ec.get("sha256"):
            bad.append("S_old e2e_config sha256 differs")
    except (subprocess.CalledProcessError, KeyError, TypeError):
        bad.append("S_old e2e_config missing or not in git")
    so_dt = (m["sets"].get("S_old") or {}).get("dt_sim_honest")
    if not isinstance(so_dt, list) or len(so_dt) != 1 or not isinstance(so_dt[0], dict):
        bad.append("S_old dt_sim_honest: exactly one record is required (A25's primary_v3 DT)")
        so_dt = []
    for e in so_dt:
        try:
            if sha(git_show(e["commit"], f"experiments/2026-10-14_p2_dt_scaleup/results/{e['record']}.json")) != e["sha256"]:
                bad.append(f"S_old DT {e.get('record')}: sha256 differs")
        except (subprocess.CalledProcessError, KeyError, TypeError):
            bad.append(f"S_old DT {e.get('record') if isinstance(e, dict) else e}: not in git")
    try:  # the real-arm manifest copies membership710 names (the runner reads them; sha must equal membership's)
        for f, fi in json.loads(mp.read_bytes())["folders"].items():
            for k, h in fi["manifests"].items():
                cp = f"man710/{f}__{k}.json"
                if CP.check(cp) or sha(CP.read(cp)) != h:
                    bad.append(f"man710 {f}__{k}: copy missing or sha256 differs from membership710")
    except (OSError, KeyError, ValueError, AttributeError):
        bad.append("membership710 unreadable (real-arm manifest copies not checked)")
    try:  # per-consumer pools: the stage_a entry must equal a recomputation from this manifest's own pins (fail closed)
        if json.dumps(m.get("pools"), sort_keys=True) != json.dumps(stage_a_pools(m), sort_keys=True):
            bad.append("pools.stage_a differs from its recomputation from the manifest's pins")
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError, OSError) as e:
        bad.append(f"pools.stage_a cannot be recomputed ({e!r})")
    sm = m["sets"]["S_newmains"].get("s4_source_map") or {}
    try:
        smb = git_show(sm["commit"], sm["path"])
        if sha(smb) != sm.get("sha256"):
            raise KeyError("sha256")
        smap = json.loads(smb)
    except (subprocess.CalledProcessError, KeyError) as e:
        bad.append(f"S_newmains s4_source_map missing or sha256 differs ({e})")
        smap = None
    if smap is not None and mp.exists():
        bad += s4_orphan_problems(json.loads(mp.read_bytes()), smap)
        osha = {r["sha256"] for r in smap["outside_source_map"]}  # no orphan .eval sha256 in any input this manifest pins
        oroots = {r["root"] for r in smap["outside_source_map"]}
        for name, b_ in pinned_blobs(m):
            if name == sm["path"] or (name.startswith("man710/") and name[7:].split("__")[0] in oroots):
                continue  # the map itself; and the orphan roots' own folder manifests, whose rows are taken only by key ownership (checked above + orphan_guard)
            hits = sorted(h for h in osha if h.encode() in b_)
            if hits:
                bad.append(f"S4 map: orphan .eval sha256 {hits[0][:12]}… ({len(hits)}) in pinned input {name}")
    led = m["ledger"]
    keyset = lambda xs: {(x["set"], x["batch"], x["cell_prefix"]) for x in xs}  # noqa: E731
    ex, rep, pend = keyset(led["excluded"]), keyset(led["replacements_admitted"]), keyset(led["pending_at_cutoff"])
    if rep & pend:
        bad.append(f"ledger: admitted and pending at once {sorted(rep & pend)}")
    for sec in ("excluded", "pending_at_cutoff"):
        for x in led[sec]:
            if not isinstance(x.get("m_r"), int) or x.get("class") not in ("terminal", "standing", "RL", "infrastructure", "unknown"):
                bad.append(f"ledger {sec} {x.get('cell_prefix')}: m_r or class invalid")
    if ex & pend:
        bad.append(f"ledger: excluded and pending at once {sorted(ex & pend)}")
    for x in led.get("structural_no_rows", []):  # 09's class @ 51a612b29 (nm014): listed by key, verbatim text, and in the filled label
        if not (isinstance(x, dict) and x.get("key") and x.get("text")) or x["text"] not in str((m.get("label") or {}).get("filled")):
            bad.append(f"ledger structural no-row {x.get('key') if isinstance(x, dict) else x}: key/text missing or not in the filled label")
    for x in led.get("empty_replacement_roots", []):  # an empty replacement root is never a mapped folder; its cells stay excluded by key
        mapped = {r for rs in m["sets"][x["set"]]["roots"].values() for r in rs}
        if x["root"] in mapped or x.get("status") != "empty, excluded by key":
            bad.append(f"ledger empty root {x['root']}: in the mappings or status not 'empty, excluded by key'")
        for c0 in x["cells"]:
            if (x["set"], x["batch"], c0) not in ex:
                bad.append(f"ledger empty root {x['root']}: cell {c0} not excluded by key")
    for x in led["replacements_admitted"]:
        mapped = {r for rs in m["sets"][x["set"]]["roots"].values() for r in rs}
        if x["root"] not in mapped:
            bad.append(f"ledger replacement {x['cell_prefix']}: root {x['root']} not in the pinned mappings")
        if (x["set"], x["batch"], x["cell_prefix"]) not in ex:
            bad.append(f"ledger replacement {x['cell_prefix']}: original not listed as excluded")
    e2 = m["epoch2_honest"]
    if e2.get("status") == "in":  # release gate (09 @ d9013b2e7): note + committed pool-key artifact (re-hashed) + 09's by-key verification sha
        for what, pin_ in (("Append 38 §4 note", e2.get("note")), ("pool-key artifact", e2.get("artifact"))):
            try:
                b_ = git_show(pin_["commit"], pin_["path"])
                if what == "pool-key artifact" and sha(b_) != pin_.get("sha256"):
                    bad.append("epoch2_honest: pool-key artifact sha256 differs")
            except (subprocess.CalledProcessError, TypeError, KeyError):
                bad.append(f"epoch2_honest: status 'in' but the {what} is not at its pin")
        try:
            subprocess.run(["git", "cat-file", "-e", f"{(e2.get('verified_09') or {})['commit']}^{{commit}}"], check=True, cwd=REPO, capture_output=True)
        except (subprocess.CalledProcessError, KeyError):
            bad.append("epoch2_honest: status 'in' without 09's by-key verification commit (release gate)")
        bad.append("epoch2_honest: 'in' is not supported by run_peraction311.py (fail closed)")
    elif e2.get("status") != E2_OFF or e2.get("roots"):
        bad.append(f"epoch2_honest: status must be 'in' or {E2_OFF!r} (with no roots)")
    elif e2.get("artifact"):  # pointed at while OFF (2e 18:0x): re-hashed; nothing reads it until 'in' with 09's verification
        try:
            if sha(git_show(e2["artifact"]["commit"], e2["artifact"]["path"])) != e2["artifact"].get("sha256"):
                bad.append("epoch2_honest: pool-key artifact sha256 differs")
        except (subprocess.CalledProcessError, KeyError, TypeError):
            bad.append("epoch2_honest: the pool-key artifact is not at its pin")
    fx = m["nm360_fold_extension"]
    for k, v in fx["gates"].items():  # passed gates use the nm360 files this manifest already pins (fold -1-only pools, all-unit per-model moments)
        if not str(v).startswith(("gate passed", "verified", "cleared", "held-out gate pending", "pending")):  # the runner's vocabulary (FOLD_EXT_PASSED)
            bad.append(f"nm360_fold_extension: gate status for {k} must start with 'gate passed', 'verified', 'cleared' or 'held-out gate pending'")
    if fx["files"]:
        bad.append("nm360_fold_extension: extra fold-extension files are not read by run_peraction311.py (fail closed)")
    ext = m["extensions"]
    if ext.get("S_paper493") != "not in this build; second refresh":
        bad.append("extensions.S_paper493: only 'not in this build; second refresh' is accepted tonight")
    oa = ext.get("own_all_L48") or {}
    if oa.get("status") == "in":  # 09-verified own_all (READ_LOG @ d5ccf685d): every pointer pin re-hashed, coverage of every folder Stage A reads
        bad += own_all_problems(m, oa)
    elif oa.get("status") not in ("not in this build", "not in this build; second refresh"):
        bad.append("extensions.own_all_L48: status must be 'in', 'not in this build' or 'not in this build; second refresh'")
    for nm_ in ("gapfill_sim2", "sel97"):
        x = m[nm_]
        if not isinstance(x, dict) or x.get("status") not in ("pending routing", "unreadable (09's note pending)", "readable", "second refresh"):
            bad.append(f"{nm_}: status must be 'pending routing', 'unreadable (09's note pending)', 'readable' or 'second refresh'")
            continue
        if x["status"] == "readable":
            try:
                git_show(x["note"]["commit"], x["note"]["path"])
            except (subprocess.CalledProcessError, TypeError, KeyError):
                bad.append(f"{nm_}: 'readable' without 09's note at its pin (fail closed)")
            if nm_ == "gapfill_sim2" and not (x.get("files") and x.get("root")):
                bad.append("gapfill_sim2: 'readable' without its files and root (fail closed)")
            if nm_ == "sel97" and not (x.get("cells") and x.get("arms")):
                bad.append("sel97: 'readable' without its cells pin and arms (fail closed)")
    if not isinstance(m["gapfill_sim2"], dict) or "never a lead" not in str(m["gapfill_sim2"].get("role", "")):
        bad.append("gapfill_sim2: role must say 'never a lead'")
    s97 = m["sel97"] if isinstance(m["sel97"], dict) else {}
    if bool(s97.get("cells")) != bool(s97.get("cells_vendored")):
        bad.append("sel97: cells and cells_vendored must be pinned together")
    elif s97.get("cells_vendored") and (s97.get("cells") or {}).get("sha256") != s97["cells_vendored"].get("sha256"):
        bad.append("sel97: the vendored cells copy differs from the original pin by sha256")
    r = m["rms"] if isinstance(m["rms"], dict) else {}  # stage-B inputs must all be present before their hashes mean anything
    for k in ("five_arm_cells", "r36", "S_old_rms_inputs", "s4_source_map"):
        if not (isinstance(r.get(k), dict) and {"commit", "path", "sha256"} <= set(r[k])):
            bad.append(f"rms.{k}: missing or not a {{commit, path, sha256}} pin")
    if r.get("s4_source_map") != m["sets"]["S_newmains"].get("s4_source_map"):
        bad.append("rms.s4_source_map differs from sets.S_newmains.s4_source_map (one map for every consumer)")
    if not (isinstance(r.get("arm_pin_sources"), list) and r["arm_pin_sources"]
            and all(isinstance(e, dict) and {"commit", "path", "sha256"} <= set(e) for e in r["arm_pin_sources"])):
        bad.append("rms.arm_pin_sources: must be a non-empty list of {commit, path, sha256} pins")
    hb = r.get("H_B") if isinstance(r.get("H_B"), dict) else {}
    if set(hb) != {"S_new", "S_newmains"} or not all(isinstance(e, dict) and {"commit", "path", "sha256", "key", "n"} <= set(e) for e in hb.values()):
        bad.append("rms.H_B: needs S_new and S_newmains, each {commit, path, sha256, key, n}")
    fa = r.get("five_arm_cells") if isinstance(r.get("five_arm_cells"), dict) else {}
    try:
        fad = json.loads(git_show(fa["commit"], fa["path"])) if fa.get("lists") else None
    except (subprocess.CalledProcessError, KeyError):
        fad = None
    for o, e in (fa.get("lists") or {}).items():  # the Z fork's cell lists (by key, with their counts)
        e = e if isinstance(e, dict) else {}
        v = resolve(fad, e.get("key")) if fad is not None else None
        if not isinstance(v, list) or len(v) != e.get("n") or len(set(map(str, v))) != len(v):
            bad.append(f"rms.five_arm_cells.lists.{o}: key {e.get('key')} does not resolve to {e.get('n')} unique cells")
    hs = r.get("H_sim_roots") if isinstance(r.get("H_sim_roots"), dict) else {}  # RMS H_sim (09 17:4x, option (a)): S_newmains = S1-S3 + S5
    sr = {st: ((m["sets"].get(st) or {}).get("roots") or {}) for st in ("S_new", "S_newmains")}
    want = {"S_new": sorted({x for rs in sr["S_new"].values() for x in rs}),
            "S_newmains": sorted({x for bt, rs in sr["S_newmains"].items() if not bt.endswith(("_s4d", "_s4r")) for x in rs})}
    for o, w in want.items():
        if hs.get(o) != w:
            bad.append(f"rms.H_sim_roots.{o}: must equal the pinned mappings' roots ({len(w)}){' without S4d/S4r' if o == 'S_newmains' else ''}")
    if (r.get("scope") or {}).get("S_newmains") != RMS_SCOPE_NM:
        bad.append(f"rms.scope.S_newmains must be exactly {RMS_SCOPE_NM!r}")
    for e in pinned_entries(m):
        try:
            b = git_show(e["commit"], e["path"])
        except (subprocess.CalledProcessError, KeyError):
            bad.append(f"pin {e.get('path')} @ {e.get('commit')}: not in git")
            continue
        if sha(b) != e.get("sha256"):
            bad.append(f"pin {e['path']} @ {e['commit']}: sha256 differs")
            continue
        if e.get("key"):
            v = resolve(json.loads(b), e["key"])
            if v is None or (hasattr(v, "__len__") and not len(v)):
                bad.append(f"pin {e['path']}: key {e['key']} does not resolve to a non-empty value")
            elif "n" in e and len(v) != e["n"]:
                bad.append(f"pin {e['path']}: key {e['key']} has {len(v)} entries, expected {e['n']}")
    return bad


def s4_orphan_problems(mem, smap):
    """09 (17:3x): d9's S4 source map is authoritative for every arm and pool (honest_a, honest_b, run1, run2, sim; DT readers too, since they take
    rows by the same folder_keys). Problems: a key owned by two folders; an outside_source_map key owned by its orphan root's folder; a 4r cell's
    world owned by a non-s4r folder; a 4r cell's 'done' world with no row and no excluded_runs entry."""
    bad, owners = [], {}
    for f, ks in mem["folder_keys"].items():
        for k in ks:
            owners.setdefault(k, []).append(f)
    bad += [f"S4 map: {k} owned by two folders {fs}" for k, fs in sorted(owners.items()) if len(fs) > 1]
    for r in smap["outside_source_map"]:
        bad += [f"S4 map: orphan row {r['key']} @ {r['root']} ({r['staging']}) owned in the membership" for f in owners.get(r["key"], [])
                if f.split("__")[0] == r["root"]]
    for c in smap["cells"]:
        for w, a in c["arms"].items():
            k, fs = f"{w}|{c['cell']}", owners.get(f"{w}|{c['cell']}", [])
            bad += [f"S4 map: {k} owned by {f}, not an s4r folder" for f in fs if not f.startswith("nm360_s4r_")]
            if not fs and a.get("state") == "done" and k not in mem.get("excluded_runs", {}):
                bad.append(f"S4 map: {k} is done in the map but has no row and no exclusion")
    return bad


def stage_a_pools(m):
    """Stage A's pools per origin, as run_peraction311.py row() resolves them (keys from pinned records, roots from the pinned mappings)."""
    led = m["ledger"]
    excl = lambda st: sorted(f"{x['batch']}|{x['cell_prefix']}" for x in led["excluded"] + led["pending_at_cutoff"] if x["set"] == st)  # noqa: E731
    reps = lambda st: sorted({x["root"] for x in led["replacements_admitted"] if x["set"] == st})  # noqa: E731
    roots = lambda st, keep=lambda bt: True: sorted({r for bt, rs in m["sets"][st]["roots"].items() if keep(bt) for r in rs})  # noqa: E731
    fin = {"commit": FINAL_REC[0], "path": FINAL_REC[1], "sha256": sha(git_show(*FINAL_REC))}
    po = resolve(json.loads(git_show(*FINAL_REC)), POOL_REF + ".pool_keys_old")
    mm = m["sets"]["S_newmains"]["membership710"]
    mem = json.loads((HERE / mm["path"]).read_bytes())
    hb_nm = set(mem["pool_honest_b"]) | set(mem["pool_honest_b_4r_sol_only"])
    ha_rule = "honest_a on H_sim's cells (per probe, after exclusions and fold rules)"
    return {"stage_a": {
        "rule": ("run_peraction311.py row(): H_B by key; H_A = honest_a on H_sim's cells (S_old: honest_c runs of H_B); H_sim = the pinned mappings' roots, "
                 "excluded/pending cells removed by key, admitted replacements in; without S4 = S4 cells out of the attack cohort only (51a612b29)"),
        "S_old": {"H_B": fin | {"key": POOL_REF + ".pool_keys_old", "n": len(po)},
                  "H_A": {"rule": "honest_c runs of H_B", "n": sum(1 for k in po if k.startswith("honest_c|"))},
                  "H_sim": {"source": "A25 primary_v3 sim-honest via the S_old E2E config", "e2e_config": m["sets"]["S_old"]["e2e_config"]}},
        "S_new": {"H_B": fin | {"key": POOL_REF + ".pool_keys_new", "n": len(resolve(json.loads(git_show(*FINAL_REC)), POOL_REF + ".pool_keys_new"))},
                  "H_A": {"rule": ha_rule + "; the 25f honest-A roots join where their cell is in H_sim",
                          "honest_a_25f": sorted({e["folder"].strip("/").split("/")[1] if e["kind"] == "p2" else e["folder"] for e in m["sets"]["S_new"]["honest_a_25f"]})},
                  "H_sim_roots": roots("S_new"), "excluded_cells": excl("S_new"), "replacement_roots": reps("S_new")},
        "S_newmains": {"H_B": {"path": mm["path"], "sha256": mm["sha256"], "keys": ["pool_honest_b", "pool_honest_b_4r_sol_only"], "n": len(hb_nm),
                               "n_without_4r_lists": len(set(mem["pool_honest_b"])),
                               "rule": "the 4r list joins for a probe whose files cover every S4r folder (710's rule)"},
                       "H_A": {"rule": ha_rule}, "H_sim_roots": roots("S_newmains"), "excluded_cells": excl("S_newmains"), "replacement_roots": reps("S_newmains"),
                       "s4_source_map": m["sets"]["S_newmains"]["s4_source_map"], "without_S4": "attack cohort only; pools whole (51a612b29)"}},
        "rms": {"see": "rms.H_B, rms.H_sim_roots, rms.five_arm_cells, rms.scope (RMS consumer: S_newmains = S1-S3 + S5)"}}
OWN_ALL_PINS = ("score", "score_addendum", "census", "census_addendum", "fit", "inputs", "verify", "verification_09")
T196_WORLDS = ("honest_a", "honest_b", "honest_c", "run1", "run2", "sim")


def own_all_problems(m, oa):
    """own_all slot 'in': the pointer's pins (4f's score record + routing-57 addendum, census + addendum, fit, inputs, artifact check, 09's READ_LOG)
    re-hashed; every folder Stage A reads own_all from (S_new arms, 25f honest-A roots, membership710 folders, every mapped sim-honest root, A25's
    primary_v3, T196's six worlds) has an entry in the score records, and its local manifest copy equals that entry's manifest_sha256."""
    bad, shas = [], {}
    pt = oa.get("pointer") if isinstance(oa.get("pointer"), dict) else {}
    for k in OWN_ALL_PINS + tuple(sorted(k_ for k_ in pt if k_ not in OWN_ALL_PINS)):  # later addenda (score_addendum2, census_addendum2, ...) too
        e = pt.get(k)
        try:
            if sha(git_show(e["commit"], e["path"])) != e.get("sha256"):
                bad.append(f"extensions.own_all_L48.pointer.{k}: sha256 differs")
        except (subprocess.CalledProcessError, KeyError, TypeError):
            bad.append(f"extensions.own_all_L48.pointer.{k}: missing or not in git")
    for k in sorted(k_ for k_ in pt if k_.startswith("score")):  # the parent record and every routing addendum
        try:
            for v in json.loads(git_show(pt[k]["commit"], pt[k]["path"]))["results"].values():
                for f, fi in v["folders"].items():
                    if f in shas:
                        bad.append(f"own_all: folder {f} in two score records")
                    shas[f] = fi["manifest_sha256"]
        except (subprocess.CalledProcessError, KeyError, TypeError, ValueError):
            pass  # reported above
    need = {"primary_v3"} | {f"t196__{w}" for w in T196_WORLDS}
    need |= {r for st in ("S_new", "S_newmains") for rs in ((m["sets"].get(st) or {}).get("roots") or {}).values() for r in rs
             if "own_all" not in ((m["sets"].get(st) or {}).get("families_pending") or {}).get(r, [])}  # own_all pending at cut-off: listed, not read
    need |= {str(e.get("folder", "")).strip("/").split("/")[1] for e in (m["sets"].get("S_new") or {}).get("honest_a_25f") or [] if e.get("kind") == "p2"}
    sol455 = CP.names("man455", "*__sol.json")  # the 24 S_new real arm folders, by the committed index (a glob on disk was silent when missing)
    if len(sol455) != 24:
        bad.append(f"own_all coverage: man455 lists {len(sol455)} S_new arm sol manifests in the copies index, not 24")
    need |= {f.split("/", 1)[1].rsplit("__", 1)[0] for f in sol455}
    try:
        need |= set(json.loads((HERE / m["sets"]["S_newmains"]["membership710"]["path"]).read_bytes())["folders"])
    except (OSError, KeyError, TypeError, ValueError):
        bad.append("own_all: membership710 unreadable (coverage not checked)")
    for f in sorted(need):
        cp = f"ownall_man/{f}__own_all.json"
        if f not in shas:
            bad.append(f"own_all coverage: {f} has no entry in the score records (own_all pending at cut-off)")
        elif CP.check(cp) or sha(CP.read(cp)) != shas[f]:
            bad.append(f"own_all {f}: local manifest copy missing or sha256 differs from the score record")
    return bad


def pinned_entries(m):
    """Every {commit, path, sha256} entry outside the sets: rms, ledger notes and coverage records, label overrides, sel97 cells."""
    r = m.get("rms") or {}
    out = [r[k] for k in ("five_arm_cells", "r36", "S_old_rms_inputs", "s4_source_map") if isinstance(r.get(k), dict)]
    out += [e for e in (r.get("arm_pin_sources") or []) if isinstance(e, dict)] + [e for e in (r.get("H_B") or {}).values() if isinstance(e, dict)]
    out += list(m["ledger"].get("notes", [])) + list(m["ledger"].get("coverage_records", []))
    out += [e for x in m["ledger"].get("empty_replacement_roots", []) for e in x["evidence"]]
    oa = (m.get("extensions") or {}).get("own_all_L48") or {}
    if oa.get("status") == "in" and isinstance(oa.get("pointer"), dict):
        out += [e for e in oa["pointer"].values() if isinstance(e, dict) and "commit" in e]
    out += [e["label_override"] for st in ("S_new", "S_newmains") for e in m["sets"].get(st, {}).get("sol_mappings", {}).values() if e.get("label_override")]
    for k in ("cells", "cells_vendored", "prep_config"):
        if isinstance(m.get("sel97"), dict) and m["sel97"].get(k):
            out.append(m["sel97"][k])
    return out


def pinned_blobs(m):
    """(name, bytes) of every input the manifest pins or the runner reads through it: Sol mappings, sim-honest manifest copies, 25f and DT records,
    the S_old E2E config, membership710 and the real-arm manifest copies it names (man710), ledger notes, coverage, rms, sel97. An unreadable
    git input is skipped here (verify() reports it); a manifest copy is read through copies.read, so a missing or changed copy STOPs here
    (SystemExit), not skipped."""
    out = []

    def add(name, fn):
        try:
            out.append((name, fn()))
        except (subprocess.CalledProcessError, KeyError, TypeError, AttributeError, OSError):
            pass

    S = m.get("sets") or {}
    for st in ("S_new", "S_newmains"):
        s = S.get(st) or {}
        for e in (s.get("sol_mappings") or {}).values():
            add(e.get("file"), lambda e=e: git_show(e["commit"], f"{MAPDIR}/{e['file']}.json"))
        for r in s.get("manifest_copies") or {}:
            for k in ("p2", "famexp", "own_a"):
                add(f"{r}__{k}", lambda r=r, k=k: CP.read(copy_path(r, k) or f"v2man/{r}__{k}.json"))
        for e in s.get("honest_a_25f") or []:
            add(e.get("record"), lambda e=e: git_show(e["commit"], e["record"]))
    for st in ("S_new", "S_newmains", "S_old"):
        for e in (S.get(st) or {}).get("dt_sim_honest") or []:
            add(e.get("record"), lambda e=e: git_show(e["commit"], f"experiments/2026-10-14_p2_dt_scaleup/results/{e['record']}.json"))
    ec = (S.get("S_old") or {}).get("e2e_config") or {}
    add(ec.get("path"), lambda: git_show(ec["commit"], ec["path"]))
    mm = (S.get("S_newmains") or {}).get("membership710") or {}
    add("membership710", lambda: (HERE / mm["path"]).read_bytes())
    try:
        mem = json.loads((HERE / mm["path"]).read_bytes())
        for f, fi in mem["folders"].items():
            for k in fi["manifests"]:
                add(f"man710/{f}__{k}", lambda f=f, k=k: CP.read(f"man710/{f}__{k}.json"))
    except (KeyError, TypeError, OSError, ValueError):
        pass
    for e in pinned_entries(m):
        add(e.get("path"), lambda e=e: git_show(e["commit"], e["path"]))
    return out


def resolve(d, key):
    """'a.b["c"].schemes[<primary>].x': dotted keys, ["..."] literal keys; [<primary>] = the 'primary' field of the dict that holds the previous key
    (record["schemes"][record["primary"]]); <any> = the single key at that level (fails if there are several)."""
    import re

    if not isinstance(key, str) or not isinstance(d, dict):
        return None
    toks = re.findall(r'\["([^"]+)"\]|\[(<primary>)\]|([^.\[\]]+)', key)
    cur, parent = d, None
    for lit, prim, plain in toks:
        if not isinstance(cur, dict):
            return None
        k = lit or plain
        if prim:
            if not isinstance(parent, dict) or parent.get("primary") not in cur:
                return None
            k = parent["primary"]
        elif k == "<any>":
            if len(cur) != 1:
                return None
            k = next(iter(cur))
        parent, cur = cur, cur.get(k)
        if cur is None:
            return None
    return cur


def fill_label(m):
    """09's template filled from the ledger (no free text)."""
    led = m["ledger"]
    adm = "; ".join(f"{x['set']} {x['cell_prefix']} ({x['class']})" for x in led["replacements_admitted"]) or "none"
    exc = "; ".join(f"{x['set']} {x['cell_prefix']} ({x['class']}: {x['reason']})" for x in led["excluded"]) or "none"
    pen = "; ".join(f"{x['set']} {x['cell_prefix']} ({x['class']})" for x in led["pending_at_cutoff"]) or "none"
    e2 = m["epoch2_honest"]
    e2t = (f"Epoch-2 honest pools in per Append 38 §4 @ {e2['note']['commit']} (artifact verified by 09 @ {e2['verified_09']['commit']})" if e2["status"] == "in"
           else E2_OFF)
    gates = m["nm360_fold_extension"]["gates"]
    g = "; ".join(f"{k}: {v}" for k, v in gates.items())
    sn = " ".join(f"{x['text']}" for x in led.get("structural_no_rows", []))
    fpl = []  # a scorer that missed the cut-off for a root: its keys (and their H_A cells) are absent from that family only
    for st in ("S_new", "S_newmains"):
        for r, kinds in sorted(((m["sets"].get(st) or {}).get("families_pending") or {}).items()):
            cells = sorted(x["cell_prefix"] for x in led["replacements_admitted"] if x["set"] == st and x["root"] == r) or [f"every key of {r}"]
            fpl.append(f"{st} {', '.join(cells)} for {'/'.join(kinds)} (root {r}; the matching H_A cells leave those families' pools too)")
    pen = pen + ("; pending at cut-off for one scorer only, by key: " + "; ".join(fpl) if fpl else "")
    return (f"+311 Q1–Q4 and nm360 S1–S5 sim-honest FINAL as of {m['cutoff_pt']} PT. Replacements admitted by key: {adm}. Excluded cells by key: {exc}. "
            f"Pending at cut-off: {pen}. {e2t}. S4 is a labelled batch; every pooled headline is also shown without S4. nm360 Sol (a) / c_asis / own_frac rows: "
            f"held-out gate {g}." + (f" Structural no-row (by key): {sn}" if sn else "") + " PRELIMINARY; POST HOC where marked.")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "verify":
        raise SystemExit("usage: inputs_final.py verify <INPUTS_FINAL.json>")
    m_ = json.loads(Path(sys.argv[2]).read_text())
    probs = verify(m_)
    print(json.dumps({"file": sys.argv[2], "sha256": sha(Path(sys.argv[2]).read_bytes()), "problems": probs, "pass": not probs}, indent=1))
    sys.exit(1 if probs else 0)
