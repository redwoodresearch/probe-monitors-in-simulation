"""CUT fork (STEP-2, 10-17; 09 via 2e): keys-only records for the 20 registered pilot runs (launch 2026-10-14_140858_simh_pilot, world
simh_pilot) as their own S_old (b) root, in d9's S_old forms, so the scorers' pilot pins can be filled. Three phases; commit each output
before the next phase reads it:
1. provenance: the pilot's stager provenance in PROV's form (stager_provenance_b_sold_2026-10-15.json @ 9527f70d2): one numbered section
   whose ONE record entry names the registered 32k stage record "results/<name> @ <commit>" and lists "<arm file> <staged sha256>".
2. attest: attestations_sold__simh_q4f_v1_pilot_cutfull.json, make_sold_attestation.py's fields, revision 1, failing {}; repro_gate is the
   pilot's OWN per-root step-0 gate (09 10-17 ~01:15), staging_identity the HEAD vs c4cc344da staging identity record, the extraction
   attestation an attest_metas.py record of this root (all three are command-line pins). No prefix_check.py record exists for this root.
3. routing: sold__simh_q4f_v1_pilot_cutfull.json, written by d9's make_sold_routing.main itself (its TARGETS / TOTAL set to the pilot;
   its file untouched).
Committed inputs only (git show <commit>:<path>); names, keys, counts and hashes only. The record inputs are sha256-pinned here or on
the command line. The stager sources the provenance names (run_simh_stage_c.py, stage_simh.py) are hashed at the registered record's
commit and recorded, not pinned. The one uncommitted input is --cut-stage-sha, the sha256 of a Volume record, which the step-0 record
must name.
Phases 1 and 2 publish their output once, atomically (write_once): a temp file in the same folder, fsynced, then hard-linked to the
final name, which fails if that name exists, then the temp file is removed. A failed check writes nothing; a failed publish before the
link leaves neither the output nor a temp file (a hard kill can leave a hidden temp file, never the final name).
Phase 3 has NOT these guarantees. Its output is written by d9's make_sold_routing.main, whose pinned bytes this builder runs unchanged:
it writes a fixed-name temp .sold__<target>_cutfull.json.tmp without closing or fsyncing it, then hard-links it, so a failed write
(ENOSPC) can publish an empty or truncated record, and a leftover temp blocks every rerun. So phase 3 refuses to start while such a
temp exists, and after the router returns it parses the published record and STOPs, naming it "published but invalid", unless batch,
target, revision, the attestation's sha256 and the epochs sum are as expected. The root fix belongs in d9's router under a new pin.

  P=experiments/2026-10-14_q3_overnight_chain/cutfull/make_pilot_records.py
  uv run --no-project --python 3.12 python $P provenance --cut-stage-sha <sha256>
  uv run --no-project --python 3.12 python $P attest --provenance <commit>:<path> --step0 <path>@<commit>=<sha256> \
      --attest-metas <path>@<commit>=<sha256> --cut-stage-sha <sha256>
  uv run --no-project --python 3.12 python $P routing
(--cut-stage-sha: the sha256 of the registered 32k Volume stage record simh_q4f_v1_pilot/stage_2026-10-14_165444.json; it is not committed,
and the step-0 record's per-run checks.pilot_cut_stage_record must name it.)
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
REL = "experiments/2026-10-14_q3_overnight_chain"
SIMH = "experiments/2026-10-14_q3_q35_simhonest_purchase"
OUT_P, OUT_A, OUT_R = CHAIN / "results/cutfull", CHAIN / "results/cutfull/attest", CHAIN / "results/cutfull/routing"
# committed inputs (path, commit, sha256)
COUNTED = (f"{SIMH}/results/simh_counted_keys_pilot_v3.json", "344829af6", "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27")  # 09: the 20 keys
REG = (f"{SIMH}/results/simh_stage_c_pilot_2026-10-14_165444.json", "300652707", "1f455a785b0eef2facfdd59ca21352b8d43399f4b677b2cadc219d6287360db6")  # registered 32k staging
REG_X = (f"{SIMH}/results/extract_simh_simh_q4b_extract_v1_pilot_c_a1_2026-10-14_171107.json", "c389286cf",
         "9ba914b378369bd40f09db16fea82d99b717c7f9bed73a9f3c28c14c8981b76e")  # its registered extraction (config binds the staged sha)
STAGE = (f"{REL}/results/chain_stage_simh_pilot_cutfull_2026-10-17_010154.json", "1c075d834", "d832eb27e2ba61ab5b325114e6afd2244a6b0f2c1eadabd9221f4bfab728622c")
IDENT = (f"{REL}/results/cutfull/compare_staged_simh_pilot_2026-10-17_010504.json", "1c075d834", "045b1367d50eda95527a613c2efd5f9f4ac6c6e64b6a54f2b65b059f8f933c0e")
FMT = (f"{REL}/results/format_full_simh_pilot_cutfull_sim_honest_2026-10-17_010534.json", "1c075d834", "7dc94d94b84c1e4a20315a0a5b28b7019592fecd4ba2171e6a105ce51afd3f5f")
XREC = (f"{REL}/results/extract_chain_simh_pilot_cutfull_extract_v1_a1_2026-10-17_014605.json", "f91317f83", "a38707bb07fdf1d7d86beb8958b24c0e0ee10acdfb84e219a767f719bf7849c4")
ROUTER = (f"{REL}/cutfull/make_sold_routing.py", "8efb04c58", "0184ffaa1c0ab61e42dcb6a6206b95eae39616081c51b67d9cf1cddadbd59c40")  # d9's routing writer, run by phase 3
# names: 2e's rename (10-17, approved); the staged set and root keep their launch names
P = {"original_target": "simh_q4f_v1_pilot", "original_staged_set": "simh_q4f_v1_pilot", "original_volume": "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13",
     "registered_root": "simh_q4b_extract_v1_pilot_c", "volume_stage_record": "simh_q4f_v1_pilot/stage_2026-10-14_165444.json", "cutfull_target": "simh_pilot_cutfull",
     "staged_set": "simh_pilot_cutfull_v1", "root": "simh_pilot_cutfull_extract_v1", "arm": "sim_honest@september", "folder": "simh_q4f_v1_pilot_cutfull_v1__sim_honest",
     "original_folder": "simh_q4f_v1_pilot__sim_honest", "world": "simh_pilot", "n": 20}
VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
SECTION = "5_simh_q4f_v1_pilot"  # PROV's sections 1-4 are the six S_old sets; the pilot is "not traced" there (PROV item 4 not_traced)
NAMES = ("2e's rename (10-17, approved): target simh_q4f_v1_pilot, folder simh_q4f_v1_pilot_cutfull_v1__sim_honest, records sold__simh_q4f_v1_pilot_cutfull; "
         "world simh_pilot (09 via 2e, 10-17 00:4x); staged set simh_pilot_cutfull_v1 and root simh_pilot_cutfull_extract_v1 unchanged")
NAME = re.compile(r"[A-Za-z0-9_.@-]+")
HEX64 = re.compile(r"[0-9a-f]{64}")


def git(*a, text=True):
    return subprocess.run(["git", *a], capture_output=True, text=text, check=True, cwd=REPO).stdout


def full(c: str) -> str:
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{c}^{{commit}}"], capture_output=True, text=True, cwd=REPO, check=False)
    if r.returncode or not re.fullmatch(r"[0-9a-f]{9,40}", str(c)) or not r.stdout.strip().startswith(c):
        raise SystemExit(f"{c!r}: not one commit; nothing written")
    return r.stdout.strip()


def pinned(spec: tuple) -> tuple[bytes, dict]:
    b = git("show", f"{full(spec[1])}:{spec[0]}", text=False)
    if hashlib.sha256(b).hexdigest() != spec[2]:
        raise SystemExit(f"{spec[0]} @ {spec[1]}: sha256 differs from its pin; nothing written")
    return b, json.loads(b)


def cite(spec: tuple) -> str:
    return f"{spec[0]} @ {full(spec[1])[:12]}"


def cli_pin(s: str) -> tuple:
    """'<path>@<commit>=<sha256>' -> (path, commit, sha256)."""
    p, _, rest = s.rpartition("@")
    c, _, h = rest.partition("=")
    if not p or not re.fullmatch(r"[0-9a-f]{9,40}", c) or not HEX64.fullmatch(h):
        raise SystemExit(f"{s!r}: not <path>@<commit>=<sha256>")
    return p, c, h


def strings(o):
    return [o] if isinstance(o, str) else [z for v in (o.values() if isinstance(o, dict) else o if isinstance(o, list) else []) for z in strings(v)]


def token(arm):
    return arm.split("@")[0]


def write_once(dest: Path, doc: dict) -> None:
    """Publish once, atomically: temp file in dest's folder -> write, flush, fsync -> os.link to dest (fails if dest exists, so no
    overwrite) -> unlink the temp. On any exception up to and including the link, dest does not exist and the temp is removed. Not
    covered: a hard kill (SIGKILL, power loss) can leave a hidden .<name>.*.tmp, never the final name; a failure after the link (the
    temp's unlink, the folder fsync) leaves dest published."""
    if dest.exists() or not NAME.fullmatch(dest.name):
        raise SystemExit(f"{dest.name}: exists or unsafe name; nothing written")
    body = json.dumps(doc, indent=1).encode()  # serialize first: a failure leaves no file
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, dest)
    except FileExistsError:
        raise SystemExit(f"{dest.name}: exists; nothing written") from None
    except OSError as e:
        raise SystemExit(f"{dest.name}: publish failed ({e.__class__.__name__}: {e}); nothing written") from None
    finally:
        os.unlink(tmp)
    dfd = os.open(dest.parent, os.O_RDONLY)  # make the new name durable
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    print(dest.relative_to(REPO), "written; sha256", hashlib.sha256(dest.read_bytes()).hexdigest())


def counted() -> list:
    """09: the pilot's keys are the 20 counted keys; staged key = volume_path below simh_evals_v1/ + '#e0' (sorted)."""
    ck = pinned(COUNTED)[1]
    rows = ck.get("keys") or []
    keys = sorted(r["volume_path"].removeprefix("simh_evals_v1/") + "#e0" for r in rows if str(r.get("volume_path")).startswith(f"simh_evals_v1/{r.get('launch')}/"))
    if ck.get("n") != P["n"] or len(rows) != P["n"] or len(set(keys)) != len(rows) or {r.get("world") for r in rows} != {P["world"]}:
        raise SystemExit(f"counted keys: not {P['n']} unique well-formed {P['world']} rows; nothing written")
    return keys


def registered() -> tuple[dict, str, dict]:
    """The registered 32k staging record (the authority's record) and its extraction record: out, 20 epochs, one staged sha, c_gate."""
    r = pinned(REG)[1]
    s, xc, af = r.get("simh") or {}, pinned(REG_X)[1].get("config") or {}, f"{P['arm']}.jsonl"
    sha, cg = (s.get("assembled") or {}).get("staged_inputs_sha256"), s.get("c_gate") or {}
    if s.get("out") != P["original_staged_set"] or not isinstance(sha, dict) or set(sha) != {af} or not HEX64.fullmatch(str(sha[af])) \
            or s["assembled"].get("epochs") != {P["arm"]: P["n"]} or cg.get("manifest_keys") != P["n"] or cg.get("staged") != P["n"] or cg.get("excluded") != {} \
            or not isinstance(cg.get("render_changed"), list) or not re.fullmatch(r"[0-9a-f]{40}", str(r.get("git"))):
        raise SystemExit(f"{REG[0]}: not the registered {P['original_staged_set']} staging of {P['n']} epochs with one staged sha and a c_gate; nothing written")
    if xc.get("run") != P["registered_root"] or xc.get("inputs") != P["original_staged_set"] or xc.get("inputs_sha256") != sha or xc.get("test") is not False:
        raise SystemExit(f"{REG_X[0]}: the registered extraction does not read this staged sha; nothing written")
    return r, sha[af], cg


def provenance(a) -> None:
    keys, (r, reg_sha, cg) = counted(), registered()
    if not HEX64.fullmatch(a.cut_stage_sha) or Path(P["volume_stage_record"]).stem.removeprefix("stage_") not in Path(REG[0]).stem:
        raise SystemExit("--cut-stage-sha is not a sha256, or the Volume stage record and the registered record differ in stamp; nothing written")
    if list(OUT_P.glob("stager_provenance_simh_pilot_*.json")):
        raise SystemExit("a pilot provenance file exists; nothing written")
    g, ran = r["git"], {}
    for f in (f"{SIMH}/run_simh_stage_c.py", f"{SIMH}/stage_simh.py"):  # the stager at the record's run commit (it writes results/simh_stage_c_<group>_<stamp>.json)
        ran[f] = f"{f} @ {g[:9]}: sha256 {hashlib.sha256(git('show', f'{g}:{f}', text=False)).hexdigest()}"
    launch = sorted({k.split("/")[0] for k in keys})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles"))
    doc = {"label": "stager provenance for the simh_pilot S_old set (registered 32k staging simh_q4f_v1_pilot), by key from committed records only "
                    "(STEP-2, CUT fork; PROV's form, for the pilot that PROV item 4 did not trace)",
           "builder": {"script": f"{REL}/cutfull/make_pilot_records.py", "git": git("rev-parse", "HEAD").strip(), "stamp (PT)": stamp.strftime("%Y-%m-%d_%H%M%S")},
           SECTION: {"target": {"original_target": P["original_target"], "out": P["original_staged_set"], "arm": P["arm"], "launch": launch},
                     "volume": P["original_volume"],
                     "record": f"results/{Path(REG[0]).name} @ {REG[1]} (git {g[:9]}): out {P['original_staged_set']}, {cg['staged']} staged, {len(cg['excluded'])} excluded, "
                               f"{len(cg['render_changed'])} render-changed; {P['arm']}.jsonl {reg_sha}",
                     "record_file_sha256": REG[2],
                     "volume_stage_record": {"volume": P["original_volume"], "path": P["volume_stage_record"], "sha256": a.cut_stage_sha, "committed": False,
                                             "same_stamp_as_record": True, "bound_by": "the pilot step-0 record's checks.pilot_cut_stage_record (attest phase)"},
                     "extraction": f"{REG_X[0]} @ {REG_X[1]} (sha256 {REG_X[2]}): root {P['registered_root']}, config inputs {P['original_staged_set']}, "
                                   "inputs_sha256 equal to the record's",
                     "ran_from": {"record_git": g, "script": ran[f"{SIMH}/run_simh_stage_c.py"], "stager_module": ran[f"{SIMH}/stage_simh.py"]},
                     "counted_keys": f"{COUNTED[0]} @ {COUNTED[1]} (sha256 {COUNTED[2]}): {len(keys)} keys, world {P['world']}"}}
    write_once(OUT_P / f"stager_provenance_simh_pilot_{stamp.strftime('%Y-%m-%d')}.json", doc)


def attest(a) -> None:
    keys, (_, reg_sha, cg_reg) = counted(), registered()
    af, tok = f"{P['arm']}.jsonl", token(P["arm"])
    if list(OUT_A.glob(f"attestations_sold__{P['original_target']}_cutfull*.json")) or list(OUT_R.glob(f"sold__{P['original_target']}_cutfull*.json")):
        raise SystemExit("a pilot attestation or routing record exists (revision 1 is written once); nothing written")
    # 1. the authority: the committed provenance (an ancestor of HEAD, unchanged at HEAD) names the registered record and its staged sha
    pc, _, pp = a.provenance.partition(":")
    pf = full(pc)
    prov_raw = git("show", f"{pf}:{pp}", text=False)
    if subprocess.run(["git", "merge-base", "--is-ancestor", pf, "HEAD"], cwd=REPO, check=False).returncode or git("show", f"HEAD:{pp}", text=False) != prov_raw:
        raise SystemExit("the provenance is not committed on this branch as read; nothing written")
    sec = json.loads(prov_raw).get(SECTION) or {}
    entry = [z for z in strings(sec) if f"results/{Path(REG[0]).name} @ {REG[1]}" in z]  # make_sold_attestation's rule: ONE entry names the record
    if len(entry) != 1 or f"{af} {reg_sha}" not in entry[0] or (sec.get("volume_stage_record") or {}).get("sha256") != a.cut_stage_sha:
        raise SystemExit("the provenance does not name the registered record with its staged sha, or another Volume stage record; nothing written")
    # 2. the full-length staging (stage record), its identity at c4cc344da, its every-epoch format check, its H200 extraction
    sv = [v for k, v in pinned(STAGE)[1].items() if k != "git" and isinstance(v, dict) and v.get("out") == P["staged_set"]]
    sv = sv[0] if len(sv) == 1 else {}
    asm, cg = sv.get("assembled") or {}, sv.get("c_gate") or {}
    u_sha = (asm.get("staged_inputs_sha256") or {}).get(af)
    if set(asm.get("staged_inputs_sha256") or {}) != {af} or not HEX64.fullmatch(str(u_sha)) or asm.get("epochs") != {P["arm"]: P["n"]} \
            or (asm.get("format_gate") or {}).get(P["arm"], {}).get("passes") is not True or cg.get("manifest_keys") != P["n"] or cg.get("staged") != P["n"] \
            or cg.get("excluded") != {} or not isinstance(cg.get("render_changed"), list) or not set(cg["render_changed"]) <= set(keys):
        raise SystemExit(f"{STAGE[0]}: not {P['staged_set']} with {P['n']} staged epochs, one staged sha, a passing format gate and no exclusions; nothing written")
    idn = pinned(IDENT)[1]
    if idn.get("identical") is not True or idn.get("n_keys_identical") != P["n"] or sorted(idn.get("keys") or {}) != keys or sorted(idn.get("pinned_keys") or []) != keys \
            or idn.get("fail") != [] or idn.get("arm") != P["arm"] or (idn.get("roots") or {}).get("head") != P["staged_set"] or (idn.get("files") or {}).get("head") != u_sha \
            or len((idn.get("stage_records") or {}).get("head") or []) != 1:
        raise SystemExit(f"{IDENT[0]}: not identical on exactly the {P['n']} staged keys of this staged file; nothing written")
    fr = pinned(FMT)[1]
    if (fr.get("target"), fr.get("arm")) != (P["cutfull_target"], P["arm"]) or fr.get("passes") is not True or fr.get("failing_keys") != [] \
            or fr.get("epochs_checked") != P["n"] or fr.get("staged_sha") != [u_sha] or sorted(x.get("key") for x in fr.get("rows") or []) != keys:
        raise SystemExit(f"{FMT[0]}: the format check does not pass on exactly these keys of this staged file; nothing written")
    xr = pinned(XREC)[1]
    xc = xr.get("config") or {}
    if (xc.get("run"), xc.get("inputs"), xc.get("inputs_sha256"), xc.get("cut"), xc.get("test"), xr.get("volume")) != (P["root"], P["staged_set"], {af: u_sha}, "full", False, VOLUME) \
            or not xr.get("shards") or any("error" in s or "H200" not in str(s.get("gpu_name")) for s in xr["shards"]):
        raise SystemExit(f"{XREC[0]}: not this root / staged sha / cut full / Volume, or a shard errored or ran off H200; nothing written")
    # 3. the pilot's own step-0 gate: a full pass on exactly the staged keys, C = the registered staged file + Volume stage record, U = this staged file
    G = cli_pin(a.step0)
    gb, g = pinned(G)
    runs = g.get("runs") if isinstance(g.get("runs"), list) else []
    if g.get("passes") is not True or not runs or g.get("n_runs") != len(runs) or g.get("n_pass") != len(runs) or sorted(x.get("key") for x in runs) != keys \
            or any(x.get("passes") is not True or x.get("arm") != P["arm"] for x in runs) \
            or any((x.get("checks") or {}).get("binding") != {"cut": True, "uncut": True, "cut_file_sha": reg_sha, "uncut_file_sha": u_sha} for x in runs) \
            or any(((x.get("checks") or {}).get("pilot_cut_stage_record") or {}).get("sha256") != a.cut_stage_sha
                   or x["checks"]["pilot_cut_stage_record"].get("files") != [Path(P["volume_stage_record"]).name] for x in runs):
        raise SystemExit(f"{G[0]}: the step-0 gate is not a full pass on exactly the {P['n']} staged keys, bound to the registered and full-length staged "
                         "files and to the pinned Volume stage record; nothing written")
    # 4. the extraction attestation (attest_metas.py, every epoch): this root and staged set, failing {}, every key once, this staged sha, units
    M = cli_pin(a.attest_metas)
    xa = [x for x in pinned(M)[1].get("roots") or [] if x.get("root") == P["root"] and x.get("uncut_set") == P["staged_set"]]
    v = ((xa[0].get("arms") or {}).get(tok) or {}) if len(xa) == 1 else {}
    if len(xa) != 1 or xa[0].get("failing") != {} or set(xa[0].get("arms") or {}) != {tok} or v.get("meta_arm") != P["arm"] or v.get("epochs") != P["n"] \
            or xa[0].get("epochs") != P["n"] or sorted(v.get("keys") or []) != keys or len(set(v["keys"])) != len(v["keys"]) or v.get("staged_sha256") != u_sha \
            or not all(isinstance(v.get(z), int) and not isinstance(v.get(z), bool) for z in ("units", "units_past_32768")) \
            or xa[0].get("stage_record") != idn["stage_records"]["head"][0]:
        raise SystemExit(f"{M[0]}: no passing attestation row for {P['root']} / {P['staged_set']} on exactly these keys, this staged sha and stage record; nothing written")
    # 5. template class per epoch (d9's rule: the cutfull c_gate's render_changed and the registered record's changed keys)
    changed = set(cg["render_changed"]) | set(cg_reg["render_changed"])
    folder = f"{P['original_target']}_cutfull_v1__{tok}"  # make_sold_attestation's folder rule (addendum 2: no _try)
    if folder != P["folder"] or f"{P['original_staged_set']}__{tok}" != P["original_folder"]:
        raise SystemExit("the derived folder names are not the approved ones; nothing written")
    rec = {"what": "simh_pilot (b) full-length attestation (provenance binding; STEP-2, 09 via 2e, 10-17)", "label": "(b) descriptive", "revision": 1,
           "authority": {"provenance": f"{pf[:9]}:{pp}", "provenance_sha256": hashlib.sha256(prov_raw).hexdigest(),
                         "registered_record": f"{REG[1]}:{REG[0]}", "registered_staged_sha256": {af: reg_sha},
                         "registered_volume_stage_record": {"volume": P["original_volume"], "path": P["volume_stage_record"], "sha256": a.cut_stage_sha,
                                                            "bound_by": f"{G[0]} @ {full(G[1])[:12]} checks.pilot_cut_stage_record, every run"}},
           "original_target": P["original_target"], "original_staged_set": P["original_staged_set"], "original_volume": P["original_volume"],
           "original_folders": {tok: P["original_folder"]},
           "cutfull": {"volume": VOLUME, "staged_set": P["staged_set"], "extraction_root": P["root"], "folders": {tok: folder}, "stage_record": cite(STAGE)},
           "names": NAMES, "dropped_arms": [], "dropped_note": None,
           "repro_gate": f"{G[0]} @ {full(G[1])[:12]} (the pilot's own per-root step-0 gate, 09 10-17 ~01:15; {len(runs)} staged keys; sha256 {G[2][:16]}...)",
           "staging_identity": cite(IDENT),
           "prefix_check": [], "prefix_check_note": "no prefix_check.py record for this root; its step-0 gate (repro_gate) compares every one of the "
                                                    f"{P['n']} staged keys against the registered 32k staging (binding shas asserted by this builder)",
           "extraction_attestation": [f"{M[0]} @ {full(M[1])[:12]}"], "extraction_record": cite(XREC),
           "exclusions_rule": "registered c_gate excluded keys",
           "membership": {tok: {"rule": {"kind": "keys", "ref": "simh_counted_keys_pilot_v3"}, "source": f"{COUNTED[0]} @ {COUNTED[1]} (sha256 {COUNTED[2]}); "
                                "keys <launch>/<rel>#e0 = volume_path below simh_evals_v1/", "membership_size": len(keys), "staged": len(v["keys"]),
                                "staged_not_in_membership": sorted(set(v["keys"]) - set(keys)), "membership_not_staged_in_this_set": sorted(set(keys) - set(v["keys"]))}},
           "template_class": {tok: {k: ("render changed with the cut" if k in changed else "fixed, render unchanged") for k in sorted(v["keys"])}},
           "membership_file": f"{COUNTED[0]} @ {COUNTED[1]} (sha256 {COUNTED[2]})",
           "units": {tok: {z: v.get(z) for z in ("epochs", "units", "units_past_32768")}},
           "format_check": {tok: cite(FMT)}, "failing": {}}
    write_once(OUT_A / f"attestations_sold__{P['original_target']}_cutfull.json", rec)


def routing(a) -> None:
    """d9's make_sold_routing.main on the pilot alone: it reads the committed attestation (highest revision, failing {}, format check), its
    extraction attestation and format record at their commits, and writes sold__<target>_cutfull.json once (one active routing file per root)."""
    pin_b, tree_b = git("show", f"{full(ROUTER[1])}:{ROUTER[0]}", text=False), (REPO / ROUTER[0]).read_bytes()  # 507-prep R1: run only the pinned bytes
    if hashlib.sha256(pin_b).hexdigest() != ROUTER[2] or hashlib.sha256(tree_b).hexdigest() != ROUTER[2]:
        raise SystemExit(f"{ROUTER[0]}: the bytes at {ROUTER[1]} or in the working tree differ from the pinned sha256; nothing written")
    m = types.ModuleType("make_sold_routing")  # 507-prep nit: execute exactly the hashed bytes, never a re-read of the file
    m.__file__ = str(REPO / ROUTER[0])
    exec(compile(pin_b, m.__file__, "exec"), m.__dict__)
    if m.REPO != REPO or m.BATCH != "sold" or m.VOLUME != VOLUME:
        raise SystemExit("make_sold_routing.py is not the sold builder of this repo; nothing written")
    stale = sorted(q.name for q in OUT_R.glob(f".sold__{P['original_target']}_cutfull*.tmp"))
    if stale:  # the router's fixed-name temp from a failed earlier publish would make its open("x") fail on every rerun
        raise SystemExit(f"leftover router temp file(s) {stale} in {OUT_R.relative_to(REPO)} from a failed earlier publish: move aside "
                         "(results are append-only); nothing written")
    m.TARGETS, m.TOTAL = [P["original_target"]], P["n"]
    m.main()
    d = OUT_R / f"sold__{P['original_target']}_cutfull.json"
    raw = d.read_bytes()  # the router neither closes nor fsyncs its temp before linking: check what was published (keys and ints only)
    ap_ = OUT_A / f"attestations_sold__{P['original_target']}_cutfull.json"
    a_raw = git("show", f"HEAD:{ap_.relative_to(REPO)}", text=False)  # the committed attestation the router read
    try:
        r = json.loads(raw)
    except ValueError:
        r = None
    arms = r.get("arms") if isinstance(r, dict) else None
    if not (isinstance(r, dict) and r.get("batch") == "sold" and r.get("target") == f"{P['original_target']}_cutfull"
            and type(r.get("revision")) is int and r["revision"] == 1
            and isinstance(r.get("attestation"), dict) and r["attestation"].get("sha256") == hashlib.sha256(a_raw).hexdigest()
            and isinstance(arms, list) and arms and all(isinstance(x, dict) and type(x.get("epochs")) is int for x in arms)
            and sum(x["epochs"] for x in arms) == P["n"]):
        raise SystemExit(f"{d.relative_to(REPO)}: published but invalid (move aside; results are append-only); STOP")
    print(d.relative_to(REPO), "published and valid; sha256", hashlib.sha256(raw).hexdigest())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    p1 = sub.add_parser("provenance")
    p1.add_argument("--cut-stage-sha", required=True)
    p2 = sub.add_parser("attest")
    for x in ("--provenance", "--step0", "--attest-metas", "--cut-stage-sha"):
        p2.add_argument(x, required=True)
    sub.add_parser("routing")
    a = ap.parse_args(argv)
    {"provenance": provenance, "attest": attest, "routing": routing}[a.phase](a)


if __name__ == "__main__":
    main()
