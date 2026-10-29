"""Adapter (metadata only; no activation or score is read) for the full-length (cutfull) roots of the (a'') scope: build ONE message for
ONE of d9's cutfull routing records (cutfull_routing_v2.json; all arms of one original target) and an adapter record that
score_step3_cutfull.py accepts. Gates (all must pass):
- the record: on aswin/q3-cut-option, the highest revision _r<k> of its (batch, original target) at the branch head, written under spec v2;
- its attestation: the pointer for that revision and the payload it names, both with failing {}, every epoch template class "fixed";
- the step-0 gate: the ONE registered compare (09 item 1) with passes true;
- the stage record: staged set, per-arm staged sha and epochs, every arm's format gate passes, c_gate counts and exclusions;
- every extraction attempt: root, staged set and shas, Qwen/Qwen3.5-27B, cut "full", the cutfull Volume, NVIDIA H200 on every shard;
- the manifest (copied from the original): one counted file per (arm, cell); staged = counted minus the record's exclusions;
- original_folders: top-level, one per arm token (meta_arm without "@<suffix>");
- the output batch is the record's "batch" (d9's chain batch, copied from the original routing; 09's layout ruling, option (c));
- inventory v3: every staged epoch is an inventory run with this cutfull folder, its original folder and meta arm; the inventory's runs
  for these folders are exactly the staged epochs (by key).

  python3 -I build_cutfull_message.py --record <commit>:<path of the routing record> --out-dir <dir>
"""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
BRANCH = "origin/aswin/q3-cut-option"
SPEC_V2 = "experiments/2026-10-14_q3_overnight_chain/cutfull/cutfull_routing_v2.json @ c75141175"
STEP0 = ("e6896f33b", "experiments/2026-10-14_q3_overnight_chain/results/cutfull_step0_compare_2026-10-15_214206.json")  # 09 item 1: the ONE compare
VOLUME = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
INVENTORY = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json", "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3")
# (b) scope (RULE F.10): inventory_b_v2 is pinned when 4f relays it; until then every (b) message is refused (fail closed)
INVENTORY_B = ("98575cc3471a224c87401111fbb8caee28ff5323", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v2_2026-10-17_081456.json", "f8dfc018d44c5a5bd373f88d9f7974dd9774c25a325d3ceceb385fcb9c82719f")  # STEP-2 pilot copy (1): the pilot inventory (commit, path, sha256) = fix_v2 (09 via 2e, 10-17); unset STOPs (main)
INVENTORY_B_PREV = ("a0bf963a7", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_v2_2026-10-16_014820.json",
                    "fc0e84ea5260193f7e5f249f022b2f4b04de544794e06a7137d777c222d5bbf8")
INVENTORY_B_FIX = {"folder": "nm360_s2_run2_s1_cutfull_v1__attack", "rows": 77, "fields": {"uncut_planned"}}  # the only change allowed vs v2
PILOT_STEP0 = ("e40411263", "experiments/2026-10-14_q3_overnight_chain/results/cutfull_step0_compare_2026-10-17_020344.json",
               "b9a44680b500cf17306df9db7e0525588bd951b393a9105ae4806babc33c9ed5")  # STEP-2 pilot copy (5): the pilot's own step-0 gate, exactly this record
COUNTED = ("344829af6", "experiments/2026-10-14_q3_q35_simhonest_purchase/results/simh_counted_keys_pilot_v3.json",
           "2149c5cde368aef9e5012c8d9cda5b12fe5a93db3d27ad2d525225bb9f511b27")  # (5) the counted pilot keys; staged key = volume_path below simh_evals_v1/ + "#e0"
INVENTORY_B_RECORD = ("98575cc3471a224c87401111fbb8caee28ff5323", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_pilot_rows_v1_2026-10-17_081456.json", "854ff211aaa61a3b24ca17b3104c910a377ea11808e06680f4b99e4ae1f9bf45")  # STEP-2 pilot copy (6): the pilot-rows file (commit, path, sha256) (09 via 2e, 10-17); unset STOPs
INVENTORY_B_CHECK = ("26c12ab14b5d33b57e383d9774c39ae865f7b25b", "experiments/2026-10-17_step2_pilot_sold_equal/results/inventory_b_fix_v2_subset_check_2026-10-17_115418.json", "dbbd63f66827321f56407ade32b1c2e3304dd30c13884dada3853b0a9f727583")  # (6) the committed keys-only fix_v1 ⊂ fix_v2 check record (check_fix_v2_subset.py, direct pins, run 1 of the run NOTE; commit, path, sha256); unset STOPs
# PIN-FILL NOTE (2e, 10-17): a check_fix_v2_subset record made BEFORE e5320a426 must be REGENERATED before it is pinned here. It does
# carry n_added and counted_keys (written since 86731ad54; check_fix_v2_subset.py L129-130 today), but its count is named
# rows_byte_identical, renamed rows_reserialized_identical at e5320a426 (check_fix_v2_subset.py L74, L87), and the binding below takes
# JSON ints only for rows_canonical_only, rows_reserialized_identical, n_fix_v1 and n_added (L332-336), so such a record STOPs there.
PILOT_ORIGINAL = "simh_q4f_v1_pilot__sim_honest"  # (6) fix_v2's own fix_v2.original_folder
PILOT_TARGET = "simh_q4f_v1_pilot"  # (5) with PILOT_ORIGINAL: the attestation's pilot IDENTITY chooses the gate (Astra R1), never the gate it cites
INVENTORY_B_V1 = ("09b070617", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v1_2026-10-16_025058.json",
                  "4a5d0fb3cad32740f849df5def3c7a03a40a18aa4c42bc360709f0c740973d03")  # (6) fix_v1: every row must be in the pilot inventory, identical
PILOT_FOLDER = "simh_q4f_v1_pilot_cutfull_v1__sim_honest"  # STEP-2 pilot copy (2): the planned folder by exact name (2e's rename), not from the staged set
SOLD_TEMPLATE_CLASSES = {"fixed", "fixed, render unchanged", "render changed with the cut"}  # S_old attestation vocabulary; "original" refused
S4_ORPHANS = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")  # 09: extra S4 keys OUT iff listed
# 09's S_old grammar (ACKed, shared by p2/famexp/step3) and the per-folder world map (never the arm token)
SOLD_RE = re.compile(r"^(q4f_(main|run2|c)|rh1e4_stage)_v1__(attack|honest_a|honest_b|sim_attack)$")
SIMH_Q4F_RE = re.compile(r"^simh_q4f_v1_full_b[0-9]+__sim_honest$")
SOLD_WORLD = {"q4f_main_v1__attack": "run1", "q4f_main_v1__honest_a": "honest_a", "q4f_main_v1__honest_b": "honest_b", "q4f_main_v1__sim_attack": "sim",
              "q4f_run2_v1__attack": "run2", "q4f_c_v1__honest_b": "honest_c", "rh1e4_stage_v1__honest_b": "relaunch"}
RELAUNCH_LABEL = "(b) descriptive; relaunch honest pool; listed, no registered membership"


def sold_world(original_folder):
    """World for an S_old original folder from the per-folder map; A25 sim-honest -> sim_honest; anything else None (refused)."""
    if SIMH_Q4F_RE.fullmatch(original_folder):
        return "sim_honest"
    if original_folder == "simh_q4f_v1_pilot__sim_honest":  # STEP-2 pilot copy (4), 09 via 2e: this ONE exact name, by equality (the regex unchanged)
        return "sim_honest"
    return SOLD_WORLD.get(original_folder) if SOLD_RE.fullmatch(original_folder) else None
truthy = lambda v: v is True or v == "true"  # noqa: E731  (d9's records write JSON true; some older records the string)
REC_RE = re.compile(r"^(.+__.+_cutfull)(?:_r([0-9]+))?\.json$")


def no_dup(pairs):
    """json object_pairs_hook: a JSON object with a repeated member name is refused (json.loads would keep the last one silently)."""
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate member {k!r}")
        d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="<commit>:<path> of d9's cutfull routing record (or, with --sold, the S_old attestation)")
    ap.add_argument("--sold", action="store_true", help="(b) S_old: the record is d9's S_old attestation (no chain routing record exists for S_old)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scope", choices=("a2", "b"), default="a2", help="a2 = (a'') with inventory v3; b = the (b) roots with inventory_b_v2")
    a = ap.parse_args()
    if a.scope == "b" and INVENTORY_B is None:
        raise SystemExit("(b): inventory_b_v2 is not pinned yet (4f relays it)")
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, check=True).stdout  # noqa: E731
    on_origin = lambda c: any(b.strip().startswith("origin/") for b in git("branch", "-r", "--contains", c).decode().splitlines())  # noqa: E731
    pins = {}

    def show(name, c, p, sha=None, strict=False):
        try:  # STEP-2 pilot copy: a missing commit or path of a pinned read is a reasoned STOP (other git calls can still raise)
            if not on_origin(c):
                raise SystemExit(f"{name}: commit {c} is not on origin")
            b = git("show", f"{c}:{p}")
        except subprocess.CalledProcessError:
            raise SystemExit(f"{name}: {p} @ {c} is not in git") from None
        if sha and hashlib.sha256(b).hexdigest() != sha:
            raise SystemExit(f"{name}: sha256 differs from its pin")
        pins[name] = {"commit": git("rev-parse", c).decode().strip(), "path": p, "sha256": hashlib.sha256(b).hexdigest()}
        if strict:  # the inventory rows (fix_v1, fix_v2, the pilot rows): a repeated member name STOPs
            try:
                return json.loads(b, object_pairs_hook=no_dup)
            except ValueError as e:
                raise SystemExit(f"STOP (6): {name}: {e}") from None
        return json.loads(b)

    def at(ref):  # "path @ commit" as d9 writes it
        p, c = (x.strip() for x in ref.split(" @ "))
        return c, p

    rc_, rp_ = a.record.split(":", 1)
    if a.sold:
        if a.scope != "b":
            raise SystemExit("--sold is a (b) scope")
        return build_sold(a, rc_, rp_, git, show, at, pins)
    rec = show("routing_record", rc_, rp_)
    gate = {}
    m_ = REC_RE.match(Path(rp_).name)
    if not m_:
        raise SystemExit(f"not a cutfull routing record name: {rp_}")
    stem, k = m_.group(1), int(m_.group(2) or 1)
    revs = [REC_RE.match(Path(x).name) for x in git("ls-tree", "--name-only", f"{BRANCH}:{Path(rp_).parent}").decode().split()]
    ks = sorted(int(x.group(2) or 1) for x in revs if x and x.group(1) == stem)
    gate["highest_revision"] = {"k": k, "revisions": ks, "pass": ks[-1] == k == rec["revision"] and rec.get("routing_spec", "").startswith(SPEC_V2)
                                and rec.get("cut") == "full" and rec["volume"] == VOLUME}
    ap_path = f"{Path(rp_).parent}/{Path(rp_).stem}_attest.json"
    ptr = show("attest_pointer", git("rev-parse", BRANCH).decode().strip(), ap_path)
    att = show("attestation", ptr["commit"], ptr["path"])
    arms = {r["meta_arm"]: r for r in rec["arms"]}
    tok = {m: m.split("@")[0] for m in arms}
    tc = {x for arm in (att.get("template_class") or {}).values() for x in arm.values()}
    gate["attestation"] = {"template_classes": sorted(tc), "pass": ptr["failing"] == {} and att["failing"] == {} and ptr["revision"] == att["revision"] == k
                           and ptr["routing_record"].split(" @ ")[0] == rp_ == att["routing_record"] and tc == {"fixed"}
                           and {t: v["epochs"] for t, v in att["units"].items()} == {tok[m]: r["epochs"] for m, r in arms.items()}
                           and {t: len(v) for t, v in att["template_class"].items()} == {tok[m]: r["epochs"] for m, r in arms.items()}
                           and hashlib.sha256(git("show", f"{ptr['routing_record'].split(' @ ')[1].strip()}:{rp_}")).hexdigest() == pins["routing_record"]["sha256"]}
    s0 = show("step0_gate", *STEP0)
    gate["step0_gate"] = {"pass": at(rec["repro_gate"])[1] == STEP0[1] and git("rev-parse", at(rec["repro_gate"])[0]) == git("rev-parse", STEP0[0])
                          and truthy(s0["passes"])}
    st_ = show("stage_record", *at(rec["stage_record"]))
    sets = [x for x in st_ if x != "git"]
    if len(sets) != 1:
        raise SystemExit(f"stage record: expected one staged set, found {sets}")
    st = st_[sets[0]]
    excluded = st["c_gate"]["excluded"]
    gate["staging"] = {"pass": st["out"] == rec["staged_set"] and st["assembled"]["epochs"] == {m: r["epochs"] for m, r in arms.items()}
                       and st["assembled"]["staged_inputs_sha256"] == {f"{m}.jsonl": r["staged_sha"] for m, r in arms.items()}
                       and set(excluded) == set(rec["staging_exclusions"]) and st["c_gate"]["manifest_keys"] == st["c_gate"]["staged"] + len(excluded)}
    # format: d9's every-epoch format check per arm when the record cites one (label "passed"); else the stage record's format_gate only
    fmt = {}
    for m, r in arms.items():
        if r.get("format_check"):
            fr = show(f"format_{tok[m]}", *at(r["format_check"]))
            fmt[m] = truthy(fr["passes"]) and not fr["failing_keys"] and r["staged_sha"] in fr["staged_sha"] and fr["arm"] == m and fr["epochs_checked"] == r["epochs"]
    if fmt and len(fmt) != len(arms):
        raise SystemExit("format_check cited for some arms only")
    fmt_commits = sorted({at(r["format_check"])[0][:9] for r in arms.values() if r.get("format_check")})
    format_label = f"format_check passed (d9 @ {', '.join(fmt_commits)})" if fmt else "format_check not run (stage format_gate only)"
    gate["format"] = {"label": format_label, "pass": all(fmt.values()) if fmt else all(truthy(st["assembled"]["format_gate"][m]["passes"]) for m in arms)}
    ex_ok = []
    for i, x in enumerate(rec["extraction_record"]):
        dv = show(f"extraction_a{i + 1}", x["commit"], x["path"])
        ex_ok.append(dv["config"]["run"] == rec["extraction_root"] and dv["config"]["inputs"] == rec["staged_set"] and dv["volume"] == VOLUME
                     and dv["config"]["inputs_sha256"] == {f"{m}.jsonl": r["staged_sha"] for m, r in arms.items()} and dv["config"]["model"] == "Qwen/Qwen3.5-27B"
                     and dv["config"].get("cut") == "full" and len(dv["shards"]) == dv["config"]["shards"] and all(s.get("gpu_name") == "NVIDIA H200" for s in dv["shards"]))
    gate["extraction_h200"] = {"attempts": len(ex_ok), "pass": bool(ex_ok) and all(ex_ok)}
    man = show("manifest", *at(rec["manifest"]))
    files = [f for f in man["files"] if f["arm_dir"] in arms]
    keys = {f"{f['launch']}/{f['rel']}#e0": f for f in files}
    per_arm = {m: [f for f in files if f["arm_dir"] == m] for m in arms}
    staged = {kk: f for kk, f in keys.items() if kk not in excluded}
    gate["manifest"] = {"files": {m: len(v) for m, v in per_arm.items()}, "pass": len(keys) == len(files) == st["c_gate"]["manifest_keys"]
                        and all(len({f["task"] for f in v}) == len(v) for v in per_arm.values()) and set(excluded) <= set(keys)
                        and all(sum(1 for f in per_arm[m] if f"{f['launch']}/{f['rel']}#e0" not in excluded) == arms[m]["epochs"] for m in arms)}
    of = rec.get("original_folders")
    folder = {m: f"{re.sub(r'_try[0-9]+$', '', rec['staged_set'])}__{t}" for m, t in tok.items()}  # naming addendum 2: routed folders carry no _try
    gate["original_folders"] = {"pass": isinstance(of, dict) and set(of) == set(tok.values()) and len(set(tok.values())) == len(tok)
                                and all("@" not in v and v.endswith(f"__{t}") for t, v in of.items()) and all(len(r["world"]) == 1 for r in arms.values())}
    inv = {r["key"]: r for r in show("inventory", *(INVENTORY if a.scope == "a2" else INVENTORY_B)[:2], (INVENTORY if a.scope == "a2" else INVENTORY_B)[2])["runs"]}
    if a.scope == "b":
        fix_check(inv, {r["key"]: r for r in show("inventory_prev", *INVENTORY_B_PREV)["runs"]})
    for r in inv.values():  # (b) carries full_len for tokens_before (equal, 4f); both present must agree
        if "tokens_before" in r and "full_len" in r and r["tokens_before"] != r["full_len"]:
            raise SystemExit(f"{r['key']}: tokens_before != full_len in the inventory")
    tb = lambda kk: inv[kk].get("tokens_before", inv[kk].get("full_len"))  # noqa: E731
    up = lambda kk: inv[kk]["uncut_planned"] if isinstance(inv[kk]["uncut_planned"], dict) else {}  # noqa: E731  (a pending name is no dict: refused)
    out_orphan, staged_all = {}, dict(staged)  # d9 attests every staged epoch, orphans included
    if a.scope == "b":  # 09 (b) rulings 2 (1): staged keys the (b) inventory does not list stay OUT iff on the S4 orphan list, else STOP
        orph = {f"{x['record']}#e0" for x in show("s4_orphans", *S4_ORPHANS)["outside_source_map"]}
        extra = {kk for kk in staged if kk not in inv}
        if not extra <= orph:
            raise SystemExit(f"(b): staged keys outside the inventory and outside the S4 orphan list: {sorted(extra - orph)[:3]}")
        out_orphan = {kk: "S4 orphan (outside the S4 source map @ b68ab0a53): OUT by key (09)" for kk in sorted(extra)}
        staged = {kk: f for kk, f in staged.items() if kk not in extra}
    bad = [kk for kk, f in staged.items() if kk not in inv or up(kk).get("folder") != folder[f["arm_dir"]] or inv[kk]["original_folder"] != (of or {}).get(tok[f["arm_dir"]])
           or inv[kk]["arm"] != f["arm_dir"] or inv[kk].get("original_folder_registered") is not True or not isinstance(tb(kk), int)]
    sold = {m: sold_world(of[t]) if isinstance(of, dict) and t in of and rec["stratum"] == "S_old" else None for m, t in tok.items()}
    if rec["stratum"] == "S_old" and (None in sold.values() or any(sold[m] != inv[kk]["world"] for kk, f in staged.items() for m in [f["arm_dir"]] if kk in inv)):
        raise SystemExit(f"S_old: an original folder outside 09's grammar, or a world differing from the per-folder map: {sold}")
    att_keys = {t: set(v) for t, v in att["template_class"].items()}  # the attested epochs are exactly the staged ones, by key (Codex cutfull r1 item 1)
    gate["attestation"]["pass"] &= att_keys == {tok[m]: {kk for kk, f in staged_all.items() if f["arm_dir"] == m} for m in arms}
    gate["inventory"] = {"scope": a.scope, "bad": bad[:5], "out_by_orphan_rule": len(out_orphan),
                         "pass": not bad and {kk for kk in inv if up(kk).get("folder") in folder.values()} == set(staged)}
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit(f"adapter gates failed: {json.dumps({kk: g for kk, g in gate.items() if not g['pass']})}")
    world = {m: sold[m] or arms[m]["world"][0] for m in arms}  # S_old: the per-folder map, never the arm token (09)
    epochs = [{"key": kk, "cell": f["task"], "world": world[f["arm_dir"]], "stratum": rec["stratum"], "run": rec["extraction_root"], "arm": f["arm_dir"],
               "staged": f"{rec['staged_set']}/inputs/{f['arm_dir']}.jsonl", "staged_sha256": arms[f["arm_dir"]]["staged_sha"], "folder": folder[f["arm_dir"]],
               "original_folder": of[tok[f["arm_dir"]]], "original_source": {"eval_sha256": inv[kk].get("eval_sha256"), "upload_manifest": inv[kk].get("upload_manifest")},
               "tokens_before": tb(kk), "launch_task": f.get("launch_task")} | ({"row_label": RELAUNCH_LABEL} if world[f["arm_dir"]] == "relaunch" else {})
              for kk, f in sorted(staged.items())]
    exc = [{"key": kk, "world": world[keys[kk]["arm_dir"]], "cell": keys[kk]["task"], "arm": keys[kk]["arm_dir"], "folder": folder[keys[kk]["arm_dir"]],
            "original_folder": of[tok[keys[kk]["arm_dir"]]], "reason": excluded[kk]} for kk in sorted(excluded)]
    msg = {"d9_arm": rec["target"], "batch": rec["batch"], "volume": VOLUME, "scope": a.scope, "format_check": format_label, "out_by_key": out_orphan,
           "provenance": [{"kind": "cutfull", "run": rec["extraction_root"], "staged_set": rec["staged_set"],
           "extraction_records": rec["extraction_record"], "stage_record": rec["stage_record"], "routing_record": f"{rp_} @ {pins['routing_record']['commit']}",
           "excluded_by_staging": exc, "rows": len(epochs)}], "epochs": epochs}
    mb = json.dumps(msg, indent=1, sort_keys=True).encode()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mp = od / f"msg_{rec['target']}_r{k}.json"
    mp.open("xb").write(mb)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"label": f"adapter record: d9's cutfull routing record -> message for one full-length original target, scope {a.scope} (metadata only)", "stamp (PT)": stamp,
           "git": git("rev-parse", "HEAD").decode().strip(), "target": rec["target"], "revision": k, "folders": folder, "original_folders": of,
           "message_path": str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else str(mp), "message_sha256": hashlib.sha256(mb).hexdigest(),
           "n_epochs": len(epochs), "per_folder": {fo: sum(e["folder"] == fo for e in epochs) for fo in folder.values()}, "excluded_by_staging": exc,
           "pins": pins, "gate": gate}
    rp2 = od / f"adapter_{rec['target']}_r{k}_{stamp}.json"
    rp2.open("x").write(json.dumps(out, indent=1))
    print(rp2.name, out["message_sha256"][:12], out["per_folder"], len(exc), {kk: g["pass"] for kk, g in gate.items()})


def fix_check(inv, prev):
    """4f: inventory_b_fix_v1 differs from v2 ONLY in the 77 rows of nm360_s2_run2_s1 (uncut_planned); anything else STOPs."""
    diff = [k for k in set(inv) | set(prev) if inv.get(k) != prev.get(k)]
    fields = {f for k in diff for f in set(inv.get(k, {})) | set(prev.get(k, {})) if inv.get(k, {}).get(f) != prev.get(k, {}).get(f)}
    if set(inv) != set(prev) or len(diff) != INVENTORY_B_FIX["rows"] or fields != INVENTORY_B_FIX["fields"] \
            or {inv[k]["uncut_planned"]["folder"] for k in diff} != {INVENTORY_B_FIX["folder"]}:
        raise SystemExit(f"inventory_b_fix_v1 differs from v2 beyond the 77 nm360_s2_run2_s1 rows: {len(diff)} rows, fields {sorted(fields)}")


def build_sold(a, rc_, rp_, git, show, at, pins):
    """(b) S_old (RULE F.10): d9's S_old attestation stands in for the chain routing record. Gates: the highest attestation revision at
    the branch head, failing {}, template classes within the fixed-template vocabulary (never "original"); the ONE step-0 compare; the
    stage record (one staged set; per-arm shas and epochs; exclusions); every extraction attempt of the root (H200 on every shard, cut
    "full", the cutfull Volume); each arm's every-epoch format-check record; the routed folders (no _try) and original folders; the
    staged epochs are inventory_b_v2's runs for these folders (registered, same original folder and meta arm, world from 09's per-folder
    map), equal by key to the attested keys per arm token, with the per-arm counts of the stage record."""
    att = show("routing_record", rc_, rp_)
    gate = {}
    m_ = REC_RE.match(Path(rp_).name)
    stem, k = m_.group(1), int(m_.group(2) or 1)
    revs = [REC_RE.match(Path(x).name) for x in git("ls-tree", "--name-only", f"{BRANCH}:{Path(rp_).parent}").decode().split()]
    ks = sorted(int(x.group(2) or 1) for x in revs if x and x.group(1) == stem)
    batch = re.match(r"^attestations_([a-z0-9]+)__", Path(rp_).name)
    tc = {x for arm in att["template_class"].values() for x in arm.values()}
    cf = att["cutfull"]
    gate["highest_revision"] = {"k": k, "revisions": ks, "pass": bool(batch) and ks[-1] == k == att["revision"] and att["failing"] == {} and tc <= SOLD_TEMPLATE_CLASSES
                                and cf["volume"] == VOLUME}
    rg = at(att["repro_gate"].split(" (")[0])
    pilot_att = PILOT_FOLDER in (cf.get("folders") or {}).values()  # (5) the pilot's identity: its planned folder (2e's Q1 ruling)
    if pilot_att != (PILOT_ORIGINAL in (att.get("original_folders") or {}).values()) or pilot_att != (att.get("original_target") == PILOT_TARGET) \
            or (pilot_att and set((att.get("original_folders") or {}).values()) != {PILOT_ORIGINAL}):
        raise SystemExit("STOP (5): the attestation's pilot identity disagrees: cutfull.folders, original_folders and original_target "
                         "must all name the pilot, or none")
    if pilot_att:  # (5) a pilot attestation: exactly the pinned record (path, commit, sha256), passes JSON true, run keys == the counted staged keys
        s0p, ck = show("step0_gate", *PILOT_STEP0), show("counted_keys", *COUNTED)
        gate["step0_gate"] = {"pass": rg[1] == PILOT_STEP0[1] and git("rev-parse", rg[0]) == git("rev-parse", PILOT_STEP0[0]) and s0p.get("passes") is True
                              and sorted(r["key"] for r in s0p["runs"]) == sorted(x["volume_path"].removeprefix("simh_evals_v1/") + "#e0" for x in ck["keys"])}
    else:  # S_old's path: unchanged, the ONE registered compare @ e6896f33b
        gate["step0_gate"] = {"pass": at(att["repro_gate"].split(" (")[0])[1] == STEP0[1] and git("rev-parse", at(att["repro_gate"].split(" (")[0])[0]) == git("rev-parse", STEP0[0])
                              and truthy(show("step0_gate", *STEP0)["passes"])}
    st_ = show("stage_record", *at(cf["stage_record"]))
    sets = [x for x in st_ if x != "git"]
    if len(sets) != 1:
        raise SystemExit(f"stage record: expected one staged set, found {sets}")
    st = st_[sets[0]]
    metas = sorted(st["assembled"]["epochs"])
    tok = {m: m.split("@")[0] for m in metas}
    sha = {m: st["assembled"]["staged_inputs_sha256"][f"{m}.jsonl"] for m in metas}
    excluded = st["c_gate"]["excluded"]
    gate["staging"] = {"pass": st["out"] == cf["staged_set"] and set(tok.values()) == set(cf["folders"]) == set(att["original_folders"]) == set(att["units"])
                       and len(set(tok.values())) == len(tok) and st["c_gate"]["manifest_keys"] == st["c_gate"]["staged"] + len(excluded)
                       and st["c_gate"]["staged"] == sum(st["assembled"]["epochs"].values())
                       and all(att["units"][tok[m]]["epochs"] == st["assembled"]["epochs"][m] for m in metas)}
    tree = git("ls-tree", "-r", "--name-only", git("rev-parse", BRANCH).decode().strip(), "--", "experiments/2026-10-14_q3_overnight_chain/results").decode().split()
    ex_ok = []
    for i, p in enumerate(sorted(x for x in tree if re.search(rf"/extract_chain_{re.escape(cf['extraction_root'])}_a[0-9]+_20", x))):
        dv = show(f"extraction_a{i + 1}", git("rev-parse", BRANCH).decode().strip(), p)
        ex_ok.append(dv["config"]["run"] == cf["extraction_root"] and dv["config"]["inputs"] == cf["staged_set"] and dv["volume"] == VOLUME
                     and all(dv["config"]["inputs_sha256"].get(f"{m}.jsonl") == sha[m] for m in metas) and dv["config"]["model"] == "Qwen/Qwen3.5-27B"
                     and dv["config"].get("cut") == "full" and len(dv["shards"]) == dv["config"]["shards"] and all(s_.get("gpu_name") == "NVIDIA H200" for s_ in dv["shards"]))
    gate["extraction_h200"] = {"attempts": len(ex_ok), "pass": bool(ex_ok) and all(ex_ok)}
    fmt = {}
    for m in metas:
        fr = show(f"format_{tok[m]}", *at(att["format_check"][tok[m]]))
        fmt[m] = (fr["passes"] is True if pilot_att else truthy(fr["passes"])) and not fr["failing_keys"] and sha[m] in fr["staged_sha"] \
            and fr["epochs_checked"] == st["assembled"]["epochs"][m] and fr["arm"] == m  # the pilot: JSON true only
    fmt_commits = sorted({at(att["format_check"][t])[0][:9] for t in tok.values()})
    format_label = f"format_check passed (d9 @ {', '.join(fmt_commits)})"
    gate["format"] = {"label": format_label, "pass": bool(fmt) and all(fmt.values())}
    folder = {m: cf["folders"][tok[m]] for m in metas}
    of = att["original_folders"]
    gate["original_folders"] = {"pass": all(folder[m] == PILOT_FOLDER and sold_world(of[tok[m]]) for m in metas)}
    inv_doc = show("inventory", *INVENTORY_B[:2], INVENTORY_B[2], strict=True)
    k2 = [r["key"] for r in inv_doc["runs"]]
    if len(set(k2)) != len(k2):  # BEFORE any key map: last-wins could hide a changed fix_v1 row behind its original copy among the added rows
        raise SystemExit("STOP (6): a key appears twice in the pilot inventory")
    inv = {r["key"]: r for r in inv_doc["runs"]}
    if INVENTORY_B_RECORD is None:  # (6) the pilot only: fix_v2's own sha pin, bound by its fields, replaces fix_check's v2 rule
        raise SystemExit("STOP (6): the pilot-rows file (INVENTORY_B_RECORD) is not pinned")
    if INVENTORY_B_CHECK is None:
        raise SystemExit("STOP (6): the fix_v1 ⊂ fix_v2 check record (INVENTORY_B_CHECK) is not pinned")
    prows = show("inventory_pilot_rows", *INVENTORY_B_RECORD, strict=True)["runs"]
    f2 = inv_doc.get("fix_v2") if isinstance(inv_doc.get("fix_v2"), dict) else {}
    for why, ok in (("fix_v2.pilot_rows_file is not the pinned pilot-rows file", f2.get("pilot_rows_file") == {"path": Path(INVENTORY_B_RECORD[1]).name, "sha256": INVENTORY_B_RECORD[2]}),
                    ("status is not COMPLETE", inv_doc.get("status") == "COMPLETE"), ("fix_v2.rows_added is not 20", f2.get("rows_added") == 20),
                    ("supersedes.sha256 is not inventory_b_fix_v1's", (inv_doc.get("supersedes") or {}).get("sha256") == INVENTORY_B_V1[2]),
                    ("fix_v2.original_folder is not the pilot's", f2.get("original_folder") == PILOT_ORIGINAL), ("fix_v2.uncut_folder is not the pilot's", f2.get("uncut_folder") == PILOT_FOLDER)):
        if not ok:
            raise SystemExit(f"STOP (6): the pilot inventory's {why}")
    ck6 = show("inventory_subset_check", *INVENTORY_B_CHECK)
    v1rows = show("inventory_fix_v1", *INVENTORY_B_V1, strict=True)["runs"]
    ints = ("rows_canonical_only", "rows_reserialized_identical", "n_fix_v1", "n_added")  # JSON ints only: false or 0.0 never pass as 0
    if ck6.get("passes") is not True or (ck6.get("fix_v1") or {}).get("sha256") != INVENTORY_B_V1[2] or (ck6.get("fix_v2") or {}).get("sha256") != INVENTORY_B[2] \
            or any(type(ck6.get(f)) is not int for f in ints) or ck6["rows_canonical_only"] != 0 \
            or ck6["rows_reserialized_identical"] != ck6["n_fix_v1"] or ck6["n_fix_v1"] != len(v1rows) or ck6["n_added"] != 20 \
            or (ck6.get("counted_keys") or {}).get("sha256") != COUNTED[2]:
        # expected values: n_fix_v1 = the pinned fix_v1's rows (INVENTORY_B_V1, L50-51; the record's writer: check_fix_v2_subset.py L130);
        # n_added = 20 (check_fix_v2_subset.py L39 N and L130; this file's fix_v2.rows_added == 20); counted_keys.sha256 = COUNTED (L40-41;
        # the writer: check_fix_v2_subset.py L129)
        raise SystemExit("STOP (6): the fix_v1 ⊂ fix_v2 check record does not pass on these two inventories")
    if any(r["key"] not in inv for r in v1rows):
        raise SystemExit("STOP (6): a fix_v1 row is missing from the pilot inventory")
    if any(json.dumps(inv[r["key"]]) != json.dumps(r) for r in v1rows):  # key order and type count (1, 1.0 and true differ)
        raise SystemExit("STOP (6): a fix_v1 row differs in the pilot inventory")
    k1 = [r["key"] for r in v1rows]
    if k2[:len(k1)] != k1:  # the fix_v2 writer appends the pilot rows to fix_v1's rows (make_inventory_b_fix_v2.py @ 67e04b0c5 L169)
        raise SystemExit("STOP (6): fix_v1's keys are not the ordered prefix of the pilot inventory's")
    if [json.dumps(r) for r in inv_doc["runs"][len(k1):]] != [json.dumps(r) for r in prows] or len(prows) != 20:
        raise SystemExit("STOP (6): the pilot-rows file is not exactly the pilot inventory's 20 added rows, in order")
    mine = {kk: r for kk, r in inv.items() if isinstance(r["uncut_planned"], dict) and r["uncut_planned"]["folder"] in folder.values()}
    fo_meta = {v: m for m, v in folder.items()}
    bad = [kk for kk, r in mine.items() if r["original_folder_registered"] is not True or r["arm"] != fo_meta[r["uncut_planned"]["folder"]]
           or r["original_folder"] != of[tok[r["arm"]]] or sold_world(r["original_folder"]) != r["world"] or r.get("tokens_before", r.get("full_len")) != r.get("full_len")
           or kk in excluded]
    att_keys = {t: set(v) for t, v in att["template_class"].items()}
    gate["inventory"] = {"scope": "b", "bad": bad[:5], "pass": not bad and {tok[m]: {kk for kk, r in mine.items() if r["arm"] == m} for m in metas} == att_keys
                         and all(sum(r["arm"] == m for r in mine.values()) == st["assembled"]["epochs"][m] for m in metas)}
    if not all(g["pass"] for g in gate.values()):
        raise SystemExit(f"adapter gates failed: {json.dumps({kk: g for kk, g in gate.items() if not g['pass']})}")
    epochs = [{"key": kk, "cell": r["task"], "world": r["world"], "stratum": "S_old", "run": cf["extraction_root"], "arm": r["arm"],
               "staged": f"{cf['staged_set']}/inputs/{r['arm']}.jsonl", "staged_sha256": sha[r["arm"]], "folder": folder[r["arm"]], "original_folder": r["original_folder"],
               "original_source": r.get("registered_source"), "tokens_before": r["full_len"], "launch_task": None}
              | ({"row_label": RELAUNCH_LABEL} if r["world"] == "relaunch" else {}) for kk, r in sorted(mine.items())]
    exc = []
    for kk in sorted(excluded):  # no manifest here: the arm dir is the key's second path component (an arm dir holds no "/")
        m = kk.split("/")[1]
        if m not in folder:
            raise SystemExit(f"{kk}: a staging exclusion outside this set's arms")
        exc.append({"key": kk, "cell": kk.split("/")[2], "arm": m, "folder": folder[m], "original_folder": of[tok[m]], "reason": excluded[kk]})
    target = re.sub(r"_r[0-9]+$", "", stem).split("__", 1)[1]
    msg = {"d9_arm": target, "batch": batch.group(1), "volume": VOLUME, "scope": "b", "format_check": format_label, "out_by_key": {},
           "template_classes": {t: dict(__import__("collections").Counter(v.values())) for t, v in att["template_class"].items()},
           "provenance": [{"kind": "cutfull_sold", "run": cf["extraction_root"], "staged_set": cf["staged_set"], "stage_record": cf["stage_record"],
                           "extraction_records": [pins[x] for x in sorted(pins) if x.startswith("extraction_a")], "routing_record": f"{rp_} @ {pins['routing_record']['commit']}",
                           "excluded_by_staging": exc, "rows": len(epochs)}], "epochs": epochs}
    mb = json.dumps(msg, indent=1, sort_keys=True).encode()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mp = od / f"msg_{target}_r{k}.json"
    mp.open("xb").write(mb)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"label": "adapter record: d9's S_old attestation -> message for one full-length S_old target, scope b (metadata only)", "stamp (PT)": stamp,
           "git": git("rev-parse", "HEAD").decode().strip(), "target": target, "revision": k, "folders": folder, "original_folders": of,
           "message_path": str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else str(mp), "message_sha256": hashlib.sha256(mb).hexdigest(),
           "n_epochs": len(epochs), "per_folder": {fo: sum(e["folder"] == fo for e in epochs) for fo in folder.values()}, "excluded_by_staging": exc,
           "pins": pins, "gate": gate}
    rp2 = od / f"adapter_{target}_r{k}_{stamp}.json"
    rp2.open("x").write(json.dumps(out, indent=1))
    print(rp2.name, out["message_sha256"][:12], out["per_folder"], len(exc), {kk: g["pass"] for kk, g in gate.items()})


if __name__ == "__main__":
    main()
