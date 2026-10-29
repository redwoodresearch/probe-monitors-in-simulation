"""IL spec fragment `arms` for 09's IL analysis job (aswin/il-analysis-09 @ 47cb134d4; make_il_spec.py --arms <this file>@<commit>; d9 via
09, 10-15 ~11:30): for each authoritative IL scorer folder 09's parse lists (il_parse_keys @ 1f15b9d04), {hygiene, p2_manifest,
sol_manifest} in FINAL's shape (final_spec_3_launch.json @ 3ce6ea55e), each pin path @ commit + sha256:
- hygiene from the batch's LIVE attestation (the routing pointer, following `superseded` -> revisions[-1]): longest_epoch_h200_gate
  {extract, compare, staging}, corrected_render, tool_error_table (checked: its staged_sha is the folder's, every source_evals entry has a
  path and a sha256); every attestation entry for the folder's staged sha must pass;
- p2_manifest and sol_manifest from the scorers' work Volume (sol_r2/ when it exists, as the loader requires); a missing one is listed;
- the folder is <staged set>__<attack|sim_attack> from the chain state, cross-checked against the retries input index @ 7a203a695.
Two scopes, two files (09 via d9 ~11:45: the primary stays byte-identical whether or not the extension exists): `primary` = exactly the
folders 09's parse lists (il311 + ilold); `extended` = every attested ilnm IL folder (S1, S2, S3, S5, S4d, snapshots and retries) as
`arms_extended`. An extraction retry's folder is the input index's authoritative scorer folder. Also per folder (outside FINAL's keys): the attestation record and the IL pairwise approach provenance file it cites. The p2 manifests'
cell-list pins (inputs.cells_plus311.sha256) are reported by value, for p2_cells_list_sha256_allowed. Keys, paths and hashes only.

  python make_il_arms_fragment.py <chain state dir> primary <parse result path@commit> <out fragment>
  python make_il_arms_fragment.py <chain state dir> extended - <out fragment>
  python make_il_arms_fragment.py <chain state dir> primary <parse path@commit> <out v2> <added folders, comma-separated> <v1 path@commit>
     (an append-only revision, e.g. q4c's two folders after its routing: 09's folders plus the added ones; every v1 row must come out
     byte-identical, else nothing is written)
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

import chain as C
from make_xerr_spec_newmains import WORK, git, pin, write_new

SFX = {"attack": "attack", "sim_attack@september": "sim_attack"}
INDEX = {"commit": "7a203a695", "path": f"{C.REL}/results/hygiene/input_index_retries_2026-10-15.json"}
PLUS311 = "scaleup_tasks_plus311_2026-10-14_125135.json"  # the FINAL loader's PINS key for the +311 list


def live_attestation(st_dir: Path, bid: str) -> dict:
    """The batch's live attestation pointer: follow superseded -> revisions[-1].pointer until a pointer is not superseded."""
    p = st_dir / "routing" / f"{bid}_attest.json"
    seen = []
    while True:
        d = json.loads(p.read_text())
        seen.append(p.name)
        if not d.get("superseded"):
            assert d.get("failing") == {} and len(d["commit"]) == 40, (bid, seen)
            return d | {"pointer_chain": seen}
        p = st_dir / "routing" / d["revisions"][-1]["pointer"]


def main():
    st_dir, scope, out_p = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[4])
    prefixes = {"primary": ("il311", "ilold"), "extended": ("ilnm",)}[scope]
    st = json.loads((st_dir / "state.json").read_text())["batches"]
    idx_raw = subprocess.run(["git", "show", f"{INDEX['commit']}:{INDEX['path']}"], capture_output=True, check=True, cwd=C.HERE).stdout
    idx = {(e["batch"], e["target"]): e["authoritative"] for e in json.loads(idx_raw)["entries"]}
    work = modal.Volume.from_name(WORK, version=2)

    def vol(path: str):
        try:
            raw = b"".join(work.read_file(path))
        except Exception as e:  # noqa: BLE001 (only a missing file is expected; anything else is reported, not hidden)
            return None if type(e).__name__ in ("FileNotFoundError", "NotFoundError") else {"error": type(e).__name__}
        return raw

    arms, where, missing, p2_lists = {}, {}, [], {}
    for bid, b in sorted(st.items()):
        if b["kind"] not in C.IL_KINDS or b.get("name_prefix") not in prefixes or not b.get("targets") or "attest" not in b["done"]:
            continue
        for t in b["targets"]:
            sg = b["stage"][t]
            for arm in [a for a in sg["epochs"] if a != "repro-c"]:
                folder = f"{sg['out']}__{SFX[arm]}"
                ix = idx.get((bid, t))
                if ix is not None:  # the authoritative artifact of a retried logical batch (an extraction retry names its own folder)
                    assert ix["staged_set"] == sg["out"] and ix["scorer_folder"] in (folder, f"{sg['out']}_try2__{SFX[arm]}"), (folder, ix["scorer_folder"])
                    folder = ix["scorer_folder"]
                assert folder not in where, folder
                where[folder] = (bid, t, arm)
    if scope == "primary":
        pp, pc = sys.argv[3].rsplit("@", 1)
        parse_raw = subprocess.run(["git", "show", f"{pc}:{pp}"], capture_output=True, check=True, cwd=C.HERE).stdout
        want = json.loads(parse_raw)["result"]["authoritative_folders_needed"]
        source = {"commit": pc, "path": pp, "sha256": hashlib.sha256(parse_raw).hexdigest()}
        added = [x for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else []) if x]
        assert not set(added) & set(want), added
        want = [*want, *added]
    else:
        want, source = sorted(where), "every attested ilnm IL folder in the chain state"
    extra = sorted(set(where) - set(want))
    absent = sorted(set(want) - set(where))
    assert not absent, f"09's folders without an attested chain staged set: {absent}"
    for folder in want:
        bid, t, arm = where[folder]
        b, sg = st[bid], st[bid]["stage"][t]
        ssha = sg["staged_inputs_sha256"][f"{arm}.jsonl"]
        ap = live_attestation(st_dir, bid)
        att_raw = subprocess.run(["git", "show", f"{ap['commit']}:{ap['path']}"], capture_output=True, check=True, cwd=C.HERE).stdout
        att_pin = {"commit": ap["commit"][:12], "path": ap["path"], "sha256": hashlib.sha256(att_raw).hexdigest()}  # exactly the pointer's commit
        ents = json.loads(att_raw)["attestations"][ssha]
        assert all(e["pass"] for e in ents), f"{folder}: a failing attestation entry"
        ev = {e["check"]: e for e in ents}
        rec_sha = {}  # every evidence file the attestation recorded, path -> the sha256 it recorded (a pin must be exactly that file)
        for e in ents:
            for x in [e.get("evidence")] + list(e.get("evidence_extra") or []):
                if x:
                    assert rec_sha.setdefault(x["path"], x["sha256"]) == x["sha256"], (folder, x["path"])

        def apin(path: str) -> dict:
            q = pin(path)
            assert q["sha256"] == rec_sha[path], f"{folder}: {path} differs from the file its attestation recorded"
            return q

        gate = ev["longest_epoch_h200_gate"]
        extra_ev = [x["path"] for x in gate.get("evidence_extra", [])]
        ext = sorted([gate["evidence"]["path"]] + [p for p in extra_ev if "/extract_chain_" in p])
        te_path = ev["tool_error_table"]["evidence"]["path"]
        te = json.loads((C.REPO / te_path).read_text())
        assert te["staged_sha"] == ssha and te["source_evals"] and all(
            isinstance(s["path"], str) and isinstance(s["sha256"], str) and len(s["sha256"]) == 64 for s in te["source_evals"]), f"{folder}: tool-error table"
        hyg = {"longest_epoch_h200_gate": {"extract": [apin(p) for p in ext], "compare": apin(next(p for p in extra_ev if "/compare_repro_" in p)),
                                           "staging": apin(next(p for p in extra_ev if "/staged_stats_" in p))},
               "corrected_render": apin(ev["corrected_render"]["evidence"]["path"]), "tool_error_table": apin(te_path)}
        sol_dir = "sol_r2" if vol(f"scaleup/{folder}/sol_r2/manifest_sol.json") is not None else "sol"
        p2_raw, sol_raw = vol(f"scaleup/{folder}/p2/manifest_p2.json"), vol(f"scaleup/{folder}/{sol_dir}/manifest_sol.json")
        sup = vol(f"scaleup/{folder}/{sol_dir}/SUPERSEDED.json")
        entry = {"hygiene": hyg}
        for name, raw, path in (("p2_manifest", p2_raw, f"scaleup/{folder}/p2/manifest_p2.json"),
                                ("sol_manifest", sol_raw, f"scaleup/{folder}/{sol_dir}/manifest_sol.json")):
            if not isinstance(raw, bytes) or (name == "sol_manifest" and sup is not None):
                missing.append({"folder": folder, "missing": name, "detail": "SUPERSEDED" if sup is not None and name == "sol_manifest" else raw})
                entry[name] = None
            else:
                entry[name] = {"path": f"/work/{path}", "sha256": hashlib.sha256(raw).hexdigest()}
        if isinstance(p2_raw, bytes):
            cl = ((json.loads(p2_raw).get("inputs") or {}).get("cells_plus311") or {}).get("sha256")
            p2_lists.setdefault(str(cl), []).append(folder)
        entry["attestation_record"] = att_pin | {"pointer_chain": ap["pointer_chain"], "staged_sha": ssha}
        pv = ev.get("approach_provenance")
        if pv is not None:
            entry["il_approach_provenance"] = apin(pv["evidence"]["path"]) | {"pass": pv["pass"]}
        arms[folder] = entry
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
    frag = {("arms" if scope == "primary" else "arms_extended"): arms, "scope": {"scope": scope, "folders_from": source, "folders": len(want),
                                    "attested_il_folders_not_in_scope": extra, "q4c": "added after its routing"},
            "missing_scorer_manifests": missing, "p2_cells_list_sha256": {k: sorted(v) for k, v in p2_lists.items()},
            "p2_cells_list_note": f"the FINAL loader pins {PLUS311}; any other value here needs 4f's name for p2_cells_list_sha256_allowed",
            "input_index": INDEX | {"sha256": hashlib.sha256(idx_raw).hexdigest()}, "d9_commit": git("rev-parse", "HEAD")[:12], "written": now,
            "note": "PRELIMINARY; keys, paths and hashes only"}
    if scope == "primary" and len(sys.argv) > 6:  # an append-only revision: the earlier version's rows byte-identical
        bp, bc = sys.argv[6].rsplit("@", 1)
        base_raw = subprocess.run(["git", "show", f"{bc}:{bp}"], capture_output=True, check=True, cwd=C.HERE).stdout
        base = json.loads(base_raw)["arms"]
        assert set(base) <= set(arms) and all(json.dumps(arms[f], sort_keys=True) == json.dumps(base[f], sort_keys=True) for f in base), \
            "a v1 row differs (pins moved since v1): nothing written"
        frag["revision"] = {"supersedes": {"commit": bc, "path": bp, "sha256": hashlib.sha256(base_raw).hexdigest()},
                            "added_folders": sorted(set(arms) - set(base)), "earlier_rows_byte_identical": True}
    write_new(out_p, frag)
    print(json.dumps({"arms": len(arms), "missing": missing, "attested_not_in_scope": extra,
                      "p2_cells_list_sha256": {k: len(v) for k, v in p2_lists.items()}}, indent=1))


if __name__ == "__main__":
    main()
