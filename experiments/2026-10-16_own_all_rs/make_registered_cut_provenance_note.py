"""Cut-provenance NOTE for the REGISTERED own_all fits (APPEND-ONLY; 4f for 2e's OOD table, 10-16): own_all{,25,10}_L48 (v1, fit @ 66800141f) and
own_all_rs{,25,10}_L48 (fit @ 52067c020) have no cut field. Both trained on own_all v1's sha-pinned rows (ROWS_SHA). For every training root this NOTE
gives the staged set, its staging record / config (path @ commit, sha256) and the stager's cut (CUT = 32768, both readouts bounded), taken from the
signed OOD table r4's per-root bindings (09 r4 §2), and asserts them BY KEY against the fit inputs' training epochs (root, arm, staged sha256, epoch
counts) and the fit records' row pins. The cut is read from the staging code / records, never from a dump name. Git only."""

import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R4 = ("36f0341bd", "experiments/2026-10-14_q35_sol_avg_POSTHOC/results/training_length_table_2026-10-16_030702_r4.json", "b206daba6140838b953c565dcf76cccac9dbfbc2154a1de6048bacdedac894b9")
V1_FIT = ("66800141f", "experiments/2026-10-14_p2_family_expansion/results/own_all_fit_2026-10-15_173309.json")
RS_FIT = ("52067c020", "experiments/2026-10-14_p2_family_expansion/results/own_all_rs_fit_2026-10-16_003049.json")
INPUTS = ("807c52f78", "experiments/2026-10-14_p2_family_expansion/results/own_all_inputs_2026-10-15.json")
ROWS_SHA = {"T196": {"R": "8b375c6174a12740165cd483a6cfa1be58c9cec6caed95387bdc82c48ba53fcb", "C": "26edc3300421ee915f08d9f0d8f3cff0488d5c66c596286c016df2e6aff478f7"},
            "scaleup": {"R": "ec50594b39d6f63e83df0918c9773e25c9c60921975248610fcf11bb1a1ebd24", "C": "09cbe4995372aa100ca682d53d8ac3c481756a3bf42c311f9001f63b9a56e308"}}
CUT_LINE = 'ACTING, CUT, SCRUB_LEVEL, LABELER = "vllm/Qwen/Qwen3.5-27B", 32768, 1, "eb014307"'


def main(out_rel):
    show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
    r4b = show(R4[0], R4[1])
    if hashlib.sha256(r4b).hexdigest() != R4[2]:
        raise SystemExit("the r4 table differs from its pin")
    rows = [x for x in json.loads(r4b)["rows"] if x["probe"].split(" (")[0] in ("own_all_rs_L48", "own_all_rs25_L48", "own_all_rs10_L48")]
    v1b, rsb, ib = show(*V1_FIT), show(*RS_FIT), show(*INPUTS)
    inp = json.loads(ib)
    bad = []
    if len(rows) != 6 or any(json.dumps(r["class_source"]["roots"]) != json.dumps(rows[0]["class_source"]["roots"]) for r in rows):
        bad.append("r4 does not hold the 6 own_all_rs rows with one shared root binding")
    cs = rows[0]["class_source"]
    if cs["fit_record"]["sha256"] != hashlib.sha256(rsb).hexdigest() or cs["training_rows"]["from"]["sha256"] != hashlib.sha256(v1b).hexdigest() or \
            cs["training_inputs"]["sha256"] != hashlib.sha256(ib).hexdigest():
        bad.append("r4's fit / rows-from / inputs pins differ from the git records")
    if {p: {rd: x["sha256"] for rd, x in v.items()} for p, v in cs["training_rows"]["rows"].items()} != ROWS_SHA:
        bad.append("r4's training-row shas differ from own_all v1's ROWS_SHA")
    ep = collections.defaultdict(list)
    for e in inp["training_epochs"]:
        ep[e["run"]].append(e)
    roots = {x["root"]: x for x in cs["roots"]}
    if set(roots) != set(ep):
        bad.append(f"roots differ: r4 only {sorted(set(roots) - set(ep))}, inputs only {sorted(set(ep) - set(roots))}")
    out = {}
    for rt, x in sorted(roots.items()):
        es = ep.get(rt, [])
        arms = sorted({e["arm"] for e in es})
        sha_ok = all(e.get("staged_sha256") in (None, "from_config") or e["staged_sha256"] == x["staged_inputs_sha256"].get(e["arm"]) for e in es)
        n_keyed = sum(1 for e in es if e.get("staged_sha256") not in (None, "from_config"))
        cut_ok = x["stager"]["cut_constant"][1] == CUT_LINE and len(x["stager"].get("readouts_bounded_by_cut", [])) == 2
        if x["epochs"] != len(es) or sorted(x["arms"]) != arms or not sha_ok or not cut_ok:
            bad.append(f"{rt}: epochs {x['epochs']} vs {len(es)}, arms, staged sha by key {sha_ok}, or cut line {cut_ok}")
        out[rt] = {"staged_set": x.get("staged_set") or x.get("extraction_config", {}).get("inputs"), "arms": arms, "training_epochs": len(es),
                   "training_epochs_staged_sha_checked_by_key": n_keyed, "staged_inputs_sha256": x["staged_inputs_sha256"], "binding": x["binding"],
                   "stage_record": x["stage_record"], "extraction": x.get("extraction_records") or x.get("extraction_config"), "stager": x["stager"],
                   "cut": 32768, "truncated_runs_at_staging": x["truncated_runs"]}
    note = {"label": "Cut provenance of the registered own_all fits (APPEND-ONLY NOTE; 4f / 2e, 10-16): own_all{,25,10}_L48 v1 and own_all_rs{,25,10}_L48 trained within the "
                     "32,768-token staging cut; keys, shas and counts only",
            "applies_to": {"own_all v1 (own_all_L48, own_all25_L48, own_all10_L48)": {"git_fit_record": f"{V1_FIT[1]} @ {V1_FIT[0]}", "sha256": hashlib.sha256(v1b).hexdigest()},
                           "own_all_rs (own_all_rs_L48, own_all_rs25_L48, own_all_rs10_L48)": {"git_fit_record": f"{RS_FIT[1]} @ {RS_FIT[0]}", "sha256": hashlib.sha256(rsb).hexdigest(),
                                                                                              "rows_from": "own_all v1's rows (ROWS_SHA)"}},
            "fit_inputs": {"path": INPUTS[1], "commit": INPUTS[0], "sha256": hashlib.sha256(ib).hexdigest(), "training_epochs": len(inp["training_epochs"])},
            "binding_source": {"path": R4[1], "commit": R4[0], "sha256": R4[2], "rule": cs["rule"]},
            "training_rows_sha256": ROWS_SHA, "roots": out, "cut": 32768, "pass": not bad, "problems": bad,
            "note": "T196 epochs (q4b roots) carry no staged sha in the fit inputs (pass 1's spec 5e2409a7); their binding is r4's extraction config inputs_sha256 == the "
                    "staging record (q4b main) or q4f's changed_inputs_sha256 (rendercut). The rendercut record has no truncation counts (changed-epoch compare)."}
    (REPO / out_rel).open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"pass": note["pass"], "roots": len(out), "epochs": sum(v["training_epochs"] for v in out.values()),
                      "staged_sha_checked_by_key": sum(v["training_epochs_staged_sha_checked_by_key"] for v in out.values()), "problems": bad[:3]}))
    if bad:
        raise SystemExit("STOP")


if __name__ == "__main__":
    main(sys.argv[1])
