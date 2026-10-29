"""il-s4d's corrected counted manifests (09 via d9 10-15 ~07:05; key check results/hygiene/keycheck_il_s4d_donors_2026-10-15.json
@ 4ee851f80): the il-s4d box read ran on xbox2, whose copy of the S4 main holds only launch.json and references/, so its manifests
have no honest donor rows and staging excluded every epoch ("no usable donor"). The v2 manifests are the originals' rows unchanged
plus the S4 main's counted honest_a / honest_b rows exactly as the chain read them on box4 (the s4d_sim follow-up's donor rows, the
same counted() rule and row format as an interleaved box read), each checked by path and sha256 against the chain's relayed S4
main manifests, and the 4d cells' 38 against the key check's box4 hashes. New files (open 'x'); each original stays, with a
SUPERSEDED note beside it. Keys and hashes only.

  python3 make_s4d_v2_manifests.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / "config"
KEYCHECK = HERE / "results" / "hygiene" / "keycheck_il_s4d_donors_2026-10-15.json"
REASON = ("re-stage from v2 manifests: donor rows missing on xbox2 (its copy of the S4 main holds only launch.json and references/); "
          "key check 4ee851f80; registered 09 @ 65ef3a9ff section 1)")


def main():
    main_rows = {}
    for name in ("manifest_nm360_s4_main_s1.json", "manifest_nm360_s4_main.json"):  # the chain's relayed S4 main (snapshot 1 + remainder)
        for f in json.loads((CFG / name).read_text())["files"]:
            if f["arm_dir"] in ("honest_a", "honest_b"):
                assert f["path"] not in main_rows, f["path"]
                main_rows[f["path"]] = f
    donors = [f for f in json.loads((CFG / "manifest_nm360_s4d_sim.json").read_text())["files"] if f["arm_dir"] in ("honest_a", "honest_b")]
    assert {f["path"] for f in donors} == set(main_rows), "the s4d_sim donors are not exactly the S4 main's counted honest rows"
    assert all(f["sha256"] == main_rows[f["path"]]["sha256"] and f["task"] == main_rows[f["path"]]["task"] for f in donors)
    kc = json.loads(KEYCHECK.read_text())
    box4 = {r["key"]: r["box4"]["sha256"] for r in kc["rows"]}
    by_key = {f"{f['arm_dir']}/{f['task']}": f["sha256"] for f in donors}
    assert len(box4) == 38 and all(by_key.get(k) == v for k, v in box4.items()), "a 4d donor differs from box4's hash"
    out = {}
    for w in ("run3", "sim2"):
        src = CFG / f"manifest_ilnm_s4d_{w}.json"
        orig = json.loads(src.read_text())
        assert not any(f["arm_dir"] in ("honest_a", "honest_b") for f in orig["files"]), f"{src.name} already has donors"
        assert not {f["path"] for f in orig["files"]} & set(main_rows)
        files = sorted(orig["files"] + donors, key=lambda f: f["path"])
        doc = {"note": orig["note"] + " (v2: honest donors from the S4 main's counted rows on box4, which the xbox2 box read could not see)",
               "counts": {"files": len(files), "by_arm": {a: sum(f["arm_dir"] == a for f in files) for a in sorted({f["arm_dir"] for f in files})}},
               "counted_attempts_without_eval": orig["counted_attempts_without_eval"], "files": files,
               "supersedes": src.name, "supersedes_reason": REASON,
               "donor_source": {"rows": "config/manifest_nm360_s4d_sim.json (honest_a/honest_b)", "checked_against": ["config/manifest_nm360_s4_main_s1.json",
                                "config/manifest_nm360_s4_main.json", f"results/hygiene/{KEYCHECK.name} (box4, the 38 rows of the 4d cells)"]}}
        dst = CFG / f"manifest_ilnm_s4d_{w}_v2.json"
        with dst.open("x") as fh:
            fh.write(json.dumps(doc, indent=1) + "\n")
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        note = CFG / f"manifest_ilnm_s4d_{w}.SUPERSEDED.md"
        with note.open("x") as fh:
            fh.write(f"# SUPERSEDED\n\n`{src.name}` is superseded by `{dst.name}` (sha256 {sha}): {REASON}. The v2 file keeps every row of this one "
                     f"unchanged and adds the S4 main's {len(donors)} counted honest donor rows. This file stays unchanged; its all-excluded staging "
                     f"(`ilnm_s4d_{w}_v1`) is a non-authoritative audit artifact, never routed or scored.\n")
        out[dst.name] = {"sha256": sha, "counts": doc["counts"], "original_rows": len(orig["files"]), "donors": len(donors)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
