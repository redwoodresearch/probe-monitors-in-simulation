"""APPEND-ONLY companion to an own_all_rs cutfull score record (4f / 2e, 10-16): per scored cutfull folder, the staged inputs file, staged sha256 and
extraction root the scoring job READ, by key. Source: each message's routing pin in the score record (path @ commit, sha256 re-checked), whose arm
staged_sha is what score() hashed the staged file against (a mismatch is a STOP there). The manifests carry no staged-sha field; the only template
sha in them is manifest.record.chat_template_sha256.fixed_staged, copied from the routing record (the chat TEMPLATE, not a staged input). Git only."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INV = ("b8a39f329", "experiments/2026-10-15_cut_sensitivity/results/inventory_v3_2026-10-15.json", "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3")
show = lambda c, p: subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731


def _arms(m, rr, rb):
    """(folder, staged inputs file, staged sha256, extraction root) per arm: a routing record's arms, or an S_old attestation's cutfull block + stage record."""
    if m["routing_pin"].get("kind") == "S_old attestation":
        cf = rr["cutfull"]
        sp_, sc_ = cf["stage_record"].split(" @ ")
        sr = json.loads(show(sc_, sp_))
        shas = next(v["assembled"]["staged_inputs_sha256"] for v in sr.values() if isinstance(v, dict) and "assembled" in v)
        ma = {Path(f).stem.split("@", 1)[0]: Path(f).stem for f in shas}
        return [(fo, f"{cf['staged_set']}/inputs/{ma[t]}.jsonl", shas[f"{ma[t]}.jsonl"], cf["extraction_root"]) for t, fo in cf["folders"].items()]
    ss = re.sub(r"_cutfull_v1_try[0-9]+$", "_cutfull_v1", rr["staged_set"])
    return [(f"{ss}__{a['meta_arm'].split('@', 1)[0]}", a["staged_inputs_file"], a["staged_sha"], rr["extraction_root"]) for a in rr["arms"]]


def main(score_rel: str, out_rel: str, inv: str = ""):
    sb = (REPO / score_rel).read_bytes()
    if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{score_rel}"], capture_output=True).stdout != sb:
        raise SystemExit("the score record is not committed as read")
    rec = json.loads(sb)
    inv_ = tuple(inv.split(":", 2)) if inv else INV  # (A): 2e's inventory v3; (B): the record's inventory_b pin, given as <commit>:<path>:<sha256>
    ib = show(inv_[0], inv_[1])
    if hashlib.sha256(ib).hexdigest() != inv_[2]:
        raise SystemExit("the inventory differs from its pin")
    keys_of = {}
    for r in json.loads(ib)["runs"]:
        keys_of.setdefault(r["uncut_planned"]["folder"], []).append(r["key"])
    folders, by_key, tpl_eq = {}, {}, 0
    for h, m in sorted(rec["messages"].items()):
        rp = m["routing_pin"]
        rb = show(rp["commit"], rp["path"])
        if hashlib.sha256(rb).hexdigest() != rp["sha256"]:
            raise SystemExit(f"{rp['path']}: sha256 differs from the score record's pin")
        rr = json.loads(rb)
        tpl = (rr.get("chat_template_sha256") or {}).get("fixed_staged")
        for fo, sf, ssha, root in _arms(m, rr, rb):
            if fo not in m["original_of"]:
                raise SystemExit(f"{fo}: an arm of the routing record is not a scored folder of this message")
            res = rec["results"][f"scaleup|{h}"] if f"scaleup|{h}" in rec["results"] else rec["results"][f"simh|{h}"]
            if fo not in res["folders"]:
                raise SystemExit(f"{fo}: not in the score record's results")
            ks = sorted(keys_of.get(fo, []))
            n_ep = res["folders"][fo]["counts"]["R"]["runs"] + res["folders"][fo]["counts"]["R"]["runs_without_rows"]
            if not ks or len(ks) != n_ep:
                raise SystemExit(f"{fo}: inventory keys {len(ks)} != scored epochs {n_ep} (runs with R rows + runs without)")
            tpl_eq += ssha == tpl
            folders[fo] = {"original_folder": m["original_of"][fo], "staged_inputs_file": sf, "staged_sha256": ssha,
                           "extraction_root": root, "epochs": len(ks), "manifest_sha256": res["folders"][fo]["manifest_sha256"],
                           "routing_record": f"{rp['path']} @ {rp['commit']} (revision {rp['revision']}, sha256 {rp['sha256']})", "message_sha256": h,
                           "chat_template_sha256_fixed_staged_NOT_a_staged_sha": tpl}
            for k in ks:
                if k in by_key:
                    raise SystemExit(f"{k}: in two folders")
                by_key[k] = [fo, ssha]
    if tpl_eq:
        raise SystemExit(f"{tpl_eq} arms have staged sha == the template sha")
    note = {"note": "APPEND-ONLY companion (4f / 2e, 10-16): the staged inputs each cutfull folder's scoring job READ, by key. The manifests have no staged-sha "
                    "field; manifest.record.chat_template_sha256.fixed_staged is the chat TEMPLATE sha copied from the routing record, never a staged input. "
                    "score() hashed each staged file against the staged_sha256 below and would have STOPPED on a mismatch.",
            "note_on": {"path": score_rel, "sha256": hashlib.sha256(sb).hexdigest()}, "inventory": {"commit": inv_[0], "path": inv_[1], "sha256": inv_[2]},
            "folders_count": len(folders), "keys_count": len(by_key), "arms_with_staged_sha_equal_template_sha": 0,
            "s_old": "S_old folders: staged file / sha from the attestation's cutfull block and its stage record (d9's standard sold__ records are asserted equal by key in the uncut fit record)",
            "folders": folders, "by_key": dict(sorted(by_key.items()))}
    (REPO / out_rel).open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"folders": len(folders), "keys": len(by_key), "su311_folders": sum(f.startswith("su311_") for f in folders),
                      "su311_keys": sum(v[0].startswith("su311_") for v in by_key.values()), "template_sha_as_staged": 0,
                      "example": folders.get("su311_q1_main_cutfull_v1__attack", {}).get("staged_sha256")}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
