"""Input index for the retried logical batches (09 @ 65ef3a9ff section 1 and 09's s5 sim2 conditions, via d9 10-15): per logical
batch (batch, target), the ONE authoritative artifact (staged set, extraction root, scorer folder, staged sha) with the reason, and
every earlier artifact as a non-authoritative audit record, never routed or scored. Read from the chain's state (the re-stages
il-s4d and il-q3c: first_staging_v1 vs stage) and from the committed extraction records (il-s5's sim2 extraction retry, same staged
set: the record's inputs_sha256 must equal the staged sha). Keys and hashes only.

  python3 make_input_index_retries.py <chain state dir> <out.json>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = "sim_attack@september"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"


def arm_of(t: str) -> str:
    return "attack" if t.endswith("_run3") else SIM


def folder(out: str, t: str) -> str:
    """An interleaved target's scorer folder (<staged set>__<arm>); None for other targets (their routing file names the arms)."""
    if t.endswith("_run3") or "_run3_s" in t:
        return f"{out}__attack"
    if t.endswith("_sim2") or "_sim2_s" in t:
        return f"{out}__sim_attack"
    return None


def main():
    st = json.loads((Path(sys.argv[1]) / "state.json").read_text())["batches"]
    rows = []
    for bid, kc in (("il-s4d", "keycheck_il_s4d_donors_2026-10-15.json @ 4ee851f80"), ("il-q3c", "keycheck_main_donors_2026-10-15.json @ a7c5e0db7")):
        b = st[bid]
        assert b.get("rev") == "_v2" and "attest" in b["done"], (bid, b.get("rev"), b["done"])
        first = b["first_staging_v1"]
        for t in b["targets"]:
            cur, old = b["stage"][t], first["stage"][t]
            assert cur["out"] == f"{t}_v1_try2" and old["out"] == f"{t}_v1" and old["epochs"] == {}, (t, cur["out"], old["out"])
            rows.append({"batch": bid, "target": t, "kind": "re-stage from v2 manifests (09 @ 65ef3a9ff section 1)",
                         "authoritative": {"staged_set": cur["out"], "extraction_root": b["extract"][t], "scorer_folder": folder(cur["out"], t),
                                           "staged_sha": cur["staged_inputs_sha256"].get(f"{arm_of(t)}.jsonl"), "epochs": cur["epochs"],
                                           "manifest": f"config/manifest_{t}_v2.json", "staging_record": cur["record"]},
                         "non_authoritative": [{"staged_set": old["out"], "extraction_root": first["extract"][t], "scorer_folder": folder(old["out"], t),
                                                "epochs": old["epochs"], "excluded": old["excluded"], "staging_record": old["record"],
                                                "status": "audit artifact only: all epochs excluded (no usable donor); never routed or scored"}],
                         "reason": f"re-stage from v2 manifests: donor rows missing on the family's box; key check {kc}"})
    b = st["il-s5"]
    t = "ilnm_s5_sim2"
    staged = b["stage"][t]["staged_inputs_sha256"][f"{SIM}.jsonl"]
    recs = sorted((HERE / "results").glob("extract_chain_ilnm_s5_sim2_extract_v1_try2_a*_20*.json"))
    assert recs, "no committed record of the retry yet"
    rec = json.loads(recs[-1].read_text())
    assert rec["config"]["inputs"] == b["stage"][t]["out"] == "ilnm_s5_sim2_v1" and rec["config"]["inputs_sha256"][f"{SIM}.jsonl"] == staged, "staged set or sha differs"
    rows.append({"batch": "il-s5", "target": t, "kind": "extraction retry of the same staged set (not a re-stage)",
                 "authoritative": {"staged_set": "ilnm_s5_sim2_v1", "extraction_root": rec["config"]["run"], "scorer_folder": "ilnm_s5_sim2_v1_try2__sim_attack",
                                   "staged_sha": staged, "extraction_record": recs[-1].name, "staged_sha_equal_to_v1": True},
                 "non_authoritative": [{"extraction_root": "ilnm_s5_sim2_extract_v1", "scorer_folder": "ilnm_s5_sim2_v1__sim_attack",
                                        "status": "never routed further or scored: its chunks were written by an attempt-1 writer whose GPU is unrecorded"}],
                 "reason": "extraction retry; staged set ilnm_s5_sim2_v1, staged sha unchanged; reason: attempt-1 writer's GPU unrecorded (fc-01M4ZX3RPFHWZPPFWQFBYMJHVW)"})
    # staging retries after a failed try (Modal error, no record): try n is the ONE authoritative staged set; each earlier try is
    # listed with what it left on the Volume (09 via d9 10-15 ~08:15)
    import modal

    vol = modal.Volume.from_name(VOL, version=2)

    def left(name: str) -> dict:
        try:
            files = [e.path for e in vol.listdir(name, recursive=True)]
        except (FileNotFoundError, modal.exception.NotFoundError) as e:  # only a confirmed missing folder; any other failure raises
            return {"on_volume": False, "listing": type(e).__name__}
        return {"on_volume": True, "files": len(files), "staged_inputs": sorted(f.split("/")[-1] for f in files if "/inputs/" in f and f.endswith(".jsonl"))}

    for bid, b in sorted(st.items()):
        if b.get("rev"):
            continue  # the re-stages above
        for t, n in sorted((b.get("stage_tries") or {}).items()):
            if n <= 1 or t not in b.get("stage", {}):
                continue
            cur = b["stage"][t]
            base = cur["out"][: -len(cur["suffix"])] if cur["suffix"] else cur["out"]
            assert cur["out"] == f"{base}_try{n}", (bid, t, cur["out"], n)
            earlier = []
            for k in range(1, n):
                name = base if k == 1 else f"{base}_try{k}"
                earlier.append({"staged_set": name, "status": "failed on Modal, no record; never routed or scored", **left(name)})
            rows.append({"batch": bid, "target": t, "kind": f"staging retry (try {n}; the earlier tries failed on Modal with no record)",
                         "authoritative": {"staged_set": cur["out"], "extraction_root": b.get("extract", {}).get(t), "scorer_folder": folder(cur["out"], t),
                                           "staged_sha": cur["staged_inputs_sha256"], "epochs": cur["epochs"], "staging_record": cur["record"]},
                         "non_authoritative": earlier, "reason": "infrastructure retry of the same manifest (not a re-stage)"})
    doc = {"rule": "09 @ 65ef3a9ff section 1; 09's s5 sim2 conditions (via d9 10-15 ~07:45); staging retries (via d9 ~08:15): one authoritative artifact per logical batch",
           "entries": rows}
    Path(sys.argv[2]).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps([(r["batch"], r["target"], r["authoritative"]["extraction_root"], r["authoritative"]["scorer_folder"]) for r in rows], indent=1))


if __name__ == "__main__":
    main()
