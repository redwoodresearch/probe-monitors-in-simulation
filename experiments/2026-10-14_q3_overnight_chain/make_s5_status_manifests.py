"""S5 sim snapshot cutoffs and status manifests (09 @ bffe22e3e, via d9; 10-15 ~03:55), committed before any 5r sim run is routed for
scoring. One manifest per S5 sim batch (nm360-s5fu_s1 = snapshot 1, nm360-s5fu = the remainder): its cutoff (the chain log's boxfiles
time, when the batch read the launch), and per key sim|<cell>: the launcher state at that cutoff, whether the batch staged it, and the
stager's exclusion reason. Keys, states and reasons only. Sources: the chain's boxfiles records (the launch.json subset it read), the
committed staging records, and the chain log.
Run: python make_s5_status_manifests.py --state <chain_run_state dir> --out <json>"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCHES = ["nm360-s5fu_s1", "nm360-s5fu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = Path(a.state)
    st = json.loads((sd / "state.json").read_text())
    log = (sd / "chain.log").read_text().splitlines()
    out = {"note": "per-key status at each S5 sim batch's cutoff; PRELIMINARY", "batches": {}}
    for bid in BATCHES:
        b = st["batches"][bid]
        box_p = sd / f"boxfiles_{bid}.json"
        box = json.loads(box_p.read_text())
        cut = [ln[:22] for ln in log if re.match(rf"^\S+ \S+ PT  {re.escape(bid)}: boxfiles done$", ln)]
        (t,) = b["stage"]
        rec_p = HERE / "results" / b["stage"][t]["record"]
        rec = json.loads(rec_p.read_text())
        main = rec[next(k for k in rec if k not in ("git", "stamp"))]
        excl = {k.split("#")[0].split("/")[-2]: v for k, v in main["c_gate"]["excluded"].items()}
        sub = json.loads(box["launch_subset"])["runs"]
        rows = {}
        for r in sub:
            c = r["task"]
            rows[f"sim|{c}"] = {"run_key": r.get("key"), "state_at_cutoff": r.get("state"), "staged": c not in excl, "exclusion_reason": excl.get(c),
                                "counted": True}
        for x in box["facts"]["not_counted"]:
            if x["arm"] == "sim_attack@september":
                rows.setdefault(f"sim|{x['task']}", {"run_key": f"sim_attack@september/{x['task']}", "state_at_cutoff": x["state"], "staged": False,
                                                     "exclusion_reason": None, "counted": False})
        if b["kind"] == "fus":  # runs not done at this snapshot's cutoff: the ones a later batch of the family took (launcher state then: running)
            for later in BATCHES[BATCHES.index(bid) + 1:]:
                for m in st["batches"][later].get("members", []):
                    rows.setdefault(f"sim|{m}", {"run_key": f"sim_attack@september/{m}", "state_at_cutoff": "running (not done at the cutoff)",
                                                 "staged": False, "exclusion_reason": None, "counted": False, "taken_by": later})
        for x in box.get("counted_without_eval", []):
            rows[f"sim|{x['task']}"] |= {"staged": False, "exclusion_reason": "counted attempt without an .eval"}
        out["batches"][bid] = {"cutoff_pt": cut[-1] if cut else None, "launch": b["run"], "kind": b["kind"], "staged_set": b["stage"][t]["out"],
                               "staged_sha": b["stage"][t]["staged_inputs_sha256"], "stage_record": f"experiments/2026-10-14_q3_overnight_chain/results/{rec_p.name}",
                               "boxfiles_record_sha256": hashlib.sha256(box_p.read_bytes()).hexdigest(), "launch_states_at_cutoff": box["facts"]["states"],
                               "counts": {"rows": len(rows), "staged": sum(r["staged"] for r in rows.values()), "excluded": sum(bool(r["exclusion_reason"]) for r in rows.values()),
                                          "not_counted": sum(not r["counted"] for r in rows.values())}, "rows": dict(sorted(rows.items()))}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({b: {k: v for k, v in x.items() if k in ("cutoff_pt", "counts", "launch_states_at_cutoff")} for b, x in out["batches"].items()}, indent=1))


if __name__ == "__main__":
    main()
