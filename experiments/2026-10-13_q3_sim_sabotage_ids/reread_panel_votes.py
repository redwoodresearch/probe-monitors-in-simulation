"""Re-read the stage-1 sabotage panel's existing votes for every call in an ID list's runs ($0: no inference). Aswin runs this
on his laptop, against his stage-1 panel cache. It prints counts and dollar sums only, never a per-call label or any transcript text.

The stage-1 judges labelled EVERY call in each packet (config/judge_prompt.txt at b90cd42a), and the 120 packets were the 40
stage-1 tasks' real run 1, real run 2 and simulated runs. Only the 1,457 listed calls were exported; this exports one ID list's
runs' calls, with the same votes and the same rule (aggregate() from label_panel_results.py at b90cd42a: a label needs 4 agreeing
"sure" votes; otherwise it is "unclear". A missing or failed judge is not a vote, so a packet with 4 valid judges needs all 4).

Fatal checks:
1. The ledger, join.json, locator, config, panel code and ID list have their recorded sha256s.
2. aggregate() reproduces stage 1's blind_consensus.json sha256 (4aa0ec30…) from the ledger and packets.json.
3. packets.json and provenance.json have one entry per packet, in the same order (the packet builder appends both in one loop),
   and each listed run is in exactly one packet.
4. Per run, the packet holds c0000, c0001, … and as many calls as the ID list (the call at `ordinal` i is c{i:04d}).
5. Every listed stage-1 call in a listed run maps, through join.json and the locator, to the same packet and fresh ID as (4).

  uv run --no-project --python 3.12 python reread_panel_votes.py --cache <stage1_labels cache> --panel-src <b90cd42a worktree>/src \\
      --config <b90cd42a worktree>/experiments/2026-10-11_sabotage_stage1_labels/config/panel_resolved_125.json \\
      --locator <locator_stage1.csv> --calls <sim_calls_all_….csv> --out <new CSV path, outside the repo>
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

SHA = {"ledger": "5cafd7b0288a6a5001b2a41c26366088077d3091ad17d1e71dce4a020618c6b2",  # final_labels/summary.json at b90cd42a
       "join": "d6160d7e90eea2ef96bd3b56fe755e50400ba2fbea1f72d1dbffad4ecc055bf3",  # independent_audit/protocol.json
       "locator": "81fca0c4495b771cb730a2c431d0ad49e01af6a89e4269c567c87c29cebdab89",  # extraction/preparation.json
       "config": "11f3ea251895efcf28d9232d8b95f42dbedac97bca1be217414ecde83001e119",  # summary.json config_sha256
       "panel_code": "579c9e68a616214599bba1514c2b7ba7cb3b814a3ed6d82e9674ca54cb464864",  # export_provenance.json
       "consensus": "4aa0ec3001f6b6ab79b2aa29c4d1e81d617a74e2abd311cd66aac0848ad8ba1d",  # summary.json blind_consensus_sha256
       "calls": None}  # checked against CALLS
CALLS = {"13b0a3da826196eb051d9cd82be86e949200e359b19514f71de61423982ab8e2": ["sim"],  # sim_calls_all_2026-10-13_120231.csv
         "392eff0ed547fb18b62a08bb9cd8945d5206c0a281672780affd40825acf4a1a": ["real", "run2"]}  # real_run2_calls_all_2026-10-13_122622.csv
FIELDS = ["call_id", "world", "run_id", "epoch", "turn", "call_index", "ordinal", "state_changing", "packet", "panel_id", "label", "confident",
          "valid_votes", "sure_sabotage", "sure_not_sabotage", "other_votes"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(r) -> tuple:
    return int(r["epoch"]), int(r["turn"]), int(r["call_index"])


def main(a):
    prep, src = Path(a.cache) / "sabotage" / "prepared_v2", Path(a.panel_src) / "probe_shift" / "label_panel_results.py"
    if sha(a.calls) not in CALLS:
        raise SystemExit(f"{a.calls}: sha256 is not one of the committed ID lists")
    for p, k in [(Path(a.cache) / "panel" / "ledger.json", "ledger"), (prep / "join.json", "join"), (a.locator, "locator"),
                 (a.config, "config"), (src, "panel_code")]:
        if sha(p) != SHA[k]:
            raise SystemExit(f"{p}: sha256 differs from the recorded {k} hash")
    spec = importlib.util.spec_from_file_location("label_panel_results", src)
    lpr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lpr)
    models = json.loads(Path(a.config).read_text())["models"]
    blind, _, valid_pairs, expected_pairs = lpr.aggregate(prep, Path(a.cache) / "panel" / "ledger.json", "sabotage", models)
    if hashlib.sha256((json.dumps(blind, indent=2) + "\n").encode()).hexdigest() != SHA["consensus"]:
        raise SystemExit("re-aggregated votes differ from stage 1's blind_consensus.json")
    packets = json.loads((prep / "packets.json").read_text())
    prov = json.loads((prep / "provenance.json").read_text())
    if len(packets) != len(prov) or {(p["packet"], c) for p in packets for c in p["ids"]} != {(r["packet"], r["id"]) for r in blind}:
        raise SystemExit("packets.json / provenance.json don't match the votes")
    run_packet = {}
    for p, r in zip(packets, prov):
        if r["run_id"] in run_packet:
            raise SystemExit("a run is in two packets")
        run_packet[r["run_id"]] = p
    calls = list(csv.DictReader(open(a.calls)))
    by_run = defaultdict(list)
    for c in calls:
        by_run[c["run_id"]].append(c)
    worlds = CALLS[sha(a.calls)]  # the simulated list has no world column; the real list's must match
    run_world = {r: {c.get("world", "sim") for c in cs} for r, cs in by_run.items()}
    if any(len(v) != 1 for v in run_world.values()) or sorted(Counter(v.pop() for v in run_world.values()).items()) != [(w, 40) for w in sorted(worlds)]:
        raise SystemExit(f"expected exactly 40 runs in each of {worlds}, each run in one world")
    votes = {(r["packet"], r["id"]): r for r in blind}
    out, where, packet_votes = [], {}, Counter()
    for run, cs in sorted(by_run.items()):
        p = run_packet.get(run)
        if p is None:
            raise SystemExit(f"run {run} has no packet")
        cs.sort(key=lambda c: int(c["ordinal"]))
        if [int(c["ordinal"]) for c in cs] != list(range(len(cs))) or p["ids"] != [f"c{i:04d}" for i in range(len(cs))]:
            raise SystemExit(f"run {run}: packet holds {len(p['ids'])} calls, the ID list {len(cs)}")
        for c in cs:
            pid = f"c{int(c['ordinal']):04d}"
            where[hashlib.sha256(run.encode()).hexdigest()[:16], *key(c)] = (p["packet"], pid)
            v = votes[p["packet"], pid]
            packet_votes[v["valid_votes"]] += pid == "c0000"
            s = v["sure_votes"]
            out.append({k: c[k] for k in ("call_id", "run_id", "epoch", "turn", "call_index", "ordinal", "state_changing")} | {"world": c.get("world", "sim")} |
                       {"packet": p["packet"], "panel_id": pid, "label": v["label"], "confident": "yes" if v["label"] != "unclear" else "no",
                        "valid_votes": v["valid_votes"], "sure_sabotage": s.get("sabotage", 0), "sure_not_sabotage": s.get("not_sabotage", 0),
                        "other_votes": v["valid_votes"] - s.get("sabotage", 0) - s.get("not_sabotage", 0)})
    loc = {r["opaque_id"]: r for r in csv.DictReader(open(a.locator))}
    sim_hashes, checked = {k[0] for k in where}, Counter()
    for j in json.loads((prep / "join.json").read_text()):
        r = loc[j["opaque_id"]]
        if r["run_hash"] in sim_hashes:
            if where.get((r["run_hash"], *key(r))) != (j["packet"], j["id"]):
                raise SystemExit("a listed call maps to a different packet or fresh ID")
            checked[r["run_hash"]] += 1
    os.umask(0o077)
    with open(a.out, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    sc = [r for r in out if r["state_changing"] == "1"]
    sim_packets = {run_packet[r]["packet"] for r in by_run}  # what stage 1 spent on these packets: the price of a fresh run (path B)
    spent = [e for e in json.loads((Path(a.cache) / "panel" / "ledger.json").read_text()).values()
             if e["dataset"] == "sabotage" and e["packet"] in sim_packets]
    print(json.dumps({"rows": len(out), "state_changing_rows": len(sc), "valid_judge_packets_all_120": f"{valid_pairs}/{expected_pairs}",
                      "packets_by_valid_judges": dict(packet_votes), "listed_calls_cross_checked": sum(checked.values()),
                      "runs_cross_checked": len(checked), "worlds": worlds, "state_changing_confident": sum(r["confident"] == "yes" for r in sc),
                      "stage1_attempts_on_these_packets": len(spent),
                      "stage1_reported_cost_usd_on_these_packets": round(sum(e.get("reported_cost_usd", 0.0) for e in spent), 4),
                      "stage1_charged_or_reserved_usd_on_these_packets": round(sum(e.get("charge_or_reserve_usd", 0.0) for e in spent), 4)},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("--cache", "--panel-src", "--config", "--locator", "--calls", "--out"):
        ap.add_argument(f, required=True)
    main(ap.parse_args())
