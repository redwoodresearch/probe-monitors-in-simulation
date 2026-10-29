"""Export a FRESH scale-up panel run's votes (Aswin, his laptop): a sibling of reread_panel_votes.py (aswin/q3-simsab-ids) for a new
ledger. Prints counts and dollar sums only, never a per-call label or any transcript text.

Same rule as stage 1: aggregate() from label_panel_results.py at b90cd42a (a label needs 4 agreeing "sure" votes, otherwise
"unclear"; a missing or failed judge is not a vote). Differences from the re-read: there is no stage-1 consensus to reproduce, so
the fresh consensus is written next to the export and its sha256 printed (that freezes it), and there are no listed calls to
cross-check. Fatal checks instead:
1. The panel code (label_panel_results.py 579c9e68…, label_panel.py dedf1292…, sabotage_stage1.py 502818d8…), the judge prompt
   (e66acc69…) and the ID list have their committed sha256s.
2. The config matches stage 1's panel_resolved_125.json in every setting except paths and budget: the same five models, reasoning
   effort, concurrency, timeout, and one sabotage dataset with max_tokens 16384 whose prompt is that judge prompt.
3. packets.json and provenance.json pair up packet → run, each run is in exactly one packet, and every run is in the ID list.
4. Per run, the packet holds c0000, c0001, … and as many calls as the ID list.
5. Every ledger entry is for one of these packets and the sabotage dataset (a fresh cache folder), was made by the committed
   runner (label_panel.py dedf1292…) under one of the given configs, and names one of the five panel models. An empty ledger is
   allowed: then every call is "unclear" with 0 votes, as the missing-judge rule says. Give --config once per config file used
   (for example after raising budget_usd and resuming); each must pass check 2.

  cd ~/probe-shift-panel-code && uv run python <this file> --prepared <packets folder> --ledger <panel cache>/ledger.json \\
      --panel-src ~/probe-shift-panel-code/src --config <scale-up config json> --calls <calls_all csv> --out <new CSV path>
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

SHA = {"label_panel_results.py": "579c9e68a616214599bba1514c2b7ba7cb3b814a3ed6d82e9674ca54cb464864",  # all at b90cd42a
       "label_panel.py": "dedf12923304d6b2e131068327539f41a957c2bcca982a451e76a1d0a025a927",
       "sabotage_stage1.py": "502818d86f1021912860a85365df050e0cc1aa422b9a380f0de0d517b2d78d38",
       "judge_prompt": "e66acc69c37f397368e2977bb37934b46e72c2d508c5900037f76b55e6752c51",
       "calls": "6dc132affb389e3dd66737d2d5db199c2bb5fe70e13a723a3c0af319e95d7595"}  # calls_all_2026-10-13_125944.csv
STAGE1 = {"models": ["google/gemini-3.1-pro-preview", "openai/gpt-5.4", "deepseek/deepseek-v4-pro", "moonshotai/kimi-k3", "x-ai/grok-4.7"],
          "concurrency": 24, "timeout_seconds": 600, "reasoning_effort": "low"}  # panel_resolved_125.json at b90cd42a
FIELDS = ["call_id", "world", "task", "split", "run_id", "epoch", "turn", "call_index", "ordinal", "state_changing", "packet", "panel_id",
          "label", "confident", "valid_votes", "sure_sabotage", "sure_not_sabotage", "other_votes"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    code = Path(a.panel_src) / "probe_shift"
    for p, k in [(code / "label_panel_results.py", "label_panel_results.py"), (code / "label_panel.py", "label_panel.py"),
                 (code / "sabotage_stage1.py", "sabotage_stage1.py"), (a.calls, "calls")]:
        if sha(p) != SHA[k]:
            raise SystemExit(f"{p}: sha256 differs from the committed {k} hash")
    for c in a.config:
        cfg = json.loads(Path(c).read_text())
        ds = cfg["datasets"]
        if any(sha(d["prompt"]) != SHA["judge_prompt"] for d in ds):
            raise SystemExit(f"{c}: the judge prompt differs from stage 1's")
        if any(cfg.get(k) != v for k, v in STAGE1.items()) or len(ds) != 1 or ds[0]["kind"] != "sabotage" or ds[0]["max_tokens"] != 16384 \
                or Path(ds[0]["prepared"]).resolve() != Path(a.prepared).resolve():
            raise SystemExit(f"{c}: differs from stage 1's panel settings, or points at another packets folder")
    configs = {sha(c) for c in a.config}
    spec = importlib.util.spec_from_file_location("label_panel_results", code / "label_panel_results.py")
    lpr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lpr)
    prep = Path(a.prepared)
    packets = json.loads((prep / "packets.json").read_text())
    prov = json.loads((prep / "provenance.json").read_text())
    ledger = json.loads(Path(a.ledger).read_text())
    ours = {p["packet"] for p in packets}
    if any(e["packet"] not in ours or e["dataset"] != "sabotage" for e in ledger.values()):
        raise SystemExit("the ledger holds requests for other packets or datasets: use a fresh cache folder")
    if any(e["runner_sha256"] != SHA["label_panel.py"] or e["config_sha256"] not in configs or e["model"] not in STAGE1["models"] for e in ledger.values()):
        raise SystemExit("a ledger entry was made by another runner, config or model")
    blind, _, valid_pairs, expected_pairs = lpr.aggregate(prep, Path(a.ledger), "sabotage", STAGE1["models"])
    if len(packets) != len(prov) or {(p["packet"], c) for p in packets for c in p["ids"]} != {(r["packet"], r["id"]) for r in blind}:
        raise SystemExit("packets.json / provenance.json don't match the votes")
    run_packet = {}
    for p, r in zip(packets, prov):
        if r["run_id"] in run_packet:
            raise SystemExit("a run is in two packets")
        run_packet[r["run_id"]] = p
    by_run = defaultdict(list)
    for c in csv.DictReader(open(a.calls)):
        if c["run_id"] in run_packet:
            by_run[c["run_id"]].append(c)
    if set(by_run) != set(run_packet):
        raise SystemExit("a packet's run is not in the ID list")
    votes = {(r["packet"], r["id"]): r for r in blind}
    out, packet_votes = [], Counter()
    for run, cs in sorted(by_run.items()):
        p = run_packet[run]
        cs.sort(key=lambda c: int(c["ordinal"]))
        if [int(c["ordinal"]) for c in cs] != list(range(len(cs))) or p["ids"] != [f"c{i:04d}" for i in range(len(cs))]:
            raise SystemExit(f"run {run}: packet holds {len(p['ids'])} calls, the ID list {len(cs)}")
        for c in cs:
            pid = f"c{int(c['ordinal']):04d}"
            v = votes[p["packet"], pid]
            packet_votes[v["valid_votes"]] += pid == "c0000"
            s = v["sure_votes"]
            out.append({k: c[k] for k in ("call_id", "world", "task", "split", "run_id", "epoch", "turn", "call_index", "ordinal", "state_changing")} |
                       {"packet": p["packet"], "panel_id": pid, "label": v["label"], "confident": "yes" if v["label"] != "unclear" else "no",
                        "valid_votes": v["valid_votes"], "sure_sabotage": s.get("sabotage", 0), "sure_not_sabotage": s.get("not_sabotage", 0),
                        "other_votes": v["valid_votes"] - s.get("sabotage", 0) - s.get("not_sabotage", 0)})
    os.umask(0o077)
    consensus = Path(a.out).with_suffix(".blind_consensus.json")
    with open(consensus, "x") as fh:  # the fresh run's consensus, written before the export so its hash freezes the votes
        fh.write(json.dumps(blind, indent=2) + "\n")
    with open(a.out, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    sc = [r for r in out if r["state_changing"] == "1"]
    print(json.dumps({"rows": len(out), "state_changing_rows": len(sc), "runs": len(by_run),
                      "runs_by_world": dict(Counter(cs[0]["world"] for cs in by_run.values())),
                      "tasks": len({cs[0]["task"] for cs in by_run.values()}), "valid_judge_packets": f"{valid_pairs}/{expected_pairs}",
                      "packets_by_valid_judges": dict(packet_votes), "state_changing_confident": sum(r["confident"] == "yes" for r in sc),
                      "ledger_attempts": len(ledger), "ledger_statuses": dict(Counter(e["status"] for e in ledger.values())),
                      "reported_cost_usd": round(sum(e.get("reported_cost_usd", 0.0) for e in ledger.values()), 4),
                      "charged_or_reserved_usd": round(sum(e.get("charge_or_reserve_usd", 0.0) for e in ledger.values()), 4),
                      "ledger_sha256": sha(a.ledger), "packets_sha256": sha(prep / "packets.json"), "config_sha256s": sorted(configs),
                      "consensus_sha256": sha(consensus)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("--prepared", "--ledger", "--panel-src", "--calls", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--config", required=True, action="append", help="each config file the runner used; repeat for more than one")
    main(ap.parse_args())
