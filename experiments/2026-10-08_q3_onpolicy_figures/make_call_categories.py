"""Per call of Phase 2's attack runs: the labeler's label and primary category (eb014307), keyed by request id and call
index, for joining with per-call probe scores. $0, local; no text out.

Same calls as run_replay_readouts.py: the acting model's events, the counted attempt per run, terminal submits and calls
to tools not offered excluded and counted; a call's index is its position in its completion.

  PYTHONPATH=<#163 src> uv run --no-project --python 3.14 python make_call_categories.py --run-dir <run folder> \
      --arms attack,sim_attack --unoffered str_replace_editor,str_replace
"""

import argparse
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.calls import IMPACTFUL, fingerprint, label_call

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--unoffered", default="")
    ap.add_argument("--acting-model", default="vllm/Qwen/Qwen3.5-27B")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    launch = json.loads((rd / "launch.json").read_text())
    rows, skipped = [], Counter()
    for r in launch["runs"]:
        if r["arm"] not in args.arms.split(","):
            continue
        counted = [a for a in r.get("attempts") or [] if not a.get("redrawn_because") and a.get("kind") != "rerun"]
        if not counted:  # a selected run with no counted attempt (e.g. never started) would leave the table incomplete
            raise ValueError(f"{r['key']}: no counted attempt")
        for a in counted[:1]:
            evs = sorted((rd / a["log_dir"].split(f"/{rd.name}/", 1)[1]).glob("*.eval"))
            if len(evs) != 1:  # a missing or extra log would make an incomplete table look complete
                raise ValueError(f"{r['key']}: {len(evs)} .eval files for the counted attempt; one is expected")
            for ev in evs:
                with zipfile.ZipFile(ev) as z:
                    names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
                    if len(names) != 1:
                        raise ValueError(f"{ev}: {len(names)} samples; one is expected")
                    s = json.loads(z.read(names[0]))
                att = s.get("attachments") or {}
                for e in s.get("events") or []:
                    if e.get("event") != "model" or e.get("model") != args.acting_model:
                        continue
                    irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
                    tcs = ((((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])
                    for j, tc in enumerate(tcs):
                        name, a_ = tc.get("function"), tc.get("arguments") or {}
                        a_ = {k: (att.get(v.removeprefix("attachment://"), v) if isinstance(v, str) and v.startswith("attachment://") else v)
                              for k, v in a_.items()}
                        if name == "submit" or name in args.unoffered.split(","):
                            skipped["terminal submit" if name == "submit" else f"tool not offered: {name}"] += 1
                            continue
                        lab = label_call(name, a_)
                        rows.append({"arm_dir": ev.relative_to(rd).parts[0], "task": r["task"], "irid": irid, "call": j, "tool": name,
                                     "label": lab["label"], "category": lab["category"], "categories": lab["categories"],
                                     "net_method": lab["net_method"], "net_dest": lab["net_dest"], "net_sends_data": lab["net_sends_data"],
                                     # Q2's localhost_reads_read_only (run_q2_probes.py y_of): read-only under that robustness rule
                                     "localhost_read_only": {c for c in lab["categories"] if c in IMPACTFUL} == {"network"}
                                     and lab["net_method"] in ("GET", "HEAD") and lab["net_dest"] == "localhost" and not lab["net_sends_data"]})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"phase2_call_categories_{rd.name}_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"labeler": fingerprint(), "run_dir": str(rd.resolve()), "arms": args.arms, "unoffered": args.unoffered,
                             "acting_model": args.acting_model, "skipped": dict(skipped), "rows": rows}))
    print("labeler", fingerprint(), "| skipped", dict(skipped))
    arms = sorted({x["arm_dir"] for x in rows})
    cats = sorted({(x["label"], x["category"]) for x in rows})
    print("calls by category (label / category):", " | ".join(arms))
    for lab, cat in cats:
        print(f"  {lab:9s} {cat:22s}", "  ".join(f"{sum(x['arm_dir'] == a and x['category'] == cat and x['label'] == lab for x in rows):5d}" for a in arms),
              "| tasks:", "  ".join(f"{len({x['task'] for x in rows if x['arm_dir'] == a and x['category'] == cat and x['label'] == lab}):3d}" for a in arms))
    net = [x for x in rows if x["category"] == "network"]
    print("network calls by (method, destination, sends data):")
    for a in arms:
        print(f"  {a}:", dict(Counter((x["net_method"], x["net_dest"], x["net_sends_data"]) for x in net if x["arm_dir"] == a).most_common()))
        print(f"  {a}: localhost GET or HEAD without data: {sum(x['net_method'] in ('GET', 'HEAD') and x['net_dest'] == 'localhost' and not x['net_sends_data'] for x in net if x['arm_dir'] == a)}"
              f" of {sum(x['arm_dir'] == a for x in net)} network calls")
    print("localhost_reads_read_only would relabel:", {a: sum(x["arm_dir"] == a and x["localhost_read_only"] for x in rows) for a in arms})
    print("->", path)


if __name__ == "__main__":
    main()
