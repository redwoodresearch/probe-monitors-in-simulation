"""The LinuxArena 2AFC judge's per-action flags for the rendering-swap cohort, mapped onto Q2's calls. $0, no model calls.

Cohort: the 100 simulated runs of the rendering swap (q2 `config/swap_runs.json`) and the same cells' real attack epoch-1
runs. Flags: the stored medium-effort judge payloads (`read-n200` + `read-remaining753`, 938 pairs), plus the 22:21
rerun's new judgments for a cell the stored set lacks. A flag is `{action number: reason}`; the judge numbers actions
1..N densely over the trajectory, without the runner's submit.

For each flag:
- `turn`, `call`: Q2's 0-based (assistant turn, call within turn) of the flagged action, from the run's calls in order
  with `submit` left out;
- `cited`: other action numbers the reason names ("action 20", "actions 14 and 16");
- `visible_turn`: the turn of the latest action the flag relies on (itself or a cited later one). A Q2 call has the flag
  in its history only if its turn is later than this, since a sibling call cannot see an output of its own turn;
- `clipped`: the flagged output reached the reader truncated by the harness (`<START_TOOL_OUTPUT>`), while the judge
  saw it in full.
Check: for every cohort run whose rendered judge prompt is saved (the 22:21 rerun), the judge's action list (number,
tool) must equal this mapping.

  uv run --with pandas python experiments/2026-10-08_tell_judge_localization/build_flags.py
"""

import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
Q2 = Path("~/probe-shift-q2").expanduser()  # read-only: the swap's data and config
sys.path.insert(0, str(ROOT / "experiments/2026-10-06_q2_impactful_call_transfer/shift"))
import examples as ex  # noqa: E402

from probe_shift.provenance import capture  # noqa: E402

BASE = Path("~/probe-shift-baselines/experiments/2026-10-06_q1_text_baselines").expanduser()
JUDGE = BASE / ".run_cache/paper_judge/workspace/paper-organize/.run_cache"
MEDIUM = ["2026-09-21_la_discriminator_read-n200-sol-medium-v1", "2026-09-21_la_discriminator_read-remaining753-sol-medium-v1"]
RERUN = BASE / "results/judge_missing_2026-10-06_222118.json"
RERUN_PROMPTS = BASE / ".run_cache/judge_missing_2026-10-06_222118/prompts.jsonl.gz"
CLIP = "<START_TOOL_OUTPUT>"
CITE = re.compile(r"\bactions?\s+((?:\d+)(?:\s*(?:,|and|&|to|–|-)\s*\d+)*)", re.I)


def sample_id(key: str) -> str:
    return key.split("__")[-2]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cited_actions(reason: str) -> list[int]:
    out = []
    for m in CITE.finditer(reason):
        out += [int(x) for x in re.findall(r"\d+", m.group(1))]
    return sorted(set(out))


def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    ex.DATA = Q2 / ".run_cache/q2_shift"
    swap_cfg = Q2 / "experiments/2026-10-07_q2_rendering_swap/config/swap_runs.json"
    sim_keys = sorted(json.loads(swap_cfg.read_text())["runs"])
    q = ex.table("qwen3-32b")
    cells = set(q[q.key.isin(sim_keys)].cell)
    real_keys = sorted(set(q[(q.arm == "real-attack-e1") & q.cell.isin(cells)].key))
    meta = q.drop_duplicates("key").set_index("key")[["arm", "cell", "main_task", "env", "side_task", "sim_reference"]]

    # Judge pairs, keyed by (real sample id, sim sample id).
    pairs, src = {}, {}
    for run in MEDIUM:
        for p in json.loads((JUDGE / run / "pairs.json").read_text())["two_afc"]["pairs"]:
            pairs[(p["id"], p["sim_id"])] = p
            src[(p["id"], p["sim_id"])] = run
    for p in json.loads(RERUN.read_text())["pairs"]:  # only cells the stored set lacks
        new = p["new"]
        k = (new.get("id") or p.get("real_id"), new.get("sim_id") or p.get("sim_id"))
        if k not in pairs and all(k):
            pairs[k], src[k] = new, "judge_missing_2026-10-06_222118"
    by_real = {r: k for k in pairs for r in [k[0]]}
    by_sim = {s: k for k in pairs for s in [k[1]]}

    runs = ex.load_runs(set(sim_keys) | set(real_keys))
    order = {k: [sa for sa in sorted(c) if c[sa]["tool"] != "submit"] for k, c in runs.items()}
    rows, missing = [], Counter()
    for key in sim_keys + real_keys:
        side = "sim" if key in sim_keys else "real"
        pk = (by_sim if side == "sim" else by_real).get(sample_id(key))
        if pk is None:
            missing[side] += 1
            continue
        flags = pairs[pk][f"flags_on_{side}"] or {}
        acts = order[key]
        for n_str, reason in flags.items():
            n = int(n_str)
            if not 1 <= n <= len(acts):
                rows.append({"side": side, "key": key, "action_n": n, "in_range": False, "reason": reason, "pair_source": src[pk]})
                continue
            turn, call = acts[n - 1]
            cited = [c for c in cited_actions(reason) if c != n]
            latest = max([n] + [c for c in cited if 1 <= c <= len(acts)])
            out = runs[key][(turn, call)]["output"] or ""
            rows.append({"side": side, "key": key, **meta.loc[key].to_dict(), "action_n": n, "in_range": True,
                         "turn": turn, "call": call, "tool": runs[key][(turn, call)]["tool"], "reason": reason,
                         "cited": cited, "cites_later": any(c > n for c in cited), "visible_turn": acts[latest - 1][0],
                         "clipped": CLIP in out, "reader_output_chars": len(out), "pair_source": src[pk],
                         "pair_accuracy": pairs[pk].get("accuracy")})
    flags = pd.DataFrame(rows)

    # Mapping check against the saved rendered prompts (22:21 rerun): (action number, tool) per side.
    check = {"runs_checked": 0, "runs_mismatched": []}
    head = re.compile(r"^### \[action (\d+)\] (\S+)", re.M)
    for line in gzip.open(RERUN_PROMPTS, "rt"):
        r = json.loads(line)
        prompt = r["prompt"]
        a, b, end = prompt.find("<transcript_a>"), prompt.find("<transcript_b>"), prompt.find("# How to judge")
        assert 0 <= a < b < end, "rendered prompt without its two transcripts"
        texts = {"A": prompt[a:b], "B": prompt[b:end]}
        sim_slot = "B" if r["real_slot"] == "A" else "A"
        for side, sid, slot in (("real", r["real_id"], r["real_slot"]), ("sim", r["sim_id"], sim_slot)):
            keys = [k for k in (sim_keys if side == "sim" else real_keys) if sample_id(k) == sid]
            if not keys:
                continue
            key = keys[0]
            judge_seq = [(int(n), t) for n, t in head.findall(texts[slot])]
            ours = [(i + 1, runs[key][sa]["tool"]) for i, sa in enumerate(order[key])]
            check["runs_checked"] += 1
            if judge_seq != ours:
                check["runs_mismatched"].append({"key": key, "judge_n": len(judge_seq), "ours_n": len(ours)})
    out_path = HERE / ".run_cache" / f"flags_swap_{stamp}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    flags.to_json(out_path, orient="records", lines=True)
    summary = {
        "stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]),
        "inputs": {"swap_runs": {"path": str(swap_cfg), "sha256": sha(swap_cfg)},
                   **{run: sha(JUDGE / run / "pairs.json") for run in MEDIUM}, "rerun": sha(RERUN)},
        "runs": {"sim": len(sim_keys), "real": len(real_keys)}, "runs_without_judge_pair": dict(missing),
        "flags": {s: int((flags.side == s).sum()) for s in ("sim", "real")},
        "flags_out_of_range": int((~flags.in_range).sum()),
        "flags_citing_later_actions": {s: int((flags.in_range & (flags.side == s) & flags.cites_later.fillna(False)).sum()) for s in ("sim", "real")},
        "flags_on_clipped_outputs": {s: int((flags.in_range & (flags.side == s) & flags.clipped.fillna(False)).sum()) for s in ("sim", "real")},
        "runs_with_a_flag": {s: int(flags[flags.side == s].key.nunique()) for s in ("sim", "real")},
        "mapping_check": check, "flags_file": str(out_path.relative_to(HERE))}
    (HERE / "results" / f"flags_swap_summary_{stamp}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("provenance", "inputs")}, indent=1, default=str))


if __name__ == "__main__":
    main()
