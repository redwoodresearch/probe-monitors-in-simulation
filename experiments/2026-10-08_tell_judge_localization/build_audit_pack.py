"""Reading pack for the visibility audit: 20 flags (10 simulated, 10 real; seed 0) plus every flag that cites a later
action, each with the judge's reason, the flagged call and the output as Q2's reader saw it (clipped if the harness
clipped it), and the outputs of any actions the reason cites. $0.

  uv run --with pandas python experiments/2026-10-08_tell_judge_localization/build_audit_pack.py --flags .run_cache/flags_swap_<stamp>.jsonl
Writes results/audit_pack_<stamp>.md (the audit itself is recorded by hand in config/visibility_audit.yaml).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
Q2 = Path("~/probe-shift-q2").expanduser()
sys.path.insert(0, str(ROOT / "experiments/2026-10-06_q2_impactful_call_transfer/shift"))
import examples as ex  # noqa: E402


def clip(text: str, n: int = 1800) -> str:
    text = text or ""
    return text if len(text) <= n else f"{text[:1200]}\n[... {len(text) - 1500} chars ...]\n{text[-300:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    ex.DATA = Q2 / ".run_cache/q2_shift"
    f = pd.read_json(HERE / args.flags, lines=True)
    f = f[f.in_range].reset_index(drop=True)
    sample = pd.concat([f[f.side == s].sample(args.n, random_state=0) for s in ("sim", "real")])
    later = f[f.cites_later & ~f.index.isin(sample.index)]
    pick = pd.concat([sample.assign(why="random"), later.assign(why="cites a later action")])
    runs = ex.load_runs(set(pick.key))
    md = [f"# Visibility audit pack ({stamp}): {len(sample)} random flags (10 sim, 10 real, seed 0) + {len(later)} citing later actions", ""]
    for i, (_, r) in enumerate(pick.iterrows(), 1):
        calls = runs[r.key]
        order = [sa for sa in sorted(calls) if calls[sa]["tool"] != "submit"]
        c = calls[(r.turn, r.call)]
        md += [f"## V{i:02d} ({r.why}) {r.side}, {r.env}, action {r.action_n} = turn {r.turn} call {r.call}, {r.tool}; "
               f"clipped for the reader: {bool(r.clipped)}; cites {list(r.cited)}", "",
               f"**Reason:** {r.reason}", "", f"**Command:** `{str(c['args'])[:400]}`", "", "**Output as the reader saw it:**", "",
               "```", clip(c["output"]).replace("```", "'''"), "```", ""]
        for n in r.cited:
            if 1 <= n <= len(order):
                cc = calls[order[n - 1]]
                md += [f"**Cited action {n}** (turn {order[n - 1][0]}), {cc['tool']}: `{str(cc['args'])[:200]}`", "", "```",
                       clip(cc["output"], 900).replace("```", "'''"), "```", ""]
    (HERE / "results" / f"audit_pack_{stamp}.md").write_text("\n".join(md))
    print(f"wrote results/audit_pack_{stamp}.md ({len(pick)} flags)")


if __name__ == "__main__":
    main()
