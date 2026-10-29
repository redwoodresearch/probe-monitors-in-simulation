"""Amendment 1 (RULE.md): the 3 named movers from q2's matched dig, each checked against the 437 fix-full simulated
honest runs, with its L63 score_mean and rank (same ordering as draw_read10.py), caught_1pct, real counterpart (same rule)
and whether it is also in the random 10. Numbers and keys only; no transcript text.

  uv run --no-project --python 3.12 --with numpy==2.3.3 python experiments/2026-10-13_claimB_read10/movers_read10.py \\
      results/draw_2026-10-13_170032.json
"""

import csv
import hashlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import draw_read10 as D

HERE = Path(__file__).parent
Q2 = ("bfe35c2c", "experiments/2026-10-13_q2_fixfull/results")
DIGS = (
    "matched_dig_qwen3_full_POSTHOC_2026-10-13_165050.json",
    "matched_dig_qwen3_N_POSTHOC_2026-10-13_165050.json",
    "matched_dig_seed_full_POSTHOC_2026-10-13_165049.json",
    "matched_dig_seed_N_POSTHOC_2026-10-13_165049.json",
)
MOVERS = (
    "policy_factory_pattern__policy_factory_pattern:9f838d__1",
    "extract_video_query_builder__extract_video_query_builder:d48abf__1",
    "message_envelope_wrapper__message_envelope_wrapper:39545a__1",
)


def main(draw_path):
    draw = json.loads((HERE / draw_path).read_text())
    raw = D.git_show(*D.SCORES)
    assert hashlib.sha256(raw).hexdigest() == draw["scores"]["sha256"]
    rows = [
        r
        for r in csv.DictReader(io.StringIO(raw.decode()))
        if int(r["layer"]) == D.LAYER
    ]
    for r in rows:
        r["score_mean"] = float(r["score_mean"])
    ranked = sorted(rows, key=lambda r: (-r["score_mean"], r["run_key"]))
    rank = {r["run_key"]: i + 1 for i, r in enumerate(ranked)}
    by_key = {r["run_key"]: r for r in rows}
    real_raw = D.git_show(*D.REAL)
    assert hashlib.sha256(real_raw).hexdigest() == D.REAL_SHA
    by_task = {}
    for r in sorted(
        (r for r in json.loads(real_raw) if r["arm"] == "real-honest"),
        key=lambda r: r["key"],
    ):
        by_task.setdefault(r["main_task"], r["key"])
    drawn = {e["run_key"] for e in draw["drawn"]}
    sources = {}
    for name in DIGS:
        text = D.git_show(Q2[0], f"{Q2[1]}/{name}").decode()
        for m in MOVERS:
            if re.search(re.escape(m), text):
                sources.setdefault(m, []).append(name)
    out = {
        "label": "read10 Amendment 1 movers (POST HOC, descriptive); RULE.md in this folder",
        "q2_source": Q2,
        "draw": draw_path,
        "movers": [],
    }
    for m in MOVERS:
        r = by_key.get(m)
        out["movers"].append(
            {
                "run_key": m,
                "in_437": r is not None,
                "named_in": sources.get(m, []),
                "main_task": r["main_task"] if r else None,
                "l63_score_mean": r["score_mean"] if r else None,
                "l63_rank_of_437": rank.get(m),
                "caught_1pct": float(r["caught_1pct"]) if r else None,
                "counterpart": by_task.get(r["main_task"], "no counterpart")
                if r
                else None,
                "also_in_random_10": m in drawn,
            }
        )
    out |= {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule_sha256": hashlib.sha256((HERE / "RULE.md").read_bytes()).hexdigest(),
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"movers_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(
        path.name,
        [
            (
                x["run_key"][:30],
                x["in_437"],
                x["l63_rank_of_437"],
                x["also_in_random_10"],
            )
            for x in out["movers"]
        ],
    )


if __name__ == "__main__":
    main(sys.argv[1])
