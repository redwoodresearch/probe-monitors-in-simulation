"""Dates in tool outputs, per arm (Modal CPU, pennies; PREDICTIONS.md first). Is there a date leak between arms?

Reads the staged reader messages of the v4 inputs (original text, five arms incl. real and simulated honest) and finds
every ISO date (`YYYY-MM-DD`, with `HH:MM` if present) and log-style date (`DD/Mon/YYYY:HH:MM`) in tool outputs.
Per arm: counts by year, by year-month-day and by hour; per run, its most common date, and how the arms' most common
dates overlap. A seeded sample (seed 0, 5 runs per arm) records run ID, step, action and the matched strings in
results/; the raw output lines around them go to data/ (raw text, not results/). Step = the agent turn's index;
action = the call's index in its turn (as the swap's helpers count).

  uv run modal run experiments/2026-10-08_q2_date_histogram/run_date_histogram.py
"""
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE
INPUTS = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"
META = "2026-10-06_la_segment-end-v4_qwen3-32b_alllayers_v1"  # its extraction: which outputs end inside the 32,768-token cut
SEED = 0
app = modal.App("probe-shift_aswin-surya_sub-q2-dates_kill-2026-11-08")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
image = modal.Image.debian_slim(python_version="3.12")
MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
ISO = re.compile(r"(?<!\d)((?:19|20)\d{2})-(\d{2})-(\d{2})(?:[T ](\d{2}):\d{2})?(?!\d)")
LOG = re.compile(r"(?<!\d)(\d{2})/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/((?:19|20)\d{2}):(\d{2}):\d{2}")


def dates(text: str) -> list[tuple[str, str | None]]:
    """(YYYY-MM-DD, HH or None) for every date in `text`."""
    out = [(f"{y}-{mo}-{d}", h) for y, mo, d, h in ISO.findall(text)]
    return out + [(f"{y}-{MON[mo]:02d}-{d}", h) for d, mo, y, h in LOG.findall(text)]


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=1800, retries=0)
def histogram(visible_only: bool) -> dict:
    """visible_only: count only tool outputs whose last token the extraction kept (inside the reader's token cut)."""
    import hashlib

    years, days, hours = (defaultdict(Counter) for _ in range(3))
    run_mode, hits, tied = defaultdict(dict), defaultdict(dict), defaultdict(list)
    runs, check = Counter(), Counter()  # check: in untruncated runs every tool output should be in the extraction's positions
    for shard in sorted((Path("/acts") / INPUTS / "inputs").glob("shard_*.jsonl")):
        for it in map(json.loads, shard.read_text().splitlines()):
            arm, step, per_run = it["arm"], -1, Counter()
            runs[arm] += 1
            if visible_only:
                k = it["key"]
                name = k if len(k.encode()) <= 240 else f"{k[:150]}__{hashlib.sha1(k.encode()).hexdigest()[:12]}"  # the extractor's file stem
                meta = json.loads((Path("/acts") / META / "meta" / arm / f"{name}.json").read_text())
                seen = {(q["step"], q["action"]) for q in meta["positions"] if q["kind"] == "output_end" and q.get("kept", True)}
            action = 0
            for m in it["messages"]:
                if m["role"] == "assistant":
                    step, action = step + 1, 0
                    continue
                if visible_only and m["role"] == "tool" and meta["truncated"] in (False, "False"):
                    check["untruncated outputs"] += 1
                    check["...of them missing from the positions"] += (step, action) not in seen
                    if (step, action) not in seen:
                        c = m["content"] if isinstance(m["content"], str) else ""
                        check["...missing and empty"] += not c.strip()
                        check["...dates in the missing ones"] += len(dates(c))
                if m["role"] != "tool" or not isinstance(m["content"], str) or (visible_only and (step, action) not in seen):
                    action += m["role"] == "tool"
                    continue
                for day, hour in dates(m["content"]):
                    years[arm][day[:4]] += 1
                    days[arm][day] += 1
                    hours[arm][hour or "none"] += 1
                    per_run[day] += 1
                    h = hits[arm].setdefault(it["key"], [])
                    if len(h) < 5:  # enough for the seeded sample
                        h.append((step, action, day, hour, m["content"]))
                action += 1
            if per_run:
                top = per_run.most_common(2)
                run_mode[arm][it["key"]] = top[0][0]  # on a tie, Counter keeps the first-seen date
                if len(top) > 1 and top[1][1] == top[0][1]:
                    tied[arm].append(it["key"])
    modes = {a: set(v.values()) for a, v in run_mode.items()}
    overlap = {f"{a} vs {b}": {f"share of {a} runs whose most common date is some {b} run's":
                               round(sum(d in modes[b] for d in run_mode[a].values()) / max(len(run_mode[a]), 1), 4)}
               for a in modes for b in modes if a != b}
    rng, sample, raw = random.Random(SEED), {}, {}
    for arm in sorted(hits):
        keys = rng.sample(sorted(hits[arm]), min(5, len(hits[arm])))
        sample[arm] = [{"key": k, "hits": [{"step": s, "action": a, "date": d, "hour": h} for s, a, d, h, _ in hits[arm][k][:5]]} for k in keys]
        raw[arm] = [{"key": k, "step": s, "action": a, "lines": [ln[:200] for ln in c.split("\n") if d[:4] in ln][:3]}
                    for k in keys for s, a, d, h, c in hits[arm][k][:5]]
    return {"inputs": INPUTS, "visible only": visible_only, "position check": dict(check), "runs per arm": dict(runs), "runs with a date": {a: len(v) for a, v in run_mode.items()},
            "dates per arm": {a: sum(c.values()) for a, c in years.items()},
            "by year": {a: dict(sorted(c.items())) for a, c in years.items()},
            "by day (all)": {a: dict(sorted(c.items())) for a, c in days.items()},
            "by hour": {a: dict(sorted(c.items())) for a, c in hours.items()},
            "most common day per run": {a: dict(v) for a, v in run_mode.items()},
            "modal-day overlap": overlap, "runs whose most common date is tied": {a: sorted(v) for a, v in tied.items()},
            "sample (seed 0)": sample, "raw": raw}


@app.local_entrypoint()
def main(visible_only: bool = True):
    res = histogram.remote(visible_only)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    raw = ROOT / "data" / "date_histogram" / f"raw_lines_{stamp}.json.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw, "wt") as f:
        json.dump(res.pop("raw"), f)
    out = HERE / "results" / f"dates{'_visible' if visible_only else ''}_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res | {"raw lines (not in git)": str(raw.relative_to(ROOT))}, indent=1))
    print(out)
    for a in sorted(res["by year"]):
        n = res["dates per arm"][a]
        print(a, "runs", res["runs per arm"][a], "with a date", res["runs with a date"].get(a, 0), "dates", n,
              "years:", {y: f"{c / n:.1%}" for y, c in res["by year"][a].items() if c / n >= 0.005})
    print(json.dumps(res["modal-day overlap"], indent=1))
    print("runs with a date / of them with a tied most common date:",
          {a: (res["runs with a date"][a], len(res["runs whose most common date is tied"].get(a, []))) for a in sorted(res["runs with a date"])})
