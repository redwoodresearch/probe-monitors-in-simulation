"""Step 0 of the eval-awareness plan: audit Stage 1's contrast pairs ($0, saved data only, 2026-10-10).

Reads the exact set file Stage 1's extraction and fits used (sets_2026-10-07_212051.jsonl, checked against the sha256
in experiments/2026-10-07_stage1_awareness/results/sets_manifest_2026-10-08_103826.json). For every awareness family:
  - which messages differ between a pair's two sides (system, user, a tool output, a tool call), and whether the final
    call is byte-identical;
  - surface separability: how well total length alone, and the single best word's presence, tell the two sides
    apart (AUROC over items; 0.5 = no signal);
  - task overlap between families (the same task id in two families means a family-held-out score is not
    task-held-out).
A seeded random draw of pairs per family is written with the text that differs on each side. The dialogues are
model-written text, so the draw goes to the git-ignored .run_cache; results/ gets numbers and pair ids only.

  uv run python experiments/2026-10-10_eval_awareness_audit/analyze_stage1_pairs.py
"""

import hashlib
import json
import random
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / ".run_cache" / "eval_awareness_audit" / "sets_2026-10-07_212051.jsonl"
SHA = "6dfe1106229c348fce4d31ca49d0672b253b17e91ab691c6f620d5f5c6669925"  # the manifest's sha256 for this file
SEED, PER_FAMILY = 20261010, 3
WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(a random label-1 value > a random label-0 value), ties half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def parts(item: dict) -> list[tuple[str, str]]:
    """The dialogue as (place, text) pieces: system, user, each tool call's arguments, each tool output."""
    out, k = [], 0
    for m in item["messages"]:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                out.append((f"call {k}", json.dumps(c["function"], sort_keys=True)))
                k += 1
        else:
            out.append((m["role"] if m["role"] != "tool" else f"output {k - 1}", m["content"] or ""))
    return out


def main():
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError(f"{DATA.name} does not match the manifest's sha256")
    items = [json.loads(l) for l in raw.decode().splitlines()]
    pairs = defaultdict(dict)
    for it in items:
        if it["probe"] != "impact":
            pairs[(it["set"], it["family"], it["pair"])][it["label"]] = it
    fam = defaultdict(list)
    for (set_, family, pid), sides in pairs.items():
        if set(sides) != {0, 1}:
            raise ValueError(f"{pid}: sides {sorted(sides)}")
        fam[f"{set_} / {family}"].append((pid, sides[1], sides[0]))
    rng = random.Random(SEED)
    report, draw = {}, []
    tasks = {f: {a["task"] for _, a, _ in v} for f, v in fam.items()}
    for f, v in sorted(fam.items()):
        where, final_same = defaultdict(int), 0
        len1, len0 = [], []
        for pid, a, b in v:
            pa, pb = parts(a), parts(b)
            if len(pa) != len(pb):
                where["different number of messages"] += 1
            for (wa, ta), (wb, tb) in zip(pa, pb):
                if ta != tb:
                    where[wa if wa == wb else f"{wa} vs {wb}"] += 1
            final_same += pa[-2][1] == pb[-2][1]  # the readout call (the last call; its "ok" output follows)
            len1.append(sum(len(t) for _, t in pa))
            len0.append(sum(len(t) for _, t in pb))
        words = defaultdict(lambda: [0, 0])
        for _, a, b in v:
            for side, it in ((0, a), (1, b)):
                for w in set(x.lower() for x in WORD.findall(" ".join(t for _, t in parts(it)))):
                    words[w][side] += 1
        n = len(v)
        best = sorted(((max(c1, c0) / n + 1 - min(c1, c0) / n) / 2, w, c1, c0) for w, (c1, c0) in words.items())[-3:][::-1]
        report[f] = {"pairs": n, "where the sides differ (pairs)": dict(where), "final call byte-identical (pairs)": final_same,
                     "length alone, AUROC (label 1 over 0)": round(auroc(len1, len0), 3),
                     "label-1 side longer (share of pairs)": round(sum(x > y for x, y in zip(len1, len0)) / n, 3),
                     "mean length, label 1 / label 0 (chars)": [round(sum(len1) / n), round(sum(len0) / n)],
                     "best single words by presence (AUROC, in label-1 / label-0 items)": [
                         {"word": w, "auroc": round(s, 3), "in label 1": c1, "in label 0": c0} for s, w, c1, c0 in best],
                     "tasks": len(tasks[f])}
        for pid, a, b in rng.sample(v, min(PER_FAMILY, n)):
            pa, pb = parts(a), parts(b)
            diff = [(wa, ta, tb) for (wa, ta), (_, tb) in zip(pa, pb) if ta != tb]
            draw.append({"family": f, "pair": pid, "task": a["task"], "differs": diff})
    fams = sorted(tasks)
    overlap = {f"{x} & {y}": len(tasks[x] & tasks[y]) for i, x in enumerate(fams) for y in fams[i + 1:] if tasks[x] & tasks[y]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    ex = ROOT / ".run_cache" / "eval_awareness_audit" / f"stage1_pair_draw_{stamp}.json"
    ex.write_text(json.dumps({"seed": SEED, "per_family": PER_FAMILY, "draw": draw}, indent=1, ensure_ascii=False))
    out = {"git": git, "data": DATA.name, "data_sha256": SHA, "seed": SEED, "per_family": PER_FAMILY,
           "draw (pair ids; text in .run_cache)": [d["pair"] for d in draw], "draw_file": ex.name,
           "draw_sha256": hashlib.sha256(ex.read_bytes()).hexdigest(), "families": report, "shared task ids between families": overlap}
    path = HERE / "results" / f"stage1_pairs_audit_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items() if k != "families") +
                    ',\n "families": {\n' + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in report.items()) + "\n }\n}\n")
    json.loads(path.read_text())  # one family per line, still valid JSON
    print(path.read_text())
    print(f"-> {path} (draw: {ex})")


if __name__ == "__main__":
    main()
