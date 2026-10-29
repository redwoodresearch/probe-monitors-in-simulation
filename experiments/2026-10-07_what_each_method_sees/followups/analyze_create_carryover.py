"""Probe carry-over on byte-identical `create` outputs: per-output AUROC (real e1 vs simulated) by position in the run,
at levels 0 and 7, held-out environments, for outputs that are exactly the editor's "File created successfully at:
<path>" message (the path differs; nothing else does). Saved scores only (../.run_cache/joined.npz); 95% intervals by
bootstrapping main tasks.

  uv run python experiments/2026-10-07_what_each_method_sees/followups/analyze_create_carryover.py
"""

import gzip
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from probe_shift.provenance import capture
from probe_shift.scoring import auroc_ci

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parent.parent
TEMPLATE = re.compile(r"\AFile created successfully at: \S+\s*\Z")
BUCKETS = ((1, 10), (11, 25), (26, 10_000))


def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    cfg = yaml.safe_load((EXP / "config/what_sees.yaml").read_text())
    T = dict(np.load(EXP / ".run_cache/joined.npz"))
    text = {(r["key"], r["step"], r["action"]): r["readout_text"]
            for r in map(json.loads, gzip.open(Path(cfg["staging_dir"]).expanduser() / "outputs.jsonl.gz", "rt")) if r["kept"]}
    key, arm = T["key"], T["arm"]
    order = np.lexsort((T["action"], T["step"], key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    ordinal = np.empty(len(key), int)
    ordinal[order] = np.arange(len(key)) - np.maximum.accumulate(np.where(first, np.arange(len(key)), 0)) + 1
    exact = np.array([bool(TEMPLATE.match(text[(k, int(s), int(a))])) for k, s, a in zip(key, T["step"], T["action"])])
    out = {"stamp": stamp, "provenance": capture(ROOT, [Path(__file__)]), "rows": {}}
    md = [f"# Carry-over on exact 'File created successfully at:' outputs, held-out environments ({stamp})", "",
          "| level | position | real e1 | simulated | TF-IDF | probe |", "|---|---|---|---|---|---|"]
    for level in (0, 7):
        sc = {m: T[f"{m}_L{level}_env_real_vs_sim"] for m in ("tfidf", "probe")}
        base = np.isin(arm, ["real-attack-e1", "sim-attack"]) & exact & ~np.isnan(sc["tfidf"]) & ~np.isnan(sc["probe"])
        for lo, hi in BUCKETS + ((1, 10_000),):
            m = np.flatnonzero(base & (ordinal >= lo) & (ordinal <= hi))
            y = arm[m] == "sim-attack"
            r = {mt: auroc_ci(y, sc[mt][m], T["main_task"][m]) for mt in sc}
            name = f"{lo}-{hi if hi < 10_000 else 'end'}" if (lo, hi) != (1, 10_000) else "all"
            out["rows"][f"L{level} {name}"] = {"n_real": int((~y).sum()), "n_sim": int(y.sum()), **r}
            f = lambda v: f"{v['auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
            md.append(f"| {level} | {name} | {(~y).sum()} | {y.sum()} | {f(r['tfidf'])} | {f(r['probe'])} |")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"create_carryover_{stamp}.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / f"create_carryover_{stamp}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
