"""One task file per box for tonight's +311 launch (RULE.md, "Overnight +311 launch"): the box's stage files
concatenated in stage order (pilot, est, sel), cells verbatim; its config = the pilot-stage config with this task file
and cap = the sum of the box's stage caps. Writes config/tasks_il_q<k>_all.json, config/pilot_il_q<k>_all.yaml and
config/tasks_il_all_manifest.json.

  python3 experiments/2026-10-15_q3_interleaved_run3_sim2/concat_stages.py
"""

import hashlib
import json
import re
from pathlib import Path

CONFIG = Path(__file__).parent / "config"
STAGES = ("pilot", "est", "sel")
manifest = {}
for k in "1234":
    docs = [
        json.loads((CONFIG / f"tasks_il_q{k}_{s}.json").read_text()) for s in STAGES
    ]
    cells = [c for d in docs for c in d["cells"]]
    full = json.loads((CONFIG / f"tasks_il_q{k}.json").read_text())["cells"]
    assert sorted(c["key"] for c in cells) == sorted(c["key"] for c in full), k
    doc = {kk: v for kk, v in docs[0].items() if kk not in ("cells", "stage")} | {
        "stages": list(STAGES),
        "cells": cells,
    }
    out = CONFIG / f"tasks_il_q{k}_all.json"
    assert not out.exists(), out
    out.write_text(json.dumps(doc, indent=1) + "\n")
    caps = [
        int(
            re.search(
                r"\n  cap_usd: (\d+)", (CONFIG / f"pilot_il_q{k}_{s}.yaml").read_text()
            ).group(1)
        )
        for s in ("pilot", "est")
    ]  # sel is empty for every +311 box
    assert not docs[2]["cells"], f"q{k}: a non-empty selection stage"
    cfg = (CONFIG / f"pilot_il_q{k}_pilot.yaml").read_text()
    cfg = cfg.replace(
        f"tasks: config/tasks_il_q{k}_pilot.json   # this launch's {len(docs[0]['cells'])} cells",
        f"tasks: config/tasks_il_q{k}_all.json   # this launch's {len(cells)} cells, stages pilot+est+sel",
    )
    cfg = re.sub(
        r"\n  cap_usd: \d+   # [^\n]*",
        f"\n  cap_usd: {sum(caps)}   # the box's stage caps {caps} summed",
        cfg,
    )
    cfg = re.sub(
        r"\n  worst_case_limit_usd: \d+", f"\n  worst_case_limit_usd: {sum(caps)}", cfg
    )
    assert f"tasks_il_q{k}_all.json" in cfg and f"cap_usd: {sum(caps)}" in cfg, k
    (CONFIG / f"pilot_il_q{k}_all.yaml").write_text(
        cfg.replace(
            "# INTERLEAVED launch", "# INTERLEAVED +311 launch (one per box)"
        ).replace(f"q{k}_pilot (", f"q{k}_all (")
    )
    manifest[f"q{k}"] = {
        "tasks": out.name,
        "tasks_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "config": f"pilot_il_q{k}_all.yaml",
        "config_sha256": hashlib.sha256(
            (CONFIG / f"pilot_il_q{k}_all.yaml").read_bytes()
        ).hexdigest(),
        "cells": len(cells),
        "cap_usd": sum(caps),
    }
(CONFIG / "tasks_il_all_manifest.json").write_text(
    json.dumps(manifest, indent=1) + "\n"
)
print(json.dumps(manifest, indent=1))
