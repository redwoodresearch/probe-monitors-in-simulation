"""2e/09 (10-15 17:3x): by-key check, read-only, no compute. Did any of my nm360 records use an S4 orphan row, i.e. one of the 115
`outside_source_map` rows of d9's S4 source map (4r part @ b68ab0a53), by (root, key) or by .eval sha256? Reads committed files only.

  python3 experiments/2026-10-15_il_z_e2e/check_s4_orphans.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
SMAP = ("b68ab0a53", "experiments/2026-10-14_q3_overnight_chain/results/s4_source_map/s4_source_map_4r_2026-10-15.json")
RECS = ["rmshits_S_newmains_2026-10-15_145501.json", "rmshits_S_newmains_deviation_2026-10-15_150228.json"]
LEAK = "leakage_addendum_2026-10-15_164923.json"
root_of = lambda folder: folder.strip("/").split("/")[-1].split("__")[0]  # noqa: E731

smap_b = git("show", f"{SMAP[0]}:{SMAP[1]}")
orph = json.loads(smap_b)["outside_source_map"]
assert len(orph) == 115, len(orph)
o_pairs = {(r["root"], r["key"]): r for r in orph}
o_roots, o_cells, o_shas = {r["root"] for r in orph}, {r["cell"] for r in orph}, {r["sha256"] for r in orph}
four_r_cells = {c["cell"] for c in json.loads(smap_b)["cells"]}


def path_roots(paths):
    return sorted({p for p in paths if any(f"/{r}__" in "/" + p.strip("/") for r in o_roots)})


out = {"label": "S4 orphan check (by key; read-only; PRELIMINARY)", "source_map": {"commit": SMAP[0], "path": SMAP[1], "sha256": sha(smap_b),
       "outside_rows": len(orph), "orphan_roots": sorted(o_roots), "orphan_cells": len(o_cells), "four_r_cells": len(four_r_cells)}, "records": {}}
for name in RECS:
    rb = (HERE / "results" / name).read_bytes()
    rec = json.loads(rb)
    cb = git("show", f"{rec['meta']['git']}:experiments/2026-10-15_il_z_e2e/{rec['meta']['config']}")
    assert sha(cb) == rec["meta"]["config_sha256"], name
    cfg = json.loads(cb)
    arm_pairs = [(a, c, root_of(f), k) for a, rows in cfg["arm_rows"].items() for c, (f, k) in rows.items()]
    files = sorted({p for d in rec["diag"].values() for p in d.get("files", {})})
    files_sha = {s for d in rec["diag"].values() for s in d.get("files", {}).values()}
    out["records"][name] = {
        "record_sha256": sha(rb), "config": f"{rec['meta']['git'][:9]}:{rec['meta']['config']}", "config_sha256": sha(cb),
        "arm_rows": {"n": len(arm_pairs), "orphan_pairs_used": [p for p in arm_pairs if (p[2], p[3]) in o_pairs],
                     "rows_from_orphan_roots": [p for p in arm_pairs if p[2] in o_roots],
                     "cells_in_four_r": sorted({p[1] for p in arm_pairs} & four_r_cells),
                     "cells_with_orphan_rows": sorted({p[1] for p in arm_pairs} & o_cells)},
        "H_B": {"folders": cfg["H_B"]["folders"], "folders_at_orphan_roots": [f for f in cfg["H_B"]["folders"] if root_of(f) in o_roots],
                "keys": len(cfg["H_B"]["keys"]), "keys_of_orphan_cells": sorted(k for k in cfg["H_B"]["keys"] if k.split("|", 1)[1] in o_cells),
                "keys_of_four_r_cells": sorted(k for k in cfg["H_B"]["keys"] if k.split("|", 1)[1] in four_r_cells)},
        "H_sim": {"roots": len(cfg["H_sim"]["roots"]), "roots_at_orphan_roots": [r["root"] for r in cfg["H_sim"]["roots"] if root_of(r["root"]) in o_roots]},
        "files_read": {"n": len(files), "under_orphan_roots": path_roots(files)},
        "orphan_eval_sha256_in_record_or_config": sorted(s for s in o_shas if s in rb.decode() or s in cb.decode() or s in files_sha)}
lb = (HERE / "results" / LEAK).read_bytes()
leak = json.loads(lb)
out["records"][LEAK] = {"record_sha256": sha(lb), "rms_S_newmains_config_sha256": leak["sources"].get("rms_S_newmains_config_sha256"),
                        "files_read": {"n": len(leak["files"]), "under_orphan_roots": path_roots(leak["files"])},
                        "orphan_eval_sha256_in_record": sorted(s for s in o_shas if s in lb.decode())}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
b = json.dumps(out, indent=1).encode()
fn = f"s4_orphan_check_{stamp}.json"
(HERE / "results" / fn).open("xb").write(b)
print(fn, sha(b))
