"""Build config/inputs_z.json from committed sources only: the e2e config (rows, local pins, sim-honest specs) and 09's IL interim record @ c3d904c0a
(its embedded resolution index v4 and IL key list, and its pinned per-folder sol/p2 manifest shas). S_old (ilold) cells only.

  python3 experiments/2026-10-15_il_z_e2e/make_config_z.py
"""

import base64
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
E2E = HERE.parent / "2026-10-15_e2e_sim_pipeline"
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
IL_REC = "c3d904c0a:experiments/2026-10-15_il_analysis_09/results/il_interim_s1_result_il_2026-10-15_115759.json"
LABELS = ["Sol (a) R", "Sol (a) C", "FIBS R", "FIBS C", "harmfulness R", "harmfulness C", "refusal R", "refusal C", "high-stakes R", "high-stakes C",
          "c_asis R", "c_asis C"]
SRC = {("Sol (a)", "R"): ("sol", "a_tight__avg_top2.npz"), ("Sol (a)", "C"): ("sol", "a_tight__cmd_A.npz"),
       ("c_asis", "R"): ("sol", "c_asis_tight__avg_top2.npz"), ("c_asis", "C"): ("sol", "c_asis_tight__cmd_A.npz")}
P2 = {"FIBS": "fibs", "harmfulness": "harmfulness", "refusal": "refusal", "high-stakes": "highstakes"}
for fam, nm in P2.items():
    SRC[(fam, "R")], SRC[(fam, "C")] = ("p2", f"{nm}__reply_mean.npz"), ("p2", f"{nm}__command.npz")

e2e_b = (E2E / "config" / "inputs_e2e.json").read_bytes()
cfg = json.loads(e2e_b)
il_b = git("show", IL_REC)
il = json.loads(il_b)["result_il"]
idx = json.loads(base64.b64decode(il["record_inputs"]["index"]["content_b64"]))
kl = json.loads(base64.b64decode(il["record_inputs"]["keylist"]["content_b64"]))
assert hashlib.sha256(base64.b64decode(il["record_inputs"]["index"]["content_b64"])).hexdigest() == il["record_inputs"]["index"]["sha256"]
assert hashlib.sha256(base64.b64decode(il["record_inputs"]["keylist"]["content_b64"])).hexdigest() == il["record_inputs"]["keylist"]["sha256"]
est_old = set(kl["estimation_set"]["t196_tasks"])
by = {}
for r in idx["rows"]:
    if r["launch"] == "old":
        assert (r["cell"], r["world"]) not in by
        by[(r["cell"], r["world"])] = r
ok = lambda c, w: by.get((c, w), {}).get("status") == "scored" and by[(c, w)].get("attempt_label") == "attempted"  # noqa: E731
cells = sorted(c for c in est_old if ok(c, "run3") and ok(c, "sim2"))
rows_il = {w: {c: [by[(c, w)]["scorer_folder"], by[(c, w)]["key"]] for c in cells} for w in ("run3", "sim2")}
folders = sorted({v[0] for w in rows_il for v in rows_il[w].values()})
man = {f: {"sol": il["inputs"][f]["sol_manifest"], "p2": il["inputs"][f]["p2_manifest"]} for f in folders}
n_old_09 = il["results"]["R|0.05"]["co_primary"]["sim2_minus_run3"]["S_old"]["cells"]
assert len(cells) == n_old_09, (len(cells), n_old_09)
local = {k: v for k, v in cfg["local"].items() if k != "rec_prev"}
E2E_REC = "51b6b2a28:experiments/2026-10-15_e2e_sim_pipeline/results/e2e_2026-10-15_124407.json"
local["rec_e2e"] = {"git": E2E_REC, "sha256": hashlib.sha256(git("show", E2E_REC)).hexdigest()}
for k, v in local.items():
    if "path" in v and not v["path"].startswith("/"):
        v["path"] = str(Path("..") / E2E.name / v["path"]) if not v["path"].startswith("..") else v["path"]
rows = []
for r in cfg["rows"]:
    if r["label"] in LABELS:
        sub, fn = SRC[(r["family"], r["readout"])]
        rows.append(r | {"il_src": {"sub": sub, "file": fn}})
assert len(rows) == 12
out = {"label": "Z (IL same-launch, separate thresholds, per action), S_old; POST HOC", "crossrecipe_config": cfg["crossrecipe_config"],
       "local": local, "rows": rows, "expected": cfg["expected"], "fprs": [0.01, 0.001, 0.05], "a25_alphas": cfg["a25_alphas"],
       "il": {"source_record": IL_REC, "source_record_sha256": hashlib.sha256(il_b).hexdigest(), "index_sha256": il["record_inputs"]["index"]["sha256"],
              "il_lists_record_sha256": il["record_inputs"]["keylist"]["sha256"], "origin": "S_old (ilold, T196 estimation cells)", "expected_cells": n_old_09,
              "cells": cells, "rows": rows_il, "manifests": man},
       "e2e_config_sha256": hashlib.sha256(e2e_b).hexdigest()}
(HERE / "config").mkdir(exist_ok=True)
(HERE / "config" / "inputs_z.json").write_text(json.dumps(out, indent=1) + "\n")
print(len(cells), "cells;", len(folders), "folders:", folders)
