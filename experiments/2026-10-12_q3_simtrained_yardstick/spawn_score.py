"""Deploy-and-spawn launcher for run_score_onpolicy.py. Twice on 2026-10-10 a laptop network loss killed `modal run
--detach` clients and Modal stopped their apps; after that, the orchestrator ruled that launches must not depend on the
laptop's connection. This computes the same checkpoint key as run_score_onpolicy.main() (the scorer file is unchanged,
so its hash and folders are too), checks that the folder already holds draws, then spawns one call of the DEPLOYED
`score` function per shard and returns. The calls run on Modal without the laptop; saved draws are skipped. Collect
the results file afterwards with the usual `modal run ... --shards 1`, which reads the saved draws.

  modal deploy experiments/2026-10-08_q3_onpolicy_figures/run_score_onpolicy.py
  python experiments/2026-10-08_q3_onpolicy_figures/spawn_score.py --config score_D_i_196.yaml --draws 0-500 --shards 3
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

import run_score_onpolicy as R

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--draws", required=True)
ap.add_argument("--shards", type=int, required=True)
ap.add_argument("--per-call", action="store_true")
ap.add_argument("--new", action="store_true", help="start a run whose folder does not exist yet")
ap.add_argument("--cache-as", default="", help="spawn cache_d (rows read once on CPU) under this name instead of score")
ap.add_argument("--honest-key", default="", help="spawn honest_scores on this save_probes draw's folder instead of score")
ap.add_argument("--ext-config", default="", help="with --honest-key: the config whose honA, honB and simH rows are scored")
ap.add_argument("--main-percall", default="", help="with --honest-key: the main run's per-call observed draw (gate b)")
a = ap.parse_args()
cfg = yaml.safe_load((R.HERE / "config" / a.config).read_text())
if cfg.get("label_rule"):
    raise ValueError("this launcher does not cover the category-table configs")
folds = R.frozen_folds(R.HERE.parents[1] / cfg["pilot_tasks"])
lo, hi = map(int, a.draws.split("-"))
ids = list(range(lo, hi + 1))
if cfg.get("null_test") and lo == 0:
    raise ValueError("a null config takes draws from 1")
extractions = {r: json.loads(b"".join(R.acts_vol.read_file(f"{r}/complete.json"))) for r in cfg["extractions"]}
ident = {"config": cfg, "folds": folds, "extractions": extractions,
         "scorer_sha256": hashlib.sha256((R.HERE / "run_score_onpolicy.py").read_bytes()).hexdigest()}
if cfg.get("mode") == "D":
    ident["phase1_table_sha256"] = hashlib.sha256((R.HERE.parents[1] / cfg["phase1_table"]).read_bytes()).hexdigest()
if cfg.get("mode") == "D" and "d_redo_196" in cfg.get("amendments", {}):  # main()'s guard (Codex)
    vc = json.loads((R.HERE.parents[1] / cfg["variant_check"]).read_text())
    if vc["primary"] != cfg["variant"] or vc["i"]["draws"] != 500:
        raise ValueError(f"the label check names variant {vc['primary']}; this config is variant {cfg['variant']}")
p1rows = json.loads((R.HERE.parents[1] / cfg["phase1_table"]).read_text())["rows"] if cfg.get("mode") == "D" else None
if a.honest_key:  # the follow-ups' redesign: gate b and the honest scores, on CPU, spawned on the deployed app
    f = json.loads(Path(a.main_percall).read_text())
    pc = f["draws"][0]["cells"]["real->real"]["per_call"]
    main = {"choices": f["draws"][0]["choices"], "meta": [[m["world"], m["task"], m["y"]] for m in f["meta"]], "rows": pc["row"], "scores": pc["score"]}
    ext = yaml.safe_load((R.HERE / "config" / a.ext_config).read_text())
    c = modal.Function.from_name(R.APP, "honest_scores").spawn(cfg, ext, folds, a.honest_key, main)
    print(json.dumps({"spawned_at": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S PT"), "honest_key": a.honest_key, "call": c.object_id}))
    raise SystemExit(0)
if a.cache_as:  # the CPU row cache, spawned on the deployed app; it refuses to overwrite an existing cache folder
    c = modal.Function.from_name(R.APP, "cache_d").spawn(cfg, folds, p1rows, a.cache_as)
    print(json.dumps({"spawned_at": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S PT"), "cache": a.cache_as, "call": c.object_id}))
    raise SystemExit(0)
key = f"{cfg['label']}_{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]}" + ("_percall" if a.per_call else "")
saved = [e for e in R.score_vol.listdir(key) if Path(e.path).name.startswith("draw_")] if not a.new else []
if not saved and not a.new:
    raise ValueError(f"{key} holds no draws: pass --new to start a run")
fn = modal.Function.from_name(R.APP, "score")
calls = [fn.spawn(cfg, folds, ids[s::a.shards], key, p1rows, a.per_call, None, False) for s in range(a.shards)]
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S PT")
print(json.dumps({"spawned_at": stamp, "key": key, "already_saved": len(saved), "draws": a.draws, "shards": a.shards,
                  "calls": [c.object_id for c in calls]}, indent=1))
