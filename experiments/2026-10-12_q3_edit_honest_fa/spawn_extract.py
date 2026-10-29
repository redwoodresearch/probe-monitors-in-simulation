"""Spawn run_extract_all_layers.py's `orchestrate` on its DEPLOYED app (`modal deploy` that file first), so a run survives
the client going away. Builds cfg exactly as that file's main() does (pinned revision, provenance, the same-settings check),
writes config.json, spawns one orchestrator call (it spawns the shards and writes stats/summary.json), records the call
ids, returns. `--dry` changes nothing. The extractor itself is used unchanged.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with pyyaml python \
      experiments/2026-10-12_q3_edit_honest_fa/spawn_extract.py --config alllayers_q3_edithonest_smoke.yaml [--dry]
"""
import argparse
import importlib.util
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_shift.models import QWEN3_32B  # noqa: E402
from probe_shift.provenance import capture  # noqa: E402

spec = importlib.util.spec_from_file_location("rx", ROOT / "experiments/2026-10-06_all_layers_extraction/run_extract_all_layers.py")
spec.loader.exec_module(R := importlib.util.module_from_spec(spec))
SMOKE_CONFIG = "alllayers_q3_edithonest_smoke.yaml"
KEYS = {"run", "inputs_from", "model", "dtype", "attn_implementation", "max_tokens", "shards", "chunk_size", "ema_final_decays",
        "ema_only", "mean_final", "max_final", "layers"}


def main(config: str, dry: bool, commit: str, smoke: str):
    cfg = yaml.safe_load((R.HERE / "config" / config).read_text())
    if config != SMOKE_CONFIG:  # RULE gate 2: the edited extraction only after a passing smoke check of this staging
        rec = json.loads((Path(__file__).parent / smoke).read_text()) if smoke else {}
        stage = json.loads((Path(__file__).parent / "results" / rec["stage_record"]).read_text()) if rec.get("stage_record") else {}
        if not rec.get("pass") or stage.get("staged") != yaml.safe_load((R.HERE / "config" / config).read_text())["inputs_from"]:
            raise ValueError("no passing smoke-check record for this staging (--smoke results/smoke_check_<stamp>.json)")
    if set(cfg) - KEYS or cfg["model"] != QWEN3_32B["name"]:
        raise ValueError("unknown settings or an unpinned model")
    cfg["revision"] = QWEN3_32B["revision"]
    cfg["provenance"] = capture(ROOT, [Path(R.__file__), R.HERE / "config" / config, *sorted((ROOT / "src" / "probe_shift").glob("*.py"))],
                                commit)
    if cfg["provenance"]["git_dirty"]:
        raise ValueError("tracked files differ from HEAD: commit first")
    try:
        stored = json.loads(b"".join(R.out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    same = lambda c: {k: v for k, v in c.items() if k != "provenance"} | {"source_sha": (c.get("provenance") or {}).get("source_sha")}
    if stored is not None and same(stored) != same(cfg):
        raise ValueError(f"{cfg['run']} exists with different settings or code; use a new run name")
    staged = [e.path for e in R.out_vol.listdir(f"{cfg['inputs_from']}/inputs") if e.path.endswith(".jsonl")]
    if not staged:
        raise FileNotFoundError(f"no staged inputs under {cfg['inputs_from']}/inputs")
    fn = modal.Function.from_name(R.APP, "orchestrate")
    fn.hydrate()  # fails here if the app is not deployed
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {"run": cfg["run"], "app": R.APP, "function": fn.object_id, "git": git, "config_stored": stored is not None, "dry": dry,
           "smoke_record": smoke or None}
    if not dry:
        with R.out_vol.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
        rec = Path(__file__).parent / "results" / f"launch_{cfg['run']}_{datetime.now(ZoneInfo('America/Los_Angeles')):%Y-%m-%d_%H%M%S}.jsonl"
        rec.parent.mkdir(exist_ok=True)
        with rec.open("x") as fh:  # written before the spawn, then the accepted call
            print(json.dumps(out | {"config": cfg}), file=fh, flush=True)
            out["call"] = fn.spawn(cfg, cfg["shards"]).object_id
            print(json.dumps({"orchestrate_call": out["call"], "shards": cfg["shards"]}), file=fh, flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--smoke", default="", help="results/smoke_check_<stamp>.json; required for the edited extraction")
    ap.add_argument("--dry", action="store_true")
    main(**vars(ap.parse_args()))
