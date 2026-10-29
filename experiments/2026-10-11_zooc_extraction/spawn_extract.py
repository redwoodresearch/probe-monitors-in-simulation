"""Spawn run_extract_qwen35.py's shards on its DEPLOYED app (`modal deploy` that file first), so a run survives the client
going away. Builds cfg as its main() does, writes config.json, spawns one call per shard, returns; `--dry` changes nothing."""
import argparse, importlib.util, io, json, os, subprocess, sys  # noqa: E401
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import modal, yaml  # noqa: E401
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from probe_shift.models import QWEN35_27B  # noqa: E402
spec = importlib.util.spec_from_file_location("rx", ROOT / "experiments/2026-10-06_qwen35_replication/run_extract_qwen35.py")
spec.loader.exec_module(R := importlib.util.module_from_spec(spec))


def main(config: str, test: bool, dry: bool):
    cfg = yaml.safe_load((R.HERE / "config" / config).read_text())
    if cfg["model"] != QWEN35_27B["name"] or cfg.get("transcripts") or os.environ.get("QWEN35_OUTPUT_VOLUME"):
        raise ValueError("unpinned model, a `transcripts` subset, or QWEN35_OUTPUT_VOLUME set (the deployed app writes to the default)")
    cfg = R.THROUGHPUT_DEFAULTS | cfg | {"revision": QWEN35_27B["revision"]}
    if test:
        cfg |= {"run": cfg["run"] + "_test", "test_items": 8, "chunk_size": 4, "repeat_forward": True}
    try:
        stored = json.loads(b"".join(R.out_vol.read_file(f"{cfg['run']}/config.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        stored = None
    if stored is not None and R.THROUGHPUT_DEFAULTS | stored != cfg:
        raise ValueError(f"{cfg['run']} exists with different settings; use a new run name")
    fn = modal.Function.from_name(R.APP, "extract_shard")
    fn.hydrate()  # fails here if the app is not deployed
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {"run": cfg["run"], "app": R.APP, "function": fn.object_id, "git": git, "config_stored": stored is not None, "dry": dry, "calls": {}}
    if not dry:
        with R.out_vol.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(json.dumps(cfg, indent=1).encode()), f"{cfg['run']}/config.json")
        rec = Path(__file__).parent / "results" / f"launch_{cfg['run']}_{datetime.now(ZoneInfo('America/Los_Angeles')):%Y-%m-%d_%H%M%S_%f}.jsonl"
        rec.parent.mkdir(exist_ok=True)
        with rec.open("x") as fh:  # written before any spawn, then one flushed line per accepted call
            print(json.dumps(out | {"config": cfg}), file=fh, flush=True)
            for i in range(1 if test else cfg["shards"]):
                out["calls"][i] = fn.spawn(cfg, i).object_id
                print(json.dumps({"shard": i, "call": out["calls"][i]}), file=fh, flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    for flag in ("--test", "--dry"):
        ap.add_argument(flag, action="store_true")
    main(**vars(ap.parse_args()))
