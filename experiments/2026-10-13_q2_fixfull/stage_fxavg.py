"""Fix-full averaged readout + F10 (PLAN_F10.md Amendment 8b), POST HOC: stage one input set for run_extract_qwen3_fxavg.py.
Modal CPU; counts and hashes only, no message text leaves the container. Never overwrites.

Arms, every row byte-identical to its source row except as noted:
  - sim-attack-fx-fullF9 (939), sim-honest-fx-fullF9 (437): F9's staged run, each run's messages checked against F9's stage
    record (sha256 pinned by stage_f9_2026-10-13_145822.json);
  - sim-attack-fx-fullF9F10 (74), sim-honest-fx-fullF9F10 (36): F10's staged run, each shard file checked against
    stage_f10_stage_2026-10-13_175148.json;
  - repro-sim-attack (24), repro-sim-honest (24): as-is runs drawn with seed 20261013 from the runs d9's mean-pool dumps hold
    (2026-10-12_la_mean-pool36_qwen3-32b_l7_v1, arm sim-attack; 2026-10-13_q3_mean-pool36-honestsim_qwen3-32b_l7_v1, arm
    sim-honest), each copied from that dump's own inputs. Only `arm` changes, and `tokens` (the extractor's sort key, never
    read by the model) is set to -1..-48 so these 48 sort last and are dealt to 48 different shards: one reproduction run
    per shard, so the bit-exact gate covers every shard's GPU. Their source `tokens` is kept as `source_tokens`.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run experiments/2026-10-13_q2_fixfull/stage_fxavg.py
"""
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-fxavg_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
F9SRC, F10SRC, DST = "2026-10-13_q2_fixfull_inputs_f9_v1", "2026-10-13_q2_f10_inputs_v1", "2026-10-13_q2_fxavg_inputs_v1"
F9_STAGE = HERE / "results" / "stage_f9_2026-10-13_145822.json"
F10_STAGE = HERE / "results" / "stage_f10_stage_2026-10-13_175148.json"
D9 = {"repro-sim-attack": ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "sim-attack"),
      "repro-sim-honest": ("2026-10-13_q3_mean-pool36-honestsim_qwen3-32b_l7_v1", "sim-honest")}
EXPECT = {"sim-attack-fx-fullF9": 939, "sim-honest-fx-fullF9": 437, "sim-attack-fx-fullF9F10": 74, "sim-honest-fx-fullF9F10": 36}
N_REPRO, SHARDS, SEED = 24, 48, 20261013

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")


class Gate(Exception):
    """A failed check; its message holds counts only."""


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def run_job(f9_record_sha: str, f10_shards: dict) -> dict:
    root = Path("/acts")

    def rows(run, arms, files=None):
        out = {}
        for p in sorted((root / run / "inputs").glob("shard_*.jsonl")):
            if files is not None and hashlib.sha256(p.read_bytes()).hexdigest() != files.get(p.name):
                raise Gate("a staged shard differs from its stage record")
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if it["arm"] in arms:
                    if (it["arm"], it["key"]) in out:
                        raise Gate("a run twice in a staged run")
                    out[(it["arm"], it["key"])] = it
        return out

    rec = root / F9SRC / "stage_record.json"
    if hashlib.sha256(rec.read_bytes()).hexdigest() != f9_record_sha:
        raise Gate("F9's stage record is not the pinned one")
    f9sha = {(r["arm"], r["key"]): r["messages_sha256"] for r in json.loads(rec.read_text())["runs"]}
    items = rows(F9SRC, {"sim-attack-fx-fullF9", "sim-honest-fx-fullF9"})
    bad = sum(msha(it["messages"]) != f9sha.get(ak) for ak, it in items.items())
    f10 = rows(F10SRC, {"sim-attack-fx-fullF9F10", "sim-honest-fx-fullF9F10"}, f10_shards)
    if {p.name for p in (root / F10SRC / "inputs").glob("shard_*.jsonl")} != set(f10_shards):
        raise Gate("F10's shard files differ from its stage record's list")
    items |= f10
    counts = {a: sum(1 for (x, _) in items if x == a) for a in EXPECT}
    if bad or counts != EXPECT:
        raise Gate(f"source gate failed: {bad} F9 rows off their record; counts {counts}")

    rng, repro, chosen = random.Random(SEED), {}, {}
    n = 0
    for new_arm, (dump, arm) in D9.items():
        cfg = json.loads((root / dump / "config.json").read_text())
        have = sorted(p.stem for p in (root / dump / "meta" / arm).glob("*.json"))
        src = {filename(k): it for (a, k), it in rows(cfg["inputs_from"], {arm}).items()}
        pool = sorted(s for s in have if s in src)
        if len(pool) < N_REPRO:
            raise Gate(f"{new_arm}: {len(pool)} runs in d9's dump with inputs, fewer than {N_REPRO}")
        pick = rng.sample(pool, N_REPRO)
        chosen[new_arm] = {"dump": dump, "dump arm": arm, "inputs_from": cfg["inputs_from"], "pool": len(pool),
                           "keys": sorted(src[s]["key"] for s in pick)}
        for s in pick:
            it = src[s]
            n += 1
            repro[(new_arm, it["key"])] = it | {"arm": new_arm, "tokens": -n, "source_tokens": it["tokens"]}
    if any(it["tokens"] < 0 for it in items.values()):
        raise Gate("a source row has a negative token count: the reproduction runs would not sort last")
    items |= repro

    # the extractor's deal (sort by -tokens, arm, key; round-robin), to prove one reproduction run per shard
    order = sorted(items.values(), key=lambda it: (-it["tokens"], it["arm"], it["key"]))
    shard_of = {(it["arm"], it["key"]): i % SHARDS for i, it in enumerate(order)}
    rshards = [shard_of[k] for k in repro]
    if sorted(rshards) != list(range(SHARDS)):
        raise Gate("the reproduction runs do not cover every shard exactly once")
    per_shard = [0] * SHARDS
    for it in order:
        per_shard[shard_of[(it["arm"], it["key"])]] += max(it["tokens"], it.get("source_tokens", 0))

    dst = root / DST
    if dst.exists():
        raise FileExistsError(f"{DST} exists: a staged run is never overwritten")
    (dst / "inputs").mkdir(parents=True)
    shas, by_arm = {}, {}
    for i, arm in enumerate(sorted({a for a, _ in items})):
        rs = sorted((it for (a, _), it in items.items() if a == arm), key=lambda x: x["key"])
        body = "".join(json.dumps(x) + "\n" for x in rs).encode()
        (dst / "inputs" / f"shard_{i}.jsonl").write_bytes(body)
        shas[f"inputs/shard_{i}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
        by_arm[arm] = [x["key"] for x in rs]
    record = {"runs": [{"arm": a, "key": k, "messages_sha256": msha(it["messages"]), "planned shard": shard_of[(a, k)]} for (a, k), it in sorted(items.items())],
              "shards": shas, "reproduction draw": chosen, "seed": SEED}
    body = json.dumps(record).encode()
    (dst / "stage_record.json").write_bytes(body)
    acts.commit()
    return {"staged": DST, "shards": shas, "runs by arm": by_arm, "reproduction draw": chosen, "record_sha256": hashlib.sha256(body).hexdigest(),
            "reproduction run per shard": {f"{a}|{k}": shard_of[(a, k)] for (a, k) in sorted(repro)},
            "planned source tokens per shard (max, median)": [max(per_shard), sorted(per_shard)[SHARDS // 2]]}


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=32768, timeout=1800, retries=0)
def job(f9_record_sha: str, f10_shards: dict) -> dict:
    try:
        return run_job(f9_record_sha, f10_shards)
    except Exception as e:  # noqa: BLE001 — type, location and our own messages only (counts), never staged text
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = str(e) if isinstance(e, (Gate, FileExistsError)) else ""
        raise RuntimeError(f"stage failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}: {msg[:160]}") from None


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "PLAN_F10.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    f9 = json.loads(F9_STAGE.read_text())["record_sha256"]
    f10 = {Path(k).name: v["sha256"] for k, v in json.loads(F10_STAGE.read_text())["shards"].items()}
    out = job.remote(f9, f10)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(p := HERE / "results" / f"stage_fxavg_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "plan": "PLAN_F10.md Amendment 8b", "git": head, "mode": "stage",
                             "inputs sha256": {F9_STAGE.name: hashlib.sha256(F9_STAGE.read_bytes()).hexdigest(),
                                               F10_STAGE.name: hashlib.sha256(F10_STAGE.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print(p)
    print({a: len(v) for a, v in out["runs by arm"].items()}, "repro shards covered:", len(set(out["reproduction run per shard"].values())),
          "tokens per shard (max, median):", out["planned source tokens per shard (max, median)"])
