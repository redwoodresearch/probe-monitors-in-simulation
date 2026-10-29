"""Attachment census, part 2: `attachment://` in the inputs the extractions actually read. Integer counts only.

On-policy (Qwen3.5-27B, q3): run_replay_readouts.py stages each context epoch's logged vLLM ids (prompt + completion)
as `<stage>/inputs/<arm>.jsonl` on the q3 on-policy Volume; run_mask_stage.py, swap_stage and strip_stage derive their
stagings from those ids. Each staged epoch's ids are decoded with the same tokenizer and "attachment://" is counted:
per (stage, arm file) epochs, tokens, occurrences, epochs with any.
Off-policy (Qwen3-32B, `--offpolicy`): the segment-end stagings write reader messages built by
`probe_shift.transcripts.reader_messages` (no attachment lookup) to `<run>/inputs/shard_<i>.jsonl` on the la-activations
Volume, one item per transcript with its `arm`. The needle is counted in each item's raw JSON line: per (run, arm)
items, occurrences, items with any.
Attack arms (and the off-policy shards, which mix arms) are refused unless `--attack-go` names the go (Aswin's 10-11
10:19:53 rule: "No new pass over attack-run text, scripted or not"; his go for this census, just before 16:57 PT via
0d: "Yes, counts only is fine"). Only counts and file names are printed or written, never matched text. Modal CPU,
read-only Volumes, retries 0, bounded timeouts and file caps (SPEND.md).

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run \
      experiments/2026-10-11_attachment_census/count_staged_attachments_modal.py --out <json> [--attack-go "<go>"] [--offpolicy]
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-census_attachments_kill-2026-11-10"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
LA = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
NEEDLE = "attachment://"
HONEST = ("honest_", "sim_honest")
CAPS = {"honest": 40, "all": 120, "offpolicy": 70}  # units per launch, each priced in SPEND.md before it runs

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))
vols = {"/acts": modal.Volume.from_name(ACTS, version=2).read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only(),
        "/la": modal.Volume.from_name(LA, version=2).read_only()}


def is_honest(arm_file: str) -> bool:
    return arm_file.startswith(HONEST) and "attack" not in arm_file


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0, volumes=vols)
def list_files(offpolicy: bool) -> list:
    if offpolicy:
        return sorted(str(p.parent.parent.relative_to("/la")) for p in Path("/la").glob("*/inputs/shard_0.jsonl"))
    return sorted(f"{p.parent.parent.name}/inputs/{p.name}" for p in Path("/acts").glob("*/inputs/*.jsonl")
                  if not p.parent.parent.name.endswith("_test"))


@app.function(image=image, cpu=2, memory=8192, timeout=300, retries=0, volumes=vols)
def count_file(rel: str, attack_go: str) -> dict:
    from transformers import AutoTokenizer

    stage, _, name = rel.split("/")
    if not (is_honest(name) or attack_go):
        raise ValueError(f"refused: {rel} is not an honest arm file and no attack go was given")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    c = {"stage": stage, "arm_file": name, "epochs": 0, "tokens": 0, "occurrences": 0, "epochs_with_any": 0}
    with open(Path("/acts") / rel) as fh:
        for line in fh:
            ids = json.loads(line)["ids"]
            k = tok.decode(ids).count(NEEDLE)
            c["epochs"] += 1
            c["tokens"] += len(ids)
            c["occurrences"] += k
            c["epochs_with_any"] += int(k > 0)
    return c


@app.function(image=image, cpu=2, memory=8192, timeout=600, retries=0, volumes=vols)
def count_run(run: str, attack_go: str) -> dict:
    if not attack_go:
        raise ValueError("refused: off-policy shards mix attack and honest arms; an attack go is required")
    per = {}
    for p in sorted((Path("/la") / run / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            arm = json.loads(line).get("arm")
            k = line.count(NEEDLE)
            a = per.setdefault(str(arm), {"items": 0, "occurrences": 0, "items_with_any": 0})
            a["items"] += 1
            a["occurrences"] += k
            a["items_with_any"] += int(k > 0)
    return {"run": run, "per_arm": per}


@app.local_entrypoint()
def main(out: str, attack_go: str = "", offpolicy: bool = False):
    files = list_files.remote(offpolicy)
    if offpolicy:
        if len(files) > CAPS["offpolicy"]:
            raise SystemExit(f"{len(files)} runs: more than the {CAPS['offpolicy']} priced; re-price first")
        rows = list(count_run.map(files, kwargs={"attack_go": attack_go}))
        tot = {"items": sum(a["items"] for r in rows for a in r["per_arm"].values()),
               "occurrences": sum(a["occurrences"] for r in rows for a in r["per_arm"].values())}
        skipped = []
    else:
        chosen = [f for f in files if attack_go or is_honest(f.rsplit("/", 1)[1])]
        cap = CAPS["all" if attack_go else "honest"]
        if len(chosen) > cap:
            raise SystemExit(f"{len(chosen)} files: more than the {cap} priced; re-price first")
        skipped = [f for f in files if f not in chosen]
        print(f"{len(files)} staged arm files, {len(chosen)} counted, {len(skipped)} skipped (not scanned)")
        rows = list(count_file.map(chosen, kwargs={"attack_go": attack_go}))
        tot = {k: sum(r[k] for r in rows) for k in ("epochs", "tokens", "occurrences", "epochs_with_any")}
    for r in rows:
        print(json.dumps(r))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
    Path(out).write_text(json.dumps({"git": {"commit": git, "dirty": dirty}, "written": stamp, "app": APP,
                                     "volume": LA if offpolicy else ACTS, "tokenizer": None if offpolicy else [MODEL, REVISION],
                                     "needle": NEEDLE, "attack_go": attack_go or None, "units_listed": len(files),
                                     "skipped_not_scanned": skipped, "totals": tot, "rows": rows}, indent=1))
    print("totals", json.dumps(tot), "->", out)
