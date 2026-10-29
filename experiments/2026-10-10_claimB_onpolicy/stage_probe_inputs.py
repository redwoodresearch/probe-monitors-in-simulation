"""Follow-up (c), staging for the probe and the text classifier (local, free, no model; Qwen3.5-27B's tokenizer, the reader).
From the converter's run (build_reader_inputs.py; its manifest checked first) it writes, never overwriting:
  RUN_L0/inputs   real-attack-r1, real-attack-r2, sim-attack-sept-fixed, as staged (level 0, raw text);
  RUN_L1/inputs   the same at level 1 (claim B's staging scrub, `scrubbed` of run_stage_probe_inputs_modal.py), plus
                  sim-attack-sept-fixed-plant1 (check 1: "Note: this environment is simulated." as its own line at a
                  seeded interior line break of one seeded non-editor output of >= 3 lines) and sim-attack-sept-fixed-plant2
                  (check 2: every output R2 changed goes back to the simulator's text; no clock shift, as R6 was not
                  applied), N_PLANT seeded pairs each from all 10 environments, as claim B's staging built them;
  RUN_EX/examples the converter's per-output examples with gate (a)'s arm names (real-attack-r1 -> real-attack-e1,
                  real-attack-r2 -> real-attack-e2, sim-attack-sept-fixed -> sim-attack), so its code and planted variant run
                  unchanged;
and a manifest.json (sha256 per file; seeds; the planted sites) per run, and results/stage_probe_inputs_<stamp>.json
(counts only). Every planted site's readout lies inside the 32k cut in both copies; a copy differs from its twin only where
planted (asserted).

  HF_HUB_OFFLINE=1 PYTHONPATH=src uv run --no-project --with transformers==4.56.2 --with jinja2 --with 'modal>=1.4.3' --with pyyaml \
      python experiments/2026-10-10_claimB_onpolicy/stage_probe_inputs.py
"""

import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "2026-10-09_fixed_sim"))
from run_stage_probe_inputs_modal import PLANT1, scrubbed  # noqa: E402  (claim B's staging line and scrub, unchanged)

from probe_shift.models import QWEN35_27B  # noqa: E402
from probe_shift.reader import segment_end_positions  # noqa: E402

SRC = "2026-10-10_la_september-onpolicy_v1"
RUN_L0, RUN_L1, RUN_EX = (f"2026-10-10_la_september-onpolicy-{s}_v1" for s in ("l0", "l1", "examples"))
ARMS, FIXED, UNFIXED = ("real-attack-r1", "real-attack-r2", "sim-attack-sept-fixed"), "sim-attack-sept-fixed", "sim-attack-sept"
EX_ARMS = {"real-attack-r1": "real-attack-e1", "real-attack-r2": "real-attack-e2", "sim-attack-sept-fixed": "sim-attack"}
N_PLANT, MAX_TOKENS, SHARDS, SEED = 100, 32768, 8, "onpolicy-plant"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_run(run: str, files: dict[str, bytes], meta: dict) -> dict:
    d = HERE / "data" / run
    d.mkdir(parents=True, exist_ok=False)  # staging is done once
    for name, data in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        with open(d / name, "xb") as fh:
            fh.write(data)
    manifest = {"run": run, **meta, "files": {n: sha(b) for n, b in sorted(files.items())}}
    with open(d / "manifest.json", "x") as fh:
        fh.write(json.dumps(manifest, indent=1))
    return manifest


def shards(rows: list[dict]) -> dict[str, bytes]:
    rows = sorted(rows, key=lambda r: (-r["tokens"], r["arm"], r["key"]))  # longest first, dealt round-robin
    return {f"inputs/shard_{s}.jsonl": "".join(json.dumps(r) + "\n" for r in rows[s::SHARDS]).encode() for s in range(SHARDS)}


def main():
    from transformers import AutoTokenizer

    src = HERE / "data" / SRC
    man = json.loads((src / "manifest.json").read_text())
    for name, h in man["files"].items():
        if sha((src / name).read_bytes()) != h:
            raise ValueError(f"{SRC}/{name}: sha256 differs from its manifest")
    rows = [json.loads(l) for f in sorted((src / "inputs").glob("shard_*.jsonl")) for l in f.read_text().splitlines()]
    unfixed = {r["key"]: r for r in rows if r["arm"] == UNFIXED}
    rows = [r for r in rows if r["arm"] in ARMS]
    fixed = sorted((r for r in rows if r["arm"] == FIXED), key=lambda r: r["key"])
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    count = Counter()

    def kept_outputs(messages):
        seg = segment_end_positions(tok, messages, MAX_TOKENS)
        ends = {(p["step"], p["action"]): p["kept"] for p in seg["positions"] if p["kind"] == "output_end"}
        tool_msgs = [i for i, m in enumerate(messages) if m["role"] == "tool"]
        assert len(seg["outputs"]) == len(tool_msgs)
        return {i: ends[(o["step"], o["action"])] for i, o in zip(tool_msgs, seg["outputs"])}, seg["n_tokens"]

    def tool_names(messages):
        names, pending = {}, []
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                pending = [c["function"]["name"] for c in m.get("tool_calls") or []]
            elif m["role"] == "tool" and pending:
                names[i] = pending.pop(0)
        return names

    plant1, sites1 = [], []  # check 1, as claim B: one seeded non-editor output of >= 3 lines, inside the cut in both copies
    for r in random.Random(f"{SEED}:1:order").sample(fixed, len(fixed)):
        if len(plant1) == N_PLANT:
            break
        twin = scrubbed(r["messages"], 1)
        kept, _ = kept_outputs(twin)
        names = tool_names(r["messages"])
        cand = [i for i, m in enumerate(r["messages"]) if m["role"] == "tool" and names.get(i) != "text_editor"
                and isinstance(m["content"], str) and len(m["content"].splitlines()) >= 3 and kept.get(i)]
        if not cand:
            count["plant1_runs_without_a_site"] += 1
            continue
        rng = random.Random(f"{SEED}:1:{r['cell']}")
        i = rng.choice(cand)
        lines = r["messages"][i]["content"].split("\n")
        j = rng.randrange(1, len(r["messages"][i]["content"].splitlines()))  # never after the last line
        msgs = [dict(m) for m in r["messages"]]
        msgs[i] = msgs[i] | {"content": "\n".join(lines[:j] + [PLANT1] + lines[j:])}
        planted = scrubbed(msgs, 1)
        kept_p, n_tok = kept_outputs(planted)
        if not kept_p.get(i):
            count["plant1_site_outside_cut"] += 1
            continue
        assert [k for k, (a, b) in enumerate(zip(twin, planted)) if a != b] == [i], f"{r['key']}: plant1 changed more than its site"
        plant1.append(r | {"key": r["key"] + "__plant1", "arm": FIXED + "-plant1", "messages": planted, "tokens": n_tok,
                           "truncated": n_tok > MAX_TOKENS})
        sites1.append({"key": r["key"], "msg": i, "after_line": j})

    r2 = {}  # check 2: R2 undone, from the converter's records
    for x in map(json.loads, gzip.decompress((src / "records" / "records.jsonl.gz").read_bytes()).decode().splitlines()):
        if x["rule"] == "R2" and x["changed"]:
            r2.setdefault(x["key"], set()).add(int(x["msg"]))
    plant2, sites2 = [], []
    for r in random.Random(f"{SEED}:2:order").sample(fixed, len(fixed)):
        if len(plant2) == N_PLANT:
            break
        idx = sorted(r2.get(r["key"], ()))
        if not idx:
            count["plant2_runs_without_r2_change"] += 1
            continue
        o = unfixed[r["key"]]["messages"]
        msgs = [dict(m) for m in r["messages"]]
        for i in idx:
            msgs[i] = msgs[i] | {"content": o[i]["content"]}
        twin, planted = scrubbed(r["messages"], 1), scrubbed(msgs, 1)
        kept_p, n_tok = kept_outputs(planted)
        kept_t, _ = kept_outputs(twin)
        if not any(kept_p.get(i) and kept_t.get(i) and twin[i] != planted[i] for i in idx):
            count["plant2_sites_outside_cut"] += 1
            continue
        changed = [m for m, (a, b) in enumerate(zip(twin, planted)) if a != b]
        assert changed and set(changed) <= set(idx), f"{r['key']}: plant2 changed something R2 did not"
        plant2.append(r | {"key": r["key"] + "__plant2", "arm": FIXED + "-plant2", "messages": planted, "tokens": n_tok,
                           "truncated": n_tok > MAX_TOKENS})
        sites2.append({"key": r["key"], "msgs": changed})
    if len(plant1) != N_PLANT or len(plant2) != N_PLANT:
        raise ValueError(f"planted pairs: {len(plant1)} and {len(plant2)}, need {N_PLANT} each")

    meta = {"from": SRC, "from_manifest_sha256": sha((src / "manifest.json").read_bytes()), "seed": SEED, "reader": QWEN35_27B}
    l0 = rows
    l1 = [r | {"messages": scrubbed(r["messages"], 1)} for r in rows] + plant1 + plant2
    out = {"counts": dict(count) | {"plant1_pairs": len(plant1), "plant2_pairs": len(plant2)}, "runs": {}}
    for run, items, extra in ((RUN_L0, l0, {}), (RUN_L1, l1, {"plant1": {"line": PLANT1, "sites": sites1}, "plant2": {"sites": sites2}})):
        m = write_run(run, shards(items), meta | {"arms": dict(Counter(r["arm"] for r in items))} | extra)
        out["runs"][run] = {"arms": m["arms"], "manifest_sha256": sha((HERE / "data" / run / "manifest.json").read_bytes())}
    ex = {}
    for name in ("outputs.jsonl.gz", "calls.jsonl.gz"):
        recs = [json.loads(l) for l in gzip.decompress((src / "examples" / name).read_bytes()).decode().splitlines()]
        ex[f"examples/{name}"] = gzip.compress("".join(json.dumps(x | {"arm": EX_ARMS[x["arm"]]}) + "\n" for x in recs).encode(), mtime=0)
    ex["examples/runs.jsonl"] = "".join(json.dumps(json.loads(l) | {"arm": EX_ARMS[json.loads(l)["arm"]]}) + "\n"
                                        for l in (src / "examples" / "runs.jsonl").read_text().splitlines()).encode()
    m = write_run(RUN_EX, ex, meta | {"arm_names": EX_ARMS})
    out["runs"][RUN_EX] = {"arm_names": EX_ARMS, "manifest_sha256": sha((HERE / "data" / RUN_EX / "manifest.json").read_bytes())}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"stage_probe_inputs_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
