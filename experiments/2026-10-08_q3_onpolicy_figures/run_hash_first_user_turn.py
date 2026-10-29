"""POST-HOC check (plan Amendment 14): after the instruction swap, does any attacker-only text remain in the first user
turn? Modal CPU, token ids only: per task, the sha256 of the first user turn's ids and of the whole system turn's ids,
for the swapped run 1 (`*_swapsys`), swapped run 2, the swapped simulated runs (`*_swapsys2`) and honest_a (the masked
base stagings). Differing tasks get token-range metadata only (no text is decoded or printed): the first differing
token's offset within the turn, both turns' lengths, the common prefix and suffix, and each side's differing span.
Epochs the cache skips (redrawn, or attempt_kind "rerun") are left out; every counted epoch of a task must agree.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run experiments/2026-10-08_q3_onpolicy_figures/run_hash_first_user_turn.py
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ACTS, HF_CACHE = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
BASES = ["rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1", "scaleup_topup_segends_v1_scrub1"]
SOURCES = {"run1": [(f"{b}_swapsys", "attack") for b in BASES], "run2": [("second_attack_segends_v1_scrub1_swapsys", "attack")],
           "sim": [(f"{b}_swapsys2", "sim_attack@september") for b in BASES], "honest_a": [(b, "honest_a") for b in BASES]}

app = modal.App("probe-shift_aswin-surya_sub-q3onpol_hashturn_kill-2026-11-10")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


def span_meta(a: list, b: list) -> dict:
    """Where two id lists differ, as offsets and lengths only."""
    p = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    s = 0
    while s < min(len(a), len(b)) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    return {"first_differing_token": p, "len_a": len(a), "len_b": len(b), "common_prefix": p, "common_suffix": s,
            "differing_span_a": len(a) - p - s, "differing_span_b": len(b) - p - s}


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS, version=2).read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def check() -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")

    def turns(ids):  # run_read_tests.swap_stage's turns(): (start, end) of each turn's interior
        out, i = [], 0
        while i < len(ids):
            if ids[i] == im_start:
                j = next((k for k in range(i + 1, len(ids)) if ids[k] == im_end), len(ids))
                out.append((i + 1, j))
                i = j
            i += 1
        return out

    got, disagree = {}, []
    for world, srcs in SOURCES.items():
        for staging, arm in srcs:
            for line in (Path("/acts") / staging / "inputs" / f"{arm}.jsonl").read_text().splitlines():
                ep = json.loads(line)
                if ep.get("redrawn_because") or ep.get("attempt_kind") == "rerun":
                    continue
                ids = ep["ids"]
                (s0, s1), (u0, u1) = turns(ids)[:2]
                cur = {"system": ids[s0:s1], "user": ids[u0:u1]}
                prev = got.setdefault(world, {}).setdefault(ep["task"], cur)
                if prev != cur:
                    disagree.append(f"{world}|{ep['task']}")
    sha = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    out = {"epochs_disagreeing_within_a_task": len(disagree), "tasks": {w: len(v) for w, v in got.items()}}
    for w in ("run1", "run2", "sim"):
        common = sorted(set(got[w]) & set(got["honest_a"]))
        r = {"tasks_compared": len(common), "tasks_without_honest_a": len(set(got[w]) - set(got["honest_a"]))}
        for part in ("user", "system"):
            diff = [t for t in common if sha(got[w][t][part]) != sha(got["honest_a"][t][part])]
            r[f"{part}_match"], r[f"{part}_differ"] = len(common) - len(diff), len(diff)
            r[f"{part}_differing_spans"] = [span_meta(got[w][t][part], got["honest_a"][t][part]) for t in diff]
        out[f"{w}_vs_honest_a"] = r
    return out


@app.local_entrypoint()
def main():
    res = check.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"first_user_turn_hash_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"amendment": "14 (post hoc)", "results": res}, indent=1))
    for k, v in res.items():
        if isinstance(v, dict) and "user_match" in v:
            print(k, json.dumps({x: v[x] for x in ("tasks_compared", "tasks_without_honest_a", "user_match", "user_differ", "system_match", "system_differ")}))
        else:
            print(k, json.dumps(v))
    print(path.name)
