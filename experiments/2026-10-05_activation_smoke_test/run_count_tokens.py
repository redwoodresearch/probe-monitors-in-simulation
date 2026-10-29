"""Exact reader token counts for every sample in the four LinuxArena cohorts (CPU, read-only S3 streaming).

Same rendering as the smoke test: Qwen chat template with thinking off, attack runs given the honest
system prompt of the same main task. Writes one row per sample to results/token_counts_<stamp>.jsonl.

  uv run --with transformers==4.56.2 --with jinja2 python experiments/2026-10-05_activation_smoke_test/run_count_tokens.py
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer

from probe_shift.transcripts import eval_files, file_samples, main_task_id, reader_messages, system_prompt

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-32B"
COHORTS = {
    "real-honest": "la_public10_real_honest_n1000",
    "real-attack-e1": "la_public10_real_attack_n1000",
    "real-attack-e2": "la_public10_real_attack_reroll_n1000",
    "sim-attack": "la_public10_sim_astra_attack_n1000",
}

tok = AutoTokenizer.from_pretrained(MODEL)


def all_samples(alias):
    with ThreadPoolExecutor(32) as pool:
        for samples in pool.map(lambda f: [(f, s) for s in file_samples(alias, f)], eval_files(alias)):
            yield from samples


honest_prompt = {}
rows = []
for arm, alias in COHORTS.items():  # real-honest first, so every attack run can take its honest prompt
    n = 0
    for file, s in all_samples(alias):
        mt = main_task_id(s)
        if arm == "real-honest":
            honest_prompt.setdefault(mt, system_prompt(s))
        prompt = honest_prompt[mt]
        text = tok.apply_chat_template(reader_messages(s, prompt), tokenize=False, enable_thinking=False)
        rows.append({"arm": arm, "file": file, "sample_id": s["id"], "epoch": s["epoch"], "main_task": mt,
                     "tokens": len(tok(text, add_special_tokens=False)["input_ids"]), "error": bool(s.get("error"))})
        n += 1
    print(f"{arm}: {n} samples", flush=True)

stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results").mkdir(exist_ok=True)
path = HERE / "results" / f"token_counts_{stamp}.jsonl"
path.write_text("".join(json.dumps(r) + "\n" for r in rows))
for arm in COHORTS:
    t = sorted(r["tokens"] for r in rows if r["arm"] == arm)
    over = sum(x > 32768 for x in t)
    print(f"{arm:15} n={len(t)} median={t[len(t)//2]} p90={t[int(.9*len(t))]} max={t[-1]} over_32k={over} ({over/len(t):.1%})")
print(f"-> {path}")
