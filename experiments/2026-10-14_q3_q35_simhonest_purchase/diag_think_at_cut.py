"""Diagnostic (counts only): for the 3 runs A's assemble gate checked (sorted by key, first 3), where the unclosed <think> sits."""
import json
from pathlib import Path

import modal

app = modal.App("probe-shift_aswin-surya_sub-q3simh_diag_kill-2026-11-13")
vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13", version=2)
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("transformers==5.19.0", "jinja2").env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})


@app.function(image=image, cpu=(1.0, 1.0), memory=(4096, 4096), timeout=600, retries=0, volumes={"/ev": vol.read_only(), "/hf": hf.read_only()})
def diag() -> list:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", local_files_only=True)
    th, eth = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
    empty = tok.encode("<think>\n\n</think>", add_special_tokens=False)
    eps = [json.loads(x) for p in sorted(Path("/ev/simh_q4f_v1_atk2/parts").glob("part*.jsonl")) for x in p.read_text().splitlines()]
    out = []
    for e in sorted(eps, key=lambda e: e["key"])[:3]:
        ids = e["ids"]
        opens = [i for i, t in enumerate(ids) if t == th]
        closes = [i for i, t in enumerate(ids) if t == eth]
        last = opens[-1] if opens else None
        out.append({"len": len(ids), "opens": len(opens), "closes": len(closes), "last_open_from_end": len(ids) - last if last is not None else None,
                    "tail_matches_empty_prefix": ids[last:] == empty[: len(ids) - last] if last is not None else None, "empty_len": len(empty)})
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(diag.remote()))
