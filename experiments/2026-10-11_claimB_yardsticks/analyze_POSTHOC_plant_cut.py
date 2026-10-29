"""POST HOC (#853 review thread PRRT_kwDOU9RNXc6rYPN1; $0, laptop tokenizer only, no text printed or saved, no
activations): I2's planted documents used every tool output of the level-1 reader inputs, while its training documents
used only the outputs kept within the reader's 32k cut. For plant 1 and plant 2 (150 copy/twin pairs each), count runs
whose documents included outputs past the cut, and planted or changed outputs past the cut.

Source: the level-1 reader inputs the job read (2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1/inputs, streamed from the
Volume; text stays in memory). Each run's output_end positions come from the reader itself
(probe_shift.reader.segment_end_positions with Qwen3-32B's tokenizer at the pinned revision, max_tokens 32768): an output
is kept (token < cut), has no content token (token None, e.g. empty, or a lone "." merged with the template newline), or
lies past the cut (token >= cut). Copy and twin outputs are matched by (step, action); an output changed by the plant is
any whose text differs.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with 'transformers==4.56.2' --with jinja2 \\
      python experiments/2026-10-11_claimB_yardsticks/analyze_POSTHOC_plant_cut.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
RUN_L1 = "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1"
ARMS = ("sim-attack-fixed", "sim-attack-fixed-plant1", "sim-attack-fixed-plant2")
MAX_TOKENS = 32768


def lines_of(vol, path, hashes):
    buf, h = b"", hashlib.sha256()
    for chunk in vol.read_file(path):
        h.update(chunk)
        buf += chunk
        *done, buf = buf.split(b"\n")
        yield from done
    if buf:
        yield buf
    hashes[path] = h.hexdigest()


def main():
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B
    from probe_shift.reader import segment_end_positions

    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    rows, hashes = {}, {}
    for s in range(4):
        for line in lines_of(vol, f"{RUN_L1}/inputs/shard_{s}.jsonl", hashes):
            r = json.loads(line)
            if r["arm"] in ARMS:
                rows[(r["arm"], r["key"])] = r["messages"]
    plant_keys = {n: sorted(k for a, k in rows if a == f"sim-attack-fixed-plant{n}") for n in (1, 2)}
    needed = {(f"sim-attack-fixed-plant{n}", k) for n, ks in plant_keys.items() for k in ks} | {
        ("sim-attack-fixed", k.removesuffix(f"__plant{n}")) for n, ks in plant_keys.items() for k in ks}

    status = {}  # (arm, key) -> {(step, action): (state, text)}
    for ak in sorted(needed):
        msgs = rows[ak]
        seg = segment_end_positions(tok, msgs, MAX_TOKENS)
        texts, step, action = {}, -1, -1
        for m in msgs:
            if m["role"] == "assistant":
                step, action = step + 1, -1
            elif m["role"] == "tool":
                action += 1
                texts[(step, action)] = m["content"]
        st = {}
        for p in seg["positions"]:
            if p["kind"] == "output_end":
                state = "kept" if p["kept"] else ("no_token" if p["token"] is None else "past_cut")
                st[(p["step"], p["action"])] = (state, texts[(p["step"], p["action"])])
        if set(st) != set(texts):
            raise ValueError(f"{ak}: positions and tool messages disagree")
        status[ak] = st

    out = {"label": "POST HOC (#853 PRRT_kwDOU9RNXc6rYPN1): plant documents vs the reader's 32k cut (counts only)", "plants": {}}
    for n in (1, 2):
        c = dict.fromkeys(["pairs", "copy_has_outputs_past_cut", "twin_has_outputs_past_cut", "either_has_outputs_past_cut",
                           "outputs_past_cut_copy", "outputs_past_cut_twin", "no_token_outputs_copy", "no_token_outputs_twin",
                           "changed_outputs", "changed_outputs_past_cut_in_copy_or_twin", "pairs_with_a_changed_output_past_cut",
                           "pairs_with_every_changed_output_kept_in_both"], 0)
        for k in plant_keys[n]:
            cp, tw = status[(f"sim-attack-fixed-plant{n}", k)], status[("sim-attack-fixed", k.removesuffix(f"__plant{n}"))]
            if set(cp) != set(tw):
                raise ValueError(f"{k}: copy and twin outputs are not aligned")
            c["pairs"] += 1
            pc = sum(v[0] == "past_cut" for v in cp.values())
            pt = sum(v[0] == "past_cut" for v in tw.values())
            c["copy_has_outputs_past_cut"] += pc > 0
            c["twin_has_outputs_past_cut"] += pt > 0
            c["either_has_outputs_past_cut"] += pc > 0 or pt > 0
            c["outputs_past_cut_copy"] += pc
            c["outputs_past_cut_twin"] += pt
            c["no_token_outputs_copy"] += sum(v[0] == "no_token" for v in cp.values())
            c["no_token_outputs_twin"] += sum(v[0] == "no_token" for v in tw.values())
            changed = [sa for sa in cp if cp[sa][1] != tw[sa][1]]
            past = [sa for sa in changed if "past_cut" in (cp[sa][0], tw[sa][0])]
            c["changed_outputs"] += len(changed)
            c["changed_outputs_past_cut_in_copy_or_twin"] += len(past)
            c["pairs_with_a_changed_output_past_cut"] += bool(past)
            c["pairs_with_every_changed_output_kept_in_both"] += bool(changed) and all(
                cp[sa][0] == "kept" and tw[sa][0] == "kept" for sa in changed)
        out["plants"][f"plant{n}"] = c
    out |= {"inputs": hashes, "tokenizer": QWEN3_32B,
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"plant_cut_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out["plants"], indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
