"""Per run set and task, the rendered tool-definition lines of the real (attack) and simulated (sim_attack@september)
runs, as token ids (Qwen3.5-27B fc05daec tokenizer), from the structured `call.request.tools` field of acting-model
events only: no message text is read. Each line is the chat template's own `tool | tojson` line inside <tools>; the
check that each line's ids appear verbatim in a full rendered prompt guards the splice. Local, $0.

  uv run --no-project --python 3.14 --with transformers --with jinja2 python build_tool_lines.py --runs <phase2 runs dir> --out <json>
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

ACTING = "vllm/Qwen/Qwen3.5-27B"
SETS = {"rerun_strict_segends_v2": "2026-10-09_000523_rerun_strict", "scaleup_canary_segends_v1": "2026-10-09_173815_scaleup_canary",
        "scaleup_l2_segends_v1": "2026-10-09_182505_scaleup", "scaleup_topup_segends_v1": "2026-10-10_001314_scaleup_topup",
        "second_attack_segends_v1": "2026-10-10_001314_second_attack"}


REF = re.compile(r"attachment://[0-9a-f]{32}")


def resolved(x, att):
    """Inspect stores long strings of model events (here, parameter descriptions) in the sample's attachments map; the
    served prompt carried the text. Whole-value references are resolved at any depth; a missing key or a reference
    left after resolution raises."""
    if isinstance(x, str) and REF.fullmatch(x):
        return att[x.removeprefix("attachment://")]
    if isinstance(x, list):
        return [resolved(v, att) for v in x]
    if isinstance(x, dict):
        return {k: resolved(v, att) for k, v in x.items()}
    return x


def tools_of(ev):
    with zipfile.ZipFile(ev) as z:
        s = json.loads(z.read([n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")][0]))
    for e in s.get("events") or []:
        if e.get("event") == "model" and e.get("model") == ACTING:
            t = resolved(e["call"]["request"]["tools"], s.get("attachments") or {})
            if REF.search(json.dumps(t)):
                raise ValueError(f"{ev.name}: an attachment reference is left in the tools")
            return t
    return None


def main(runs: str, out: str):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

    def lines(tools):
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
        m = re.search(r"<tools>\n(.*?)\n</tools>", text, re.S)
        ls = [x for x in m.group(1).split("\n") if x.strip()]
        full = tok(text, add_special_tokens=False)["input_ids"]
        res = []
        for x in ls:
            ids = tok(x, add_special_tokens=False)["input_ids"]
            hits = [i for i in range(len(full) - len(ids) + 1) if full[i:i + len(ids)] == ids]
            if len(hits) != 1:
                raise ValueError(f"a tool line is not found exactly once in its rendered prompt ({len(hits)})")
            res.append({"name": (json.loads(x).get("function") or json.loads(x)).get("name"), "ids": ids,
                        "sha256": hashlib.sha256(x.encode()).hexdigest()})
        return res

    res, counts = {}, {}
    for stage, folder in SETS.items():
        root = Path(runs) / folder
        res[stage] = {}
        for arm in ("attack", "sim_attack@september"):
            if not (root / arm).exists():
                continue
            for tdir in sorted((root / arm).iterdir()):
                evs = sorted(tdir.glob("*.eval"))
                tl = [t for t in (tools_of(ev) for ev in evs) if t]
                if not tl:
                    counts[f"{stage}/{arm}: task without an acting-model event"] = counts.get(f"{stage}/{arm}: task without an acting-model event", 0) + 1
                    continue
                if any(json.dumps(t, sort_keys=True) != json.dumps(tl[0], sort_keys=True) for t in tl):
                    raise ValueError(f"{stage}/{arm}/{tdir.name}: attempts carry different tool lists")
                res[stage].setdefault(tdir.name, {})["real" if arm == "attack" else "sim"] = lines(tl[0])
        for t, v in res[stage].items():
            if "real" in v and "sim" in v:
                if [x["name"] for x in v["real"]] != [x["name"] for x in v["sim"]]:
                    raise ValueError(f"{stage}/{t}: tool names or order differ")
                counts[f"{stage}: tasks with both"] = counts.get(f"{stage}: tasks with both", 0) + 1
                counts[f"{stage}: lines that differ"] = counts.get(f"{stage}: lines that differ", 0) + sum(a["ids"] != b["ids"] for a, b in zip(v["real"], v["sim"]))
    Path(out).write_text(json.dumps({"tokenizer": "Qwen/Qwen3.5-27B@fc05daec", "counts": counts, "sets": res}))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    main(**vars(ap.parse_args()))
