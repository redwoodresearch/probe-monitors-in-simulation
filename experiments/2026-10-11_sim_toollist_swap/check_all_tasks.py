"""Per task, real vs simulated served tool lists: equal after setting additionalProperties and required aside? Token
counts of the chat template's rendered tool block (Qwen3.5-27B fc05daec tokenizer), with a fixed dummy system and user
message, never run text. Counts and booleans only."""
import json, sys, zipfile
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
acting = "vllm/Qwen/Qwen3.5-27B"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
def tools_all(ev):
    with zipfile.ZipFile(ev) as z:
        s = json.loads(z.read([n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")][0]))
    ts = [json.dumps(e["call"]["request"]["tools"], sort_keys=True) for e in s["events"] if e.get("event") == "model" and e.get("model") == acting]
    return [json.loads(t) for t in dict.fromkeys(ts)]
def canon(tools):
    out = []
    for t in tools:
        f = json.loads(json.dumps(t["function"])); p = f.get("parameters") or {}
        p.pop("additionalProperties", None); p.pop("required", None); out.append(f)
    return out
def ntok(tools):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    text = tok.apply_chat_template(msgs, tools=[t for t in tools], tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])
root = Path(sys.argv[1]); c = Counter(); dtok = Counter()
for run, real_arm, sim_arm in [(r, a, b) for r, a, b in (x.split(":") for x in sys.argv[2].split(","))]:
    for tdir in sorted((root / run / real_arm).iterdir()):
        se = sorted((root / run / sim_arm / tdir.name).glob("*.eval")); re_ = sorted(tdir.glob("*.eval"))
        if not se or not re_:
            c[f"{run} {real_arm}/{sim_arm}: task missing in one arm"] += 1; continue
        R, Sx = tools_all(re_[0]), tools_all(se[0])
        if not R or not Sx:
            c[f"{run} {real_arm}/{sim_arm}: no acting-model event in one arm"] += 1; continue
        c[f"{run} {real_arm}: runs with >1 distinct tool list"] += len(R) > 1; c[f"{run} {sim_arm}: runs with >1 distinct tool list"] += len(Sx) > 1
        same = canon(R[0]) == canon(Sx[0]); c[f"{run} {real_arm}/{sim_arm}: equal apart from additionalProperties/required = {same}"] += 1
        rr = all((t["function"].get("parameters") or {}).get("additionalProperties") is False for t in R[0] if t["function"]["name"] != "submit")
        ss = all((t["function"].get("parameters") or {}).get("additionalProperties") is True and not (t["function"].get("parameters") or {}).get("required") for t in Sx[0] if t["function"]["name"] != "submit")
        c[f"{run}: real all addProps false={rr}, sim all addProps true & required empty={ss}"] += 1
        dtok[ntok(R[0]) - ntok(Sx[0])] += 1
print(json.dumps({"counts": dict(c), "token_difference_real_minus_sim_of_rendered_prompt": {str(k): v for k, v in sorted(dtok.items())}}, indent=1))
