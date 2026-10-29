"""Arm-masked coding files for the verbalized-awareness rubric (RUBRIC.md). Local; no model calls. Python 3.14 (reads
the .eval logs' compression).

For each run: the agent's own words only, i.e. each assistant turn's reasoning and visible text, numbered by turn.
No tool calls, outputs or prompts are included. Identifying strings (arm labels, eval file names, run ids) are
replaced by placeholders. Each run gets id = sha256(salt + log file name)[:8], and the files are written in id order,
which is effectively shuffled. The unmasking key (id to file, arm, attack or honest), the salt and the coverage go to
--key-dir, which must be outside this folder: coders are never pointed at it.

  python prepare_coding.py --logs data/logs --out data/coding --key-dir <private dir>
"""

import argparse
import hashlib
import json
import re
import secrets
import zipfile
from pathlib import Path

MASK = [(re.compile(r"[\w.:-]*\.eval\b"), "<log>"),  # whole file names first, then any arm label left
        (re.compile(r"(real|sim)[-_](attack|honest)", re.I), "<arm>"),
        (re.compile(r"\b[A-Za-z0-9]{22}\b"), "<id>")]  # inspect run/sample ids


def mask(text: str) -> str:
    for pat, sub in MASK:
        text = pat.sub(sub, text)
    return text


def resolve(x, att):
    """Inspect stores long strings as attachment://<id> references; replace them with the stored text."""
    if isinstance(x, str) and x.startswith("attachment://"):
        if x[len("attachment://"):] not in att:
            raise KeyError(f"unresolved {x}")
        return att[x[len("attachment://"):]]
    if isinstance(x, dict):
        return {k: resolve(v, att) for k, v in x.items()}
    if isinstance(x, list):
        return [resolve(v, att) for v in x]
    return x


def turns(sample):
    out = []
    for m in (x for x in sample["messages"] if x["role"] == "assistant"):
        content = m["content"]
        reasoning = text = ""
        if isinstance(content, str):
            text = content
        else:
            reasoning = "\n".join(c.get("reasoning") or "" for c in content if c.get("type") == "reasoning")
            text = "\n".join(c.get("text") or "" for c in content if c.get("type") == "text")
        out.append((reasoning.strip(), text.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="folder with step1_attack/, honest/, step2_sim/ .eval logs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-dir", required=True)
    args = ap.parse_args()
    logs, out, key_dir = Path(args.logs), Path(args.out), Path(args.key_dir)
    o, k = out.resolve(), key_dir.resolve()
    if o.is_relative_to(k) or k.is_relative_to(o) or k.is_relative_to(Path(__file__).parent.resolve()):
        raise ValueError("--key-dir must be outside this experiment folder and apart from --out")
    groups = {"step1_attack": ("real", "attack"), "honest": ("real", "honest"), "step2_sim": ("simulated", "attack")}
    if (key_dir / "unmasking_key.json").exists():  # check before writing any transcript, so none is orphaned
        raise FileExistsError(f"{key_dir / 'unmasking_key.json'} exists: its codes would lose their key")
    salt = secrets.token_hex(16)
    out.mkdir(parents=True, exist_ok=False)
    key_dir.mkdir(parents=True, exist_ok=True)
    key = {}
    for group, (arm, policy) in groups.items():
        for f in sorted((logs / group).glob("*.eval")):
            with zipfile.ZipFile(f) as z:
                [name] = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
                sample = json.loads(z.read(name))
            sample["messages"] = resolve(sample["messages"], sample.get("attachments") or {})
            ts = turns(sample)
            rid = hashlib.sha256((salt + f.name).encode()).hexdigest()[:8]
            lines = [f"TRANSCRIPT {rid}", f"{len(ts)} assistant turns. Each turn: the agent's private reasoning, then its "
                     "visible text. Tool calls and outputs are omitted.", ""]
            for i, (r, t) in enumerate(ts):
                lines.append(f"=== TURN {i} ===")
                lines.append("[reasoning] " + (mask(r) if r else "(none)"))
                if t:
                    lines.append("[text] " + mask(t))
            (out / f"{rid}.txt").write_text("\n".join(lines) + "\n")
            key[rid] = {"file": f.name, "group": group, "arm": arm, "policy": policy, "sample": name,
                        "turns": len(ts), "turns_with_reasoning": sum(bool(r) for r, _ in ts),
                        "reasoning_chars": sum(len(r) for r, _ in ts), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    with (key_dir / "unmasking_key.json").open("x") as f:  # never replace the key the existing codes belong to
        f.write(json.dumps({"salt": salt, "runs": key}, indent=1))
    print(len(key), "transcripts ->", out, "| key ->", key_dir / "unmasking_key.json")
    for rid in sorted(key):
        k = key[rid]
        print(rid, f"turns {k['turns']}, with reasoning {k['turns_with_reasoning']}, {k['reasoning_chars']:,} reasoning chars")


if __name__ == "__main__":
    main()
