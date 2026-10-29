"""RULE.md: per arm and tool, how often a call's command-end readout token and its content-only end-of-call token are the
same position, on the T196 corrected-B staged inputs plus batch C. Scores-free: it reads only the staged inputs' call rows
(tool name, command-parameter name, readout positions) and token ids at the end-of-call position (for the fixed-token rule).
No activation, score or argument value is read. All work runs in one Modal CPU container, which writes its own record.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach \\
      experiments/2026-10-14_p2_cmd_vs_callend/run_cmd_vs_callend.py --manifest <correctedb_manifest_v1.json>
"""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_cmdcallend_kill-2026-11-13"
BVOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
OUTVOL = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
FOLDER = "2026-10-14_p2_cmd_vs_callend"
TOK = "hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654/tokenizer.json"
PINS = {"manifest": "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc",
        "c_inputs": "513f621a9e71e91f8188cccc76cb222e9c9f4acb5a91fc90c45a283964729b31"}
C_STAGE = "q4f_c_v1/inputs/honest_b.jsonl"  # batch C staged inputs, world honest_c (as run_score_linear.py)
FIXED = ("0", "n")  # fixed call tokens: content-only end-of-call drops call_end rows on them (run_score_linear.py:35)
# text_editor by its command parameter, which reader.command_param picks from the sub-command (reader.py:198-216)
EDITOR_PARAM = {"file_text": "create", "new_str": "str_replace|insert(new_str)", "insert_text": "insert(insert_text)",
                "path": "view|undo_edit", None: "no command value"}

app = modal.App(APP)
b_vol = modal.Volume.from_name(BVOL, version=2).read_only()
hf_vol = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()
out_vol = modal.Volume.from_name(OUTVOL, version=2)
image = modal.Image.debian_slim(python_version="3.12")


def tool_cat(name: str, param) -> str:
    if name == "bash" or name.endswith("_bash"):
        return "bash"
    if name == "text_editor":
        return "text_editor:" + EDITOR_PARAM.get(param, f"other param ({param})")
    return "other tool"


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=900, retries=0,
              volumes={"/b": b_vol, "/hf": hf_vol, "/out": out_vol})
def count(epochs: list, meta: dict) -> dict:
    vocab = json.loads((Path("/hf") / TOK).read_text())["model"]["vocab"]
    fixed_ids = {vocab[t] for t in FIXED}
    want = {(e["staged_inputs_file"], e["key"]): e["world"] for e in epochs}
    sha_of = {e["staged_inputs_file"]: e["staged_inputs_sha256"] for e in epochs} | {C_STAGE: PINS["c_inputs"]}
    tally, seen, fails = {}, Counter(), []
    for f in sorted(sha_of):
        raw = (Path("/b") / f).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha_of[f]:
            fails.append(f"{f}: sha256 differs from the manifest's (or C's pin)")
            continue
        for line in raw.decode().splitlines():
            ep = json.loads(line)
            world = "honest_c" if f == C_STAGE else want.get((f, ep["key"]))
            if world is None:
                continue  # a staged epoch the manifest does not use
            seen[world] += 1
            by_call = {}
            for r in ep["calls"]:
                if r["kind"] in ("command", "call_end"):
                    slot = by_call.setdefault((r["irid"], r["call_index"]), {})
                    if r["kind"] in slot:
                        fails.append(f"{world}|{ep['task']}: duplicate {r['kind']} row for one call")  # Codex round 1
                    slot[r["kind"]] = r
            for (_irid, _ci), rows in by_call.items():
                cmd, ce = rows.get("command"), rows.get("call_end")
                if cmd is None or ce is None:
                    fails.append(f"{world}|{ep['task']}: a call without both rows")
                    continue
                cat = tool_cat(cmd["name"], cmd["param"])
                t = tally.setdefault(world, {}).setdefault(cat, Counter())
                t["calls"] += 1
                rc, re_ = cmd["readout"], ce["readout"]
                ce_fixed = re_ is not None and ep["ids"][re_] in fixed_ids
                if rc is None:
                    t["no command readout: " + ("no command value" if cmd["param"] is None else "beyond the cut, unmatched, or no whole token inside the value")] += 1  # reader.py:259 (Codex round 1)
                if re_ is None:
                    t["no end-of-call readout: beyond the cut or unmatched"] += 1
                elif ce_fixed:
                    t["end-of-call dropped: fixed call token"] += 1
                if rc is not None and re_ is not None and not ce_fixed:
                    t["both"] += 1
                    t["same token"] += int(rc == re_)
                    if rc != re_:
                        t["command-end before end-of-call" if rc < re_ else "command-end after end-of-call"] += 1
    expect = Counter(e["world"] for e in epochs) | Counter({"honest_c": 192})
    if seen != expect:
        fails.append(f"epochs seen {dict(seen)} != expected {dict(expect)}")
    out = {w: {c: dict(v) for c, v in sorted(cats.items())} for w, cats in sorted(tally.items())}
    overall = {}
    for w, cats in out.items():
        b = sum(v.get("both", 0) for v in cats.values())
        s = sum(v.get("same token", 0) for v in cats.values())
        overall[w] = {"both": b, "same token": s, "share same": (s / b) if b else None}
    rec = meta | {"epochs_seen": dict(seen), "fails": fails, "fixed_ids": sorted(fixed_ids), "by_world_tool": out, "overall": overall}
    d = Path("/out") / FOLDER / meta["stamp (PT)"]
    d.mkdir(parents=True, exist_ok=False)
    (d / "cmd_vs_callend.json").write_text(json.dumps(rec, indent=1))
    out_vol.commit()
    return rec


@app.local_entrypoint()
def main(manifest: str):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    raw = Path(manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS["manifest"]:
        raise SystemExit("manifest sha256 differs from its pin")
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    meta = {"label": "scores-free count; PRELIMINARY until Codex", "git": git, "stamp (PT)": stamp, "app": APP,
            "inputs": {"manifest": PINS["manifest"], "c_inputs": PINS["c_inputs"], "tokenizer": TOK},
            "volume_copy": f"{OUTVOL}:{FOLDER}/{stamp}/cmd_vs_callend.json"}
    rec = count.remote(json.loads(raw)["epochs"], meta)
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / f"cmd_vs_callend_{stamp}.json").write_text(json.dumps(rec, indent=1))
    lines = [f"cmd vs end-of-call {stamp}; git {git[:9]}; epochs {rec['epochs_seen']}; fails {len(rec['fails'])}"]
    for w, o in rec["overall"].items():
        lines.append(f"{w}: same token {o['same token']} of {o['both']} calls with both ({o['share same']:.3f})" if o["both"] else f"{w}: none")
        for c, v in rec["by_world_tool"][w].items():
            lines.append(f"  {c}: " + ", ".join(f"{k} {n}" for k, n in sorted(v.items())))
    (res / f"cmd_vs_callend_{stamp}.printout.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if rec["fails"]:
        raise SystemExit(f"{len(rec['fails'])} fails: see the record")
