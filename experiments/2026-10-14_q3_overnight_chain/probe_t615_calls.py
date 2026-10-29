"""Keys-only probe of ONE attack run's tool calls (d9 GO 10-15 ~23:50; 09 rules on the labeler gap that crashed the b_run2 dry staging).
Attack run: NO text is printed. Per assistant tool call: its index (step, call), the tool name, the argument KEY names; for text_editor the `command`
value only if it is in the known enum {view, create, str_replace, insert, undo_edit} (else "<other, len N>"); for shell tools whether a command
string exists and its type; and whether probe_shift.calls.label_call accepts the call (else the ValueError's rule kind, from a fixed list, never
its message). The .eval is read from the chain Volume, sha256-checked against the committed dry manifest.

  modal run probe_t615_calls.py --path <evals_v1-relative path> --sha256 <hex> --out <local json>
"""
import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("zstandard").add_local_python_source("probe_shift")
         .add_local_file(HERE / "evalzip.py", "/root/evalzip.py"))  # the stager's own zstd (.eval method 93) reader
app = modal.App("probe-shift_aswin-surya_sub-q3chain_probe_kill-2026-11-14")
ENUM = {"view", "create", "str_replace", "insert", "undo_edit"}
KINDS = (("text_editor command with no rule", "text_editor command with no rule"), ("call without a command string", "shell call without a command string"))


@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def probe(path: str, sha256: str) -> dict:
    import sys
    import zipfile

    sys.path.insert(0, "/root")
    import evalzip  # noqa: F401  (registers zip method 93, as stage_q4b does)
    from probe_shift import calls as C

    p = Path("/ev/evals_v1") / path
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("the .eval's sha256 differs from the manifest row")
    import io

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        s = json.loads(z.read(n))
    rows, step = [], 0
    for m in s.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for i, c in enumerate(m.get("tool_calls") or []):
            f = c.get("function")
            name = f.get("name") if isinstance(f, dict) else c.get("function")
            args = (f.get("arguments") if isinstance(f, dict) else c.get("arguments")) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"<unparsed arguments string>": None}
            row = {"step": step, "call": i, "tool": name, "arg_keys": sorted(args) if isinstance(args, dict) else f"<{type(args).__name__}>"}
            if name == "text_editor" and isinstance(args, dict):
                v = args.get("command")
                row["command"] = v if v in ENUM else (None if v is None else f"<other, len {len(str(v))}>")
            if name in C.SHELL_TOOLS and isinstance(args, dict):
                v = args.get("cmd", args.get("command"))
                row["has_command_string"], row["command_type"] = isinstance(v, str), type(v).__name__
            try:
                C.label_call(name, args if isinstance(args, dict) else {})
                row["label_call"] = "ok"
            except ValueError as e:
                row["label_call"] = next((k for pat, k in KINDS if pat in str(e)), "ValueError (other rule; message withheld)")
            except Exception as e:  # never its message
                row["label_call"] = f"{type(e).__name__} (message withheld)"
            rows.append(row)
        step += 1
    bad = [r for r in rows if r["label_call"] != "ok"]
    return {"path": path, "sha256": sha256, "assistant_steps": step, "calls": len(rows), "tools": sorted({str(r["tool"]) for r in rows}),
            "label_call_failures": bad}


@app.local_entrypoint()
def main(path: str, sha256: str, out: str):
    r = probe.remote(path, sha256)
    Path(out).open("x").write(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: r[k] for k in ("assistant_steps", "calls", "tools")} | {"failures": len(r["label_call_failures"])}))
