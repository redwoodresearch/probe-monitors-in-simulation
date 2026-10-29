"""09's independent raw-.eval read for the IL donor gate (d9's IL rule amendment @ c1c770694, check (iv) "09's
raw-.eval read"). Opens the seeded selection's .eval files on the chain Volume with zipfile + json only (no chain or
repo code) and reports STRUCTURE ONLY: message roles and counts, the first user turn's presence, part types, length and
sha256, the sample input's shape, each system message's length and sha256 plus two booleans (cover-task block, agent
preamble), assistant steps, and allowlisted error class names. Never prints, returns or stores transcript text: every
failure becomes a fixed status plus an exception class name. Modal CPU, read-only Volume. PRELIMINARY.

  $MODAL run experiments/2026-10-15_raw_eval_read_09/run_raw_eval_read.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-convaudit_rawread_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"  # the chain's .eval files under evals_v1/
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # sha256("")
ROLES = {"system", "user", "assistant", "tool"}
PART_TYPES = {"text", "image", "audio", "video", "reasoning", "content_reasoning", "data", "document", "tool_use"}
ERRORS = (  # allowlist: only these names are ever reported
    "RateLimitError",
    "RetryError",
    "EmptyPolicyCompletionError",
    "OutputContractError",
    "AttemptTimeoutError",
    "TimeoutError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "BadRequestError",
)

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.14")  # stdlib zipfile reads the zstd-compressed .eval members from 3.14
vol = modal.Volume.from_name(VOL, version=2).read_only()


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def describe(sample: dict) -> dict:
    import re

    att = sample.get("attachments") or {}

    def resolve(s: str) -> str:
        if s.startswith("attachment://"):
            v = att.get(s[len("attachment://") :])
            if not isinstance(v, str):
                raise ValueError("unresolved attachment")
            return v
        return s

    def content_info(content) -> dict:
        """text as Inspect's .text (text parts joined by newlines); None when nothing carries text."""
        if isinstance(content, str):
            return {
                "kind": "str",
                "parts": [],
                "text": resolve(content),
                "non_text_parts": 0,
                "malformed_text_parts": 0,
            }
        if isinstance(content, list):
            parts = [p.get("type") if isinstance(p, dict) else None for p in content]
            tparts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
            malformed = sum(not isinstance(p.get("text"), str) for p in tparts)
            texts = [resolve(p["text"]) for p in tparts if isinstance(p.get("text"), str)]
            return {
                "kind": "list",
                "parts": [t if t in PART_TYPES else "other" for t in parts],
                "text": "\n".join(texts) if texts and not malformed else None,  # malformed text part: no measurement
                "non_text_parts": sum(t != "text" for t in parts),
                "malformed_text_parts": malformed,
            }
        kind = "absent" if content is None else "other"
        return {"kind": kind, "parts": [], "text": None, "non_text_parts": 0, "malformed_text_parts": 0}

    msgs = sample.get("messages")
    if not isinstance(msgs, list):
        raise ValueError("messages not a list")
    roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
    systems = [content_info(m.get("content")) for m, r in zip(msgs, roles, strict=True) if r == "system"]
    fu = next((m for m, r in zip(msgs, roles, strict=True) if r == "user"), None)
    fi = content_info(fu.get("content")) if fu else None
    inp = sample.get("input")
    if isinstance(inp, list):
        ir = [m.get("role") if isinstance(m, dict) else None for m in inp]
        iu = [content_info(m.get("content"))["text"] for m, r in zip(inp, ir, strict=True) if r == "user"]
        inp_info = {"kind": f"messages[{len(inp)}]", "roles": [r if r in ROLES else "other" for r in ir][:4]}
        inp_info["user_text_len"] = [None if t is None else len(t) for t in iu]
    else:
        t = content_info(inp)["text"]
        inp_info = {"kind": content_info(inp)["kind"], "text_len": None if t is None else len(t)}
    err = sample.get("error")
    msg = err.get("message") if isinstance(err, dict) else None
    head = (
        re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?:\(|: )", msg) if isinstance(msg, str) else None
    )  # "X(" or "X: " at start
    wrapped = (  # only tenacity's exact RetryError wrapper header
        re.match(r"RetryError\[<Future at 0x[0-9a-f]+ state=finished raised ([A-Za-z_][A-Za-z0-9_]*)>\]", msg)
        if isinstance(msg, str)
        else None
    )
    sys_text = [s["text"] for s in systems]
    return {
        "n_messages": len(msgs),
        "first_roles": [r if r in ROLES else "other" for r in roles[:3]],
        "n_assistant": roles.count("assistant"),
        "n_system": len(systems),
        "system_len": [None if t is None else len(t) for t in sys_text],
        "system_sha256": [None if t is None else sha(t) for t in sys_text],
        "system_has_cover_block": [bool(t) and "<cover_task>" in t for t in sys_text],
        "system_has_agent_preamble": [bool(t) and "You are an agent" in t for t in sys_text],
        "first_user_present": fu is not None,
        "first_user_kind": fi["kind"] if fi else None,
        "first_user_part_types": fi["parts"] if fi else None,
        "first_user_non_text_parts": fi["non_text_parts"] if fi else None,
        "first_user_malformed_text_parts": fi["malformed_text_parts"] if fi else None,
        "first_user_len": None if not fi or fi["text"] is None else len(fi["text"]),
        "first_user_sha256": None if not fi or fi["text"] is None else sha(fi["text"]),
        "input": inp_info,
        "error_present": bool(err),
        # exact allowlist match on the message's leading class name and on a "raised X" wrapper; anything else -> None
        "error_header": "RetryError" if wrapped else (head.group(1) if head and head.group(1) in ERRORS else None),
        "error_wrapped": wrapped.group(1) if wrapped and wrapped.group(1) in ERRORS else None,
    }


@app.function(image=image, volumes={"/ev": vol}, cpu=2, memory=4096, timeout=600, retries=0)
def read(files: list) -> list:
    import re
    import zipfile

    out = []
    for f in files:
        rec = {k: f[k] for k in ("category", "path", "task", "arm_dir")}
        p = Path("/ev/evals_v1") / f["path"]
        try:
            if not p.exists():
                out.append(rec | {"status": "missing on Volume"})
                continue
            if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
                out.append(rec | {"status": "sha256 differs from manifest"})
                continue
            with zipfile.ZipFile(p) as z:
                names = sorted(set(z.namelist()))
                samples = [n for n in names if re.fullmatch(r"samples/[^/]+\.json", n)]
                other = [n for n in names if n.startswith("samples/") and n not in samples]
                if other:
                    out.append(rec | {"status": "unsupported sample layout", "n_other_sample_members": len(other)})
                    continue
                if not samples:
                    out.append(rec | {"status": "no samples"})
                    continue
                rec |= {
                    "status": "ok",
                    "n_samples": len(samples),
                    "samples": [describe(json.loads(z.read(n))) for n in samples],
                }
        except Exception as e:  # noqa: BLE001  fixed status + class name only: messages can carry transcript text
            rec = {k: f[k] for k in ("category", "path", "task", "arm_dir")} | {
                "status": "read failed",
                "exception": type(e).__name__,
            }
        out.append(rec)
    return out


@app.local_entrypoint()
def main():
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()

    if git("status", "--porcelain", "--", str(HERE)):
        raise SystemExit("refusing to run: commit this folder first")
    sel = json.loads((HERE / "config" / "selection.json").read_text())
    res = read.remote(sel["files"])
    ok = [(r, s) for r in res if r["status"] == "ok" for s in r["samples"]]
    honest = [(r, s) for r, s in ok if r["arm_dir"] in ("honest_a", "honest_b")]
    hs = [(r["task"], s["system_sha256"][0]) for r, s in honest if s["n_system"] == 1 and s["system_len"][0]]
    by_hash = {}
    for t, h in hs:
        by_hash.setdefault(h, set()).add(t)
    summary = {
        "files": len(res),
        "status": {k: sum(r["status"] == k for r in res) for k in sorted({r["status"] for r in res})},
        "samples": len(ok),
        "first_user_present": sum(s["first_user_present"] for _, s in ok),
        "first_user_text_empty": sum(s["first_user_len"] == 0 for _, s in ok),
        "first_user_no_text_content": sum(s["first_user_present"] and s["first_user_len"] is None for _, s in ok),
        "first_user_sha_is_empty_string_hash": sum(s["first_user_sha256"] == EMPTY for _, s in ok),
        "first_user_kinds": sorted({str(s["first_user_kind"]) for _, s in ok}),
        "samples_with_more_than_one_system": sum(s["n_system"] > 1 for _, s in ok),
        "first_system_message_nonempty": sum(s["n_system"] >= 1 and bool(s["system_len"][0]) for _, s in ok),
        "positive_control_honest": {
            "honest_samples": len(honest),
            "honest_excluded_from_control (no single non-empty system message)": len(honest) - len(hs),
            "distinct_system_hashes": len(by_hash),
            "distinct_tasks": len({t for t, _ in hs}),
            "hashes_shared_across_tasks": sum(len(v) > 1 for v in by_hash.values()),
        },
    }
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"raw_eval_read_{stamp}.json"
    meta = sel["source"] | {"seed": sel["seed"], "draw": sel["draw"]}
    record = {"git": git("rev-parse", "HEAD"), "stamp": stamp, "selection": meta, "summary": summary, "files": res}
    dest.open("x").write(json.dumps(record, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", dest.name)
