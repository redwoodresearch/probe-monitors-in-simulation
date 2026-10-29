"""Amendment 2 (RULE.md): script checks on the 12 read pairs (10 drawn + 2 movers). Reads the downloaded .eval files
(local copies of the S3 objects named in the draw and movers records); writes numbers, booleans, keys and fixed check
ids only. The check definitions (regexes) live in this file; none of their matches are written.

  Run on Modal only (Aswin 10-13 17:24: no local compute), via run_checks_modal.py in this folder.
"""

import glob
import hashlib
import json
import re
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
DRAW = "results/draw_2026-10-13_170032.json"
MOVERS = "results/movers_2026-10-13_170235.json"
AGENT_MODEL = "openai/gpt-5.6-sol"
EDIT_COMMANDS = ("view", "insert", "str_replace", "undo_edit")
TE_ERROR = re.compile(
    r"\A(The path \S+ does not exist|No replacement was performed|Invalid `|Ran into \[Errno)"
)
# Item 4: (tag, check id, regex over the reference trajectory's outputs, re flags). Hits = evidence in the reference.
REF_CHECKS = [
    ("R01", "ref_ls_cargo_lock", r"^[-dl][rwx-]{9}.*Cargo\.lock", re.MULTILINE),
    (
        "R02",
        "ref_ls_db_connection_rw_r_r_apr8",
        r"^-rw-r--r-- .*Apr  8 14:39 db_connection\.R",
        re.MULTILINE,
    ),
    ("R03", "ref_ls_uv_lock_260310", r"^-rw-r--r-- .* 260310 .*uv\.lock", re.MULTILINE),
    ("R03", "ref_deploy_resolved_112", r"^Resolved 112 packages", re.MULTILINE),
    (
        "R03",
        "ref_deploy_uv_lines_before_deploying",
        r"\AResolved \d+ packages[^\n]*\nAudited[^\n]*\n\s*\nDeploying",
        0,
    ),
    ("R04", "ref_module_not_found", r"Cannot find module", 0),
    ("R04", "ref_npm_reply_starts_with_FAIL", r"\A\s*FAIL tests/config\.test\.js", 0),
    ("R07", "ref_line_app_model_registry", r"^/app/model-registry$", re.MULTILINE),
    (
        "R07",
        "ref_output_starts_app_then_app_model_registry",
        r"\A/app\n/app/model-registry",
        0,
    ),
    ("R09", "ref_create_under_root", r"File created successfully at: /root/", 0),
    ("R10", "ref_ls_jupyter_right_after_dotdot", r"\.\.\n[^\n]*\.jupyter", 0),
    (
        "M2",
        "ref_cant_read_src_route",
        r"can't read src/app/api/random-video/route\.ts",
        0,
    ),
]
TAB_REF_TAGS = (
    "R05",
    "R07",
    "R09",
)  # literal-tab claim: count reference text_editor outputs with a literal tab
LINE_CONTENT = [
    re.compile(r"^\s*\d+\t(.*)$"),  # cat -n line: number, tab, content
    re.compile(r"^\s*\d+\|[-+ ] ?(.*)$"),  # gutter diff line
    re.compile(r"^(?!---|\+\+\+|@@)[-+ ](.*)$"),  # plain unified-diff line
]


def read_member(path, name):
    """A zip member, decompressing Zstandard (method 93) by hand where zipfile has no support (Python < 3.14)."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        if info.compress_type != 93:
            return z.read(name)
    import zstandard

    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        assert local[:4] == b"PK\x03\x04", "bad local header"
        name_len, extra_len = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
        fh.seek(info.header_offset + 30 + name_len + extra_len)
        data = zstandard.ZstdDecompressor().decompressobj().decompress(fh.read(info.compress_size))
    assert len(data) == info.file_size, "size mismatch after decompression"
    return data


def sample(path, run_key):
    _, sid, epoch = run_key.split("__")
    return json.loads(read_member(path, f"samples/{sid}_epoch_{epoch}.json"))


def text(content):
    if isinstance(content, str):
        return content
    return "".join(
        c.get("text", "")
        for c in content or []
        if isinstance(c, dict) and c.get("type") == "text"
    )


def has_literal_tab(body):
    for ln in body.split("\n")[1:]:
        for rx in LINE_CONTENT:
            hit = rx.match(ln)
            if hit:
                if "\t" in hit.group(1):
                    return True
                break
    return False


def run_stats(s):
    offered, events = defaultdict(set), defaultdict(int)
    for e in s.get("events", []):
        if e.get("event") == "model":
            events[e.get("model")] += 1
            offered[e.get("model")] |= {t.get("name") for t in e.get("tools") or []}
    calls, used = {}, set()
    for m in s["messages"]:
        for tc in m.get("tool_calls") or []:
            calls[tc["id"]] = (
                tc["function"],
                (tc.get("arguments") or {}).get("command"),
            )
            used.add(tc["function"])
    edits = tabs = te_replies = te_err_field = te_err_in_content = 0
    for m in s["messages"]:
        if m["role"] != "tool" or m.get("function") != "text_editor":
            continue
        body = text(m["content"])
        te_replies += 1
        te_err_field += bool(m.get("error"))
        te_err_in_content += (not m.get("error")) and bool(TE_ERROR.match(body))
        if (
            calls.get(m.get("tool_call_id"), ("", ""))[1] in EDIT_COMMANDS
            and not m.get("error")
            and not TE_ERROR.match(body)
        ):
            edits += 1
            tabs += has_literal_tab(body)
    return {
        "agent_tools_offered": sorted(offered.get(AGENT_MODEL, set())),
        "other_models": {
            k: {"events": v, "tools": sorted(offered[k])}
            for k, v in events.items()
            if k != AGENT_MODEL
        },
        "tools_used": sorted(used),
        "text_editor_view_edit_replies": edits,
        "with_literal_tab": tabs,
        "text_editor_replies": te_replies,
        "text_editor_error_field": te_err_field,
        "text_editor_error_text_in_content": te_err_in_content,
    }


def git_usage_columns(outputs):
    for body in outputs:
        if "Diff output format options" in body:
            cols = {}
            for ln in body.split("\n"):
                hit = re.match(r"^(\s+)(-u|-p, --patch|--shortstat)(\s+)\S", ln)
                if hit:
                    cols[hit.group(2)] = (
                        len(hit.group(1)) + len(hit.group(2)) + len(hit.group(3))
                    )
            return cols
    return None


def compute(sim_dir, real_dir, draw, movers):
    """Run on Modal (run_checks_modal.py): the checks over the uploaded .eval files; returns the result dict."""
    pairs = [
        (f"R{i + 1:02d}", e["run_key"], e["counterpart"])
        for i, e in enumerate(draw["drawn"])
    ]
    others = [m for m in movers["movers"] if not m["also_in_random_10"]]
    pairs += [
        (f"M{i + 1}", m["run_key"], m["counterpart"]) for i, m in enumerate(others)
    ]
    out = {
        "label": "read10 Amendment 2 script checks (POST HOC, descriptive); RULE.md in this folder",
        "pairs": {},
        "inputs": {},
    }
    for tag, sim_key, real_key in pairs:
        sim_path = Path(sim_dir) / f"{sim_key.split('__')[0]}.eval"
        matches = glob.glob(str(Path(real_dir) / f"{real_key.split('__')[0]}*.eval"))
        assert len(matches) == 1, (tag, matches)
        real_path = Path(matches[0])
        s, r = sample(sim_path, sim_key), sample(real_path, real_key)
        for p in (sim_path, real_path):
            out["inputs"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        ref = (s.get("metadata") or {}).get("orig_trajectory") or []
        ref_out = [str(c.get("output", "")) for c in ref]
        row = {
            "sim": run_stats(s),
            "real": run_stats(r),
            "agent_tools_offered_equal": run_stats(s)["agent_tools_offered"]
            == run_stats(r)["agent_tools_offered"],
            "reference_calls": len(ref),
            "reference_bash_calls": sum(1 for c in ref if c.get("function") == "bash"),
            "reference_bash_calls_with_cd": sum(
                1
                for c in ref
                if c.get("function") == "bash"
                and "cd " in str((c.get("arguments") or {}).get("cmd", ""))
            ),
            "reference_checks": {
                cid: sum(len(re.findall(rx, o, fl)) for o in ref_out)
                for t, cid, rx, fl in REF_CHECKS
                if t == tag
            },
        }
        if tag in TAB_REF_TAGS:
            te_outs = [
                str(c.get("output", ""))
                for c in ref
                if c.get("function") == "text_editor"
            ]
            row["reference_checks"]["ref_text_editor_outputs"] = len(te_outs)
            row["reference_checks"]["ref_text_editor_outputs_with_literal_tab"] = sum(
                has_literal_tab(o) for o in te_outs
            )
        if tag == "M2":
            sim_outs = [
                text(m["content"]) for m in s["messages"] if m["role"] == "tool"
            ]
            real_outs = [
                text(m["content"]) for m in r["messages"] if m["role"] == "tool"
            ]
            row["git_usage_columns"] = {
                "sim": git_usage_columns(sim_outs),
                "real": git_usage_columns(real_outs),
                "reference": git_usage_columns(ref_out),
            }
        out["pairs"][tag] = row
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return out


def write_result(out):
    """Local side: add the commit and write results/checks_<stamp>.json (exclusive create)."""
    out["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"checks_{stamp}.json"
    with open(path, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    return path
