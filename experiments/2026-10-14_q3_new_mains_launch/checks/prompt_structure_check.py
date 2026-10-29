"""Structure-only check (d9 / 09, 10-15 ~02:05 PT), ON THE BOX with the September venv; prints NO prompt text, only lengths,
hashes and booleans. For every real run of the given main launches (and their follow-ups' simulated runs): the system
prompt's length (characters of the first system message in the .eval sample), its sha256, and for one CELL in detail:
whether its honest_a is the pinned reference (launch.json references[cell].sha256 == sha256 of the honest_a .eval, and the
reference copy's sha256), whether the main task's description (the settings folder's main_tasks/<main>/ description files,
matched as an exact substring after stripping) appears in the system prompt or the first user message, and truncation
signs (prompt ends inside a template placeholder or with an unclosed tag; an empty task block between known headings).
  <sept venv python> prompt_structure_check.py <cell> <settings dir> <main run dir> [<more run dirs> ...]"""
import glob, hashlib, json, re, statistics, sys
from pathlib import Path
from inspect_ai.log import read_eval_log
cell, settings, runs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
def sysprompt(ev):
    s = read_eval_log(ev).samples[0]
    sysm = next((m for m in s.messages if m.role == "system"), None)
    user = next((m for m in s.messages if m.role == "user"), None)
    text = lambda m: (m.text if hasattr(m, "text") else str(m.content)) if m else ""
    return text(sysm), text(user), len(s.messages)
out = {"cell": cell, "runs": [], "honest_a_lengths": {}}
main_name = None
for run in runs:
    d = json.load(open(run + "/launch.json"))
    tasks = {c["key"]: c for c in json.load(open(Path(run).parents[2] / "nm" / "attack_check_nm" / d["tasks_file"]))["cells"]} if False else None
    for r in d["runs"]:
        t = r["task"] if isinstance(r["task"], str) else r["task"]["key"]
        a = r["attempts"][-1] if r["attempts"] else None
        ev = sorted(glob.glob(a["log_dir"] + "/*.eval")) if a and a.get("log_dir") else []
        if len(ev) != 1:
            continue
        if r["key"].startswith("honest_a/") or t.removesuffix("_r2") == cell:
            try:
                sp, up, nmsg = sysprompt(ev[0])
            except Exception as e:  # noqa: BLE001
                sp, up, nmsg = None, None, None
            if r["key"].startswith("honest_a/"):
                out["honest_a_lengths"][f"{run.rstrip('/').split('/')[-1]}:{t}"] = len(sp) if sp is not None else None
            if t.removesuffix("_r2") == cell:
                rec = {"launch": run.rstrip("/").split("/")[-1], "key": r["key"], "state": r["state"], "eval_sha256": hashlib.sha256(Path(ev[0]).read_bytes()).hexdigest(),
                       "system_chars": len(sp) if sp is not None else None, "system_sha256": hashlib.sha256(sp.encode()).hexdigest() if sp else None,
                       "first_user_chars": len(up) if up is not None else None, "messages": nmsg}
                if r["key"] == f"honest_a/{cell}":
                    ref = (d.get("references") or {}).get(cell) or {}
                    rec["pinned_reference_sha256_matches_eval"] = ref.get("sha256") == rec["eval_sha256"]
                    rec["pinned_reference_copy_sha256_matches"] = bool(ref.get("copy")) and Path(ref["copy"]).exists() and hashlib.sha256(Path(ref["copy"]).read_bytes()).hexdigest() == ref.get("sha256")
                    rec["pinned_reference_try"] = ref.get("try")
                    # the main task's description from the settings folder
                    env = cell.split("_", 1)[1]
                    main_name = main_name or next((c.get("main") for c in json.load(open(d["tasks_file"] if Path(d["tasks_file"]).is_absolute() else Path("/home/aswin/phase2/nm/attack_check_nm") / d["tasks_file"]))["cells"] if c["key"] == cell), None)
                    files = sorted(set(glob.glob(str(settings / "**" / env / "**" / "main_tasks" / main_name / "*"), recursive=True))) if main_name else []
                    descs = {}
                    for f in files:
                        if Path(f).is_file() and Path(f).suffix in (".md", ".txt", ".yml", ".yaml"):
                            txt = Path(f).read_text(errors="replace")
                            if Path(f).suffix in (".yml", ".yaml"):
                                m = re.search(r"description:\s*[|>]?-?\s*\n((?:\s{2,}.*\n?)+)", txt)
                                txt = re.sub(r"^\s{2,}", "", m.group(1), flags=re.M) if m else ""
                            descs[str(Path(f).relative_to(settings))] = txt.strip()
                    rec["main_task"] = main_name
                    rec["description_files"] = {k: {"chars": len(v), "in_system": bool(v) and v in (sp or ""), "in_first_user": bool(v) and v in (up or "")} for k, v in descs.items()}
                    rec["truncation_signs"] = {"ends_inside_placeholder": bool(re.search(r"\{[A-Za-z_]*$", (sp or "").rstrip())),
                                               "unclosed_tag_at_end": bool(re.search(r"<[A-Za-z_]+>[^<]*$", (sp or "").rstrip())),
                                               "empty_task_block": bool(re.search(r"<task>\s*</task>|task_description>\s*<", sp or ""))}
                out["runs"].append(rec)
L = [v for v in out["honest_a_lengths"].values() if v is not None]
out["honest_a_median_chars"] = statistics.median(L) if L else None
out["honest_a_under_1500"] = sorted(k for k, v in out["honest_a_lengths"].items() if v is not None and v < 1500)
print(json.dumps(out))
