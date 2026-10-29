"""Structure-only (d9 / 09, 10-15 ~02:15 PT), ON THE BOX with the September venv; prints NO prompt text. For every honest_a
(and honest_b) of a main launch: the system prompt's length, the first user message's length, and whether the main task's
description (the settings folder's main_tasks/<main>/info.yml description field, and any other .md/.txt file there,
stripped) is an exact substring of the system prompt, of the first user message, or of ANY message in the sample.
Normal-length prompts are the positive control for the matching method.
  <sept venv python> honest_task_text_check.py <settings dir> <main run dir>"""
import glob, json, re, statistics, sys, yaml
from pathlib import Path
from inspect_ai.log import read_eval_log
settings, run = Path(sys.argv[1]), sys.argv[2]
d = json.load(open(run + "/launch.json"))
tf = Path(d["tasks_file"]) if Path(d["tasks_file"]).is_absolute() else Path("/home/aswin/phase2/nm/attack_check_nm") / d["tasks_file"]
cells = {c["key"]: c for c in json.load(open(tf))["cells"]}
def text(m):
    return (m.text if hasattr(m, "text") else str(m.content)) if m else ""
rows = []
for r in d["runs"]:
    if not r["real"] or not r["key"].startswith(("honest_a/", "honest_b/")) or not r["attempts"]:
        continue
    t = r["task"] if isinstance(r["task"], str) else r["task"]["key"]
    ev = sorted(glob.glob(r["attempts"][-1]["log_dir"] + "/*.eval"))
    if len(ev) != 1:
        continue
    c = cells.get(t, {}); env, main = t.split("_", 1)[1], c.get("main")
    descs = []
    for f in glob.glob(str(settings / "**" / env / "**" / "main_tasks" / str(main) / "*"), recursive=True):
        p = Path(f)
        if p.suffix in (".yml", ".yaml"):
            try:
                y = yaml.safe_load(p.read_text())
                if isinstance(y, dict) and isinstance(y.get("description"), str): descs.append(y["description"].strip())
            except Exception: pass
        elif p.suffix in (".md", ".txt") and p.is_file():
            descs.append(p.read_text(errors="replace").strip())
    try:
        s = read_eval_log(ev[0]).samples[0]
    except Exception as e:  # noqa: BLE001
        rows.append({"key": r["key"], "error": type(e).__name__}); continue
    sysm = next((m for m in s.messages if m.role == "system"), None); user = next((m for m in s.messages if m.role == "user"), None)
    sp, up, alltext = text(sysm), text(user), "\n".join(text(m) for m in s.messages)
    descs = [x for x in descs if x]
    rows.append({"key": r["key"], "main": main, "system_chars": len(sp), "first_user_chars": len(up), "n_desc_files": len(descs),
                 "desc_chars": [len(x) for x in descs], "desc_in_system": any(x in sp for x in descs) if descs else None,
                 "desc_in_first_user": any(x in up for x in descs) if descs else None, "desc_in_any_message": any(x in alltext for x in descs) if descs else None,
                 "sample_input_chars": len(s.input) if isinstance(s.input, str) else None})
short = [x for x in rows if x.get("system_chars", 99999) < 1500]; normal = [x for x in rows if x.get("system_chars", 0) >= 1500]
summ = lambda xs, k: {"n": len(xs), "true": sum(1 for x in xs if x.get(k) is True), "false": sum(1 for x in xs if x.get(k) is False), "none": sum(1 for x in xs if x.get(k) is None)}
print(json.dumps({"run": run.rstrip("/").split("/")[-1], "short_lt1500": {k: summ(short, k) for k in ("desc_in_system", "desc_in_first_user", "desc_in_any_message")},
                  "normal_ge1500": {k: summ(normal, k) for k in ("desc_in_system", "desc_in_first_user", "desc_in_any_message")}, "rows": rows}))
