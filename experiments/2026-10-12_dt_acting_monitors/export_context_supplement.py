"""Blinded context supplement for Aswin's destructive/transfer labelling (2026-10-12; $0, local, one scripted export).

Aswin, relayed by the orchestrator at about 17:15 PT: "Have 2e provide blinded context" (CONTEXT_REQUEST_FOR_2E.md,
sha256 f5ba0e08...). For each requested blind_id: the contents of the files its command references (context_parse.py),
rebuilt only from the same run's tool calls recorded before that command's turn (context_evidence.py). Never a later
event, a final state, another run or a working copy; nothing is executed; ambiguous means no contents. A row whose
command or evidence window leaves the shell grammar these rules model (shell_constructs.py, an allowlist, after Codex's
reviews; the orchestrator: "anything it can't model must fail closed") loses its contents: construct_outside_shell_model.

Join (as export_command_bundle.py): held key -> run_key (strip #e) -> .eval -> the acting model's ModelEvent with that
x-irid -> output tool_calls[call_index], whose tool must match the v3 sheet. Evidence: that request's own input messages
(exactly what was recorded before the command's turn); a call's result is used only when the tool messages right after
its assistant message match the calls one-to-one by id, and a missing or non-text result counts as missing. The other
calls of the command's own turn are checked for possible writes and for other_programs_between. A relative path's directory: the same tool's
earlier bare `pwd` output, if no cd/pushd/popd came after it and all such outputs agree.

Output JSONL, one row per requested id, in the id file's order: blind_id, context_status (available: every referenced
file complete; partial: some; unavailable: none, or no file), files [{path, contents (complete only, else null),
complete, missing_reason, display_normalized, other_programs_between}], missing_context_reason (the first incomplete
file's reason, or no_referenced_file_parsed). Nothing else: no run, world, arm, task, epoch, time or order. One run
writes the full set and the pilot (its 50 rows, the same values).

Blinding check (Aswin, about 18:00 PT 2026-10-12: "One yes/no check first"), fixed here before it runs: over the full
set, the share of rows that are available among real rows (every arm but sim_honest and sim_attack: real honest A/B,
real attack 1/2) and among sim rows. If the two differ by more than 5 points, the labeller-facing files drop every
field that states or encodes completeness: context_status, missing_context_reason, and each file's complete,
missing_reason and other_programs_between (display_normalized stays). Only the yes/no bit is printed; the two shares
are neither printed nor stored. Whether contents is null still shows completeness either way.
The private file in --private-dir keeps every field but contents, the provenance (run_key, irid, call_index, evidence
source and message indices) and the bit. Prints counts, hashes and output paths only; never the key's path.

  uv run --no-project --python 3.14 python -I -B export_context_supplement.py --ids-full <csv> --ids-pilot <csv> \\
      --sheet <v3 sheet> --key <held key> --evals <bundle_src> --out-dir <data dir> --private-dir <key-holder dir>
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import context_evidence  # noqa: E402
import context_parse  # noqa: E402
import shell_constructs  # noqa: E402

ID_SHA = {"pilot": "8b8c4d49d409d8b8c594501e3ab9c6bd5097a92e3081589f2dc3ae405ca97850",
          "full": "60533ecd417878cf6d695a5a229e80682221ea42454b80330173b098bb9335df"}
SHEET_SHA = "b4c53e449b08bfd7497d254fa63a34f29cb76383da93545712195efe63344859"
KEY_SHA = "8af4a98744900aeb4520e76d0ec67834b0a7c5f3be03258cff5754c520561331"  # as committed in results/manifest_dt_label_blind_v2.json
LAUNCHES = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup",
            "2026-10-10_001314_scaleup_topup", "2026-10-10_001314_second_attack"]
ACTING = "vllm/Qwen/Qwen3.5-27B"
REF = re.compile(r"attachment://[0-9a-f]{32}")
SIM = {"sim_honest", "sim_attack"}
BLIND_POINTS = 5  # available shares more than 5 points apart -> completeness fields withheld from the labeller-facing files
HIDE_ROW, HIDE_FILE = {"context_status", "missing_context_reason"}, {"complete", "missing_reason", "other_programs_between"}
REASONS = {"no_referenced_file_parsed", "command_unparsed", "path_ambiguous", "module_reference_unresolved", "no_in_run_evidence",
           "edit_without_complete_base", "evidence_ambiguous", "possibly_modified_after_evidence", "evidence_truncated",
           "construct_outside_shell_model"}


def resolved(x, att, depth=0):
    if depth > 20:
        raise ValueError("attachment chain too deep")
    if isinstance(x, str) and REF.fullmatch(x):
        return resolved(att[x.removeprefix("attachment://")], att, depth + 1)
    if isinstance(x, list):
        return [resolved(v, att, depth) for v in x]
    if isinstance(x, dict):
        return {k: resolved(v, att, depth) for k, v in x.items()}
    return x


def text_of(c):
    """A message's text, or None when it is missing or not all text."""
    if isinstance(c, str):
        return c
    ok = isinstance(c, list) and c and all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) for p in c)
    return "".join(p["text"] for p in c) if ok else None


def git_head() -> str:
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    mine = ["export_context_supplement.py", "context_parse.py", "context_evidence.py", "shell_constructs.py"]
    if g("status", "--porcelain", "--untracked-files=no") or len(g("ls-files", *mine).split()) != len(mine):
        raise SystemExit("the exporter, its rules or another tracked file is uncommitted")
    return g("rev-parse", "HEAD").strip()


def evidence(s: dict, e: dict, att: dict):
    """(steps, source): [(message index, tool, args, output or None, error)] from this request's input messages."""
    pool = (s.get("events_data") or {}).get("messages") or []
    msgs = e.get("input") if isinstance(e.get("input"), list) and e.get("input") else [m for a, b in e.get("input_refs") or [] for m in pool[a:b]]
    msgs = resolved(msgs, att)
    if REF.search(json.dumps(msgs)):
        raise SystemExit("an attachment reference is left in the evidence")
    steps = []
    for k, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") != "assistant" or not isinstance(m.get("tool_calls"), list):
            continue
        calls, block = [t for t in m["tool_calls"] if isinstance(t, dict)], []
        for r in msgs[k + 1:]:
            if not isinstance(r, dict) or r.get("role") != "tool":
                break
            block.append(r)
        ids, rids = [t.get("id") for t in calls], [r.get("tool_call_id") for r in block]
        one_to_one = all(ids) and len(set(ids)) == len(ids) and len(set(rids)) == len(rids) and set(rids) <= set(ids)
        res = {r.get("tool_call_id"): r for r in block} if one_to_one else {}
        for t in calls:
            rm = res.get(t.get("id"))
            err = rm.get("error") if rm else None
            out = None if rm is None else (text_of(err.get("message")) if isinstance(err, dict) else None) if err else text_of(rm.get("content"))
            args = t.get("arguments") if isinstance(t.get("arguments"), dict) else {}
            steps.append((k, t.get("function") if isinstance(t.get("function"), str) else "", args, out, bool(err)))
    return steps, "request_input" if msgs else "no_messages"


def run_cwd(steps, tool: str):
    cwd = None
    for _, t, a, o, er in steps:
        if t != tool:
            continue
        c = context_parse.bash_cmd(a)
        if c.strip() == "pwd" and o and not er and re.fullmatch(r"/\S*", o.strip()):
            cwd = o.strip() if cwd in (None, o.strip()) else "?"
        elif re.search(r"\b(cd|pushd|popd)\b", c):
            cwd = "?" if cwd else None
    return None if cwd == "?" else cwd


def tally(rows: list) -> dict:
    c = Counter()
    for r in rows:
        c.update([r["context_status"], f"row_reason_{r['missing_context_reason']}"] + [f"file_{f['missing_reason'] or 'complete'}" for f in r["files"]]
                 + ["file_display_normalized"] * sum(f["display_normalized"] for f in r["files"])
                 + ["file_other_programs_between"] * sum(f["other_programs_between"] for f in r["files"]))
    return dict(sorted(c.items()))


def main(ids_full: str, ids_pilot: str, sheet: str, key: str, evals: str, out_dir: str, private_dir: str):
    n_tests, git = context_parse.selftest() + context_evidence.selftest() + shell_constructs.selftest(), git_head()
    for p, h in ((ids_full, ID_SHA["full"]), (ids_pilot, ID_SHA["pilot"]), (sheet, SHEET_SHA), (key, KEY_SHA)):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise SystemExit("an input's sha256 doesn't match")
    want = [r["blind_id"] for r in csv.DictReader(open(ids_full, newline=""))]
    pilot = [r["blind_id"] for r in csv.DictReader(open(ids_pilot, newline=""))]
    if len(set(pilot)) != len(pilot) or not set(pilot) <= set(want):
        raise SystemExit("the pilot ids are not a subset of the full ids")
    tool_of = {r["blind_id"]: r["tool"] for r in csv.DictReader(open(sheet, newline=""))}
    keyrows = {r["blind_id"]: r for r in csv.DictReader(open(key, newline=""))}
    if len(set(want)) != len(want) or any(b not in tool_of or b not in keyrows for b in want):
        raise SystemExit("an id is duplicated or missing from the sheet or key")
    index = {}
    for launch in LAUNCHES:
        for p in (Path(evals) / launch).rglob("*.eval"):
            rel = p.relative_to(Path(evals) / launch)
            if "references" not in rel.parts:
                if str(rel) in index:
                    raise SystemExit("a run path occurs in two launches")
                index[str(rel)] = p
    by_run = {}
    for b in want:
        by_run.setdefault(keyrows[b]["run_key"].rsplit("#e", 1)[0], []).append(b)
    rows, prov, src_count, constructs = {}, {}, Counter(), Counter()
    for run, bids in by_run.items():
        if run not in index:
            raise SystemExit("a requested run has no local .eval")
        z = zipfile.ZipFile(index[run])
        names = [n for n in z.namelist() if n.startswith("samples/")]
        if len(names) != 1:
            raise SystemExit("a run file without exactly one sample")
        s = json.loads(z.read(names[0]))
        att, events = s.get("attachments") or {}, {}
        for e in s.get("events") or []:
            irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid") if e.get("event") == "model" and e.get("model") == ACTING else None
            if irid in events:
                raise SystemExit("an x-irid occurs twice in one run")
            if irid:
                events[irid] = e
        for b in bids:
            r, ci, tool = keyrows[b], int(keyrows[b]["call_index"]), tool_of[b]
            e = events.get(r["irid"])
            tcs = (((((e or {}).get("output") or {}).get("choices") or [{}])[0] or {}).get("message") or {}).get("tool_calls") or []
            if e is None or not 0 <= ci < len(tcs) or tcs[ci].get("function") != tool:
                raise SystemExit("a key row's call doesn't match the sheet")
            tcs = resolved(tcs, att)
            if REF.search(json.dumps(tcs)):
                raise SystemExit("an attachment reference is left in the call")
            tc, steps, src = tcs[ci], *evidence(s, e, att)
            siblings = [(t.get("function") or "", t.get("arguments") if isinstance(t.get("arguments"), dict) else {}) for j, t in enumerate(tcs) if j != ci]
            files, used = [], {}
            for path, st in context_parse.referenced(tool, tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}, run_cwd(steps, tool)):
                if st != "ok":
                    files.append({"path": path, "contents": None, "complete": False, "missing_reason": st, "display_normalized": False,
                                  "other_programs_between": False})
                    continue
                c, done, why, used[path], shown, between = context_evidence.rebuild(path, context_evidence.container(tool), steps, siblings)
                files.append({"path": path, "contents": c, "complete": done, "missing_reason": why, "display_normalized": shown,
                              "other_programs_between": between})
            complete, found = [f for f in files if f["complete"]], []
            if complete:  # shell grammar the rules don't model, from the earliest evidence on (shell_constructs.py)
                start = 0 if run_cwd(steps, tool) else min(min(used[f["path"]]) for f in complete)
                calls = [context_parse.bash_cmd(a) for k, t, a, _, _ in steps if k >= start and t.endswith("bash")]
                calls += [context_parse.bash_cmd(a) for t, a in siblings if t.endswith("bash")]
                own = context_parse.bash_cmd(tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}) if tool.endswith("bash") else ""
                found = shell_constructs.hits(calls, own, [f["path"] for f in complete])
                edits = [a.get("path") for k, t, a, _, _ in steps if k >= start and t == "text_editor"] + [a.get("path") for t, a in siblings if t == "text_editor"]
                if any(isinstance(e, str) and e.startswith("//") for e in edits):  # normpath keeps a leading //, so rebuild() can't match it (Bugbot, #1147)
                    found = sorted(set(found) | {"double_slash_editor_path"})
                if any(not c.strip() for c in calls):  # a bash call with missing or non-string arguments reads as "" (Codex, #1151)
                    found = sorted(set(found) | {"malformed_bash_call"})
                for f in complete if found else []:
                    f.update(contents=None, complete=False, missing_reason="construct_outside_shell_model", display_normalized=False)
                constructs.update(found + ["rows_with_contents"] + ["rows_hit"] * bool(found))
            done = [f["complete"] for f in files]
            status = "available" if done and all(done) else "partial" if any(done) else "unavailable"
            why = "no_referenced_file_parsed" if not files else next((f["missing_reason"] for f in files if not f["complete"]), None)
            if why is not None and why not in REASONS or any(f["missing_reason"] not in REASONS | {None} for f in files):
                raise SystemExit("a reason outside the fixed vocabulary")
            rows[b] = {"blind_id": b, "context_status": status, "files": files, "missing_context_reason": why}
            prov[b] = {"run_key": r["run_key"], "irid": r["irid"], "call_index": ci, "evidence_source": src, "evidence_message_indices": used,
                       "context_status": status, "missing_context_reason": why, "shell_constructs": found,
                       "files": [{k: v for k, v in f.items() if k != "contents"} for f in files]}
            src_count[src] += 1
    # the blinding bit: only it leaves this function
    arms = {b: keyrows[b]["arm"] for b in want}
    if any(not a or ("sim" in a and a not in SIM) for a in arms.values()):
        raise SystemExit("an arm outside the fixed world rule")
    avail = {w: [rows[b]["context_status"] == "available" for b in want if (arms[b] in SIM) == (w == "sim")] for w in ("real", "sim")}
    (ra, rn), (sa, sn) = [(sum(v), len(v)) for v in (avail["real"], avail["sim"])]
    if not rn or not sn:
        raise SystemExit("a world has no requested rows")
    hide = 100 * abs(ra * sn - sa * rn) > BLIND_POINTS * rn * sn  # exact: |ra/rn - sa/sn| > 5 points
    del avail, ra, rn, sa, sn
    public = lambda r: ({k: v for k, v in r.items() if k not in HIDE_ROW} | {"files": [{k: v for k, v in f.items() if k not in HIDE_FILE} for f in r["files"]]}) if hide else r  # noqa: E731
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    od, pd = Path(out_dir), Path(private_dir)
    if not od.is_dir() or not pd.is_dir() or od.resolve() == pd.resolve():
        raise SystemExit("both output folders must exist, apart from each other")
    pv = pd / f"dt_context_provenance_{stamp}.json"
    with pv.open("x") as fh:  # the private file first: a supplement never exists without it
        fh.write(json.dumps({"git": git, "ids_sha256": {"full": ID_SHA["full"], "pilot": ID_SHA["pilot"]}, "blind_threshold_points": BLIND_POINTS,
                             "completeness_fields_withheld": hide, "rows": prov}))
    report = {"git": git, "selftest_checks": n_tests, "evidence_source": dict(src_count), "completeness_fields_withheld": hide,
              "shell_constructs": dict(sorted(constructs.items()))}
    for label, ids in (("full", want), ("pilot", pilot)):
        out = od / f"dt_context_supplement_{label}_{stamp}.jsonl"
        with out.open("x") as fh:
            fh.writelines(json.dumps(public(rows[b]), ensure_ascii=False) + "\n" for b in ids)
        report[label] = {"rows": len(ids), "counts": tally([rows[b] for b in ids]), "path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    report["private_sha256"] = hashlib.sha256(pv.read_bytes()).hexdigest()
    print(json.dumps(report))


class Parser(argparse.ArgumentParser):
    def error(self, message):  # no echo: the arguments include the held key's path
        sys.stderr.write("failed: bad arguments\n")
        sys.exit(2)


if __name__ == "__main__":
    ap = Parser()
    for k in ("--ids-full", "--ids-pilot", "--sheet", "--key", "--evals", "--out-dir", "--private-dir"):
        ap.add_argument(k, required=True)
    try:
        main(**vars(ap.parse_args()))
    except SystemExit:
        raise
    except BaseException as exc:  # no message or path: content and the key's location must not reach stderr
        print(f"failed: {type(exc).__name__} at line {traceback.extract_tb(exc.__traceback__)[-1].lineno}", file=sys.stderr)
        sys.exit(1)
