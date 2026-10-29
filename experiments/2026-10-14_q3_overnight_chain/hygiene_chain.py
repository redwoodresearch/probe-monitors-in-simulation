"""Overnight chain (RULE.md here): 09's hygiene files for one scale-up batch, on Modal CPU over the relayed .evals. Keys and counts
only; no message content is returned, stored or printed (attack arms included).
- Tool-error table, ONE JSON per arm (09's schema): rows {key "<world>|<cell>", n_tool_calls, n_tool_errors, n_tool_errors_nonparsing,
  first_error_step}, totals, staged_sha, source_evals. The per-sample reader is load_errors() from
  experiments/2026-10-12_claimB_halves_errors/analyze_POSTHOC_hidden_errors.py @ 22bc87114 (its lines 51-68, i.e. the reader at 55-67),
  vendored byte for byte (vendored/load_errors_22bc87114_L51-68.py, sha256 checked here); our wrapper only finds each .eval's one
  sample id and epoch and calls it. Counts: tool messages excluding the runner's submit; an error = a non-null `error`; its type =
  error.type or "unknown"; nonparsing excludes type "parsing" (PR #67's rule); first_error_step = the index of the first error.
- Attempt labels: from the batch's make_phase2_attempt_labels.py rows (6e1a8c86): attempted / unclear / declined as given, each with
  its `rule` recorded beside it; ONLY the script's None (no labelled attempt) -> "run_failed"; counts per arm and per (label, rule);
  staged_sha per arm.
- Run table (per quarter, once its follow-up is in): one row per cell: attack_pair_launch "same", run1_launch == run2_launch (asserted),
  sim_launch, sim_status "ran", or sim_launch "NOT_RUN" with sim_status "no_honest_A_reference" | "cap_stopped".

  modal run experiments/2026-10-14_q3_overnight_chain/hygiene_chain.py --spec <spec json> --out <dir>
"""
import hashlib
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3chain_hygiene_kill-2026-11-14"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
READER, READER_SHA = "load_errors_22bc87114_L51-68.py", "77a96b9aeaf1c368483e5fb0754e71f66a6616779ad8ff246630949078d84045"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.14").add_local_dir(HERE / "vendored", "/root/vendored")  # 3.14: zstd .eval members


def load_reader():
    src = Path("/root/vendored") / READER
    if hashlib.sha256(src.read_bytes()).hexdigest() != READER_SHA:
        raise ValueError("the vendored reader differs from 22bc87114's lines 51-68")
    import zipfile

    ns = {"zipfile": zipfile, "json": json, "Path": Path}
    exec(src.read_text(), ns)
    return ns["load_errors"]


def row_of(key: str, errs: list) -> dict:
    idx = [i for i, e in enumerate(errs) if e is not None]
    return {"key": key, "n_tool_calls": len(errs), "n_tool_errors": len(idx),
            "n_tool_errors_nonparsing": sum(1 for e in errs if e is not None and e != "parsing"), "first_error_step": idx[0] if idx else None}


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0, volumes={"/ev": modal.Volume.from_name(VOL, version=2).read_only()})
def tool_errors(rows: list) -> list:
    """rows: [{key, path (under evals_v1/), sha256}] -> table rows, in order. Each .eval holds exactly one sample."""
    import zipfile

    load_errors = load_reader()
    out = []
    for r in rows:
        p = Path("/ev/evals_v1") / r["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"{r['path']}: differs from its manifest")
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if len(names) != 1:
                raise ValueError(f"{r['path']}: {len(names)} samples")
            s = json.loads(z.read(names[0]))
            sid, ep = s["id"], s["epoch"]  # identifiers only
        errs = load_errors((str(p), [(r["key"], sid, ep)]))[r["key"]]
        out.append(row_of(r["key"], errs))
    return out


def labels_of(spec: dict) -> dict:
    """Per counted attack-arm run of this batch, 09's enum {attempted, unclear, declined, run_failed} from the 6e1a8c86 script's rows:
    the script's label as given (a no-report run stays unclear, rule "no report", as for the 196); run_failed ONLY for the script's
    None (no labelled attempt). Each row's rule is recorded beside its label; counts per (label, rule). run2 keys are run2|<base cell>."""
    labels, rules, counts, by_rule = {}, {}, {}, {}
    for r in spec["label_rows"]:
        task = r["task"]
        if spec["kind"] in ("il", "ils"):  # interleaved: its attack is world run3, its simulated attack world sim2
            world, cell = ("sim2" if r.get("arm") == "sim_attack@september" else "run3"), task
        elif spec["kind"] in ("fu", "fus"):
            world, cell = spec.get("sim_world") or "sim", task  # a follow-up family's own world (the ug family: sim_ug)
        else:
            world, cell = ("run2", task[:-3]) if task.endswith("_r2") else ("run1", task)
        lab, rule = r["label"], r.get("rule")
        if lab is None:
            v = "run_failed"  # 09: ONLY the script's None (no labelled attempt, e.g. no .eval in the counted attempt)
        elif lab in ("attempted", "unclear", "declined"):
            v = lab  # as the script gives it (a no-report run stays unclear, rule "no report"), comparable with the 196
        else:
            raise ValueError(f"unexpected label {lab!r}")
        labels[f"{world}|{cell}"] = v
        rules[f"{world}|{cell}"] = rule
        counts.setdefault(world, {}).setdefault(v, 0)
        counts[world][v] += 1
        by_rule.setdefault(world, {}).setdefault(f"{v}|{rule}", 0)
        by_rule[world][f"{v}|{rule}"] += 1
    return {"labels": labels, "rules": rules, "counts": counts, "counts_by_label_rule": by_rule}


@app.local_entrypoint()
def main(spec: str, out: str):
    s = json.loads(Path(spec).read_text())
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    written = {}
    for world, w in s["worlds"].items():  # one tool-error table per arm
        rows = tool_errors.remote(w["rows"])
        tot = {"runs": len(rows), "n_tool_calls": sum(r["n_tool_calls"] for r in rows), "n_tool_errors": sum(r["n_tool_errors"] for r in rows),
               "n_tool_errors_nonparsing": sum(r["n_tool_errors_nonparsing"] for r in rows),
               "runs_with_error": sum(r["n_tool_errors"] > 0 for r in rows)}
        doc = {"arm": w["arm"], "world": world, "staged_sha": w["staged_sha"], "staged_set": w.get("staged_set"), "reader": f"22bc87114 analyze_POSTHOC_hidden_errors.py L51-68 (sha256 {READER_SHA[:16]})",
               "source_evals": [{"path": r["path"], "sha256": r["sha256"]} for r in w["rows"]], "rows": rows, "totals": tot}
        p = d / f"toolerr_{s['batch']}_{world}.json"
        p.write_text(json.dumps(doc, indent=1) + "\n")
        written[p.name] = tot
    if s.get("label_rows") is not None:
        lab = labels_of(s)
        p = d / f"labels_{s['batch']}.json"
        p.write_text(json.dumps({"batch": s["batch"], "labeler": s["labeler"], "staged_sha": s["staged_sha"], "staged_sets": s["staged_sets"], **lab}, indent=1) + "\n")
        written[p.name] = lab["counts"]
    if s.get("run_table") is not None:
        rt = s["run_table"]
        for c in rt["cells"]:
            if "run1_launch" in c:
                assert c["run1_launch"] == c["run2_launch"], c["cell"]  # "same": both real attacks in one launch
        p = d / s["run_table_name"]
        p.write_text(json.dumps(rt | {"staged_sets": s["staged_sets"]}, indent=1) + "\n")
        written[p.name] = {"cells": len(rt["cells"]), "sim_status": {k: sum(c.get("sim_status") == k for c in rt["cells"]) for k in {c.get("sim_status") for c in rt["cells"]}}}
    print(json.dumps({"written": written}))
