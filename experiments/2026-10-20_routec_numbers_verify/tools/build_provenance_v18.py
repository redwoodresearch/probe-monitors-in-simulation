"""Route C rerun item 3 (detl48), provenance rows 119-122: build provenance v18 from final17 IN A MODAL CONTAINER (PRESTATED_rows119_122
AMEND1 section 3; 09's design via 2e, 10-20 PT). v18 = final17 with exactly 5 fields changed in rows 119-122: record, record_sha256,
key_path[1] ("commands L63" -> "commands L48"), predicate.value_string (the new record's slot string, byte for byte) and
predicate.component_string (predicates.slot_component(value_string, component, "identity")).

Values never reach the laptop's terminal. The laptop sends git bytes UNPARSED with their pinned sha256 (final17, the new rc1b_detl48 record,
the old rc1b record); predicates.py and verify_core.py ride in the image and are sha-checked in the container. The container checks every
sha before it parses, proves it can write in final17's own style (every line re-serializes byte-identically), reproduces final17's
construction of the two strings on the OLD record, builds the 5 fields from the NEW record, re-checks them, and returns the v18 bytes. The
laptop writes them create-only and prints path, sha256, line count, pins and booleans only. A failed check returns a step name only.

  run from the verifier folder's repo root, attached on aswin/routec-numbers-verify, with the repo .venv's modal 1.6.1 (COMMON_RULES L49):
  /Users/aswinsurya/probe-distribution-shift-from-sims/.venv/bin/modal run --detach \\
      experiments/2026-10-20_routec_numbers_verify/tools/build_provenance_v18.py --out <a new .jsonl path>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # tools/
VDIR = HERE.parent  # the verifier folder (predicates.py, verify_core.py)
APP = "probe-shift_aswin-surya_sub-rerun_numv18rows_kill-2026-11-19"  # 60 characters
CODE = ("predicates.py", "verify_core.py")
SELF = "tools/build_provenance_v18.py"
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6",  # the verifier's base
                                                                       "zstandard==0.25.0")
for _f in CODE:
    image = image.add_local_file(VDIR / _f, f"/root/{_f}")
app = modal.App(APP, image=image)

RC = "experiments/2026-10-17_routec_slot_readout/results"
PINS = {
    "final17": {"commit": "98b5054fbf8306e54711b1a7bacd72b5f5e0af2a",
                "path": "experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final17_0141.jsonl",
                "sha256": "220d25d21943226791637581afda92faac595a3b15b9b6d6bbd3fd1a1e575971"},
    "new_record": {"commit": "6d77ea36908000cf282e5b52e16b288da058db2d", "path": f"{RC}/slots_routeC_writing_rc1b_detl48.json",
                   "sha256": "9519ea1b7aadcfa611a4cfa012af9aa4698f543f57b8ebd9ef1bf3858332590c"},
    "old_record": {"commit": "600f63614c0ab77f74fa7e326461e2e73086c2ac", "path": f"{RC}/slots_routeC_writing_rc1b.json",
                   "sha256": "ab8dcb439b942c92d30b784ae1e6a3a193b18127391ff1753426663b726e728b"},
}
N_LINES = 251
ROWS = (119, 120, 121, 122)
OLD_REC = f"{RC}/slots_routeC_writing_rc1b.json@600f63614"
NEW_REC = f"{RC}/slots_routeC_writing_rc1b_detl48.json@6d77ea369"
OLD_TAG, NEW_TAG = "commands L63", "commands L48"
OLD_KEYS = {119: "23.spearman.sol_a_uncut C.commands L63.sim_score_vs_drop", 120: "23.spearman.sol_a_uncut C.commands L63.sim_score_vs_drop",
            121: "23.spearman.sol_a_uncut C.commands L63.sim_score_vs_drop",
            122: "23.spearman.sol_a_uncut C.commands L63.null_run2_score_vs_run2_minus_run1"}
COMPONENTS = {119: "point", 120: "interval_low", 121: "interval_high", 122: "point"}
CHANGED = ("record", "record_sha256", "key_path")  # top-level fields that change; in "predicate": value_string, component_string
STAGE = ["init"]  # the step running: an unexpected exception reports this fixed name (never a value) with its class


class Stop(Exception):
    """A failed check. Its message is a step name and row number only, never a value."""


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dumps(o):
    return json.dumps(o, ensure_ascii=False)  # final17's style: raw non-ASCII, default separators (proved by the round trip below)


def build(lines, new_rec, old_rec, slot_component):
    """final17's lines (str, no newline) -> v18's lines. Pure; raises Stop(step) on any failed check."""
    if len(lines) != N_LINES:
        raise Stop("final17 is not 251 lines")
    STAGE[0] = "parse and round trip"
    objs = [json.loads(x) for x in lines]
    if any(dumps(o) != x for o, x in zip(objs, lines)):
        raise Stop("round trip: final17's lines do not re-serialize byte-identically")
    out = list(lines)
    for r in ROWS:
        STAGE[0] = f"row {r}: final17 checks"
        p = objs[r - 1]
        pr = p.get("predicate")
        if p.get("row") != r or not isinstance(pr, dict):
            raise Stop(f"row {r}: line {r} is not row {r} with a predicate")
        if (p.get("record"), p.get("record_sha256"), p.get("key_path")) != (OLD_REC, PINS["old_record"]["sha256"], ["slots", OLD_KEYS[r]]):
            raise Stop(f"row {r}: not final17's expected record / sha / key path")
        if (pr.get("op"), pr.get("component"), pr.get("transform")) != ("slot_component_eq", COMPONENTS[r], "identity"):
            raise Stop(f"row {r}: not final17's expected op / component / transform")
        if not (isinstance(pr.get("value_string"), str) and isinstance(pr.get("component_string"), str)):
            raise Stop(f"row {r}: value_string or component_string missing")
        # real data first: final17's construction, reproduced on the OLD record
        STAGE[0] = f"row {r}: old-record construction"
        old_s = (old_rec.get("slots") or {}).get(OLD_KEYS[r])
        if not isinstance(old_s, str) or pr["value_string"] != old_s:
            raise Stop(f"row {r}: final17's value_string is not old_record.slots[key]")
        if pr["component_string"] != slot_component(old_s, COMPONENTS[r], "identity"):
            raise Stop(f"row {r}: final17's component_string is not the identity component of its value_string")
        if OLD_KEYS[r].count(OLD_TAG) != 1:
            raise Stop(f"row {r}: the key does not hold the detector tag once")
        STAGE[0] = f"row {r}: new fields"
        new_key = OLD_KEYS[r].replace(OLD_TAG, NEW_TAG)
        new_s = (new_rec.get("slots") or {}).get(new_key)
        if not isinstance(new_s, str):
            raise Stop(f"row {r}: the new record has no string slot for the L48 key")
        q = json.loads(lines[r - 1])  # a fresh copy; key order kept
        q["record"], q["record_sha256"], q["key_path"] = NEW_REC, PINS["new_record"]["sha256"], ["slots", new_key]
        q["predicate"]["value_string"] = new_s
        q["predicate"]["component_string"] = slot_component(new_s, COMPONENTS[r], "identity")
        # re-check: the 5 fields are the new record's; every other field (and key order) is final17's
        qp = q["predicate"]
        if qp["value_string"] != new_rec["slots"][new_key] or qp["component_string"] != slot_component(new_rec["slots"][new_key],
                                                                                                          COMPONENTS[r], "identity"):
            raise Stop(f"row {r}: re-check of the two strings failed")
        if list(q) != list(p) or list(qp) != list(pr):
            raise Stop(f"row {r}: key order changed")
        if any(q[k] != p[k] for k in p if k not in CHANGED + ("predicate",)) or any(
                qp[k] != pr[k] for k in pr if k not in ("value_string", "component_string")):
            raise Stop(f"row {r}: a field outside the 5 changed")
        out[r - 1] = dumps(q)
    STAGE[0] = "lines outside rows 119-122"
    if any(out[j] != lines[j] for j in range(N_LINES) if j + 1 not in ROWS):
        raise Stop("a line outside rows 119-122 changed")
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=(1024, 1024), timeout=300, startup_timeout=300, scaledown_window=10, retries=0)
def build_remote(payload: dict) -> dict:
    """The container: every sha before any parse, then build(). The reply: the v18 bytes and shas, or a step name (never a value)."""
    try:
        STAGE[0] = "code shas"
        for f_, want in payload["code_sha256"].items():
            if sha(Path("/root", f_).read_bytes()) != want:
                raise Stop(f"{f_}: the container's copy is not the pinned code")
        b = {}
        STAGE[0] = "input shas"
        for k, x in payload["inputs"].items():
            if sha(x["bytes"]) != x["sha256"]:
                raise Stop(f"{k}: sha256 differs from its pin")
            b[k] = x["bytes"]
        STAGE[0] = "import predicates"
        sys.path.insert(0, "/root")
        import predicates as PR  # the verifier's own slot_component (blob pinned by the laptop, sha-checked above)

        STAGE[0] = "final17 lines"
        text = b["final17"].decode("utf-8")
        if not text.endswith("\n") or text.count("\n") != N_LINES:
            raise Stop("final17 is not 251 newline-terminated lines")
        out = build(text[:-1].split("\n"), json.loads(b["new_record"]), json.loads(b["old_record"]), PR.slot_component)
        STAGE[0] = "encode"
        v18 = ("\n".join(out) + "\n").encode("utf-8")
        return {"ok": True, "v18": v18, "sha256": sha(v18), "lines": len(out), "rows_changed": list(ROWS)}
    except Stop as e:
        return {"ok": False, "stop": str(e)}
    except Exception as e:  # noqa: BLE001  the class only: a library message could carry a value
        return {"ok": False, "stop": f"error {type(e).__name__} at {STAGE[0]}"}


def git(*a):
    return subprocess.run(["git", *a], cwd=VDIR, capture_output=True, check=True).stdout


@app.local_entrypoint()
def main(out: str = ""):
    if not out or Path(out).exists():
        sys.exit("STOP: give a new --out path")
    if git("symbolic-ref", "-q", "HEAD").strip() != b"refs/heads/aswin/routec-numbers-verify":
        sys.exit("STOP: run from the attached branch aswin/routec-numbers-verify")
    if git("status", "--porcelain", "--", str(VDIR)).strip():
        sys.exit("STOP: the verifier folder has uncommitted changes; the run must cite committed code")
    head = git("rev-parse", "HEAD").decode().strip()
    for f_ in CODE + (SELF,):  # the code that runs is the committed code
        if git("hash-object", str(VDIR / f_)).decode().strip() != git("rev-parse", f"HEAD:./{f_}").decode().strip():
            sys.exit(f"STOP: {f_} differs from HEAD")
    code_sha = {f_: sha((VDIR / f_).read_bytes()) for f_ in CODE}
    inputs = {k: x | {"bytes": git("show", f"{x['commit']}:{x['path']}")} for k, x in PINS.items()}
    res = build_remote.remote({"inputs": inputs, "code_sha256": code_sha})
    if not res.get("ok"):
        print(json.dumps({"stop": res.get("stop")}))
        sys.exit(2)
    v18 = res["v18"]
    if sha(v18) != res["sha256"] or res["lines"] != N_LINES or res["rows_changed"] != list(ROWS):
        sys.exit("STOP: the reply's sha, line count or rows do not match")
    with open(out, "xb") as fh:
        fh.write(v18)
    print(json.dumps({"out": out, "sha256": res["sha256"], "lines": res["lines"], "rows_changed": res["rows_changed"], "git": head,
                      "app": APP, "code_sha256": code_sha, "self_sha256": sha((VDIR / SELF).read_bytes()),
                      "pins": {k: x["sha256"] for k, x in PINS.items()}}))
