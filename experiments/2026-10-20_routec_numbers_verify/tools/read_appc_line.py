"""Route C rows 119-122 re-verify (B1-X, PRESTATED_rows119_122_REVERIFY_AMEND2 section 3.3): read the numbers verifier's OWN classes for the App C
line of X. Glue only: it loads, selects and prints class words; it classifies nothing.

The laptop reads ONLY keys from the verifier's result JSON: its rows.jsonl pointer (volume, path, sha256) and its post pins for X (pins.post.artifact)
and v19 (pins.post.inputs.provenance): commit, path, sha256. So the reading is bound to the run it reads (Codex rv r1 REQUIRED-2). The container
asserts the frozen verifier modules' sha256 before importing verify_post (REQUIRED-1), reads rows.jsonl from the verifier's Volume (sha-checked), X and
v19 from git bytes (sha-checked), and uses the verifier's own tokens(). The App C line = the raw line of X holding v19 row 119's anchor (exactly once).
For every token tokens() finds on that line it returns the verifier's row for that position, inside a fixed vocabulary (REQUIRED-3): the claimer list
or class word (the slot id before "@"; elements "<class word>" or "<class word>:<id>"), unit (one of UNITS), equal and orphan (real booleans),
the non-result class (a class word or absent), checks_ok (a real boolean or absent) and, for a constants row, its row id. Anything outside that
vocabulary STOPs. Never a printed number, a matched text or a value.

  run from the verifier folder's repo root, attached on aswin/routec-numbers-verify, with the repo .venv's modal 1.6.1:
  /Users/aswinsurya/probe-distribution-shift-from-sims/.venv/bin/modal run --detach \\
      experiments/2026-10-20_routec_numbers_verify/tools/read_appc_line.py --result <results/numbers_verify_<stamp>.json>
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # tools/
VDIR = HERE.parent
APP = "probe-shift_aswin-surya_sub-rerun_reverifyread_kill-2026-11-19"  # 62 characters
CODE = ("verify_post.py", "verify_core.py", "predicates.py")
CODE_SHA256 = {"verify_post.py": "dbfc5911a38d1cb5a98a37b79828f2f6b8bef4de847adba2a6c822a74ec65d1f",  # the frozen verifier modules (blobs 027b0372,
               "verify_core.py": "084d72d14235332c77f63f35ecbe25ba3637b8e5b071bdca74686c208387c906",  # c92cf08f, 5f104735): asserted before any
               "predicates.py": "608abfec7578f2d9378659432287a54fa8abda112a65407da1d1ea713d661e8e"}  # import (Codex rv r1 REQUIRED-1)
SELF = "tools/read_appc_line.py"
OUT_VOL = "probe-shift_aswin-surya_sub-routec-numv_rows_kill-2026-11-19"  # run_numbers_verify.OUT_VOL
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6", "zstandard==0.25.0")
for _f in CODE:
    image = image.add_local_file(VDIR / _f, f"/root/{_f}")
app = modal.App(APP, image=image)
rows_vol = modal.Volume.from_name(OUT_VOL).read_only()
ROW_ANCHOR = 119
STAGE = ["init"]
UNITS = ("number", "nonresult", "deferred", "not_route_C", "preliminary", "slot", "check")  # VC.row units the post writes
_EL = r"[a-z][a-z_]{0,29}(?::[A-Za-z0-9_.#-]{1,40})?"  # one claimer: a class word, or a class word and an id (map:119, rule:<name>, ...)
CLAIMERS_RX = re.compile(rf"{_EL}(?:\+{_EL}){{0,7}}")  # all patterns are used with fullmatch (Codex rv r2 REQUIRED-4)
CLASS_RX = re.compile(r"[a-z][a-z_]{0,29}")
CONST_KP_RX = re.compile(r"constants row ([0-9]{1,4})")
SHA_RX = re.compile(r"[0-9a-f]{64}")
COMMIT_RX = re.compile(r"[0-9a-f]{40}")
PATH_RX = re.compile(r"experiments/2026-10-20_routec_numbers_verify/post/[A-Za-z0-9_.-]{1,80}")
JSONL_RX = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}/rows\.jsonl")


class Stop(Exception):
    """A failed check: a step name only, never a value."""


def sha(b):
    return hashlib.sha256(b).hexdigest()


@app.function(image=image, cpu=(1.0, 1.0), memory=(1024, 4096), timeout=300, startup_timeout=300, scaledown_window=10, retries=0,
              volumes={"/rows": rows_vol})
def read_remote(payload: dict) -> dict:
    try:
        STAGE[0] = "code shas"
        if payload["code_sha256"] != CODE_SHA256:
            raise Stop("the payload's code shas are not the frozen verifier modules'")
        for f_, want in CODE_SHA256.items():
            if sha(Path("/root", f_).read_bytes()) != want:
                raise Stop(f"{f_}: the container's copy is not the pinned code")
        STAGE[0] = "input shas"
        for k in ("x", "v19"):
            if not SHA_RX.fullmatch(str(payload[k]["sha256"])) or sha(payload[k]["bytes"]) != payload[k]["sha256"]:
                raise Stop(f"{k}: sha256 differs from its pin")
        if not JSONL_RX.fullmatch(str(payload["jsonl"]["path"])) or not SHA_RX.fullmatch(str(payload["jsonl"]["sha256"])):
            raise Stop("rows.jsonl: the pointer is malformed")
        jb = Path("/rows", payload["jsonl"]["path"]).read_bytes()
        if sha(jb) != payload["jsonl"]["sha256"]:
            raise Stop("rows.jsonl: sha256 differs from the result's jsonl.sha256")
        STAGE[0] = "import verify_post"
        sys.path.insert(0, "/root")
        import verify_post as VP

        STAGE[0] = "the App C line"
        X = payload["x"]["bytes"].decode("utf-8")
        v19 = [json.loads(x) for x in payload["v19"]["bytes"].decode("utf-8").splitlines() if x.strip()]
        r = [o for o in v19 if o.get("row") == ROW_ANCHOR]
        if len(r) != 1 or not isinstance(r[0].get("anchor"), str) or X.count(r[0]["anchor"]) != 1:
            raise Stop("v19 row 119's anchor is not once in X")
        off = X.index(r[0]["anchor"])
        ls = X.rfind("\n", 0, off) + 1
        le = X.find("\n", off)
        le = len(X) if le < 0 else le
        STAGE[0] = "tokens on the line"
        starts = [s for s, _e, _t in VP.tokens(X)[0] if ls <= s < le]
        STAGE[0] = "rows.jsonl"
        by_pos = {}
        for line in jb.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("artifact") != "post" or "@" not in str(row.get("slot")):
                continue
            head, _, p = str(row["slot"]).rpartition("@")
            if p.isascii() and p.isdigit() and ls <= int(p) < le:
                by_pos.setdefault(int(p), []).append((head, row))
        STAGE[0] = "reply"
        out = []
        for s in starts:
            rs = by_pos.get(s, [])
            if len(rs) != 1:
                raise Stop("a token on the line does not have exactly one post row")
            head, row = rs[0]
            if not CLAIMERS_RX.fullmatch(head):
                raise Stop("a slot id is not a claimer list or class word")
            if row.get("unit") not in UNITS:
                raise Stop("a row's unit is not in the fixed vocabulary")
            if not isinstance(row.get("equal"), bool) or not isinstance(row.get("orphan"), bool):
                raise Stop("a row's equal / orphan is not a boolean")
            nr = row.get("nr_class")
            if nr is not None and not (isinstance(nr, str) and CLASS_RX.fullmatch(nr)):
                raise Stop("a row's non-result class is not a class word")
            ck = row.get("checks_ok")
            if ck is not None and not isinstance(ck, bool):
                raise Stop("a row's checks_ok is not a boolean")
            kp = row.get("key_path")
            m = CONST_KP_RX.fullmatch(kp) if isinstance(kp, str) else None
            out.append({"slot_kind": head, "unit": row["unit"], "equal": row["equal"], "orphan": row["orphan"], "nr_class": nr, "checks_ok": ck,
                        "constants_row": int(m.group(1)) if m else None})
        extra = [p for p in by_pos if p not in set(starts)]
        return {"ok": True, "n_tokens": len(starts), "tokens": out, "rows_off_token_starts": len(extra)}
    except Stop as e:
        return {"ok": False, "stop": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stop": f"error {type(e).__name__} at {STAGE[0]}"}


def git(*a):
    return subprocess.run(["git", *a], cwd=VDIR, capture_output=True, check=True).stdout


def pin_of(x, what):
    """A pin object from the result (keys only): exactly {commit, path, sha256}, well-formed."""
    if not (isinstance(x, dict) and set(x) == {"commit", "path", "sha256"} and COMMIT_RX.fullmatch(str(x["commit"]))
            and PATH_RX.fullmatch(str(x["path"])) and SHA_RX.fullmatch(str(x["sha256"]))):
        sys.exit(f"STOP: the result's {what} pin is malformed")
    return x


@app.local_entrypoint()
def main(result: str = ""):
    if not result or not re.fullmatch(r"experiments/2026-10-20_routec_numbers_verify/results/numbers_verify_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}\.json", result):
        sys.exit("STOP: give --result <experiments/2026-10-20_routec_numbers_verify/results/numbers_verify_<stamp>.json>")
    if git("symbolic-ref", "-q", "HEAD").strip() != b"refs/heads/aswin/routec-numbers-verify":
        sys.exit("STOP: run from the attached branch aswin/routec-numbers-verify")
    if git("status", "--porcelain", "--", str(VDIR / "tools"), *(str(VDIR / f_) for f_ in CODE)).strip():
        sys.exit("STOP: the reader's code has uncommitted changes")
    for f_ in CODE + (SELF,):
        if git("hash-object", str(VDIR / f_)).decode().strip() != git("rev-parse", f"HEAD:./{f_}").decode().strip():
            sys.exit(f"STOP: {f_} differs from HEAD")
    code_sha = {f_: sha((VDIR / f_).read_bytes()) for f_ in CODE}
    if code_sha != CODE_SHA256:
        sys.exit("STOP: the committed verifier modules are not the frozen ones")
    res = json.loads((VDIR.parents[1] / result).read_text())  # keys only: the jsonl pointer and the post pins (written by run_numbers_verify)
    if not isinstance(res, dict):
        sys.exit("STOP: the result is not one JSON object")
    jl = res.get("jsonl") or {}
    if jl.get("volume") != OUT_VOL or not JSONL_RX.fullmatch(str(jl.get("path"))) or not SHA_RX.fullmatch(str(jl.get("sha256"))):
        sys.exit("STOP: the result's jsonl pointer is not this verifier's Volume path and sha")
    post = (res.get("pins") or {}).get("post") or {}
    px = pin_of(post.get("artifact"), "post.artifact (X)")
    pv = pin_of((post.get("inputs") or {}).get("provenance"), "post.inputs.provenance (v19)")
    payload = {"code_sha256": code_sha, "jsonl": {"path": jl["path"], "sha256": jl["sha256"]},
               "x": {"sha256": px["sha256"], "bytes": git("show", f"{px['commit']}:{px['path']}")},
               "v19": {"sha256": pv["sha256"], "bytes": git("show", f"{pv['commit']}:{pv['path']}")}}
    out = read_remote.remote(payload)
    print(json.dumps({"app": APP, "result": result, "jsonl_sha256": jl["sha256"], "x_sha256": px["sha256"], "v19_sha256": pv["sha256"],
                      "self_sha256": sha((VDIR / SELF).read_bytes()), "code_sha256": code_sha} | out, sort_keys=True))
    if not out.get("ok"):
        sys.exit(2)
