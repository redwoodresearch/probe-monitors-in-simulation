"""Route C rows 119-122 re-verify (B1-X, PRESTATED_rows119_122_REVERIFY_AMEND2 section 3.1; 09 via 2e, 10-21 PT): build X, provenance v19 and
constants v10 IN A MODAL CONTAINER.
  X   = draft 0141 with exactly the 3 characters at the visible positions where draft 0141 and draft 0029 differ (hidden comments stripped with the
        verifier's own HIDDEN) replaced by 0029's. X's visible text is 0029's; every position equals 0141's.
  v19 = provenance v18 with ONLY rows 119-121's `anchor` and `printed` rewritten (X's text at the old anchor's offset) and every row's
        `v84_sha256` set to sha256(X).
  v10 = constants v9 with ONLY row 35's `anchor` rewritten the same way.
Values never reach the laptop's terminal. The laptop sends git bytes (and draft 0029's bytes from the writing folder) UNPARSED with their pinned
sha256; verify_post.py, verify_core.py and predicates.py ride in the image and are sha-checked in the container. The container checks every sha
before it parses, proves it can write in v18's and v9's own style, runs 09's X gate, builds and re-checks v19 and v10, and returns the three files'
bytes and booleans. The laptop writes them create-only and prints paths, sha256, line counts and booleans only. A failed check returns a step name.

  run from the verifier folder's repo root, attached on aswin/routec-numbers-verify, with the repo .venv's modal 1.6.1:
  /Users/aswinsurya/probe-distribution-shift-from-sims/.venv/bin/modal run --detach \\
      experiments/2026-10-20_routec_numbers_verify/tools/build_reverify_x.py --out-x <new .md> --out-v19 <new .jsonl> --out-v10 <new .jsonl>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # tools/
VDIR = HERE.parent  # the verifier folder (verify_post.py, verify_core.py, predicates.py)
APP = "probe-shift_aswin-surya_sub-rerun_reverifyx_kill-2026-11-19"  # 59 characters
CODE = ("verify_post.py", "verify_core.py", "predicates.py")
CODE_SHA256 = {"verify_post.py": "dbfc5911a38d1cb5a98a37b79828f2f6b8bef4de847adba2a6c822a74ec65d1f",  # the frozen verifier modules (blobs 027b0372,
               "verify_core.py": "084d72d14235332c77f63f35ecbe25ba3637b8e5b071bdca74686c208387c906",  # c92cf08f, 5f104735): asserted before any
               "predicates.py": "608abfec7578f2d9378659432287a54fa8abda112a65407da1d1ea713d661e8e"}  # import (Codex rv r1 REQUIRED-1)
SELF = "tools/build_reverify_x.py"
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6",  # the verifier's base
                                                                       "zstandard==0.25.0")
for _f in CODE:
    image = image.add_local_file(VDIR / _f, f"/root/{_f}")
app = modal.App(APP, image=image)

E = "experiments/2026-10-20_routec_numbers_verify"
PINS = {
    "d0141": {"commit": "558c304674538182cde388f5b2f7f754698f2cf5", "path": f"{E}/post/draft_v84_frozen_0141.md",
              "sha256": "6e3754e1995ee9df1e86a9fea418c09750d71d019b908d3aa65dfdf7d3fdcd06"},
    "v18": {"commit": "58bf4a3ac170aefc41a9692108316a49d2abe083", "path": f"{E}/post/provenance_v84_final18_0141.jsonl",
            "sha256": "13d29fb90cac3e3e54145b5d9f41319614a0ff63ebf33b5b4de3d251419741b0"},
    "v9": {"commit": "98b5054fbf8306e54711b1a7bacd72b5f5e0af2a", "path": f"{E}/post/constants_v84_final9_0141.jsonl",
           "sha256": "02751eb210a35406100b1a9934ef5dcd419f5e11625082ddfd2002d133badcc3"},
}
D0029 = {"path": "/Users/aswinsurya/probe-shift-writing-handoff-2026-10-15/w23/drafts/draft_v84_frozen_0029.md",
         "sha256": "569be149b0c99d31df69b58a6b9728c296c3ec22161b765da3a4a27bcc592ec7"}
N_V18 = 251
ROWS = (119, 120, 121)  # rewritten
ROW_KEPT = 122  # on the same line; its printed span must be unchanged
CONST_ROW = 35
N_DIFF = 3
DIGITS = "0123456789"  # ASCII digits only
STAGE = ["init"]  # the step running: an unexpected exception reports this fixed name (never a value) with its class


class Stop(Exception):
    """A failed check. Its message is a step name and row number only, never a value."""


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dumps(o):
    return json.dumps(o, ensure_ascii=False)  # v18's and v9's style (proved by the round trips below)


def lines_of(b, what):
    t = b.decode("utf-8")
    if not t.endswith("\n"):
        raise Stop(f"{what} does not end with a newline")
    return t[:-1].split("\n")


def once(hay, needle, what):
    if not isinstance(needle, str) or not needle or hay.count(needle) != 1:
        raise Stop(f"{what}: does not occur exactly once")
    return hay.index(needle)


def build(t41, t29, v18, v9, HIDDEN):
    """Pure: draft texts (str), v18 / v9 lines (str, no newline), the verifier's HIDDEN regex -> (X, v19 lines, v10 lines, x_gate booleans)."""
    STAGE[0] = "round trips"
    if len(v18) != N_V18:
        raise Stop("v18 is not 251 lines")
    o18 = [json.loads(x) for x in v18]
    o9 = [json.loads(x) for x in v9]
    if any(dumps(o) != x for o, x in zip(o18, v18)) or any(dumps(o) != x for o, x in zip(o9, v9)):
        raise Stop("round trip: v18 or v9 does not re-serialize byte-identically")
    if [o.get("row") for o in o18[ROWS[0] - 1:ROW_KEPT]] != list(ROWS) + [ROW_KEPT]:
        raise Stop("v18 rows 119-122 are not on lines 119-122")
    if any(o.get("v84_sha256") != PINS["d0141"]["sha256"] for o in o18):
        raise Stop("real data first: a v18 row is not for draft 0141")
    c_ix = [j for j, o in enumerate(o9) if o.get("row") == CONST_ROW]
    if len(c_ix) != 1:
        raise Stop("v9 does not hold constants row 35 exactly once")
    c = o9[c_ix[0]]

    STAGE[0] = "visible diff"
    vis41, vis29 = HIDDEN.sub("", t41), HIDDEN.sub("", t29)
    if len(vis41) != len(vis29):
        raise Stop("the visible texts differ in length")
    dv = [i for i in range(len(vis41)) if vis41[i] != vis29[i]]
    if len(dv) != N_DIFF:
        raise Stop("the visible texts do not differ in exactly 3 characters")
    STAGE[0] = "visible to raw"
    spans = [m.span() for m in HIDDEN.finditer(t41)]
    raw, k = [], 0
    for a, b in spans + [(len(t41), len(t41))]:
        raw += range(k, a)
        k = b
    if "".join(t41[j] for j in raw) != vis41:
        raise Stop("the visible-to-raw index map does not reproduce 0141's visible text")
    xl = list(t41)
    for i in dv:
        xl[raw[i]] = vis29[i]
    X = "".join(xl)

    STAGE[0] = "X gate"
    diffs = [j for j in range(len(X)) if X[j] != t41[j]]
    pspan = {}
    for r in ROWS + (ROW_KEPT,):
        p = o18[r - 1]
        s = once(t41, p.get("anchor"), f"row {r} anchor in 0141") + once(p["anchor"], p.get("printed"), f"row {r} printed in its anchor")
        pspan[r] = (s, s + len(p["printed"]))
    cs = once(t41, c.get("anchor"), "constants row 35 anchor in 0141") + once(c["anchor"], c.get("text"), "constants row 35 literal in its anchor")
    gate = {"x_visible_eq_0029": HIDDEN.sub("", X) == vis29, "x_comments_eq_0141": HIDDEN.findall(X) == HIDDEN.findall(t41),
            "x_len_eq_0141": len(X) == len(t41), "x_diff_chars_eq_3": len(diffs) == N_DIFF,
            "diffs_inside_rows_119_121_printed": all(any(pspan[r][0] <= j < pspan[r][1] for r in ROWS) for j in diffs),
            "row_122_printed_unchanged": X[pspan[ROW_KEPT][0]:pspan[ROW_KEPT][1]] == t41[pspan[ROW_KEPT][0]:pspan[ROW_KEPT][1]],
            "constant_literal_unchanged": X[cs:cs + len(c["text"])] == t41[cs:cs + len(c["text"])],
            "diff_chars_digit_for_digit": all(X[j] in DIGITS and t41[j] in DIGITS for j in diffs)}   # 8th (fresh review rv3 item 4)
    bad = [k_ for k_, v_ in gate.items() if v_ is not True]
    if bad:
        raise Stop("X gate false: " + ",".join(bad))

    STAGE[0] = "v19"
    shx = sha(X.encode("utf-8"))
    out19 = []
    for j, (o, line) in enumerate(zip(o18, v18)):
        q = json.loads(line)  # a fresh copy; key order kept
        q["v84_sha256"] = shx
        r = o.get("row")
        if r in ROWS:
            a, p = o["anchor"], o["printed"]
            off, po = t41.index(a), a.index(p)
            na = X[off:off + len(a)]
            npr = na[po:po + len(p)]
            pr = o.get("predicate") or {}
            if pr.get("op") != "slot_component_eq" or npr != pr.get("component_string"):
                raise Stop(f"row {r}: the new printed is not the row's L48 component_string")
            if X.count(na) != 1 or X.count(a) != 0:
                raise Stop(f"row {r}: the new anchor is not once in X, or the old anchor is still in X")
            q["anchor"], q["printed"] = na, npr
        chg = ("v84_sha256", "anchor", "printed") if r in ROWS else ("v84_sha256",)
        if list(q) != list(o) or any(q[k_] != o[k_] for k_ in o if k_ not in chg):
            raise Stop(f"v19 line {j + 1}: a field outside the permitted ones changed, or key order changed")
        out19.append(dumps(q))

    STAGE[0] = "v10"
    out10 = list(v9)
    q = json.loads(v9[c_ix[0]])
    off = t41.index(c["anchor"])
    na = X[off:off + len(c["anchor"])]
    if X.count(na) != 1 or na.count(c["text"]) != 1 or X.count(c["anchor"]) != 0:
        raise Stop("constants row 35: the new anchor is not once in X with its literal once, or the old anchor is still in X")
    q["anchor"] = na
    if list(q) != list(c) or any(q[k_] != c[k_] for k_ in c if k_ != "anchor"):
        raise Stop("v10 row 35: a field other than anchor changed, or key order changed")
    out10[c_ix[0]] = dumps(q)
    if any(out10[j] != v9[j] for j in range(len(v9)) if j != c_ix[0]):
        raise Stop("v10: a line other than row 35's changed")
    return X, out19, out10, gate


@app.function(image=image, cpu=(1.0, 1.0), memory=(1024, 1024), timeout=300, startup_timeout=300, scaledown_window=10, retries=0)
def build_remote(payload: dict) -> dict:
    """The container: every sha before any parse, then build(). The reply: the three files' bytes, shas and the X gate's booleans, or a step name."""
    try:
        STAGE[0] = "code shas"
        if payload["code_sha256"] != CODE_SHA256:
            raise Stop("the payload's code shas are not the frozen verifier modules'")
        for f_, want in CODE_SHA256.items():
            if sha(Path("/root", f_).read_bytes()) != want:
                raise Stop(f"{f_}: the container's copy is not the pinned code")
        b = {}
        STAGE[0] = "input shas"
        for k, x in payload["inputs"].items():
            if sha(x["bytes"]) != x["sha256"]:
                raise Stop(f"{k}: sha256 differs from its pin")
            b[k] = x["bytes"]
        STAGE[0] = "import verify_post"
        sys.path.insert(0, "/root")
        import verify_post as VP  # the verifier's own HIDDEN (blob pinned by the laptop, sha-checked above)

        X, l19, l10, gate = build(b["d0141"].decode("utf-8"), b["d0029"].decode("utf-8"), lines_of(b["v18"], "v18"), lines_of(b["v9"], "v9"),
                                  VP.HIDDEN)
        STAGE[0] = "encode"
        xb, b19, b10 = X.encode("utf-8"), ("\n".join(l19) + "\n").encode("utf-8"), ("\n".join(l10) + "\n").encode("utf-8")
        return {"ok": True, "x": xb, "v19": b19, "v10": b10, "sha256": {"x": sha(xb), "v19": sha(b19), "v10": sha(b10)},
                "lines": {"v19": len(l19), "v10": len(l10)}, "x_gate": gate}
    except Stop as e:
        return {"ok": False, "stop": str(e)}
    except Exception as e:  # noqa: BLE001  the class only: a library message could carry a value
        return {"ok": False, "stop": f"error {type(e).__name__} at {STAGE[0]}"}


def git(*a):
    return subprocess.run(["git", *a], cwd=VDIR, capture_output=True, check=True).stdout


@app.local_entrypoint()
def main(out_x: str = "", out_v19: str = "", out_v10: str = ""):
    outs = {"x": out_x, "v19": out_v19, "v10": out_v10}
    if any(not v or Path(v).exists() for v in outs.values()) or len({str(Path(v).resolve()) for v in outs.values()}) != 3:
        sys.exit("STOP: give three new, distinct paths: --out-x, --out-v19, --out-v10")
    if git("symbolic-ref", "-q", "HEAD").strip() != b"refs/heads/aswin/routec-numbers-verify":
        sys.exit("STOP: run from the attached branch aswin/routec-numbers-verify")
    if git("status", "--porcelain", "--", str(VDIR)).strip():
        sys.exit("STOP: the verifier folder has uncommitted changes; the run must cite committed code")
    head = git("rev-parse", "HEAD").decode().strip()
    for f_ in CODE + (SELF,):  # the code that runs is the committed code
        if git("hash-object", str(VDIR / f_)).decode().strip() != git("rev-parse", f"HEAD:./{f_}").decode().strip():
            sys.exit(f"STOP: {f_} differs from HEAD")
    code_sha = {f_: sha((VDIR / f_).read_bytes()) for f_ in CODE}
    if code_sha != CODE_SHA256:
        sys.exit("STOP: the committed verifier modules are not the frozen ones")
    inputs = {k: x | {"bytes": git("show", f"{x['commit']}:{x['path']}")} for k, x in PINS.items()}
    inputs["d0029"] = D0029 | {"bytes": Path(D0029["path"]).read_bytes()}
    res = build_remote.remote({"inputs": inputs, "code_sha256": code_sha})
    if not res.get("ok"):
        print(json.dumps({"stop": res.get("stop")}))
        sys.exit(2)
    if any(sha(res[k]) != res["sha256"][k] for k in outs) or res["lines"] != {"v19": N_V18, "v10": len(lines_of(inputs["v9"]["bytes"], "v9"))} \
            or res["x_gate"] != {k_: True for k_ in res["x_gate"]} or len(res["x_gate"]) != 8:
        sys.exit("STOP: the reply's shas, line counts or X gate do not match")
    for k, p in outs.items():
        with open(p, "xb") as fh:
            fh.write(res[k])
    print(json.dumps({"out": outs, "sha256": res["sha256"], "lines": res["lines"], "x_gate": res["x_gate"], "git": head, "app": APP,
                      "code_sha256": code_sha, "self_sha256": sha((VDIR / SELF).read_bytes()),
                      "pins": {k: x["sha256"] for k, x in PINS.items()} | {"d0029": D0029["sha256"]}}))
