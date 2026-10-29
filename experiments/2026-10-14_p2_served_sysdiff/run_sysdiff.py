"""RULE.md: per T196 task, the SERVED system turn (vLLM's ids, up to the first <|im_end|>) of sim vs run1: differing tokens split
into tool-list tokens (between the template's <tools> ... </tools> markers, located by their token ids) and prompt-text tokens.
Token ids are never decoded, printed or logged; only counts leave the container. Modal CPU, detached, cents.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with pyyaml modal run --detach \\
      experiments/2026-10-14_p2_served_sysdiff/run_sysdiff.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-p2probes_sysdiff_kill-2026-11-13"
ACTS, P2, HF = ("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13",
                "probe-shift_aswin-surya_hf-cache")
TOK = "hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654/tokenizer.json"
REPLAY_NPZ = ("2026-10-14_p2_callfacet/2026-10-14_141350/DT_destructive_primary__rule1_command.npz",
              "e142c828bb20f16305621af64c735635f863964731bfe42a0dd3b5f9147810ed")  # run1/sim keys and irids of the counted runs
MAIN = ["rerun_segends", "scaleup_canary", "scaleup_l2", "scaleup_topup"]
FILES = {"run1": "attack.jsonl", "sim": "sim_attack@september.jsonl"}
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "tokenizers>=0.21")
VOL = {"/acts": modal.Volume.from_name(ACTS, version=2).read_only(), "/p2": modal.Volume.from_name(P2, version=2).read_only(),
       "/hf": modal.Volume.from_name(HF).read_only(), "/work": modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2)}


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=900, retries=0, scaledown_window=10, startup_timeout=120, volumes=VOL)
def sysdiff(files: dict, meta: dict) -> dict:
    try:
        r = meta | _sysdiff(files)
        out = Path("/work/sysdiff") / meta["stamp (PT)"]
        out.mkdir(parents=True)
        with (out / "record.json").open("x") as fh:  # inside the job, counts only
            fh.write(json.dumps(r, indent=1))
        VOL["/work"].commit()
        return r
    except Exception as x:  # an input line could ride on an exception: withhold every message
        raise RuntimeError(f"sysdiff: {type(x).__name__} (message withheld)") from None


def _sysdiff(files: dict) -> dict:
    import difflib

    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path("/hf") / TOK))
    raw = json.loads((Path("/hf") / TOK).read_text())
    ime = {t["content"]: t["id"] for t in raw["added_tokens"]}["<|im_end|>"]
    special = {t["id"] for t in raw["added_tokens"]}
    marks = {m: [tok.encode(v, add_special_tokens=False).ids for v in (m, "\n" + m, m + "\n", "\n" + m + "\n")] for m in ("<tools>", "</tools>")}
    zb = (Path("/p2") / REPLAY_NPZ[0]).read_bytes()
    if hashlib.sha256(zb).hexdigest() != REPLAY_NPZ[1]:
        raise ValueError("replay npz sha256 differs from its pin")
    import io

    with np.load(io.BytesIO(zb)) as z:
        keys, irids = z["key"], z["irid"]
    want = {}  # (world, task) -> irids of the counted run
    for k, i in zip(keys.tolist(), irids.tolist()):
        w, t = k.split("|")
        if w in FILES:
            want.setdefault((w, t), set()).add(i)
    by_irid = {i: wt for wt, s in want.items() for i in s}
    sysids = {}
    for w, fl in files.items():
        for f, sha in fl:
            h = hashlib.sha256()
            with open(Path("/acts") / f, "rb") as fh:
                for line in fh:
                    h.update(line)
                    ep = json.loads(line)
                    hit = {by_irid[c["irid"]] for c in ep["calls"] if c.get("irid") in by_irid and by_irid[c["irid"]][0] == w}
                    for wt in hit:
                        ids = np.asarray(ep["ids"], np.int64)
                        e = np.flatnonzero(ids == ime)
                        if not len(e):
                            raise ValueError("an epoch without <|im_end|>")
                        st = ids[: e[0] + 1]
                        if wt in sysids and not (len(sysids[wt]) == len(st) and (sysids[wt] == st).all()):
                            raise ValueError("two epochs of one run disagree on the system turn")  # Codex r1
                        sysids.setdefault(wt, st)
            if h.hexdigest() != sha:
                raise ValueError("a staged file's sha256 differs from its pin")

    def find(seq, pats, after=0):
        for p in pats:
            for i in range(after, len(seq) - len(p) + 1):
                if p and seq[i] == p[0] and list(seq[i:i + len(p)]) == p:
                    return i, i + len(p)
        return None

    def tools_span(s):
        a = find(s, marks["<tools>"])
        b = find(s, marks["</tools>"], a[1] if a else 0) if a else None
        return (a[0], b[1]) if a and b else None

    per, n_no_marks, unpaired = [], 0, 0
    cohort = sorted({t for w, t in want if ("run1", t) in want and ("sim", t) in want})
    for t in cohort:
        if ("run1", t) not in sysids or ("sim", t) not in sysids:
            unpaired += 1  # a counted run's served epoch not found in the staged files
            continue
        r, s = sysids[("run1", t)], sysids[("sim", t)]
        sr, ss = tools_span(r), tools_span(s)
        if sr is None or ss is None:  # no valid ordered marker pair on both sides: excluded from the split, counted (Codex r1)
            n_no_marks += 1
            continue
        cnt = {"tool_list": 0, "prompt_text": 0, "special": 0}
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r.tolist(), s.tolist(), autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            for seq, span, lo, hi in ((r, sr, i1, i2), (s, ss, j1, j2)):
                for p in range(lo, hi):
                    cnt["special" if int(seq[p]) in special else "tool_list" if span and span[0] <= p < span[1] else "prompt_text"] += 1
        per.append({"task": t, "len_run1": int(len(r)), "len_sim": int(len(s)), "tool_list_len_run1": None if sr is None else sr[1] - sr[0],
                    "tool_list_len_sim": None if ss is None else ss[1] - ss[0], "identical": bool(len(r) == len(s) and (r == s).all()), **cnt})
    a = lambda k: np.array([p[k] for p in per], np.float64)  # noqa: E731
    q = lambda v: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "sum": float(v.sum())} if len(v) else None  # noqa: E731
    tl, pt = a("tool_list"), a("prompt_text")
    return {"cohort_tasks_with_run1_and_sim": len(cohort), "tasks_not_found_in_staging": unpaired, "tasks": len(per), "tasks_without_both_markers": n_no_marks,
            "count_conventions": "a replace opcode counts its tokens on both sides; the tool span runs from the first token of the <tools> marker match to the last of the </tools> match (newline variants included)", "identical_system_turns": int(sum(p["identical"] for p in per)),
            "tool_list_diff": q(tl), "prompt_text_diff": q(pt), "special_diff": q(a("special")),
            "share_tasks_tool_list_diff_gt_prompt_text_diff": float((tl > pt).mean()) if len(per) else None,
            "share_tasks_only_tool_list_differs": float(((tl > 0) & (pt == 0)).mean()) if len(per) else None,
            "share_tasks_prompt_text_differs": float((pt > 0).mean()) if len(per) else None,
            "tool_list_len": {"run1": q(np.array([p["tool_list_len_run1"] for p in per if p["tool_list_len_run1"] is not None], np.float64)),
                              "sim": q(np.array([p["tool_list_len_sim"] for p in per if p["tool_list_len_sim"] is not None], np.float64))},
            "per_task": per}


@app.local_entrypoint()
def main():
    import yaml

    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    head, here = git("rev-parse", "HEAD"), "experiments/2026-10-14_p2_served_sysdiff"
    for f in ("RULE.md", "run_sysdiff.py"):
        if git("status", "--porcelain", "--", f"{here}/{f}") or not git("ls-files", "--", f"{here}/{f}"):
            raise SystemExit(f"{f} is not committed unchanged")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit("HEAD is not pushed")
    cfgdir = REPO / "experiments/2026-10-08_q3_onpolicy_figures/config"
    files = {w: [] for w in FILES}
    for c in MAIN:
        cfg = yaml.safe_load((cfgdir / f"extract_{c}.yaml").read_text())
        for w, fn in FILES.items():
            files[w].append((f"{cfg['inputs']}/inputs/{fn}", cfg["inputs_sha256"][fn]))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = sysdiff.remote(files, {"label": "PRELIMINARY; counts only (no token text); RULE.md in this folder", "git": head, "stamp (PT)": stamp,
                                 "app": APP, "staged_files": files})
    r = rec
    dest = HERE / "results" / f"sysdiff_{stamp}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.open("x").write(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "per_task"}, indent=1))
