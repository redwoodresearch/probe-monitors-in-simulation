"""k = 1 stage (PLAN.md amendment 1): reads and scores on Modal. B-dev's reader and the H1 test-swap code are imported
unchanged and hash-checked; fixed-sim v2's rules (fixes.py, clock.py) are imported unchanged.

  --mode reads   SHARDS H100 shards (spawned on the deployed app): per site, SIM (the simulated reply, fixed, level-1)
                 and SELF (the real reply through the same steps) in output t's place, the level-1 staged e1 messages
                 through output t; plus the H1 test-swap plan's planted R2-undone sites and gate slots.
  --mode score   1 H100 (spawned): the frozen per-environment probes (L63, step A's layer) on every read's slot; checks,
                 per-site scores, G's twin and the e2 yardstick from stored rows; writes <OUT>/scores_<tag>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      modal run experiments/2026-10-11_claimB_k_replies/run_k1_modal.py --mode reads|score
"""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import modal

HERE = Path(__file__).parent
H1 = HERE.parent / "2026-10-11_claimB_h1_text_vs_layout"
FS = HERE.parent / "2026-10-09_fixed_sim"
SW = HERE.parent / "2026-10-07_q2_rendering_swap"
sys.path[:0] = [str(H1) if modal.is_local() else "/root", str(FS) if modal.is_local() else "/root/exp/2026-10-09_fixed_sim"]
import run_h1_test_swap_modal as T  # noqa: E402  (B-dev's reader, frozen probes, test-run loader, the test-swap plan)

B, M, H = T.B, T.M, T.H
SIM, E1, E2, PL2 = T.SIM, T.E1, T.E2, T.PL2
LEVEL0 = "2026-10-06_la_segment-end_qwen3-32b_v1"  # the real runs' level-0 reader inputs (fixed-sim v2's INPUTS)
OUT, SHARDS, TEST_PLAN = "claimB/k1_v1", 6, "claimB/h1_test_swap_v1/cells.json"
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FIX_SHA = {f: hashlib.sha256(((FS if modal.is_local() else Path("/root/exp/2026-10-09_fixed_sim")) / f).read_bytes()).hexdigest() for f in ("fixes.py", "clock.py")}

app = modal.App("probe-shift_aswin-surya_sub-claimb-k1_kill-2026-11-11")
image = (T.image.add_local_file(H1 / "run_h1_test_swap_modal.py", "/root/run_h1_test_swap_modal.py")
         .add_local_file(FS / "fixes.py", "/root/exp/2026-10-09_fixed_sim/fixes.py")
         .add_local_file(FS / "clock.py", "/root/exp/2026-10-09_fixed_sim/clock.py")
         .add_local_file(SW / "rerender.py", "/root/exp/2026-10-07_q2_rendering_swap/rerender.py")
         .add_local_file(SW / "str_replace_tells.py", "/root/exp/2026-10-07_q2_rendering_swap/str_replace_tells.py"))


T_PIN = "199b759bb0a17e75a37b11473067de5d0b3d8a2a18ad4923df53db3ff8214515"  # run_h1_test_swap_modal.py as committed on aswin/claimb-h1-test-swap


def check():
    T.check_code()
    if T.SCRIPT_SHA != T_PIN:
        raise ValueError("the H1 test-swap code differs from the pinned copy")


def plan():
    """Sites, simulated replies and the H1 test-swap plan, with their sha256: what every read is tagged by."""
    acts = Path("/acts")
    raw = {k: (acts / p).read_bytes() for k, p in (("sites", f"{OUT}/sites.json"), ("replies", f"{OUT}/replies.jsonl"), ("test_plan", TEST_PLAN))}
    sites = json.loads(raw["sites"])["sites"]
    replies = {r["site"]: r for r in map(json.loads, raw["replies"].decode().splitlines())}
    tplan = json.loads(raw["test_plan"])
    tag = hashlib.sha256(b"".join(raw[k] for k in sorted(raw)) + SCRIPT_SHA.encode()).hexdigest()[:12]
    return sites, replies, tplan, tag, {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}


def donors() -> dict:
    import clock

    out = {}
    for r in T.MP.inputs(LEVEL0):
        if r["arm"] == E1 and r["environment"] in DEV:
            n = clock.now_of([m["content"] for m in r["messages"] if m["role"] == "tool" and isinstance(m["content"], str)])
            if n is not None:
                if r["cell"] in out:
                    raise ValueError(f"two dev real e1 runs share cell {r['cell']}")
                out[r["cell"]] = n
    return out


def build_jobs():
    """Per site: SIM and SELF message lists (level-1 staged e1 run through output t, output t replaced), with counts."""
    import clock
    import fixes
    from probe_shift.scrub import scrub

    sites, replies, tplan, tag, shas = plan()
    want = {s["cell"] for s in sites}
    l0 = {r["cell"]: r for r in T.MP.inputs(LEVEL0) if r["arm"] == E1 and r["cell"] in want}
    l1 = {c: r for (c, a), r in T.test_runs().items() if a == E1 and c in want}
    dn = donors()
    jobs, info, count = [], {}, Counter()
    for s in sites:
        rep = replies.get(s["site"])
        if rep is None or rep["status"] != "ok":
            count["no_ok_reply"] += 1
            continue
        m0, m1 = l0[s["cell"]]["messages"], l1[s["cell"]]["messages"]
        if len(m0) != len(m1) or any(a["role"] != b["role"] for a, b in zip(m0, m1)):
            raise ValueError(f"site {s['site']}: level-0 and level-1 messages don't align")
        sa = {v: k for k, v in H.step_action(m1).items()}
        m = sa[(s["step"], s["action"])]

        def post_sim(text):
            """fixed-sim v2's corrections on the simulated reply, then the level-1 scrub."""
            msgs = [dict(x) for x in m0]
            msgs[m] = msgs[m] | {"content": text}
            new, recs = fixes.apply(msgs, ("R1", "R2", "R5"))
            elsewhere = sum(1 for x in recs if x["changed"] and x["msg"] != m)
            out = new[m]["content"]
            now = clock.now_of([text]) if isinstance(text, str) else None  # from the uncorrected reply, as v2 read the uncorrected outputs
            if now is not None and isinstance(out, str):
                _, donor_now = clock.donor_for(s["cell"], dn)
                out, _ = clock.shift_text(out, donor_now - now, now)
            return scrub(out, 1), elsewhere, now is not None
        sim_t, sim_else, sim_clock = post_sim(rep["reader_output"])
        self_t, self_else, self_clock = scrub(m0[m]["content"], 1), 0, False  # real outputs: untouched by the rules, as in v2
        count["fix_changes_elsewhere_sim"] += sim_else
        count["fix_changes_elsewhere_self"] += self_else
        count["clock_shifted_sim"] += sim_clock
        count["clock_shifted_self"] += self_clock
        count["self_text_equals_staged"] += self_t == m1[m]["content"]
        count["sim_text_equals_self_text"] += sim_t == self_t
        cut = [dict(x) for x in m1[: m + 1]]
        for version, text in (("SIM", sim_t), ("SELF", self_t)):
            msgs = [dict(x) for x in cut]
            msgs[-1] = msgs[-1] | {"content": text}
            jobs.append((s["cell"], E1, f"{version}|{s['site']}|{tag}", msgs, [], m1[: m + 1], m1))
        info[s["site"]] = {"msg": m, "identical": sim_t == self_t}
    runs = T.test_runs()
    for n, p in enumerate(tplan["planted"]):
        twin, pm = runs[(p["cell"], SIM)]["messages"], runs[(p["cell"], PL2)]["messages"]
        pl = [dict(x) for x in twin[: p["i"] + 1]]
        pl[-1] = pl[-1] | {"content": pm[p["i"]]["content"]}
        jobs.append((p["cell"], SIM, f"PL|{n}|{tag}", pl, [], twin[: p["i"] + 1], twin))
    for g in tplan["gate_slots"]:
        full = runs[(g["cell"], SIM)]["messages"]
        jobs.append((g["cell"], SIM, f"gate|{g['i']}|{tag}", [dict(x) for x in full[: g["i"] + 1]], [], full[: g["i"] + 1], full))
    return jobs, dict(count), tag, shas, info


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=90 * 60, retries=0)
def reads(shard: int, model: str, revision: str) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    check()
    B.P.vol.reload()
    jobs, count, tag, shas, _ = build_jobs()
    mine = jobs[shard::SHARDS]
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    bad = []
    for cell, arm, version, _, _, cut, full in mine:  # the pilot's prefix check, unchanged
        sc, sf = segment_end_positions(tok, cut, M.MAX_TOKENS), segment_end_positions(tok, full, M.MAX_TOKENS)
        ends = [p["token"] for p in sc["positions"] if p["kind"] == "output_end" and p["kept"]]
        if ends and sc["ids"][: max(ends) + 1] != sf["ids"][: max(ends) + 1]:
            bad.append(version)
    if bad:
        raise ValueError(f"{len(bad)} cut prefixes render differently from their full runs; no reads")
    B.OUT = OUT
    return B._read([j[:5] for j in mine], model, revision, f"shard{shard}") | {"jobs": len(mine), "counts": count, "tag": tag}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol}, timeout=40 * 60, retries=0)
def score(env_to_fold: dict) -> dict:
    import numpy as np

    check()
    B.P.vol.reload()
    sites, replies, tplan, tag, shas = plan()
    _, count, _, _, info = build_jobs()
    fz = T.frozen(env_to_fold)
    s, XP, row, stored = fz["s"], fz["XP"], fz["row"], fz["stored"]
    env_of = {r["cell"]: r["environment"] for r in fz["pruns"]}

    def slot(cell, arm, version, sa):
        f = Path("/acts") / OUT / "runs" / f"{hashlib.sha256(f'{cell}|{arm}|{version}|{tag}'.encode()).hexdigest()[:16]}.npz"
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        if (meta["version"], meta["script_sha256"]) != (f"{version}|{tag}", B.SCRIPT_SHA):
            raise ValueError(f"{f.name}: made for another plan or reader")
        steps = [tuple(x) for x in meta["steps"]]
        return {l: z[f"L{l}"][steps.index(tuple(sa))].astype(np.float32) for l in M.LAYERS} if tuple(sa) in steps else None

    def sc(cell, x, which):
        e = env_of[cell]
        layer = 63 if which == "63" else M.A_LAYER[e]
        return float(s(e, layer, x[layer][None])[0])

    gate = []
    for g in tplan["gate_slots"]:
        x = slot(g["cell"], SIM, f"gate|{g['i']}", g["sa"])
        k = row.get((g["cell"], SIM, *g["sa"]))
        bitwise = x is not None and k is not None and np.array_equal(x[63].astype(np.float16), XP[63][k])
        gate.append({"bitwise": bool(bitwise), "score_diff": abs(sc(g["cell"], x, "63") - g["stored_63"]) if x is not None else None})
    planted, lost_p = [], 0
    for n, p in enumerate(tplan["planted"]):
        x = slot(p["cell"], SIM, f"PL|{n}", p["sa"])
        if x is None:
            lost_p += 1
            continue
        planted.append({"main_task": p["main_task"], "reference": p["reference_63"], "read_shift": sc(p["cell"], x, "63") - p["twin_63"]})
    rec, lost = [], Counter()
    for st in sites:
        rep = replies.get(st["site"])
        if rep is None or rep["status"] != "ok":
            lost["no_ok_reply"] += 1
            continue
        sa = (st["step"], st["action"])
        xs = slot(st["cell"], E1, f"SIM|{st['site']}", sa)
        xf = slot(st["cell"], E1, f"SELF|{st['site']}", sa)
        if xs is None or xf is None:
            lost["slot_outside_cut"] += 1
            continue
        e1 = stored.get((st["cell"], E1, *sa))
        twin = stored.get((st["cell"], SIM, *sa))
        if e1 is None or twin is None:
            lost["no_stored_twin"] += 1
            continue
        e2 = stored.get((st["cell"], E2, *sa))
        r = {"site": st["site"], "main_task": st["main_task"], "environment": st["environment"], "tool": rep.get("tool"),
             "sim_text_equals_self_text": info[st["site"]]["identical"]}
        for w in ("63", "A"):
            r[w] = {"sim": sc(st["cell"], xs, w), "self": sc(st["cell"], xf, w), "stored_e1": e1[w], "stored_fixed_sim": twin[w],
                    "stored_e2": e2[w] if e2 else None}
        rec.append(r)
    res = {"label": "k = 1 stage: per-site scores (keys and numbers only)", "gate": {"slots": gate, "pass": len(gate) == len(tplan["gate_slots"]) and all(
               x["bitwise"] and x["score_diff"] is not None and x["score_diff"] < 1e-5 for x in gate)},
           "fits": fz["fits"], "fits_converged": all(v["converged"] for v in fz["fits"].values()), "planted": planted, "planted_lost_to_cut": lost_p,
           "sites": rec, "lost": dict(lost), "build_counts": count, "plan_tag": tag, "inputs_sha256": shas, "script_sha256": SCRIPT_SHA,
           "fixes_sha256": FIX_SHA, "b_script": B.SCRIPT_SHA, "t_script": T.SCRIPT_SHA}
    (Path("/acts") / OUT / f"scores_{tag}.json").write_text(json.dumps(res))
    B.P.vol.commit()
    return {k: res[k] for k in ("lost", "build_counts", "planted_lost_to_cut", "plan_tag", "fits_converged")} | {"sites_scored": len(rec), "gate_pass": res["gate"]["pass"]}


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


@app.local_entrypoint()
def main(mode: str = "reads", sim_spent: float = -1.0):
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    if modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    folds = json.loads((HERE.parents[1] / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    if mode == "reads":
        if sim_spent < 0 or sim_spent + 35.0 > 400.0:  # the simulator's ledger spend, read from its results file before launch
            raise SystemExit(f"--sim-spent {sim_spent}: the reads launch only if the simulator spend + $35 <= $400")
        print(f"start the watcher with --stop {400.0 - sim_spent:.2f}", flush=True)
        f = modal.Function.from_name(app.name, "reads")
        for sh in range(SHARDS):
            print(sh, f.spawn(sh, QWEN3_32B["name"], QWEN3_32B["revision"]).object_id, flush=True)
    elif mode == "score":
        print(modal.Function.from_name(app.name, "score").spawn(folds).object_id)
    else:
        raise SystemExit(f"unknown mode {mode}")
