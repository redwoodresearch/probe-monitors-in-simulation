"""F12 (RULE.md Addition 2), POST HOC: CRLF line endings in the HTTP header blocks of simulated bash outputs, derived from
real honest runs. Modal CPU. Counts, hashes and token counts only; never message text (attack runs included). Staged inputs
go to the Volume, never overwritten; nothing is staged unless every gate passes.

Steps, in one container:
  1. Derive: in every bash output, HTTP header blocks are found line by line (a trailing \\r is ignored for matching):
     a status line (`HTTP/x NNN`, plain or curl -v's `< `) opens a received block, a request line (`> GET /x HTTP/1.1`) a
     sent block; header lines (`name: value`, same prefix) continue it; a blank line (same prefix) ends it. Line types:
     status, header, end (plain), status<, header<, end<, req>, header>, end>; each counted by the call's family (curl
     -i/-I/-v, curl other, other bash) and by whether the line ends in \\r. A (family, type) cell is IN F12 if REAL2 has
     >= 20 such lines with >= 95% ending in \\r, and REAL1 confirms it (>= 20 lines, >= 95%).
  2. F12: every simulated line of an IN cell that does not end in \\r gets one; nothing else changes.
  3. Gates: fixtures (seeded synthetic outputs); only bash outputs of IN families change, only by inserted \\r; F12 is
     idempotent on every changed run; real-arm control (REAL1, REAL2, real e1/e2 attack): F12 changes <= 5% of their IN-cell
     lines; base shard files equal their pins.
  4. Stage `2026-10-14_q2_f12q_inputs_v1`: changed runs only (arms sim-{attack,honest}-fx-fullF9F11F12, built on full + F9 +
     F11), plus one reproduction run per extractor shard copied byte-identically from d9's mean-pool dumps' inputs (F11's
     layout and dealing). Qwen3-32B token counts (segment_end_positions, cut 32,768) and the price.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/stage_f12.py
"""
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
FOLDER = "experiments/2026-10-14_q2_ffm_iter1"
if modal.is_local():
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-q2-ffm1_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"  # REAL1 / REAL2 (b7's cohort)
RBASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # #36's staged inputs: real e1 / e2 (control)
DST = "2026-10-14_q2_f12q_inputs_v1"
RUNS36 = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
B7_SHA = "8bdaa55d0a395e5a36415bb001f1eff13c9eaabb9ff9f5c71e56bdd26e9e5042"
# side -> ((full + F9 file, sha256, arm, runs), (F11 overlay file, sha256, arm, runs)); stage records on aswin/q2-f10-tabs
SIM = {"attack": (("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_2.jsonl", "009466e7b8eb0f3f265a840157c800826ff459cb387f9824f05fabdd6f09fe55", "sim-attack-fx-fullF9", 939),
                  ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_2.jsonl", "7a97837283ab161820ff78fbb87d645fc8753ab19a13aa694f77b0a78a7bdd6b", "sim-attack-fx-fullF9F11", 427)),
       "honest": (("2026-10-13_q2_fxavg_inputs_v1/inputs/shard_4.jsonl", "eaa15a3c136de18916ae66b4bfd28a9a9779f7d14e94dc9e83cb82e20c7bdd0c", "sim-honest-fx-fullF9", 437),
                  ("2026-10-13_q2_f11q_inputs_v1/inputs/shard_3.jsonl", "78445677dc32700b49ccd9f9d30e73fe096c895584ce5bce109db0b663df3e91", "sim-honest-fx-fullF9F11", 226))}
D9 = {"repro-sim-attack": ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "sim-attack"),
      "repro-sim-honest": ("2026-10-13_q3_mean-pool36-honestsim_qwen3-32b_l7_v1", "sim-honest")}
PRICE = {"qwen3 $/1M forward tokens (fix-full v2, +15% for 7 layers and the means)": 0.78 * 1.15, "qwen3 $/model load": 0.26}
PER_SHARD, MAX_TOKENS, SEED = 1.0e6, 32768, 20261014
CONTROL_MAX = 0.05

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "joblib==1.6.0", "boto3==1.43.113",
                      "pyyaml==6.0.3", "transformers==4.56.2", "jinja2==3.1.6")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "AUTHOR_SPLIT_STAGE": "/tmp/author_split_stage"})
         .add_local_python_source("probe_shift")
         .add_local_file(REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json")
         .add_local_file(HERE / "analyze_POSTHOC_author_split.py", f"/root/{FOLDER}/analyze_POSTHOC_author_split.py")
         .add_local_file(HERE / "f12_lib.py", f"/root/{FOLDER}/f12_lib.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", "/root/swap/rerender.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", "/root/fsim/fixes.py"))


class Gate(Exception):
    """A failed check; its message holds counts only."""


def say(*a):
    print(datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"), *a, flush=True)


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def run_job(git: str, real_keys: dict, qwen_rev: str) -> dict:
    sys.path[:0] = ["/root/swap", "/root/fsim", f"/root/{FOLDER}"]
    import f12_lib as L
    import fixes
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    if hashlib.sha256(Path(f"/root/{FOLDER}/analyze_POSTHOC_author_split.py").read_bytes()).hexdigest() != B7_SHA:
        raise Gate("b7's analysis module differs from its pin")
    import analyze_POSTHOC_author_split as A

    root, c = Path("/acts"), Counter()

    def pinned(spec):
        p, h, arm, n = spec
        raw = (root / p).read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise Gate(f"{p}: sha256 differs from its stage record")
        rows = {r["key"]: r for r in map(json.loads, raw.decode().splitlines())}
        if len(rows) != n or {r["arm"] for r in rows.values()} != {arm}:
            raise Gate(f"{p}: not {n} runs of {arm}")
        return rows

    def staged_rows(run, arms, keys=None):
        out = {}
        for p in sorted((root / run / "inputs").glob("shard_*.jsonl")):
            for line in open(p):
                it = json.loads(line)
                if it["arm"] in arms and (keys is None or it["key"] in keys):
                    if (it["arm"], it["key"]) in out:
                        raise Gate("a run twice in a staged run")
                    out[(it["arm"], it["key"])] = it
        return out

    sim = {}
    for side, (full, ov) in SIM.items():
        rows, over = pinned(full), pinned(ov)
        if not set(over) <= set(rows):
            raise Gate(f"{side}: F11 keys outside full + F9")
        sim[side] = rows | over  # an F11 row replaces its full + F9 row (its arm label says which)
        c[f"sim-{side}|runs (full + F9 + F11)"] = len(sim[side])
        c[f"sim-{side}|of which F11 rows"] = len(over)

    s_, first, second, _ = A.cohort_rows()
    v4 = staged_rows(V4, {"real-honest"}, {r["key"] for r in [*first.values(), *second.values()]})
    real = {"REAL1": [v4[("real-honest", r["key"])]["messages"] for r in first.values()],
            "REAL2": [v4[("real-honest", r["key"])]["messages"] for r in second.values()]}
    rb = staged_rows(RBASE, {"real-attack-e1", "real-attack-e2"})
    for arm, keys in real_keys.items():
        got = [rb[(arm, k)]["messages"] for k in keys if (arm, k) in rb]
        if len(got) != len(keys):
            raise Gate(f"{arm}: {len(got)} of {len(keys)} runs found")
        real[arm] = got
    say("rows loaded", {k: len(v) for k, v in real.items()}, {k: len(v) for k, v in sim.items()})

    # 1. derive (REAL2), confirm (REAL1)
    cells, cell_table = L.derive(real["REAL2"], real["REAL1"], fixes, c)
    for side in sim:
        for it in sim[side].values():
            L.tally(it["messages"], fixes, c, f"sim-{side} before")
    if not cells:
        return {"counts": dict(sorted(c.items())), "cells": cell_table, "gates pass": False, "stop": "no cell is IN F12"}

    # 2-3. apply, gates
    fixture_ok = L.fixtures(cells, c)
    staged, record, tok_in = {}, [], []
    for side, rows in sim.items():
        g = f"sim-{side}"
        for k, it in sorted(rows.items()):
            m = it["messages"]
            new, changed = L.f12(m, cells, fixes)
            ok_fams = {i for i, fam in L.outputs(m, fixes) if cells.get(fam)}
            if len(new) != len(m) or any(i not in ok_fams or set(m[i]) != set(new[i]) or any(m[i][f] != new[i][f] for f in new[i] if f != "content")
                                         or new[i]["content"].replace("\r", "") != m[i]["content"].replace("\r", "")
                                         or len(new[i]["content"]) <= len(m[i]["content"]) for i in changed):
                raise Gate("F12 changed a message outside its cells, a field other than content, or more than \\r")
            c[f"{g}|outputs changed"] += len(changed)
            c[f"{g}|\\r inserted"] += sum(len(new[i]["content"]) - len(m[i]["content"]) for i in changed)
            if changed:
                c[f"{g}|runs changed"] += 1
                c[f"{g}|runs changed that are F11 rows"] += it["arm"].endswith("F11")
                again, ch2 = L.f12(new, cells, fixes)
                c[f"{g}|runs where F12 is idempotent"] += (msha(again) == msha(new) and not ch2)
                dst_arm = f"sim-{side}-fx-fullF9F11F12"
                row = it | {"arm": dst_arm, "messages": new}
                c[f"{g}|staged rows holding attachment://"] += "attachment://" in json.dumps(row)
                staged.setdefault(dst_arm, []).append(row)
                record.append({"arm": dst_arm, "key": k, "base arm": it["arm"], "changed_messages": changed,
                               "messages_sha256": msha(new), "base_messages_sha256": msha(m)})
                tok_in.append((g, k, m, new))
            L.tally(new, fixes, c, f"{g} after")
    real_ctl = {}
    for arm, ms in real.items():
        lines = chg = 0
        for m in ms:
            for i, fam in L.outputs(m, fixes):
                if cells.get(fam):
                    lines += sum(t in cells[fam] for _, t in L.typed_lines(m[i]["content"]))
                    chg += len(L.f12_text(m[i]["content"], cells[fam])) - len(m[i]["content"])
        real_ctl[arm] = {"IN-cell lines": lines, "lines F12 would change": chg, "share": chg / lines if lines else None,
                         "pass (<= 5%)": bool(lines) and chg <= CONTROL_MAX * lines}
    ok = (fixture_ok and all(c[f"sim-{s}|runs changed"] > 0 for s in sim)
          and all(c[f"sim-{s}|runs where F12 is idempotent"] == c[f"sim-{s}|runs changed"] for s in sim)
          and all(c[f"sim-{s}|staged rows holding attachment://"] == 0 for s in sim)
          and all(v["pass (<= 5%)"] for v in real_ctl.values()))

    # tokens (Qwen3-32B, exact)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", revision=qwen_rev)
    tokens = Counter()
    for g, k, old, new in tok_in:
        q_old, q_new = (len(segment_end_positions(tok, x, MAX_TOKENS)["ids"]) for x in (old, new))
        tokens[f"{g}|qwen3 tokens before F12"] += q_old
        tokens[f"{g}|qwen3 tokens after F12"] += q_new
    q_tok = sum(v for kk, v in tokens.items() if kk.endswith("after F12"))
    S = min(64, max(8, 2 * math.ceil(q_tok / PER_SHARD / 2)))
    qr = q_tok / 1e6 * PRICE["qwen3 $/1M forward tokens (fix-full v2, +15% for 7 layers and the means)"] + S * PRICE["qwen3 $/model load"]
    out = {"git": git, "counts": dict(sorted(c.items())), "cells": cell_table, "IN F12": {f: sorted(t) for f, t in cells.items()},
           "tokens": dict(sorted(tokens.items())), "fixtures pass": fixture_ok, "real-arm control (REAL1/REAL2 pass by construction; e1/e2 are the check)": real_ctl, "gates pass": ok,
           "shards": S, "price": {"basis": PRICE, "qwen3 forward tokens (F12 rows)": q_tok, "$ raw": round(qr, 2), "$ with 30%": round(qr * 1.3, 2)},
           "runs by arm": {a: len(rs) for a, rs in sorted(staged.items())}}
    if not ok:
        return out | {"stop": "a gate failed: nothing is staged"}
    if (root / DST).exists():
        raise FileExistsError(f"{DST} exists: a staged run is never overwritten")

    # 4. stage, with one reproduction run per extractor shard (F11's dealing: sorted by -tokens, arm, key; round-robin)
    n_rep = {"repro-sim-attack": (S + 1) // 2, "repro-sim-honest": S // 2}
    rng, repro, chosen, n = random.Random(SEED), {}, {}, 0
    for new_arm, (dump, arm) in D9.items():
        cfg = json.loads((root / dump / "config.json").read_text())
        have = {p.stem for p in (root / dump / "meta" / arm).glob("*.json")}
        srcr = {filename(k): it for (a, k), it in staged_rows(cfg["inputs_from"], {arm}).items()}
        pool = sorted(s for s in have if s in srcr)
        pick = rng.sample(pool, n_rep[new_arm])
        chosen[new_arm] = {"dump": dump, "dump arm": arm, "inputs_from": cfg["inputs_from"], "pool": len(pool), "keys": sorted(srcr[s]["key"] for s in pick)}
        for s in pick:
            n += 1
            repro.setdefault(new_arm, []).append(srcr[s] | {"arm": new_arm, "tokens": -n, "source_tokens": srcr[s]["tokens"]})
    allrows = [r for rs in staged.values() for r in rs] + [r for rs in repro.values() for r in rs]
    if any(r["tokens"] < 0 for rs in staged.values() for r in rs):
        raise Gate("a staged row has a negative token count")
    order = sorted(allrows, key=lambda it: (-it["tokens"], it["arm"], it["key"]))
    shard_of = {(it["arm"], it["key"]): i % S for i, it in enumerate(order)}
    if sorted(shard_of[(r["arm"], r["key"])] for rs in repro.values() for r in rs) != list(range(S)):
        raise Gate("the reproduction runs do not cover every shard exactly once")
    (root / DST / "inputs").mkdir(parents=True)
    shas = {}
    for i, (arm, rs) in enumerate(sorted((staged | repro).items())):
        body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
        (root / DST / "inputs" / f"shard_{i}.jsonl").write_bytes(body)
        shas[f"inputs/shard_{i}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
    rec = json.dumps({"shards": shas, "runs": record, "reproduction draw": chosen, "seed": SEED, "git": git,
                      "planned shard": {f"{a}|{k}": s for (a, k), s in sorted(shard_of.items())}}).encode()
    (root / DST / "stage_record.json").write_bytes(rec)
    acts.commit()
    return out | {"staged": DST, "shard files": shas, "record_sha256": hashlib.sha256(rec).hexdigest(), "reproduction draw": chosen}


@app.function(image=image, volumes={"/acts": acts, "/hf": hf_cache}, cpu=16, memory=65536, timeout=3600, retries=0)
def job(git: str, real_keys: dict, qwen_rev: str) -> dict:
    try:
        return run_job(git, real_keys, qwen_rev)
    except Exception as e:  # noqa: BLE001 — type, location and our own count-only messages; never staged text
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = str(e) if isinstance(e, (Gate, FileExistsError)) else ""
        raise RuntimeError(f"stage failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}: {msg[:200]}") from None


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "RULE.md", HERE / "analyze_POSTHOC_author_split.py", HERE / "f12_lib.py"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    from probe_shift.models import QWEN3_32B
    runs = json.loads(RUNS36.read_text())
    real_keys = {a: sorted(r["key"] for r in runs if r["arm"] == a) for a in ("real-attack-e1", "real-attack-e2")}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = job.remote(git, real_keys, QWEN3_32B["revision"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"stage_f12_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "status": "preliminary", "rule": "RULE.md Addition 2", "stamp": stamp,
                             "inputs sha256": {"#36 runs.json": hashlib.sha256(RUNS36.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print("wrote", p, "gates pass:", out["gates pass"], "price:", out.get("price"), "runs:", out.get("runs by arm"))
