"""F11 (PLAN_F11.md), POST HOC: simulated text_editor insert replies re-rendered as the real `cat -n` snippet, on full + F9.
Modal CPU. Counts, hashes, token counts and the price only; never message text (attack runs included). Staged inputs go to
the Volume, never overwritten.

  --mode survey: gates, fixtures, real-arm control, counts, token counts, shard plan and price; nothing written.
  --mode stage:  the same, then two staged runs:
                 - 2026-10-13_q2_f11_inputs_v1 (Seed): arms sim-attack-fx-fullF9F11 and sim-honest-fx-fullF9F11, changed
                   runs only, one shard file per arm;
                 - 2026-10-13_q2_f11q_inputs_v1 (Qwen3-32B): the same rows plus one as-is reproduction run per extractor
                   shard, copied byte-identically from d9's two mean-pool dumps' inputs (`tokens` set to -1..-N so they
                   sort last and are dealt one per shard; the source value kept as `source_tokens`).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_q2_fixfull/stage_f11.py --mode survey
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
SWAP, FSIM = REPO / "experiments" / "2026-10-07_q2_rendering_swap", REPO / "experiments" / "2026-10-09_fixed_sim"
APP = "probe-shift_aswin-surya_sub-q2-f11_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
G3OUT = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"  # Seed metas (forward tokens), read-only
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
F9SRC, DST_SEED, DST_Q = "2026-10-13_q2_fixfull_inputs_f9_v1", "2026-10-13_q2_f11_inputs_v1", "2026-10-13_q2_f11q_inputs_v1"
BASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # #36's staged inputs: the real arms (invariance control)
F9_STAGE = HERE / "results" / "stage_f9_2026-10-13_145822.json"  # its record_sha256 pins F9SRC's stage record
SEED_MAP = HERE / "results" / "f9_mapping_seed_2026-10-13_154954.json"
RUNS36 = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
SRC_ARMS = {"sim-attack-fx-fullF9": ("attack", 939), "sim-honest-fx-fullF9": ("honest", 437)}
REAL = {"real-attack-e1": 939, "real-attack-e2": 939, "real-honest": 990}
SEED_RUNS = {"full": "2026-10-13_fixfull_seed-oss-36b_v1", "fullF9": "2026-10-13_fixfull_f9_seed-oss-36b_v1"}
D9 = {"repro-sim-attack": ("2026-10-12_la_mean-pool36_qwen3-32b_l7_v1", "sim-attack"),
      "repro-sim-honest": ("2026-10-13_q3_mean-pool36-honestsim_qwen3-32b_l7_v1", "sim-honest")}
PRICE = {"qwen3 $/1M forward tokens (fix-full v2, +15% for 7 layers and the means)": 0.78 * 1.15, "qwen3 $/model load": 0.26,
         "seed $/1M forward tokens (F9 measured, loads included)": 1.59}
PER_SHARD = {"qwen3": 1.0e6, "seed": 0.9e6}  # tokens per shard so no container runs past about 15 min
MAX_TOKENS, SEED = 32768, 20261013

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
g3out = modal.Volume.from_name(G3OUT, version=2).read_only()
hf_cache = modal.Volume.from_name(HF_CACHE).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "jinja2==3.1.6", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(SWAP / "rerender.py", "/root/swap/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")  # fixes.py imports it
         .add_local_file(FSIM / "fixes.py", "/root/fsim/fixes.py"))


class Gate(Exception):
    """A failed check; its message holds counts only."""


def msha(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()


def filename(key: str) -> str:  # the extractors' file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def inserts_of(msgs, fixes):
    return {x["msg"]: x for x in fixes.calls_of(msgs) if x["tool"] == "text_editor" and x["args"].get("command") == "insert"}


def f11(msgs, rr, fixes, tally=None, group=""):
    """PLAN_F11.md: every insert reply F1 rendered (diff marker, ends in "\\n" + the review note) loses the note and is
    re-rendered by rerender's real snippet; every other message is returned unchanged."""
    ins = inserts_of(msgs, fixes)
    if any(m["role"] == "tool" and not isinstance(m.get("content"), str) for m in msgs):  # as fixed-sim's R1 (fixes.r1_insert)
        if tally is not None:
            tally[f"{group}|runs skipped (a non-text tool output; rerender reads outputs as text)"] += 1
        return [dict(m) for m in msgs], ins
    stripped, noted = [dict(m) for m in msgs], set()
    for i in ins:
        o = msgs[i]["content"]
        if isinstance(o, str) and rr.DIFF in o[:300] and o.endswith("\n" + rr.REVIEW):
            stripped[i] = msgs[i] | {"content": o[: -len("\n" + rr.REVIEW)]}
            noted.add(i)
    new_s, recs = rr.rerender_messages(stripped, "full")
    out = [dict(m) for m in msgs]
    for r in recs:
        cat = r["category"]
        if tally is not None:
            tally[f"{group}|insert replies|category {cat}|{'with F1 note' if r['msg'] in noted else 'without F1 note'}"] += 1
        if cat in rr.RENDERED and r["msg"] in noted:
            out[r["msg"]] = msgs[r["msg"]] | {"content": new_s[r["msg"]]["content"]}
    return out, ins


def fixtures(rr, c: Counter) -> bool:
    """Synthetic files (seed 20261013): the real insert rendering, from text_editor.py's insert semantics (file and new_str
    expanded; the 4 lines before insert_line, the inserted lines, the 4 after; numbered from max(1, insert_line - 3); the
    review note; no trailing newline), must equal F11 applied to the simulator-style diff (3 context lines, tabs kept,
    trailing newline) plus F1's note, after a whole-file simulator view of the file. No transcript text."""
    import difflib

    rng = random.Random(SEED)
    words = ["func", "x", "if", "return", "err", "nil", "{", "}", "ab", "abcdefghi", "//", "1", "a\tb"]
    for n in range(300):
        lines = ["".join(rng.choice(["\t", "", " ", "  "]) for _ in range(rng.randint(0, 2))) + " ".join(rng.choice(words) for _ in range(rng.randint(1, 4)))
                 + f" L{j}" for j in range(rng.randint(3, 14))]
        il = rng.randint(0, len(lines))
        new = [rng.choice(["\t", "", "    "]) + f"INS{n}_{j} " + rng.choice(words) for j in range(rng.randint(1, 3))]
        after = lines[:il] + new + lines[il:]
        ex = lambda s: s.expandtabs()  # noqa: E731
        snip = [ex(x) for x in lines[max(0, il - 4):il]] + [ex(x) for x in new] + [ex(x) for x in lines[il:il + 4]]
        start = max(1, il - 3)
        real = ("The file /app/x.go has been edited. Here's the result of running `cat -n` on a snippet of the edited file:\n"
                + "\n".join(f"{i + start:6}\t{x}" for i, x in enumerate(snip)) + "\n" + rr.REVIEW)
        d = list(difflib.unified_diff(lines, after, fromfile="old_version", tofile="edited_version", lineterm="", n=3))
        sim = "The file /app/x.go has been edited. Here's a diff showing the changes made:\n" + "\n".join(d) + "\n"
        view = "Here's the result of running `cat -n` on /app/x.go:\n" + "\n".join(f"{i + 1:6}\t{x}" for i, x in enumerate(lines)) + "\n"
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "text_editor", "arguments": {"command": "view", "path": "/app/x.go"}}}]},
                {"role": "tool", "content": view},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "function": {"name": "text_editor", "arguments": {"command": "insert", "path": "/app/x.go", "insert_line": il, "new_str": "\n".join(new)}}}]},
                {"role": "tool", "content": sim.rstrip("\n") + "\n" + rr.REVIEW}]
        got = rr.rerender_messages([m | ({"content": msgs[3]["content"][: -len("\n" + rr.REVIEW)]} if k == 3 else {}) for k, m in enumerate(msgs)], "full")
        c["fixture|cases"] += 1
        c["fixture|F11 equals the real rendering"] += got[0][3]["content"] == real
        c[f"fixture|category {got[1][0]['category'] if got[1] else 'none'}"] += 1
    return c["fixture|cases"] == c["fixture|F11 equals the real rendering"]


def run_job(mode: str, pins: dict, real_keys: dict, seed_overrides: dict, qwen_rev: str) -> dict:
    sys.path[:0] = ["/root/swap", "/root/fsim"]
    import fixes
    import rerender as rr
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    root = Path("/acts")

    def rows(run, arms):
        out = {}
        for p in sorted((root / run / "inputs").glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                it = json.loads(line)
                if it["arm"] in arms:
                    if (it["arm"], it["key"]) in out:
                        raise Gate("a run twice in a staged run")
                    out[(it["arm"], it["key"])] = it
        return out

    rec_path = root / F9SRC / "stage_record.json"
    if hashlib.sha256(rec_path.read_bytes()).hexdigest() != pins["f9 record"]:
        raise Gate("F9's stage record is not the pinned one")
    f9_sha = {(r["arm"], r["key"]): r["messages_sha256"] for r in json.loads(rec_path.read_text())["runs"]}
    src = rows(F9SRC, set(SRC_ARMS))
    c = Counter()
    for arm, (_, n) in SRC_ARMS.items():
        if (got := sum(1 for (a, _) in src if a == arm)) != n:
            raise Gate(f"{arm}: {got} runs, not {n}")
    c["gate 1|base runs equal to F9's stage record (message sha256)"] = sum(msha(it["messages"]) == f9_sha.get(ak) for ak, it in src.items())

    staged, record, tok_in = {}, [], []
    for (arm, k), it in sorted(src.items()):
        group = f"sim-{SRC_ARMS[arm][0]}"
        m = it["messages"]
        new, ins = f11(m, rr, fixes, c, group)
        changed = [i for i, (a, b) in enumerate(zip(m, new)) if a != b]
        if len(new) != len(m) or any(i not in ins or set(m[i]) != set(new[i]) or any(m[i][f] != new[i][f] for f in new[i] if f != "content") for i in changed):
            raise Gate("F11 changed a non-insert message, the message count, or a field other than content")
        diff_left = sum(isinstance(new[i]["content"], str) and rr.DIFF in new[i]["content"][:300] for i in ins)
        c[f"{group}|insert replies"] += len(ins)
        c[f"{group}|insert replies in diff format before F11"] += sum(isinstance(m[i]["content"], str) and rr.DIFF in m[i]["content"][:300] for i in ins)
        c[f"{group}|insert replies in diff format after F11"] += diff_left
        c[f"{group}|runs changed"] += bool(changed)
        c[f"{group}|messages changed"] += len(changed)
        if changed:
            again, _ = f11(new, rr, fixes)
            c[f"{group}|runs where F11 is idempotent"] += msha(again) == msha(new)
            dst_arm = arm.replace("fullF9", "fullF9F11")
            row = it | {"arm": dst_arm, "messages": new}
            c[f"{group}|staged rows holding attachment://"] += "attachment://" in json.dumps(row)
            staged.setdefault(dst_arm, []).append(row)
            record.append({"arm": dst_arm, "key": k, "changed_messages": changed, "messages_sha256": msha(new),
                           "base_messages_sha256": msha(m), "first_changed_message": changed[0]})
            tok_in.append((group, k, m, new))

    # real-arm invariance control (counts only)
    real = rows(BASE, set(REAL))
    for arm, n in REAL.items():
        keys = set(real_keys[arm])
        got = {k: it for (a, k), it in real.items() if a == arm and k in keys}
        if len(got) != n:
            raise Gate(f"{arm}: {len(got)} of {n} runs found")
        for k, it in got.items():
            new, ins = f11(it["messages"], rr, fixes)
            c[f"{arm}|insert replies"] += len(ins)
            c[f"{arm}|insert replies changed"] += sum(new[i] != it["messages"][i] for i in ins)
            c[f"{arm}|insert replies in diff format"] += sum(isinstance(it["messages"][i]["content"], str) and rr.DIFF in it["messages"][i]["content"][:300] for i in ins)

    # tokens: Qwen3-32B exact (the extractor's segment_end_positions, cut 32,768); Seed scaled from full + F9's metas
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", revision=qwen_rev)
    tokens = Counter()
    for group, k, old, new in tok_in:
        q_old, q_new = (len(segment_end_positions(tok, x, MAX_TOKENS)["ids"]) for x in (old, new))
        side = group.removeprefix("sim-")
        run_, arm_ = (SEED_RUNS["fullF9"], f"sim-{side}-fx-fullF9") if k in seed_overrides[side] else (SEED_RUNS["full"], f"sim-{side}-fx-full")
        s_old = json.loads((Path("/g3") / run_ / "meta" / arm_ / f"{filename(k)}.json").read_text())["forward_tokens"]
        tokens[f"{group}|qwen3 tokens before F11"] += q_old
        tokens[f"{group}|qwen3 tokens after F11"] += q_new
        tokens[f"{group}|seed forward tokens before F11 (full + F9 meta)"] += s_old
        tokens[f"{group}|seed forward tokens after F11 (scaled)"] += round(s_old * q_new / q_old)

    fixture_ok = fixtures(rr, c)
    g2 = all(c[f"{g}|insert replies in diff format after F11"] == sum(v for kk, v in c.items() if kk.startswith(f"{g}|insert replies|category failed|"))
             for g in ("sim-attack", "sim-honest"))
    g2_alarm = {g: c[f"{g}|insert replies in diff format after F11"] / max(1, c[f"{g}|insert replies in diff format before F11"]) for g in ("sim-attack", "sim-honest")}
    ok = (fixture_ok and g2 and c["gate 1|base runs equal to F9's stage record (message sha256)"] == len(src)
          and all(c[f"{g}|runs changed"] > 0 for g in ("sim-attack", "sim-honest"))
          and all(c[f"{g}|runs where F11 is idempotent"] == c[f"{g}|runs changed"] for g in ("sim-attack", "sim-honest"))
          and all(c[f"{g}|staged rows holding attachment://"] == 0 for g in ("sim-attack", "sim-honest")))
    real_ok = {a: c[f"{a}|insert replies"] > 0 and c[f"{a}|insert replies changed"] < 0.01 * c[f"{a}|insert replies"] for a in REAL}
    q_tok = sum(v for kk, v in tokens.items() if kk.endswith("qwen3 tokens after F11"))
    s_tok = sum(v for kk, v in tokens.items() if kk.endswith("seed forward tokens after F11 (scaled)"))
    shards = {"qwen3": min(64, max(8, 2 * math.ceil(q_tok / PER_SHARD["qwen3"] / 2))), "seed": min(64, max(8, math.ceil(s_tok / PER_SHARD["seed"])))}
    out = {"counts": dict(sorted(c.items())), "tokens": dict(sorted(tokens.items())), "gates pass": ok, "fixtures pass": fixture_ok,
           "gate 2: diff-format inserts left equal rerender's failed count": g2, "gate 2 alarm: share of diff-format inserts left": g2_alarm,
           "positive control: real arm changes under 1% of its insert replies": real_ok, "shards": shards,
           "runs by arm": {a: sorted(r["key"] for r in rs) for a, rs in sorted(staged.items())}}
    if mode != "stage":
        return out
    if not ok or not all(real_ok.values()) or any(v > 0.10 for v in g2_alarm.values()):
        raise Gate("a gate, the 10% alarm or the positive control failed: nothing is staged")
    for d in (DST_SEED, DST_Q):
        if (root / d).exists():
            raise FileExistsError(f"{d} exists: a staged run is never overwritten")

    def write_run(dst, by_arm, extra):
        (root / dst / "inputs").mkdir(parents=True)
        shas = {}
        for n, (arm, rs) in enumerate(sorted(by_arm.items())):
            body = "".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["key"])).encode()
            (root / dst / "inputs" / f"shard_{n}.jsonl").write_bytes(body)
            shas[f"inputs/shard_{n}.jsonl"] = {"arm": arm, "runs": len(rs), "sha256": hashlib.sha256(body).hexdigest()}
        rec = json.dumps({"shards": shas, "base": F9SRC, "base record sha256": pins["f9 record"]} | extra).encode()
        (root / dst / "stage_record.json").write_bytes(rec)
        return shas, hashlib.sha256(rec).hexdigest()

    shas_seed, rec_seed = write_run(DST_SEED, staged, {"runs": record})
    # Qwen3-32B: + one reproduction run per extractor shard (the extractor deals rows sorted by -tokens, arm, key round-robin)
    S = shards["qwen3"]
    n_rep = {"repro-sim-attack": (S + 1) // 2, "repro-sim-honest": S // 2}
    rng, repro, chosen, n = random.Random(SEED), {}, {}, 0
    for new_arm, (dump, arm) in D9.items():
        cfg = json.loads((root / dump / "config.json").read_text())
        have = {p.stem for p in (root / dump / "meta" / arm).glob("*.json")}
        srcr = {filename(k): it for (a, k), it in rows(cfg["inputs_from"], {arm}).items()}
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
    per_shard = [0] * S
    for it in order:
        per_shard[shard_of[(it["arm"], it["key"])]] += max(it["tokens"], it.get("source_tokens", 0))
    shas_q, rec_q = write_run(DST_Q, staged | repro, {"reproduction draw": chosen, "seed": SEED,
                                                      "planned shard": {f"{a}|{k}": s for (a, k), s in sorted(shard_of.items())}})
    acts.commit()
    return out | {"staged": DST_SEED, "shards_files": shas_seed, "record_sha256": rec_seed, "qwen staged": DST_Q, "qwen shard files": shas_q,
                  "qwen record_sha256": rec_q, "reproduction draw": chosen, "planned source tokens per qwen shard (max, median)": [max(per_shard), sorted(per_shard)[S // 2]]}


def price(tok: dict, shards: dict) -> dict:
    q = sum(v for k, v in tok.items() if k.endswith("qwen3 tokens after F11"))
    s = sum(v for k, v in tok.items() if k.endswith("seed forward tokens after F11 (scaled)"))
    qr = q / 1e6 * PRICE["qwen3 $/1M forward tokens (fix-full v2, +15% for 7 layers and the means)"] + shards["qwen3"] * PRICE["qwen3 $/model load"]
    sr = s / 1e6 * PRICE["seed $/1M forward tokens (F9 measured, loads included)"]
    return {"basis": PRICE, "shards": shards, "qwen3 forward tokens (F11 rows; reproduction runs add about 1 run per shard)": q, "seed forward tokens": s,
            "qwen3 $ raw": round(qr, 2), "qwen3 $ with 30%": round(qr * 1.3, 2), "seed $ raw": round(sr, 2), "seed $ with 30%": round(sr * 1.3, 2),
            "total $ raw": round(qr + sr, 2), "total $ with 30%": round((qr + sr) * 1.3, 2)}


@app.function(image=image, volumes={"/acts": acts, "/g3": g3out, "/hf": hf_cache}, cpu=16, memory=65536, timeout=3600, retries=0)
def job(mode: str, pins: dict, real_keys: dict, seed_overrides: dict, qwen_rev: str) -> dict:
    try:
        return run_job(mode, pins, real_keys, seed_overrides, qwen_rev)
    except Exception as e:  # noqa: BLE001 — type, location and our own count-only messages; never staged text
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = str(e) if isinstance(e, (Gate, FileExistsError)) else ""
        raise RuntimeError(f"stage failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}: {msg[:200]}") from None


@app.local_entrypoint()
def main(mode: str = "survey"):
    if mode not in ("survey", "stage"):
        raise SystemExit("--mode survey|stage")
    for f in (Path(__file__), HERE / "PLAN_F11.md"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    from probe_shift.models import QWEN3_32B
    pins = {"f9 record": json.loads(F9_STAGE.read_text())["record_sha256"]}
    runs = json.loads(RUNS36.read_text())
    real_keys = {a: sorted(r["key"] for r in runs if r["arm"] == a) for a in REAL}
    smap = json.loads(SEED_MAP.read_text())["arms"]
    seed_overrides = {"attack": smap["full + F9, attack"]["override keys"], "honest": smap["full + F9, honest"]["override keys"]}
    out = job.remote(mode, pins, real_keys, seed_overrides, QWEN3_32B["revision"])
    out["price"] = price(out["tokens"], out["shards"])
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"stage_f11_{mode}_{stamp}.json"
    with open(p, "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "status": "preliminary", "plan": "PLAN_F11.md", "git": head, "mode": mode,
                             "inputs sha256": {"#36 runs.json": hashlib.sha256(RUNS36.read_bytes()).hexdigest(),
                                               F9_STAGE.name: hashlib.sha256(F9_STAGE.read_bytes()).hexdigest(),
                                               SEED_MAP.name: hashlib.sha256(SEED_MAP.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print(p)
    print("gates pass:", out["gates pass"], "| fixtures:", out["fixtures pass"], "| gate 2:", out["gate 2: diff-format inserts left equal rerender's failed count"],
          out["gate 2 alarm: share of diff-format inserts left"], "| real control:", out["positive control: real arm changes under 1% of its insert replies"])
    print({a: len(v) for a, v in out["runs by arm"].items()}, out["price"])
