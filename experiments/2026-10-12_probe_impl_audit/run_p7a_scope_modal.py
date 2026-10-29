"""P7a audit, check 2 (mask and endpoint), with a scope pin: run_p7a_spans_modal.py@268cea1d copied, plus --scope-pin.

--scope-pin (default on here) skips the re-derivation and pins the scope of the plant comparisons' identity: for plant1
and plant2, the packed outputs NOT in the per-output score file, split by (a) whether the run's environment is a claim B
test environment (split.json@bd44f35f "test" = run_claimB_b_grid.py@857cb3cc:43 TEST_ENVS; the environment is the
packed run's `environment` field, as the grid derives it at run_claimB_b_grid.py@857cb3cc:74 and :186) and (b) whether
the run is in the comparison's scored pairs (q1_eval.Scorer@857cb3cc:169-175: a run of the comparison's two arms whose
cell holds both arms); and the score file's rows against the packed outputs of the test-environment pair runs.
Predicate: "plant comparisons score exactly the packed outputs of their test-environment pair runs". Counts only.
  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run run_p7a_scope_modal.py --scores <score gz> --reader <reader copy>
Expected with --scope-pin: about 1-3 min of container time (metadata reads only), about $0.01-0.03 billed (4 CPUs,
~$0.32/h list x ~1.45); the 30-min timeout bounds it at about $0.25 plus any image build.
The rest of the original script runs unchanged (pack replay, stored spans, full identity for the real comparisons).

Original docstring:

Claim B amendment 7(a), Qwen/Qwen3-32B, level 1. One Modal CPU container (4 cores, 16 GiB, timeout 30 min, retries 0);
both Volumes mounted READ-ONLY; no activation array is opened. Counts and booleans come back, never text (exceptions are
counted by type name only). Per packed output, as run_extract_all_layers.py@041859e9:164,182 and
run_claimB_max_pack.py@8bad6873:71-87 define:
  - the amax range is the position's stored `span` [a, b): a < b, b - 1 == its readout token, b <= the transcript's kept
    tokens, and no two content spans (outputs and calls) of a transcript overlap;
  - with --recompute (default), the span equals the output's content-token span re-derived by
    reader.segment_end_positions@041859e9 (pinned copy, git blob checked) from the dump's own staged inputs with the
    pinned Qwen3-32B tokenizer (HF cache Volume, offline); text stays in the container. --no-recompute skips the
    tokenizer and never reads transcript text;
  - the pack's alignment, replayed from A's pack (packed_39f842231a46) and the dump's metas: A outputs, token mismatches
    (dropped), A outputs missing from B, B outputs not in A, against the 1% and 0.1% gates; packedB_3adf85556e82's
    runs equal the replay (runs, order, outputs, tokens, n_tokens) at all 8 layers, and its digest recomputes;
  - identity with the per-output score file (claimB_max_output_scores_2026-10-10_154917.json.gz, sha256 checked):
    per comparison, its (key, step, action) rows equal the packed outputs of that comparison's arms, with equal n_tokens.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run run_p7a_spans_modal.py [--no-recompute]
Expected: about 5-10 min of container time, about $0.05-0.10 billed (4 CPUs, ~$0.32/h list x ~1.45); the timeout
bounds it at about $0.25 plus the image build.
"""

import gzip
import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-implaudit-p7a-scope_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
RUN = "2026-10-10_la_fixed-sim-v2-max-scrub1_qwen3-32b_l8_v1"
A_PACK, B_PACK, LEVEL = "39f842231a46", "3adf85556e82", "level1"
LAYERS = [19, 32, 41, 44, 47, 52, 58, 63]
MISSING_MAX, TOKEN_MISMATCH_MAX = 0.01, 0.001
QWEN3_32B_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"  # probe_shift/models.py
READER_BLOB = "f78d80a85051948bc2f898bc7fbb14348f8bb6a5"  # git rev-parse 041859e9:src/probe_shift/reader.py
SCORES_SHA = "e5d22b386a0b42a2a74083e97d332a1dcb22700c99eee83ddc08e2d3ded9078f"  # results_manifest_v9.json@5916699d
COMPS = {"fixed_vs_e1": ("real-attack-e1", "sim-attack-fixed"), "e2_vs_e1": ("real-attack-e1", "real-attack-e2"),
         "plant1": ("sim-attack-fixed", "sim-attack-fixed-plant1"), "plant2": ("sim-attack-fixed", "sim-attack-fixed-plant2")}
TEST_ENVS = ("clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking")  # split.json@bd44f35f "test"
HERE = Path(__file__).parent

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
hf = modal.Volume.from_name(HF_CACHE)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "transformers==4.56.2", "jinja2")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "TZ": "UTC"}))

_tok = None


def _silence():
    import os

    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)


def _init_worker(revision: str):
    global _tok
    _silence()
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", revision=revision)


def _spans(job):
    """Worker: one transcript's re-derived output_end spans, numbers only; an exception comes back as its type name."""
    idx, messages, max_tokens = job
    import reader_041859e9 as R

    try:
        r = R.segment_end_positions(_tok, messages, max_tokens)
    except BaseException as e:  # counted by type name; the message may quote a token, so it is never returned
        return idx, None, type(e).__name__
    pos = {(p["step"], p["action"]): (p["span"], p["token"], p["kept"]) for p in r["positions"] if p["kind"] == "output_end"}
    return idx, {"pos": pos, "kept_tokens": len(r["ids"]), "n_tokens": r["n_tokens"]}, None


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/acts": acts.read_only(), "/hf": hf.read_only()})
def verify(reader_src: str, scores_gz: bytes, recompute: bool, scope_pin: bool) -> dict:
    import sys
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime, timezone

    _silence()
    n, err = Counter(), Counter()
    recompute = recompute and not scope_pin  # the scope pin skips the re-derivation
    out = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recompute": recompute, "scope_pin": scope_pin,
           "reader_blob_ok": hashlib.sha1(f"blob {len(reader_src.encode())}\0".encode() + reader_src.encode()).hexdigest() == READER_BLOB}
    try:
        root = Path("/acts")
        cfg = json.loads((root / RUN / "config.json").read_text())
        prov = cfg.get("provenance") or {}
        out["dump_config"] = {"model_is_qwen3_32b": cfg.get("model") == "Qwen/Qwen3-32B", "revision_pinned": cfg.get("revision") == QWEN3_32B_REVISION,
                              "max_tokens": cfg.get("max_tokens"), "max_final": cfg.get("max_final"), "ema_only": cfg.get("ema_only"),
                              "provenance_git_is_041859e9": str(prov.get("git", "")).startswith("041859e9"),
                              "source_matches_commit_is_041859e9": str(prov.get("source_matches_commit", "")).startswith("041859e9"),
                              "git_dirty": prov.get("git_dirty")}
        pk = root / "claimB" / LEVEL
        A = {l: json.loads((pk / f"packed_{A_PACK}" / f"runs.L{l}.json").read_text()) for l in LAYERS}
        B = {l: json.loads((pk / f"packedB_{B_PACK}" / f"runs.L{l}.json").read_text()) for l in LAYERS}
        out["A_layers_differing_from_L19"] = sum(A[l] != A[19] for l in LAYERS)
        out["B_layers_differing_from_L19"] = sum(B[l] != B[19] for l in LAYERS)
        a_runs = A[19]
        meta = {}
        for arm in sorted({m["arm"] for m in a_runs}):
            for f in sorted((root / RUN / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                meta[(arm, m["key"])] = m
        out["dump_metas_read"] = len(meta)

        # ---- replay of the pack (run_claimB_max_pack.py@8bad6873:46-87), then packedB against it, at every layer
        for l in LAYERS:
            c = Counter()
            for i, m in enumerate(A[l]):
                a_pos = {(p["step"], p["action"]): p for p in m["positions"] if p["kind"] == "output_end" and p["kept"]}
                bm = meta.get((m["arm"], m["key"]))
                if bm is None:
                    c["A runs missing from the dump"] += 1
                    continue
                pooled = {(p["step"], p["action"]): p for p in bm["positions"] if p["kind"] == "output_end_max" and p["kept"]}
                both = set(a_pos) & set(pooled)
                bad = {o for o in both if pooled[o]["token"] != a_pos[o]["token"]}
                common = sorted(both - bad)
                c["A outputs"] += len(a_pos)
                c["token mismatches (dropped)"] += len(bad)
                c["A outputs missing from B"] += len(set(a_pos) - set(pooled))
                c["B outputs not in A (dropped)"] += len(set(pooled) - set(a_pos))
                c["runs without a common output"] += not common
                bp = B[l][i]
                if (bp["arm"], bp["key"]) != (m["arm"], m["key"]):
                    c["packedB run order differs"] += 1
                    continue
                got = [(p["step"], p["action"]) for p in bp["positions"] if p["kind"] == "output_end"]
                c["packedB runs whose outputs differ from the replay"] += got != common
                for p in bp["positions"]:
                    o = (p["step"], p["action"])
                    if o not in pooled or o not in a_pos:
                        c["packedB outputs absent from the dump or A"] += 1
                        continue
                    a_, b_ = pooled[o]["span"]
                    c["packedB token != A token"] += p["token"] != a_pos[o]["token"]
                    c["packedB n_tokens != span length"] += p["n_tokens"] != b_ - a_
            c["dropped share of A outputs"] = (c["token mismatches (dropped)"] + c["A outputs missing from B"]) / max(c["A outputs"], 1)
            c["missing gate ok (<= 1%)"] = c["A outputs missing from B"] <= MISSING_MAX * c["A outputs"]
            c["token gate ok (<= 0.1%)"] = c["token mismatches (dropped)"] <= TOKEN_MISMATCH_MAX * c["A outputs"]
            out[f"pack_replay_L{l}"] = dict(c)
        runs_b = B[19]
        dig = hashlib.sha1(json.dumps({"a_pack": A_PACK, "b_run": RUN, "readouts": ["output_end_max", "output_end"],
                                       "runs": [(m["arm"], m["key"]) for m in runs_b]}, sort_keys=True).encode()).hexdigest()[:12]
        out["packedB_digest_recomputes"] = dig == B_PACK
        out["packedB_runs"] = len(runs_b)
        out["packedB_outputs"] = sum(len(r["positions"]) for r in runs_b)

        # ---- the amax span of every packed output (the dump's stored span, which lines 164 and 182 read)
        s = Counter()
        for i, r in enumerate(runs_b):
            bm = meta[(r["arm"], r["key"])]
            pooled = {(p["step"], p["action"]): p for p in bm["positions"] if p["kind"] == "output_end_max" and p["kept"]}
            spans = sorted(tuple(p["span"]) for p in bm["positions"] if p["kind"] in ("output_end_max", "call_end_max") and p["kept"])
            s["transcripts with overlapping content spans"] += any(spans[k][1] > spans[k + 1][0] for k in range(len(spans) - 1))
            for p in r["positions"]:
                s["packed output not kept"] += p.get("kept") is not True
                q = pooled[(p["step"], p["action"])]
                a_, b_ = q["span"]
                s["outputs"] += 1
                s["span empty or reversed"] += not a_ < b_
                s["span end != readout token + 1"] += b_ - 1 != q["token"]
                s["readout token outside span"] += not a_ <= q["token"] < b_
                s["span beyond kept tokens"] += b_ > bm["kept_tokens"]
        out["stored_spans"] = dict(s)

        # ---- re-derived content-token spans (text stays here; numbers only leave)
        if recompute:
            sys.path.insert(0, "/tmp/pinned")
            Path("/tmp/pinned").mkdir(exist_ok=True)
            Path("/tmp/pinned/reader_041859e9.py").write_text(reader_src)
            src = root / cfg["inputs_from"] / "inputs"
            need = {(r["arm"], r["key"]): i for i, r in enumerate(runs_b)}
            msgs = {}
            for p in sorted(src.glob("shard_*.jsonl")):
                for line in p.read_text().splitlines():
                    it = json.loads(line)
                    k = (it["arm"], it["key"])
                    if k in need:
                        n["staged inputs found twice"] += k in msgs
                        msgs[k] = it["messages"]
            n["packed runs without staged input"] = len(need) - len(msgs)
            jobs = [(need[k], v, cfg["max_tokens"]) for k, v in msgs.items()]
            rc = Counter()
            with ProcessPoolExecutor(4, initializer=_init_worker, initargs=(cfg["revision"],)) as ex:
                for i, res, e in ex.map(_spans, jobs, chunksize=8):
                    if e is not None:
                        err[e] += 1
                        continue
                    r = runs_b[i]
                    bm = meta[(r["arm"], r["key"])]
                    rc["kept_tokens equal"] += res["kept_tokens"] == bm["kept_tokens"]
                    rc["n_tokens equal"] += res["n_tokens"] == bm["n_tokens"]
                    pooled = {(p["step"], p["action"]): p for p in bm["positions"] if p["kind"] == "output_end_max" and p["kept"]}
                    for p in r["positions"]:
                        o = (p["step"], p["action"])
                        got = res["pos"].get(o)
                        if got is None:
                            rc["output missing in re-derivation"] += 1
                            continue
                        rc["outputs compared"] += 1
                        rc["span equal"] += list(got[0]) == list(pooled[o]["span"])
                        rc["span differs (amax outside the content span)"] += list(got[0]) != list(pooled[o]["span"])
                        rc["token differs"] += got[1] != pooled[o]["token"]
                        rc["not kept in re-derivation"] += not got[2]
            out["recomputed_spans"] = dict(rc) | {"transcripts": len(jobs)}

        # ---- identity with the per-output score file
        S = json.loads(gzip.decompress(scores_gz))
        ident = {}
        for comp, arms in COMPS.items():
            P = {(r["key"], p["step"], p["action"]): (r["arm"], p["n_tokens"]) for r in runs_b if r["arm"] in arms for p in r["positions"]}
            rows = S[comp]["rows"]
            Q = {(x["key"], x["step"], x["action"]): (x["arm"], x["n_tokens"]) for x in rows}
            ident[comp] = {"score_rows": len(rows), "duplicate_score_rows": len(rows) - len(Q), "packed_outputs": len(P),
                           "in_scores_not_packed": len(set(Q) - set(P)), "packed_not_in_scores": len(set(P) - set(Q)),
                           "arm_differs": sum(P[k][0] != Q[k][0] for k in set(P) & set(Q)),
                           "n_tokens_differs": sum(P[k][1] != Q[k][1] for k in set(P) & set(Q))}
        out["identity_with_score_file"] = ident

        # ---- scope pin: plant comparisons against the packed outputs of their test-environment pair runs
        if scope_pin:
            scope = {}
            for comp in ("plant1", "plant2"):
                neg, pos = COMPS[comp]
                mine = [r for r in runs_b if r["arm"] in (neg, pos)]
                sides = {}
                for r in mine:  # q1_eval.Scorer@857cb3cc:173-175: a cell counts only if it holds both sides
                    sides.setdefault(r["cell"], set()).add(r["arm"] == pos)
                Q = {(x["key"], x["step"], x["action"]) for x in S[comp]["rows"]}
                not_scored, runs_by, scoped = Counter(), Counter(), set()
                for r in mine:
                    cat = f"test_env={r['environment'] in TEST_ENVS}|scored_pair={len(sides[r['cell']]) == 2}"
                    runs_by[cat] += 1
                    for p in r["positions"]:
                        k = (r["key"], p["step"], p["action"])
                        not_scored[cat] += k not in Q
                        if r["environment"] in TEST_ENVS and len(sides[r["cell"]]) == 2:
                            scoped.add(k)
                scope[comp] = {"runs_by_env_and_pair": dict(runs_by), "packed_not_scored_by_env_and_pair": dict(not_scored),
                               "packed_not_scored_total": sum(not_scored.values()), "scored_rows": len(S[comp]["rows"]),
                               "scoped_packed_outputs": len(scoped), "scored_not_in_scoped": len(Q - scoped),
                               "scoped_not_scored": len(scoped - Q)}
            out["scope_pin"] = scope
    except BaseException as e:  # fail loudly by type name only: a message could quote stored text
        out["fatal_error_type"] = type(e).__name__
    out["counts"], out["errors_by_type"] = dict(n), dict(err)
    out["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


@app.local_entrypoint()
def main(recompute: bool = True, scope_pin: bool = True, scores: str = str(HERE / "data_p7a" / "claimB_max_output_scores_2026-10-10_154917.json.gz"),
         reader: str = str(HERE / "pinned" / "reader_041859e9.py")):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import subprocess

    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=HERE)  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip() or git("ls-files", "--error-unmatch", Path(__file__).name).returncode:
        raise SystemExit("commit first: the result must record the commit of the code that made it")
    gz, src = Path(scores).read_bytes(), Path(reader).read_text()
    if hashlib.sha256(gz).hexdigest() != SCORES_SHA:
        raise SystemExit("score file sha256 differs from results_manifest_v9.json")
    if hashlib.sha1(f"blob {len(src.encode())}\0".encode() + src.encode()).hexdigest() != READER_BLOB:
        raise SystemExit("reader copy differs from 041859e9:src/probe_shift/reader.py")
    res = verify.remote(src, gz, recompute, scope_pin)
    res |= {"git": git("rev-parse", "HEAD").stdout.strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scores_sha256": SCORES_SHA, "reader_blob": READER_BLOB}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"p7a_scope_{stamp}.json"
    dst.parent.mkdir(exist_ok=True)
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: res.get(k) for k in ("fatal_error_type", "errors_by_type", "stored_spans", "recomputed_spans", "packedB_digest_recomputes", "scope_pin")}, indent=1))
    print("->", dst)
    st, rc, ident = res.get("stored_spans", {}), res.get("recomputed_spans", {}), res.get("identity_with_score_file", {})
    replay = [res.get(f"pack_replay_L{l}", {}) for l in LAYERS]
    checks = {"no errors": not res.get("fatal_error_type") and not res.get("errors_by_type"),
              "outputs checked > 0": st.get("outputs", 0) > 0,
              "stored spans sound": all(st.get(k, 1) == 0 for k in ("span empty or reversed", "span end != readout token + 1",
                                                                  "readout token outside span", "span beyond kept tokens",
                                                                  "transcripts with overlapping content spans", "packed output not kept")),
              "pack gates and replay": all(r.get("missing gate ok (<= 1%)") and r.get("token gate ok (<= 0.1%)")
                                           and r.get("packedB runs whose outputs differ from the replay", 1) == 0 for r in replay)
                                       and res.get("packedB_digest_recomputes") is True,
              "identity with score file": bool(ident) and all(v["in_scores_not_packed"] == 0 and v["packed_not_in_scores"] == 0
                                                              and v["arm_differs"] == 0 and v["n_tokens_differs"] == 0 for v in ident.values())}
    if scope_pin:  # the plants are judged on their scored scope; the real comparisons keep the full identity
        sp = res.get("scope_pin", {})
        checks["identity with score file"] = bool(ident) and all(
            v["in_scores_not_packed"] == 0 and v["packed_not_in_scores"] == 0 and v["arm_differs"] == 0 and v["n_tokens_differs"] == 0
            for c, v in ident.items() if c not in ("plant1", "plant2")) and all(
            ident.get(c, {}).get("in_scores_not_packed", 1) == 0 and ident.get(c, {}).get("arm_differs", 1) == 0
            and ident.get(c, {}).get("n_tokens_differs", 1) == 0 for c in ("plant1", "plant2"))
        checks["plant comparisons score exactly the packed outputs of their test-environment pair runs"] = set(sp) == {"plant1", "plant2"} and all(
            v["scoped_packed_outputs"] > 0 and v["scored_not_in_scoped"] == 0 and v["scoped_not_scored"] == 0
            and v["scored_rows"] == v["scoped_packed_outputs"] for v in sp.values())
        checks["content-span containment"] = "not run (--scope-pin; established by p7a_spans_2026-10-12_143904)"
    elif recompute:  # containment of the amax span in the content span is established only by the re-derivation
        checks["re-derived spans: all packed outputs compared and equal"] = (
            rc.get("outputs compared", 0) == st.get("outputs", -1) and rc.get("span differs (amax outside the content span)", 1) == 0)
    else:
        checks["content-span containment"] = "BLOCKED (--no-recompute)"
    print(json.dumps(checks, indent=1))
    with (dst.with_suffix(".checks.json")).open("x") as fh:
        fh.write(json.dumps(checks, indent=1))
    if not all(v is True for v in checks.values() if not isinstance(v, str)):
        raise SystemExit("a check failed or compared nothing: see the result file")
