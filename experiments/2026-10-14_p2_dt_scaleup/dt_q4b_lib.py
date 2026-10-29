"""Pure helpers for run_dt_q4b.py (RULE.md): the q4b plan with the shared scorer's prep() checks, the exact-q2 labelling of one
sample, the replay match (positive control and bridge), the scale-up fold rule, and the npz writer. Imported inside the containers."""

import hashlib
import json
from pathlib import Path


def plan_epochs(epochs: list, root: str = "/ev", declared_unstaged: frozenset = frozenset()) -> list:
    """Per epoch: (world, task, key, root, chunk, chunk rows, [(abs row, irid, call_index, turn, impactful)] of its command rows),
    with prep's checks (meta key, run and inputs sha256; the chunk index entry). Epochs with no staged epoch must be exactly the
    staging records' declared exclusions (Amendment 6); they are dropped, anything else stops."""
    plan, chunks, staged = [], {}, {}
    if len({e["join"] for e in epochs}) != len(epochs):
        raise ValueError("a duplicate (world, task)")
    for f in sorted({e["staged_inputs_file"] for e in epochs}):  # prep(): staged-file hash, then per-epoch call identity
        raw = (Path(root) / f).read_bytes()
        if {hashlib.sha256(raw).hexdigest()} != {e["staged_inputs_sha256"] for e in epochs if e["staged_inputs_file"] == f}:
            raise ValueError(f"{f}: sha256 differs from the manifest's")
        for line in raw.decode().splitlines():
            ep = json.loads(line)
            staged[(f, ep["key"])] = [(r["irid"], r["call_index"], r["kind"], r["readout"] is not None,
                                       r["kind"] == "output_end" and r["readout"] is not None and bool(r.get("span"))) for r in ep["calls"]]
        del raw
    gone = {e["key"] for e in epochs if (e["staged_inputs_file"], e["key"]) not in staged}
    if gone != set(declared_unstaged):
        raise ValueError(f"{len(gone)} manifest epochs have no staged epoch, the staging records declare {len(declared_unstaged)}: they must match exactly")
    for e in [e for e in epochs if e["key"] not in gone]:
        m = json.loads((Path(root) / e["meta_file"]).read_text())
        run, st = e["meta_file"].split("/meta/")[0], staged.get((e["staged_inputs_file"], e["key"]))
        if run != e["extraction_run"] or m["key"] != e["key"] or m.get("inputs_sha256") != e["staged_inputs_sha256"] or st is None:
            raise ValueError(f"{e['join']}: meta key, run or inputs sha256 differs from the manifest's, or no staged epoch")
        if len(st) != len(m["calls"]) or any(s_ != (c["irid"], c["call_index"], c["kind"], c.get("row") is not None, c.get("row_mean") is not None)
                                             for s_, c in zip(st, m["calls"])):
            raise ValueError(f"{e['join']}: meta calls differ from the staged calls")
        lt, mn = [c["row"] for c in m["calls"] if c.get("row") is not None], [c["row_mean"] for c in m["calls"] if c.get("row_mean") is not None]
        n, r0 = m.get("n_rows"), m.get("row_start")
        if not (isinstance(n, int) and isinstance(r0, int) and n >= 0 and r0 >= 0 and lt + mn == list(range(n))):
            raise ValueError(f"{e['join']}: rows are not numbered as the extractor numbers them, or row_start invalid")
        ck = (run, m["chunk"])
        if ck not in chunks:
            chunks[ck] = json.loads((Path(root) / run / "chunks" / f"{m['chunk']}.json").read_text())
        ent = [x for x in chunks[ck]["epochs"] if x["key"] == e["key"]]
        if len(ent) != 1 or (ent[0]["arm"], ent[0]["row_start"], ent[0]["n_rows"]) != (m["arm"], r0, n) or r0 + n > chunks[ck]["rows"]:
            raise ValueError(f"{e['join']}: chunk index entry differs from the meta")
        cmd = [(m["row_start"] + c["row"], c["irid"], c["call_index"], c["turn"], int(c["label"] == "impactful"))
               for c in m["calls"] if c["kind"] == "command" and c.get("row") is not None]
        if len({(x[1], x[2]) for x in cmd}) != len(cmd):
            raise ValueError(f"{e['join']}: a duplicate (irid, call_index) among command rows")
        plan.append({"world": e["world"], "task": e["task"], "key": e["key"], "root": run, "chunk": m["chunk"], "eval_sha256": e.get("eval_sha256"),
                     "chunk_rows": chunks[ck]["rows"], "rows": cmd, "stratum": e.get("stratum")})
    return plan


def sample_labels(S, D, T, s: dict) -> dict:
    """index_sample's enumeration (stage_q4b @ 26d036293) with the frozen q2 rules on each kept call's raw arguments:
    {"irid|call_index": [impactful, destructive_raw, transfer]}, booleans only."""
    res = {}
    att = s.get("attachments") or {}
    for e in s.get("events") or []:
        if e.get("event") != "model" or e.get("model") != S.ACTING:
            continue
        irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
        if irid is None:
            raise ValueError("an acting-model event without x-irid")
        offered = {t.get("name") for t in (e.get("tools") or [])}
        for ci, tc in enumerate((((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []):
            name, args = tc.get("function"), S.resolved(tc.get("arguments") or {}, att)
            if S.transcripts.refs_left(args):
                raise ValueError("an attachment reference left")
            if name == "submit" or (bool(offered) and name not in offered) or (name == "text_editor" and args.get("command") is None):
                continue
            dr = any(D.DESTRUCTIVE.search(v if isinstance(v, str) else json.dumps(v)) for v in args.values())  # build_recal_labels.py l.36
            tr = T.executed(name or "") and T.is_transfer(args)
            res[f"{irid}|{ci}"] = [int(S.label_call(name, args)["label"] == "impactful"), int(dr), int(tr)]
    return res


def scaleup_folds(tasks: dict, cells: list) -> dict:
    """A cell's fold by its main, with frozen_folds' rule (stage_q4b l.205); a main outside T196 gets -1 (Append 21b)."""
    h = lambda s: hashlib.sha256((tasks["salt"] + s).encode()).hexdigest()  # noqa: E731
    envs, by_env = sorted(tasks["eligible_mains_by_env"], key=h), {}
    for c in tasks["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: ((by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 if c["main"] in by_env.get(c["env"], []) else -1) for c in cells}


def match_replay(plan, rows, sc, imp, cls, fold_of, designs, replay, attack, bar) -> tuple:
    """The positive control (dt_class and impactful on matched attack rows) and the bridge (match rate, Spearman with average
    ranks, median |dz| on the replay honest B u C impactful same-fold SD), on (world, task, irid, rank of the call within its turn):
    the replay files number a turn's calls 1, 3, 5, ... and q4b 0, 1, 2, ... (T196 attempt 1), so each layout's own order within the irid
    is the key; the share of matched rows with replay call = 2 * q4b call_index + 1 is reported as a consistency count."""
    import numpy as np

    def ranked(keys):  # keys: (world, task, irid, call order) per row -> (world, task, irid, rank within the irid)
        grp = {}
        for n, k in enumerate(keys):
            grp.setdefault(k[:3], []).append((k[3], n))
        out = [None] * len(keys)
        for g, v in grp.items():
            for r, (_, n) in enumerate(sorted(v)):
                out[n] = g + (r,)
        return out

    qk = ranked([(plan[i]["world"], plan[i]["task"], plan[i]["rows"][j][1], plan[i]["rows"][j][2]) for i, j in rows])
    pos, bridge, idx = {}, {}, {k: n for n, k in enumerate(qk)}
    qci = {n: plan[i]["rows"][j][2] for n, (i, j) in enumerate(rows)}
    for des in designs:
        f = Path("/p2/2026-10-14_p2_callfacet/2026-10-14_141350") / f"DT_{des}__rule1_command.npz"
        if hashlib.sha256(f.read_bytes()).hexdigest() != replay[f"DT_{des}"]:
            raise ValueError(f"{f.name}: sha256 differs from the replay record")
        with np.load(f) as zf:  # every array read once (Codex round 5: NpzFile item access reloads the array)
            z = {k: zf[k] for k in zf.files}
        keep = np.isin(np.array([k.split("|")[0] for k in z["key"]]), ["run1", "run2", "sim", "honest_a", "honest_b", "honest_c"])
        kr = np.flatnonzero(keep)
        zk = ranked([(str(z["key"][r]).split("|")[0], str(z["key"][r]).split("|")[1], str(z["irid"][r]), int(z["call_index"][r])) for r in kr])
        m = [(r, idx.get(k)) for r, k in zip(kr, zk)]
        hit = np.array([r for r, n in m if n is not None], np.int64), np.array([n for _, n in m if n is not None], np.int64)
        if len(hit[0]) < 1000:
            raise ValueError(f"{des}: fewer than 1,000 rows match the replay layout")
        if (z["fold"][hit[0]] != np.array([fold_of[plan[rows[n][0]]["task"]] for n in hit[1]])).any():
            raise ValueError(f"{des}: a matched row's fold differs between the layouts")
        a = np.isin(np.array([str(k).split("|")[0] for k in z["key"][hit[0]]]), attack)
        pos[des] = {"matched_attack_rows": int(a.sum()), "dt_class_mismatches": int((z["dt_class"][hit[0]][a] != cls[des][hit[1]][a]).sum()),
                    "impactful_mismatches": int((z["impactful"][hit[0]] != imp[hit[1]]).sum())}
        hon = np.isin(np.array([str(k).split("|")[0] for k in z["key"]]), ["honest_b", "honest_c"]) & (z["impactful"] == 1)
        sdk = np.array([z["score"][hon & (z["fold"] == k)].std() for k in range(5)])
        if not (np.isfinite(sdk).all() and (sdk > 0).all() and all((hon & (z["fold"] == k)).sum() >= 100 for k in range(5))):
            raise ValueError(f"{des}: an honest B u C calibration SD is missing, non-finite or zero")
        dz = (sc[des][hit[1]] - z["score"][hit[0]]) / sdk[z["fold"][hit[0]]]
        def rk(v):  # average ranks (ties share their mean rank): Spearman = Pearson on these
            u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
            first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            return (first + (cnt - 1) / 2.0)[inv]
        rho = float(np.corrcoef(rk(sc[des][hit[1]]), rk(z["score"][hit[0]]))[0, 1])
        bridge[des] = {"call_number_consistent": int(sum(int(z["call_index"][r]) == 2 * qci[n] + 1 for r, n in zip(hit[0], hit[1]))),
                       "replay_rows": int(keep.sum()), "matched": len(hit[0]), "match_rate": len(hit[0]) / int(keep.sum()), "spearman": rho,
                       "median_abs_dz": float(np.median(np.abs(dz))), "p95_abs_dz": float(np.percentile(np.abs(dz), 95)), "sd_honest_BuC_by_fold": sdk.tolist()}
        bridge[des]["pass"] = bool(bridge[des]["match_rate"] >= bar["match_min"] and rho >= bar["spearman_min"] and bridge[des]["median_abs_dz"] <= bar["median_abs_dz_max"])
    return pos, bridge


def write_npz(out: Path, plan: list, rows: list, sc, byf, lay, imp, cls, fold_of: dict, designs: tuple, nest: bool = True, label: str = "", leaf: str = "dt") -> dict:
    """One npz per design: per world under <world>/dt/ (T196), or every world of the arm in one dt/ file (a d9 message; keys carry
    the world). Exclusive creation: an existing file (a failed earlier attempt) stops the job, so nothing is overwritten."""
    import io

    import numpy as np

    files = {}
    for grp in ([[w] for w in sorted({p["world"] for p in plan})] if nest else [sorted({p["world"] for p in plan})]):
        sel = [n for n, (i, j) in enumerate(rows) if plan[i]["world"] in grp]
        sel.sort(key=lambda n: (plan[rows[n][0]]["world"], plan[rows[n][0]]["task"], plan[rows[n][0]]["rows"][rows[n][1]][3], plan[rows[n][0]]["rows"][rows[n][1]][2]))
        unit, seen = [], {}
        for n in sel:
            t = (plan[rows[n][0]]["world"], plan[rows[n][0]]["task"])
            unit.append(seen.get(t, 0))
            seen[t] = unit[-1] + 1
        g = lambda f: np.array([f(plan[rows[n][0]], plan[rows[n][0]]["rows"][rows[n][1]]) for n in sel])  # noqa: E731
        for des in designs:
            p = (out / grp[0] if nest else out) / leaf / f"DT_{des}.npz"  # T196: <world>/dt/; a scale-up arm: dt/ (09); sim-honest: leaf "" (Amendment 31)
            p.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            np.savez(buf, key=g(lambda e, r: f"{e['world']}|{e['task']}"), unit=np.array(unit, np.int64), call_index=g(lambda e, r: r[2]).astype(np.int64),
                     irid=g(lambda e, r: r[1]), score=sc[des][sel], fold=g(lambda e, r: fold_of[e["task"]]).astype(np.int64),
                     layer=lay[des][sel], impactful=imp[sel], dt_class=cls[des][sel], score_by_fold=byf[des][sel],
                     **({"readout_label": np.array(label)} if label else {}))
            with p.open("xb") as fh:
                fh.write(buf.getvalue())
            files[f"{'+'.join(grp)}/DT_{des}"] = {"path": str(p), "sha256": hashlib.sha256(buf.getvalue()).hexdigest(), "rows": len(sel)}
    return files


def convert_su311(runs_rows: list, root: str) -> tuple:
    """d9's scale-up manifests (.files rows: launch, rel, path, sha256, arm_dir, task, world, stratum, launch_task) and each
    manifest's extraction run -> the corrected-B epoch schema plan_epochs reads. The staged key is "<launch>/<rel>#e0", the staged
    file <run's config inputs>/inputs/<arm_dir>.jsonl with the config's sha256, the meta file <run>/meta/<arm_dir>/<sha1(key)[:20]>.json
    (extract_q4b's naming). The run must be complete with the config's input hashes."""
    epochs, skipped, skipped_keys = [], {}, set()
    for run, rows in runs_rows:
        cfg = json.loads((Path(root) / run / "config.json").read_text())
        done = json.loads((Path(root) / run / "complete.json").read_text())
        if cfg.get("run") != run or done.get("inputs_sha256") != cfg["inputs_sha256"]:
            raise ValueError(f"{run}: config or complete.json differs (run name or input hashes)")
        for f in rows:
            if f"{f['arm_dir']}.jsonl" not in cfg["inputs_sha256"]:  # e.g. run2 manifests list honest donors; run2 stages attack only (Codex r7)
                skipped[f"{run}|{f['arm_dir']}"] = skipped.get(f"{run}|{f['arm_dir']}", 0) + 1
                skipped_keys.add(f"{f['launch']}/{f['rel']}#e0")
                continue
            if f["path"] != f"{f['launch']}/{f['rel']}":
                raise ValueError(f"{run}: a manifest row's path is not launch/rel")
            key, fn = f"{f['launch']}/{f['rel']}#e0", f"{f['arm_dir']}.jsonl"
            epochs.append({"key": key, "world": f["world"], "task": f["task"], "join": f"{f['world']}|{f['task']}", "extraction_run": run, "staged_set": cfg["inputs"],
                           "staged_inputs_file": f"{cfg['inputs']}/inputs/{fn}", "staged_inputs_sha256": cfg["inputs_sha256"][fn],
                           "meta_file": f"{run}/meta/{f['arm_dir']}/{hashlib.sha1(key.encode()).hexdigest()[:20]}.json",
                           "stratum": f.get("stratum"), "launch_task": f.get("launch_task"), "eval_sha256": f["sha256"], "state": f.get("state"), "stage": f.get("stage")})
    return epochs, skipped, skipped_keys
