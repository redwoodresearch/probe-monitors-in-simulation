"""RULE.md (this folder): the uncut (whole-run) inputs of the sim-vs-real detector. The run sets are the 32k detector's, by key; only
where their activations come from changes. Every record is read with git show <commit>:<path> and its sha256 checked.

Local side (build): the T196 rows of inventory_b_v2 for the named worlds give each run's key, task, arm and its uncut root (staged
set, extraction root, staged sha256). Each root's chain extraction record must be the run's own (config.run, config.inputs, cut
"full", every arm's staged sha256 equal to the inventory's, a clean git), with shards unique and contiguous from 0, every shard on an
H200, done_before 0 and every chunk listed; its chunk list is what the loader may read. Routing: d9's standard <batch>__<target>
_cutfull[_r<k>] records (0d option (a), 10-16) must name the same volume, staged set, extraction root, cut and per-arm staged sha;
until they are pinned, only a --dry-run passes, checked against the S_old attestations (failing {}).
Remote side (uncut_skip): base.skip_rows (the fixed-call-token rule, unchanged) on the staged inputs, plus each epoch's meta: key,
cut "full", truncated False and n_tokens equal to the inventory's full_len.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import run_simh_control as base

VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"


def git_json(repo: Path, spec: dict):
    b = subprocess.run(["git", "show", f"{spec['commit']}:{spec['path']}"], capture_output=True, check=True, cwd=repo).stdout
    if hashlib.sha256(b).hexdigest() != spec["sha256"]:
        raise SystemExit(f"{spec['path']} @ {spec['commit'][:9]}: sha256 differs from the config's")
    return json.loads(b)


def build(repo: Path, cfg: dict, worlds: dict, dry_run: bool) -> dict:
    """worlds: inventory world -> batch label. Returns the epochs (sorted later by the caller), want/shas for the skip list, the
    allowed chunks per extraction root, and the provenance record. Fail-closed throughout."""
    inv = git_json(repo, cfg["inventory"])
    rows = [r for r in inv["runs"] if r["task_set"] == cfg["task_set"] and r["world"] in worlds]
    if inv.get("status") != cfg["inventory_status"]:
        raise SystemExit(f"inventory status {inv.get('status')!r} is not {cfg['inventory_status']!r}")
    for r in rows:
        u = r.get("uncut_planned") or {}
        if u.get("volume") != VOL or not r.get("original_folder_registered") or r["full_len"] != r["tokens_before"]:
            raise SystemExit(f"{r['key']}: inventory row is not on {VOL}, not registered, or full_len != tokens_before")
    if len({r["key"] for r in rows}) != len(rows):
        raise SystemExit("a key twice in the inventory rows")
    roots = sorted({r["uncut_planned"]["extraction_root"] for r in rows})
    if not set(roots) <= set(cfg["extraction_records"]):
        raise SystemExit(f"extraction roots {sorted(set(roots) - set(cfg['extraction_records']))} have no pinned extraction record")
    # per (staged set, arm): one staged sha and one root
    arm_sha, root_of = {}, {}
    for r in rows:
        u = r["uncut_planned"]
        k = (u["staged_set"], r["arm"])
        if arm_sha.setdefault(k, u["staged_sha256"]) != u["staged_sha256"] or root_of.setdefault(u["staged_set"], u["extraction_root"]) != u["extraction_root"]:
            raise SystemExit(f"{k}: two staged shas or two roots in the inventory")
        if u["staged_inputs_file"] != f"{r['arm']}.jsonl":
            raise SystemExit(f"{r['key']}: staged inputs file {u['staged_inputs_file']!r} is not the arm's")
    allowed, prov = {}, {}
    for root in roots:  # the roots these worlds use
        spec = cfg["extraction_records"][root]
        ex = git_json(repo, spec)
        c, s = ex["config"], ex["shards"]
        sets = [st for st, rt in root_of.items() if rt == root]
        if c["run"] != root or len(sets) != 1 or c["inputs"] != sets[0] or c.get("cut") != "full" or ex["git"].get("dirty") is not False:
            raise SystemExit(f"{root}: extraction record is not this root's uncut extraction of {sets} (or its git is dirty)")
        for (st, arm), sha in arm_sha.items():
            if st == sets[0] and c["inputs_sha256"].get(f"{arm}.jsonl") != sha:
                raise SystemExit(f"{root}: {arm}.jsonl staged sha256 differs between the record and the inventory")
        if sorted(x["shard"] for x in s) != list(range(len(s))) or any("H200" not in x["gpu_name"] for x in s) or any(x["done_before"] for x in s):
            raise SystemExit(f"{root}: shards not contiguous, not all on H200, or resumed")
        ch = [cc["chunk"] for x in s for cc in x["chunks"]]
        if len(set(ch)) != len(ch) or sum(cc["epochs"] for x in s for cc in x["chunks"]) != sum(x["epochs"] for x in s):
            raise SystemExit(f"{root}: the record does not list every chunk once")
        allowed[root] = sorted(ch)
        prov[root] = {"record": spec, "git": ex["git"], "staged_set": sets[0], "shards": len(s), "gpus": sorted({x["gpu_name"] for x in s}),
                      "chunks": len(ch), "epochs": sum(x["epochs"] for x in s)}
    n_arm = {}
    for r in rows:
        k = (r["uncut_planned"]["staged_set"], r["arm"])
        n_arm[k] = n_arm.get(k, 0) + 1
    routing = check_routing(repo, cfg["routing"], arm_sha, root_of, dry_run, n_arm)
    epochs = []
    for r in rows:
        u = r["uncut_planned"]
        epochs.append({"key": r["key"], "world": r["world"], "task": r["task"], "arm": r["arm"], "extraction_run": u["extraction_root"],
                       "staged_inputs_run": u["staged_set"], "batch": worlds[r["world"]], "vol": "u", "full_len": r["full_len"]})
    want, shas = {}, {}
    for e in epochs:
        want.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    for (st, arm), sha in arm_sha.items():
        shas[f"{st}/inputs/{arm}.jsonl"] = sha
    return {"epochs": epochs, "want": want, "shas": shas, "allowed": allowed,
            "provenance": {"inventory": cfg["inventory"], "roots": prov, "routing": routing, "volume": VOL}}


def check_routing(repo: Path, rc: dict, arm_sha: dict, root_of: dict, dry_run: bool, n_arm: dict) -> dict:
    """Each (staged set, arm) must be routed by exactly one pinned record. mode "routing" = d9's standard cutfull routing records
    (volume, staged_set, extraction_root, cut, arms[].meta_arm / staged_inputs_file / staged_sha); mode "attestation" = the S_old
    attestations (cutfull.volume / staged_set / extraction_root, failing {}) and only for a --dry-run."""
    if rc["mode"] not in ("routing", "attestation"):
        raise SystemExit(f"routing mode {rc['mode']!r}")
    if rc["mode"] == "attestation" and not dry_run:
        raise SystemExit("S_old routing records are not pinned yet (0d option (a)): only --dry-run may run on the attestations")
    seen, out = {}, []
    for spec in rc["records"]:
        d = git_json(repo, spec)
        st_ = d["staged_set"] if rc["mode"] == "routing" else d["cutfull"]["staged_set"]
        if st_ not in root_of:  # a record for a root these worlds do not use (e.g. run 2's for the honest problems)
            continue
        if rc["mode"] == "routing":
            st, root, vol = d["staged_set"], d["extraction_root"], d["volume"]
            if d.get("cut") != "full":
                raise SystemExit(f"{spec['path']}: cut is not full")
            arms = {a["meta_arm"]: a for a in d["arms"]}
            for a_name, a in arms.items():
                if (st, a_name) in arm_sha and (a["staged_inputs_file"] != f"{st}/inputs/{a_name}.jsonl" or a["staged_sha"] != arm_sha[(st, a_name)]
                                                or a["epochs"] != n_arm[(st, a_name)]):
                    raise SystemExit(f"{spec['path']}: {a_name} staged file, sha or epoch count differs from the inventory's (the whole arm is used)")
        else:
            cf = d["cutfull"]
            st, root, vol = cf["staged_set"], cf["extraction_root"], cf["volume"]
            if d.get("failing") != {}:
                raise SystemExit(f"{spec['path']}: the attestation has failing checks")
            arms = {a: None for (s_, a) in arm_sha if s_ == st}
        if vol != VOL or root_of.get(st) != root:
            raise SystemExit(f"{spec['path']}: volume or extraction root differs from the inventory's")
        for a_name in arms:
            if (st, a_name) in arm_sha:
                if (st, a_name) in seen:
                    raise SystemExit(f"{st}/{a_name}: routed twice")
                seen[(st, a_name)] = spec["path"]
        out.append({"record": spec, "staged_set": st, "extraction_root": root})
    missing = sorted(f"{s_}/{a}" for (s_, a) in arm_sha if (s_, a) not in seen)
    if missing:
        raise SystemExit(f"not routed by any pinned record: {missing}")
    return {"mode": rc["mode"], "records": out, "routed": {f"{s_}/{a}": p for (s_, a), p in sorted(seen.items())}}


def check_cells(cells: dict, epochs: list, expected: dict) -> dict:
    """(Codex r1) uncut gates after the fits: every cell's rows per run equal the staged expectation (a run with 0 expected rows is the
    only allowed omission), and every observed fit and null draw is certified. Returns {"failures": [...], "zero_row_runs": {...}}."""
    name_key = {e["name"]: e["key"] for e in epochs}
    fails, zero = [], {}
    for cell, c in cells.items():
        ro = c["readout"]
        want_rows = {n: expected[k][ro] for n, k in name_key.items()}
        got = c["rows_per_run"]
        zero[cell] = sorted(n for n, v in want_rows.items() if v == 0)
        bad = sorted(n for n in want_rows if got.get(n, 0) != want_rows[n])
        if bad or set(c["runs_without_rows"]) != set(zero[cell]):
            fails.append(f"{cell}: rows differ from the staged expectation for {len(bad)} runs (e.g. {bad[:3]}), or runs without rows "
                         f"{sorted(c['runs_without_rows'])[:3]} differ from the zero-expected {zero[cell][:3]}")
        for p, r in c["problems"].items():
            if not r["observed_certified"]:
                fails.append(f"{cell}|{p}: an observed fit is uncertified")
            nul = r.get("null") or {}
            if nul.get("uncertified_draws"):
                fails.append(f"{cell}|{p}: {len(nul['uncertified_draws'])} null draws uncertified")
    return {"failures": fails, "zero_row_runs": zero}


def uncut_skip(root: str, want: dict, shas: dict, full_len: dict, roots: dict) -> dict:
    """Remote: base.skip_rows (unchanged rule) on the uncut staged inputs, then every requested epoch's meta under its extraction
    root: key equal, cut "full", truncated False, n_tokens equal to the inventory's full_len. roots = {staged set: extraction root}."""
    out = base.skip_rows(root, want, shas)
    expected = {}  # (Codex r1) each epoch's expected rows from its staged calls: command = call_end with a readout minus the fixed-token
    # rows (the skip list); reply = output_end with a readout and a span (the extractor's row_mean rule)
    for st, arms in want.items():
        for arm, keys in arms.items():
            ks = set(keys)
            for line in (Path(root) / st / "inputs" / f"{arm}.jsonl").read_text().splitlines():
                ep = json.loads(line)
                if ep["key"] in ks:
                    cs = [c for c in ep["calls"] if c["readout"] is not None]
                    expected[ep["key"]] = {"command": sum(c["kind"] == "call_end" for c in cs) - len(out["skip"][ep["key"]]),
                                           "reply": sum(c["kind"] == "output_end" and bool(c.get("span")) for c in cs)}
    if set(expected) != {k for arms in want.values() for ks in arms.values() for k in ks}:
        raise ValueError("expected rows not computed for every requested epoch")
    n, tok = 0, 0
    for st, arms in want.items():
        for arm, keys in arms.items():
            for k in keys:
                m = json.loads((Path(root) / roots[st] / "meta" / arm / f"{hashlib.sha1(k.encode()).hexdigest()[:20]}.json").read_text())
                if m["key"] != k or m.get("cut") != "full" or m.get("truncated") is not False or m["n_tokens"] != full_len[k]:
                    raise ValueError(f"{st}/{arm}: meta of an epoch is not its uncut extraction (key, cut, truncated or n_tokens)")
                n, tok = n + 1, tok + m["n_tokens"]
    return out | {"meta_checked": n, "meta_tokens": tok, "expected_rows": expected}


# (Codex r2) run_attack_heldout.py's Watchdog @ 648f33e61, copied unchanged, for the honest runner's independent bounded cancellation
class Watchdog:
    """The independent deadline (Amendment 2): a daemon thread that, every 5 s, cancels with terminate_containers=True any registered
    call past its own deadline (counted from when its spawn returned) and not yet marked done. Every cancel runs in its own daemon
    thread, so a stalled or failing cancel cannot delay the others; the log records each cancel as attempted, then confirmed or
    failed. The main thread's blocking SDK calls do not delay it."""

    def __init__(self):
        import threading

        self.calls, self.done, self.cancelled, self.log, self.threads = {}, set(), set(), [], {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def add(self, name: str, call, seconds: float):
        self.calls[name] = (call, time.time() + seconds)

    def _cancel(self, name: str, why: str):
        import threading

        def go(c):
            try:
                c.cancel(terminate_containers=True)
                self.log.append(f"{name}: cancel confirmed")
            except Exception as e:  # noqa: BLE001 - logged in the record; the per-attempt timeout still bounds that container
                self.log.append(f"{name}: cancel failed: {type(e).__name__}: {str(e)[:200]}")

        with self._lock:  # claim, start and publish in one acquisition (the same lock guards cancel_outstanding's snapshot)
            if name in self.done or name in self.cancelled or name not in self.calls:
                return
            self.cancelled.add(name)
            self.log.append(f"{name}: cancel attempted ({why})")
            th = threading.Thread(target=go, args=(self.calls[name][0],), daemon=True)
            th.start()
            self.threads[name] = th

    def _run(self):
        while not self._stop.wait(5):
            for name, (_, dl) in list(self.calls.items()):
                if time.time() >= dl:
                    self._cancel(name, "deadline")

    def cancel_outstanding(self, why: str, join_s: float = 30.0):
        """Cancel every registered call not marked done (independently), then wait at most join_s in total for confirmations."""
        for name in list(self.calls):
            self._cancel(name, why)
        end = time.time() + join_s
        with self._lock:
            snap = list(self.threads.items())
        for name, t in snap:
            t.join(max(0.0, end - time.time()))
            if t.is_alive():
                self.log.append(f"{name}: cancel not confirmed within {join_s:.0f} s")

    def stop(self):
        self._stop.set()


def ckpt_meta(git: str, config_sha: str, n_null: int, epochs: list, skip: dict, allowed: dict) -> dict:
    """(4f, 10-16) what a resumed run must share with the checkpointed one: the commit, the config, n_null and the exact inputs."""
    h = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()  # noqa: E731
    return {"git": git, "config_sha256": config_sha, "n_null": n_null, "epochs_sha256": h(epochs), "skip_sha256": h(skip), "allowed_sha256": h(allowed)}


def ckpt_save(d: Path, cell: str, out: dict, meta: dict, wall_s: float) -> None:
    """One file per collected cell, exclusive create (results/ is append-only)."""
    import os

    d.mkdir(parents=True, exist_ok=True)
    final = d / f"{cell.replace('|', '__')}.json"
    if final.exists():
        raise FileExistsError(f"{final}: a checkpoint for this cell exists")
    tmp = d / f".{final.name}.{os.getpid()}.tmp"  # (Codex r5) written whole, then renamed: a hard exit never leaves a partial file
    tmp.open("x").write(json.dumps({"meta": meta, "cell": cell, "wall_seconds": round(wall_s), "out": out}))
    os.replace(tmp, final)


def ckpt_load(d: Path, meta: dict) -> dict:
    """{cell: (out, wall_seconds)} from a previous run's checkpoint folder; any file with other meta stops the resume."""
    got = {}
    for f in sorted(d.glob("[!.]*.json")):  # dot-files are unfinished temporaries
        x = json.loads(f.read_text())
        if x["meta"] != meta:
            raise SystemExit(f"{f.name}: checkpoint from another commit, config, n_null or input set; not resumable")
        got[x["cell"]] = (x["out"], x["wall_seconds"])
    if not got:
        raise SystemExit(f"{d}: no checkpointed cells")
    return got


def hard_exit_after(seconds: float) -> None:
    """(Codex r4) an exit that no blocking SDK call can delay: a daemon thread ends the client process after `seconds`; the ephemeral
    app then stops (Modal terminates its containers on disconnect). Checkpointed cells survive (--resume)."""
    import os
    import threading

    def go():
        time.sleep(seconds)
        print(f"HARD EXIT after {seconds:.0f} s: the client stops so the ephemeral app stops", flush=True)
        os._exit(3)

    threading.Thread(target=go, daemon=True).start()
