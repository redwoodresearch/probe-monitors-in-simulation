"""detAll (all-sets uncut detector): detU's loader (../2026-10-16_cmd_sim_detector_uncut/uncut_inputs.py @ e7ad92c2d). build and
check_routing are copied with ONE change: the whole-arm check counts each arm's refused keys. These are the S4 orphans, kept OUT by key
by inventory_b_v2/b_fix_v1's s4_orphans_out (09 READ_LOG 01:35), so routed epochs = inventory rows + refused keys. The refused keys
are never read: epochs and the remote skip/meta lists come from the inventory rows only. Everything else is imported unchanged.
"""

from pathlib import Path

from uncut_inputs import VOL, git_json


def refused_out_counts(inv: dict, routing_specs: list, repo: Path) -> dict:
    """{(staged set, arm): number of s4_orphans_out keys}, each orphan's target mapped to its pinned routing record's staged set."""
    st_of = {}
    for spec in routing_specs:
        d = git_json(repo, spec)
        st_of[d["target"]] = d["staged_set"]
    out = {}
    for o in inv.get("s4_orphans_out") or []:
        k = (st_of[o["target"]], o["arm"])  # KeyError = an orphan whose target has no pinned routing record: stop
        out[k] = out.get(k, 0) + 1
    return out


def build(repo: Path, cfg: dict, worlds: dict, dry_run: bool, refused_out: dict) -> dict:
    """worlds: inventory world -> batch label. Returns the epochs (sorted later by the caller), want/shas for the skip list, the
    allowed chunks per extraction root, and the provenance record. Fail-closed throughout."""
    inv = git_json(repo, cfg["inventory"])
    rows = [r for r in inv["runs"] if r.get("task_set", r.get("set")) == cfg["task_set"] and r["world"] in worlds]  # (detAll) S_paper493 inventories name it "set"
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
    routing = check_routing(repo, cfg["routing"], arm_sha, root_of, dry_run, n_arm, refused_out)
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


def check_routing(repo: Path, rc: dict, arm_sha: dict, root_of: dict, dry_run: bool, n_arm: dict, refused_out: dict) -> dict:
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
                                                or a["epochs"] != n_arm[(st, a_name)] + refused_out.get((st, a_name), 0)):
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
