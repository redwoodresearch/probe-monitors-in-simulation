"""Remote input build of the all-tasks honest detector (RULE.md here). Runs in a Modal CPU container: every committed file arrives as bytes
from the local entrypoint (git show, sha256 against config/sources_v1.json); the laptop parses nothing. 4f's loader (uncut_inputs_all.build,
an unchanged copy) builds each source's epochs from configs made here; this module adds those configs, the S_new honest rows (4f's
make_inputs_all.phase_inventory rule, honest worlds), the pool join by key and the folds. Fail-closed throughout: keys, roots and shas only.

Classes (RULE.md "Pools"): sim = every sim-honest run of the pinned inventories; B = real honest B (epoch 1 + admitted epoch 2: S_old "B alone",
S_new H_B, S_newmains H_B; S_paper493 honest_b); A = real honest A (S_old C, S_new H_A (union of its families), S_newmains H_A, epoch 1 +
admitted epoch 2; S_paper493 honest_a). Epoch-1 lists: the E12 artifact (epoch2_poolkeys @ 5b19c08d8), each checked against the E12 block's
epoch1_members_sha256; epoch-2 lists: the E12 block, checked against INPUTS_FULL.freeze_checks.e12_pools (sorted-JSON sha256 of the keys).
"""

import hashlib
import json
from collections import Counter

import routec_keys as rk
import uncut_inputs as ui
import uncut_inputs_all as uia

# Route C rerun (2026-10-18): route C's frozen INPUTS_FULL_2026-10-17_233212.json (was d6923a75 = INPUTS_FULL_2026-10-16_110447.json)
INPUTS_FULL_SHA = "1e1868586fe8a786f6e1094b9d5b62e362ec0e2ce241e9b0103d6584af351126"
VOL = ui.VOL
HONEST = ("honest_a", "honest_b", "honest_c", "sim_honest")
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")
SYNTH = "0" * 40  # the commit field of derived (in-container) inventories


def hs(xs) -> str:
    """INPUTS_FULL's and E12's list hash: sha256 of json.dumps(sorted(keys)) with the default separators."""
    return hashlib.sha256(json.dumps(sorted(xs)).encode()).hexdigest()


class Blobs:
    """The shipped committed files by (full commit, path), each sha256-checked once; short commits resolve by unique prefix."""

    def __init__(self, files: list, data: dict):
        self.data, self.used, self.full = dict(data), set(), sorted({f["commit"] for f in files} | {SYNTH})
        for f in files:
            if hashlib.sha256(self.data[f"{f['commit']}:{f['path']}"]).hexdigest() != f["sha256"]:
                raise ValueError(f"{f['path']} @ {f['commit'][:12]}: shipped bytes differ from the sources pin")

    def resolve(self, commit: str) -> str:
        m = [c for c in self.full if c.startswith(commit)]
        if len(m) != 1:
            raise ValueError(f"commit {commit}: {len(m)} matches among the shipped sources")
        return m[0]

    def get(self, commit: str, path: str, sha256: str | None = None) -> bytes:
        k = f"{self.resolve(commit)}:{path}"
        if k not in self.data:
            raise KeyError(f"not shipped: {path} @ {commit}")
        if sha256 and hashlib.sha256(self.data[k]).hexdigest() != sha256:
            raise ValueError(f"{path} @ {commit[:12]}: sha256 differs from the citing record's")
        self.used.add(k)
        return self.data[k]

    def json(self, spec: dict):
        return json.loads(self.get(spec["commit"], spec["path"], spec.get("sha256")))

    def spec(self, commit: str, path: str) -> dict:
        full = self.resolve(commit)
        return {"commit": full, "path": path, "sha256": hashlib.sha256(self.get(full, path)).hexdigest()}

    def synth(self, path: str, obj) -> dict:
        b = json.dumps(obj, sort_keys=True).encode()
        self.data[f"{SYNTH}:{path}"] = b
        return {"commit": SYNTH, "path": path, "sha256": hashlib.sha256(b).hexdigest()}


def one_extraction(B: Blobs, rec: dict, ex: dict) -> None:
    if len(rec["extraction_record"]) != 1:
        raise ValueError(f"{rec['extraction_root']}: {len(rec['extraction_record'])} extraction records (the loader takes exactly one)")
    e = rec["extraction_record"][0]
    s = B.spec(e["commit"], e["path"])
    if ex.setdefault(rec["extraction_root"], s) != s:
        raise ValueError(f"{rec['extraction_root']}: two extraction records")


def cfg_cited(B: Blobs, inv_spec: dict, task_set: str, label: str) -> dict:
    """A (b)-contract inventory: the honest rows' own routing records (row.uncut_planned.routing = '<path> @ <commit>') and each record's one
    extraction record, as 4f's make_inputs_all.config_for pins them."""
    inv = B.json(inv_spec)
    rows = [r for r in inv["runs"] if r.get("task_set", r.get("set")) == task_set and r["world"] in HONEST]
    routing, ex = {}, {}
    for r in rows:
        p, c = r["uncut_planned"]["routing"].split(" @ ")
        if "/routing/" not in p:
            raise ValueError(f"{r['key']}: cites {p}, not a routing record")
        s = B.spec(c, p)
        routing[(s["commit"], s["path"])] = s
    for s in routing.values():
        one_extraction(B, B.json(s), ex)
    return {"label": label, "task_set": task_set, "inventory_status": inv["status"], "inventory": inv_spec, "extraction_records": ex,
            "routing": {"mode": "routing", "records": list(routing.values())}}


def select_routing(B: Blobs, ref: str, rdir: str, names: list) -> dict:
    """4f's make_inputs_all.select_routing over the shipped routing directory at ref: per (batch, target) the HIGHEST revision, used only
    if its own _attest.json has failing {} and the same revision; one record per staged set. -> {staged set: (spec, record)}."""
    top = {}
    for n in names:
        if not n.endswith(".json") or n.endswith("_attest.json") or n.startswith("sold__"):
            continue
        r = B.json({"commit": ref, "path": f"{rdir}/{n}"})
        k = (r["batch"], r["target"])
        if k in top and top[k][1]["revision"] == r["revision"]:
            raise ValueError(f"two records at revision {r['revision']} for {k}")
        if k not in top or r["revision"] > top[k][1]["revision"]:
            top[k] = (n, r)
    out = {}
    for n, r in top.values():
        att = n[:-5] + "_attest.json"
        if att not in names:
            continue
        a = B.json({"commit": ref, "path": f"{rdir}/{att}"})
        if a.get("failing") != {} or a.get("revision") != r["revision"]:
            continue
        if r["staged_set"] in out:
            raise ValueError(f"staged set {r['staged_set']} routed by two targets")
        out[r["staged_set"]] = (B.spec(ref, f"{rdir}/{n}"), r)
    return out


def arm_sha(rec: dict, staged_set: str, arm: str) -> tuple:
    a = [x for x in rec["arms"] if x["meta_arm"] == arm and x["staged_inputs_file"] == f"{staged_set}/inputs/{arm}.jsonl"]
    if len(a) != 1:
        raise ValueError(f"{staged_set}: routing record has {len(a)} arms named {arm}")
    return a[0]["staged_sha"], a[0]["epochs"]


def cfg_snew(B: Blobs, inv_spec: dict, ref: str, rdir: str, names: list, attack_cfg: dict) -> tuple:
    """S_new: inventory_v3 has no (b) contract; 4f's phase_inventory rule on the honest worlds (each row's extraction root and staged sha from
    its staged set's selected routing record). Cross-check: every S_new staged set 4f's attack config routes is selected identically here."""
    sel = select_routing(B, ref, rdir, names)
    for s in attack_cfg["routing"]["records"]:
        r = B.json(s)
        mine = sel.get(r["staged_set"])
        if mine and hashlib.sha256(B.get(s["commit"], s["path"])).hexdigest() != hashlib.sha256(B.get(mine[0]["commit"], mine[0]["path"])).hexdigest():
            raise ValueError(f"{r['staged_set']}: the selected routing record differs from 4f's attack config's")
    v3 = B.json(inv_spec)
    rows, per = [], Counter()
    for x in v3["runs"]:
        if x["world"] not in HONEST:
            continue
        u = x["uncut_planned"]
        if u["staged_set"] not in sel or u["volume"] != VOL:
            raise ValueError(f"{x['key']}: staged set {u['staged_set']} has no attested routing record, or another volume")
        s, r = sel[u["staged_set"]]
        if r["volume"] != VOL or r["cut"] != "full" or u["staged_inputs_file"] != f"{x['arm']}.jsonl":
            raise ValueError(f"{x['key']}: routing volume/cut or staged inputs file mismatch")
        sha, _ = arm_sha(r, u["staged_set"], x["arm"])
        per[(u["staged_set"], x["arm"])] += 1
        rows.append({"key": x["key"], "task_set": "S_new", "world": x["world"], "arm": x["arm"], "task": x["task"], "full_len": x["tokens_before"],
                     "tokens_before": x["tokens_before"], "original_folder": x["original_folder"],
                     "original_folder_registered": x["original_folder_registered"],
                     "uncut_planned": {"volume": VOL, "staged_set": u["staged_set"], "folder": u["folder"], "extraction_root": r["extraction_root"],
                                       "staged_sha256": sha, "staged_inputs_file": u["staged_inputs_file"], "routing": f"{s['path']} @ {s['commit'][:12]}",
                                       "revision": r["revision"]}})
    for (st, arm), n in per.items():
        if arm_sha(sel[st][1], st, arm)[1] != n:
            raise ValueError(f"{st}/{arm}: {n} inventory rows != routed epochs")
    used = {st for st, _ in per}
    ex = {}
    for st in sorted(used):
        one_extraction(B, sel[st][1], ex)
    spec = B.synth("derived/inventory_S_new_honest.json", {"status": "DERIVED", "runs": rows})
    cfg = {"label": "S_new honest worlds: inventory_v3 (INPUTS_FULL pin) + selected routing records (4f's rule)", "task_set": "S_new",
           "inventory_status": "DERIVED", "inventory": spec, "extraction_records": ex, "routing": {"mode": "routing", "records": [sel[st][0] for st in sorted(used)]}}
    return cfg, {"inventory_v3": inv_spec, "derived_sha256": spec["sha256"], "routing_ref": ref, "rows": len(rows)}


def refused(B: Blobs, inv: dict, specs: list) -> tuple:
    """uncut_inputs_all.refused_out_counts restricted to orphans whose target is one of these routing records (the others belong to staged
    sets this job does not read); both counts returned."""
    st_of = {}
    for s in specs:
        d = B.json(s)
        st_of[d["target"]] = d["staged_set"]
    orph = inv.get("s4_orphans_out") or []
    return Counter((st_of[o["target"]], o["arm"]) for o in orph if o["target"] in st_of), sum(o["target"] not in st_of for o in orph)


def pools(B: Blobs, full: dict, e12: dict, pk: dict) -> dict:
    """{class: {set: {"e1": {world|cell}, "e2": {inventory key: (world, cell)}}}} with every list checked against its pin."""
    fz = full["freeze_checks"]["e12_pools"]
    e12p, nmk = e12["pools"], "with S4 and without S4 (one calibration pool; S4r lists joined)"

    def e1(pool_list, pin):
        ks = list(pool_list)
        if pin is not None and hs(ks) != pin:
            raise ValueError(f"an epoch-1 member list differs from its E12 pin {pin[:12]}")
        if len(set(ks)) != len(ks):
            raise ValueError("an epoch-1 member twice")
        return set(ks)

    def e2(entries, pin):
        if hs([x["inventory_key"] for x in entries]) != pin["physical_members_sha256"] or hs([x["key_e2"] for x in entries]) != pin["e2_members_sha256"] \
                or len(entries) != pin["n"]:
            raise ValueError("an epoch-2 member list differs from its INPUTS_FULL pin")
        out = {}
        for x in entries:
            w, c = x["replicates"].split("|")
            if out.setdefault(x["inventory_key"], (w, c)) != (w, c):
                raise ValueError(f"{x['inventory_key']}: two epoch-1 members replicated")
        return out

    def by_pool(lst, start):
        (p,) = [x for x in lst if x["pool"].startswith(start)]
        return p["epoch1_member_keys"]

    sold, snew, nm = pk["origins"]["S_old"]["BuC"]["pools"], pk["origins"]["S_new"], pk["origins"]["S_newmains"][nmk]["pools"]
    fam = snew["H_A (25h §3 yardstick, held-out false alarm)"]["families"]
    a_new_e2 = {}
    for g, v in e12p["S_new"]["H_A"].items():
        if isinstance(v, dict) and "e2" in v:
            for x in v["e2"]:
                if a_new_e2.setdefault(x["inventory_key"], x) != x:
                    raise ValueError(f"{x['inventory_key']}: two S_new H_A epoch-2 entries")
    out = {
        "B": {"S_old": {"e1": e1(by_pool(sold, "B alone"), e12p["S_old"]["H_B2"]["epoch1_members_sha256"]),
                        "e2": e2(e12p["S_old"]["H_B2"]["e2"], fz["S_old"]["H_B2"])},
              "S_new": {"e1": e1(snew["B"]["pools"][0]["epoch1_member_keys"], e12p["S_new"]["H_B"]["epoch1_members_sha256"]),
                        "e2": e2(e12p["S_new"]["H_B"]["e2"], fz["S_new"]["H_B"])},
              "S_newmains": {"e1": e1(by_pool(nm, "H_B (honest_b pool)"), e12p["S_newmains"]["H_B"]["epoch1_members_sha256"]),
                             "e2": e2(e12p["S_newmains"]["H_B"]["e2"], fz["S_newmains"]["H_B"])}},
        "A": {"S_old": {"e1": e1(by_pool(sold, "C ("), e12p["S_old"]["H_A"]["epoch1_members_sha256"]), "e2": e2([], fz["S_old"]["H_A"])},
              "S_new": {"e1": e1(sorted({k for f in fam.values() for k in f["epoch1_member_keys"]}), None),
                        "e2": e2(sorted(a_new_e2.values(), key=lambda x: x["inventory_key"]), fz["S_new"]["H_A"])},
              "S_newmains": {"e1": e1(by_pool(nm, "H_A"), e12p["S_newmains"]["H_A"]["epoch1_members_sha256"]),
                             "e2": e2(e12p["S_newmains"]["H_A"]["e2"], fz["S_newmains"]["H_A"])}}}
    if snew["B"]["pools"][0]["pool"] != "H_B (B)":
        raise ValueError("S_new's B pool is not 'H_B (B)'")
    return out


def folds(B: Blobs, spec: dict) -> tuple:
    """The own_all_rs re-split fold list (by_main) and each cell's main and set, from the list's own pinned cell files (as 4f's build_inputs)."""
    F = B.json(spec)
    fold_of_main = {m: v[0] for m, v in F["by_main"].items()}
    main_of_cell = {}
    for _, (c, p, sh) in F["pins"]["cells"].items():
        for x in json.loads(B.get(c, p, sh))["cells"]:
            main_of_cell[x["key"]] = x["main"]
    c, p, pre = F["pins"]["sp493_facts"]
    b = B.get(c, p)
    if not hashlib.sha256(b).hexdigest().startswith(pre):
        raise ValueError(f"{p}: sha256 differs from the fold list's pin")
    for r in json.loads(b)["rows"]:
        if main_of_cell.setdefault(r["key"], r["main"]) != r["main"]:
            raise ValueError(f"{r['key']}: two mains")
    return fold_of_main, main_of_cell, F["cell_set"]


def build_all(files: list, data: dict, src: dict) -> dict:
    """All sources -> one fail-closed epoch list with class (batch), set, cell, main (task), fold and name; plus want / shas / allowed for the
    skip list and loader, the drops by reason, and the provenance."""
    B = Blobs(files, data)
    role = {}
    for f in files:
        role.setdefault(f["role"], []).append({k: f[k] for k in ("commit", "path", "sha256")})
    def one(r: str) -> dict:
        if len(role.get(r, [])) != 1:
            raise ValueError(f"role {r}: {len(role.get(r, []))} files")
        return role[r][0]

    full = B.json(one("inputs_full"))
    if one("inputs_full")["sha256"] != INPUTS_FULL_SHA:
        raise ValueError("INPUTS_FULL is not route C's frozen 1e186858 file")
    e12_spec, pk_spec = one("e12"), one("poolkeys")
    if e12_spec["sha256"] != full["e12"]["sha256"]:
        raise ValueError("the E12 block is not INPUTS_FULL's")
    e12 = B.json(e12_spec)
    if pk_spec["sha256"] != e12["artifact"]["sha256"]:
        raise ValueError("the pool-keys artifact is not the E12 block's")
    P = pools(B, full, e12, B.json(pk_spec))
    inv = {}
    # route C: S_old = inventory_b_fix_v2 (inv_sold, read through sold_config), S_newmains = inventory_b_fix_v1 (inv_bfix, its "new mains" rows)
    for name, want in (("inv_sold", "S_old"), ("inv_bfix", "S_newmains"), ("inv_snew", "S_new"), ("inv_ha25f", "ha25f"), ("inv_w1", "S_paper493"),
                       ("inv_w23", "S_paper493_w23"), ("inv_ep2", "pools_ep2"), ("inv_fill", "pools_fill")):
        s = one(name)
        if s["sha256"] != full["inventories"][want]["sha256"]:
            raise ValueError(f"{name}: not INPUTS_FULL's {want} inventory")
        inv[name] = s
    pk_extra = one("pilot_keys")
    rk.pilot_pin(full, pk_extra)
    _, pilot_pairs = rk.pilot_keys(B.json(pk_extra))
    new_keys = rk.new_rows_are_pilot(rk.inventory_delta(B.json(inv["inv_bfix"]), B.json(inv["inv_sold"])), pilot_pairs)
    fold_of_main, main_of_cell, cell_set = folds(B, one("folds"))
    sold_cfg = json.loads(B.get(one("sold_config")["commit"], one("sold_config")["path"]))
    if {k: sold_cfg["inventory"][k] for k in ("path", "sha256")} != {k: inv["inv_sold"][k] for k in ("path", "sha256")} \
            or not inv["inv_sold"]["commit"].startswith(sold_cfg["inventory"]["commit"]):
        raise ValueError("sold_config's inventory is not INPUTS_FULL's S_old inventory (inventory_b_fix_v2)")
    snew_attack_cfg = json.loads(B.get(one("snew_attack_config")["commit"], one("snew_attack_config")["path"]))
    snew_cfg, snew_prov = cfg_snew(B, inv["inv_snew"], src["routing_ref"], src["routing_dir"], src["routing_names"], snew_attack_cfg)
    cfgs = [("S_old", sold_cfg), ("S_newmains", cfg_cited(B, inv["inv_bfix"], "new mains", "S_newmains honest worlds")), ("S_new", snew_cfg),
            ("ha25f", cfg_cited(B, inv["inv_ha25f"], "S_new", "S_new 25f honest-A")), ("w1", cfg_cited(B, inv["inv_w1"], "S_paper493", "S_paper493 w1")),
            ("w23", cfg_cited(B, inv["inv_w23"], "S_paper493", "S_paper493 w23")), ("ep2", cfg_cited(B, inv["inv_ep2"], "pools_ep2", "epoch 2")),
            ("fill", cfg_cited(B, inv["inv_fill"], "pools_ep2", "A25 fill-in"))]
    ui.git_json = uia.git_json = lambda repo, spec: B.json(spec)  # 4f's loader reads committed files through the shipped blobs
    worlds = {w: w for w in HONEST}
    epochs, want, shas, allowed, prov = [], {}, {}, {}, {"S_new_derivation": snew_prov, "sources": {}}
    for tag, cfg in cfgs:
        ro, skipped = refused(B, B.json(cfg["inventory"]), cfg["routing"]["records"])
        u = uia.build(None, cfg, worlds, dry_run=False, refused_out=ro)
        for st, arms in u["want"].items():
            for arm, keys in arms.items():
                if want.setdefault(st, {}).setdefault(arm, keys) != keys:
                    raise ValueError(f"{st}/{arm}: two key lists")
        for tgt, new in ((shas, u["shas"]), (allowed, u["allowed"])):
            for k, v in new.items():
                if tgt.setdefault(k, v) != v:
                    raise ValueError(f"{k}: two values")
        prov["sources"][tag] = {"config": {k: cfg[k] for k in ("label", "task_set", "inventory_status", "inventory")}, "roots": sorted(u["allowed"]),
                                "routing": u["provenance"]["routing"], "refused_out": {f"{a}/{b}": n for (a, b), n in ro.items()}, "orphans_other_targets": skipped}
        epochs += [e | {"source": tag} for e in u["epochs"]]
    if len({e["key"] for e in epochs}) != len(epochs):
        raise ValueError("a run key in two sources")
    hit = {(c, s, x, m): 0 for c, d in P.items() for s, v in d.items() for x in ("e1", "e2") for m in v[x]}
    kept, drops = [], Counter()
    for e in epochs:
        if e["world"] == "sim_honest":
            cls, cell, ep = "sim", e["task"], 1
        elif e["source"] == "ep2":
            got = [(c, s) for c, d in P.items() for s, v in d.items() if e["key"] in v["e2"]]
            if not got:
                drops[(e["source"], e["world"], "epoch 2 not in a pool")] += 1
                continue
            (cls, s_), = got
            w, cell = P[cls][s_]["e2"][e["key"]]
            if w != e["world"] or cell_set.get(cell) != s_:
                raise ValueError(f"{e['key']}: epoch-2 replicate world or set differs from its pool's")
            hit[(cls, s_, "e2", e["key"])] += 1
            ep = 2
        else:
            cell, ep = e["task"], 1
            s_ = cell_set.get(cell)
            if s_ == "S_paper493":
                cls = {"honest_a": "A", "honest_b": "B"}.get(e["world"])
            else:
                got = [c for c in P if s_ in P[c] and f"{e['world']}|{cell}" in P[c][s_]["e1"]]
                cls = got[0] if len(got) == 1 else None
                if len(got) > 1:
                    raise ValueError(f"{e['key']}: in two pools")
            if cls is None:
                drops[(e["source"], e["world"], "epoch 1 not in a pool")] += 1
                continue
            if s_ != "S_paper493":
                hit[(cls, s_, "e1", f"{e['world']}|{cell}")] += 1
        if cell not in cell_set:
            raise ValueError(f"{e['key']}: cell {cell} is not in the fold list")
        main = main_of_cell[cell]
        if main not in fold_of_main:
            raise ValueError(f"COVERAGE STOP: main {main} (cell {cell}) has no fold")
        kept.append(e | {"batch": cls, "cell": cell, "set": cell_set[cell], "task": main, "fold": fold_of_main[main], "epoch": ep,
                         "name": f"{cls}|{cell}|e{ep}|{hashlib.sha1(e['key'].encode()).hexdigest()[:8]}"})
    bad = sorted(f"{k[0]}|{k[1]}|{k[2]}|{k[3]}" for k, n in hit.items() if n != 1)
    if bad:
        raise ValueError(f"{len(bad)} pool members not matched exactly once, e.g. {bad[:5]}")
    if len({e["name"] for e in kept}) != len(kept):
        raise ValueError("duplicate run name")
    kept.sort(key=lambda e: e["name"])
    # S_old sim membership = world sim_honest (L321) x the fold list's cell_set (L354) over the S_old config's T196 rows (uncut_inputs_all
    # L29); the inventories' "in_S_old_pool (FINAL pool_keys_old)" column is never read (0 references in this folder's code; it is null on
    # the 20 pilot rows). 09's STOP (a).
    prov["routec_sold_sim"] = rk.sold_sim(kept, new_keys) | {"pilot_keys": pk_extra}
    prov["routec_prior_match"] = rk.prior_match(B.json(one("prior_record")), kept, new_keys) | {"prior_record": one("prior_record")}
    want_k = {}
    for e in kept:
        want_k.setdefault(e["staged_inputs_run"], {}).setdefault(e["arm"], []).append(e["key"])
    shas_k = {f"{st}/inputs/{arm}.jsonl": shas[f"{st}/inputs/{arm}.jsonl"] for st, arms in want_k.items() for arm in arms}
    allowed_k = {e["extraction_run"]: allowed[e["extraction_run"]] for e in kept}
    counts = {c: {s: {f"e{x}": sum(1 for e in kept if e["batch"] == c and e["set"] == s and e["epoch"] == x) for x in (1, 2)} for s in SETS} for c in ("sim", "A", "B")}
    pool_sizes = {c: {s: {"e1": len(v["e1"]), "e2": len(v["e2"])} for s, v in d.items()} for c, d in P.items()}
    prov |= {"pool_sizes": pool_sizes, "pool_hashes": {c: {s: {"e1": hs(v["e1"]), "e2": hs(v["e2"])} for s, v in d.items()} for c, d in P.items()},
             "drops": [{"source": a, "world": b, "reason": c, "n": n} for (a, b, c), n in sorted(drops.items())],
             "files_used": sorted(B.used), "folds": one("folds"), "inputs_full": one("inputs_full"), "e12": e12_spec, "poolkeys": pk_spec}
    return {"epochs": kept, "want": want_k, "shas": shas_k, "allowed": allowed_k, "counts": counts, "provenance": prov}
