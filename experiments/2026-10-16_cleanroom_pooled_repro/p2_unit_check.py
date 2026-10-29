"""Why do the p2 `command` C files hold ~16% more units than sol_a_uncut__cmd_A and own_all_rs_uncut_L48__C for the same runs?
(0d / 09, 10-17; a1; the MECHANISM cross-check on a seeded sample of 10 runs, seed 20261017; E2E's exhaustive count decides.)
Revision 2 after review round 1 (review/p2_check_rounds.md): units are (irid, call_index) via the staged rows; no text is touched.

Keys, positions, token ids and booleans only. The job reads the staged inputs' `ids` and `calls` rows (kind, readout, irid,
call_index, label) and the score files' run_key / token columns; it reads no message text and no score.

Per sampled run (identity = the physical run_key, all files from one folder):
  - every unit of p2 command (four probes), sol_a_uncut cmd_A and own_all_rs_uncut C is mapped to the staged row whose readout equals its
    token (p2 expected kind "command"; Sol / own_all_rs expected kind "call_end"); unmapped units and kind mismatches are counted;
  - the unit sets are compared as sets of calls (irid, call_index);
  - each p2 call missing from Sol (and the reverse) is described by: whether the call has a call_end readout, whether that readout's
    token id is fixed-slot (15 "0", 77 "n": the registered content-only drop, RULE l.3295 / l.4577-4579), the staged kind at the p2
    token, and the call's staged label (impactful / read-only), when the row carries one.

  CR_CFG=<pinned input folder> uv run --no-project --python 3.12 --with modal==1.6.1 modal run p2_unit_check.py
"""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import modal

SCORES = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
ACTS = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
SEED, N = 20261017, 10
FIXED = {15, 77}  # the registered fixed call tokens '0', 'n' (sola_uncut_skiplist fixed_ids)
P2 = ["fibs", "harmfulness", "refusal", "highstakes"]
F = {"p2": "p2/fibs__command.npz", "sol": "sola_uncut_reselect/sol_a_uncut__cmd_A.npz",
     "oars": "own_all_rs_uncut/own_all_rs_uncut_L48__C.npz", **{f"p2_{p}": f"p2/{p}__command.npz" for p in P2[1:]}}
WANT = {"p2": "command", "sol": "call_end", "oars": "call_end"}
FULL = "INPUTS_FULL_2026-10-16_110447.json"

app = modal.App("probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
if modal.is_local():
    image = image.add_local_dir(os.environ["CR_CFG"], "/cfg")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def base(k):
    return str(k).split("#")[0]


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0,
              volumes={"/s": modal.Volume.from_name(SCORES, version=2).read_only(),
                       "/a": modal.Volume.from_name(ACTS, version=2).read_only()})
def check() -> dict:
    import numpy as np

    full = json.loads(Path("/cfg", FULL).read_bytes())
    # ---- every C folder (arms and sim-honest roots) of every lead pool that has all six files
    groups = {}
    for k, o in full["overrides"].items():
        if not k.endswith("|C"):
            continue
        ob = Path("/cfg", Path(o["path"]).name).read_bytes()
        assert sha(ob) == o["sha256"], k
        om = json.loads(ob)
        ents = list(om["arms"].items())
        simh = {}
        for fk, lst in om["simh"].items():
            for e in lst:
                simh.setdefault(e["root"], {})[fk] = e
        ents += list(simh.items())
        for folder, d in ents:
            if all(F[x] in d for x in F):
                groups.setdefault(d[F["p2"]]["path"], {x: d[F[x]] for x in F} | {"_folder": folder, "_map": k})

    def rows(e):
        """run_key -> list of token positions (one per row)."""
        b = Path("/s", e["path"]).read_bytes()
        assert sha(b) == e["sha256"], e["path"]
        z = np.load(io.BytesIO(b), allow_pickle=False)
        out = {}
        for rk, tok in zip(z["run_key"].tolist(), z["token"].tolist()):
            out.setdefault(str(rk), []).append(int(tok))
        return out

    # ---- candidates: run_keys in all three files of one folder; seeded sample without replacement
    cand, keys3 = [], {}
    for p, g in sorted(groups.items()):
        r = {x: rows(g[x]) for x in ("p2", "sol", "oars")}
        cand += [(rk, p) for rk in sorted(set(r["p2"]) & set(r["sol"]) & set(r["oars"]))]
        keys3[p] = r
    cand = sorted(set(cand))
    rng = np.random.default_rng(SEED)
    pick = [cand[i] for i in sorted(rng.choice(len(cand), size=N, replace=False).tolist())]

    out = {"seed": SEED, "n_candidates": len(cand), "n_folders": len(groups), "schema": None, "runs": []}
    for rk, p in pick:
        g = groups[p]
        mb = Path("/s", g["p2"]["manifest"]["path"]).read_bytes()
        assert sha(mb) == g["p2"]["manifest"]["sha256"], "p2 manifest sha"
        man = json.loads(mb)
        st = [s for s in man["dump"]["staged"] if Path(s["folder"].rstrip("/")).name == Path(p).parent.name]
        assert len(st) == 1, (p, len(st))
        assert st[0]["sha"] in (g["p2"].get("staged_shas") or [st[0]["sha"]]), "staged sha not among the file map's staged_shas"
        h = hashlib.sha256()
        with Path("/a", st[0]["path"]).open("rb") as fb:
            for blk in iter(lambda: fb.read(1 << 22), b""):
                h.update(blk)
        assert h.hexdigest() == st[0]["sha"], f"staged file sha ({p})"
        rec = None
        with Path("/a", st[0]["path"]).open() as fh:
            for line in fh:
                if base(rk) in line:
                    it = json.loads(line)
                    if base(it.get("key")) == base(rk):
                        assert rec is None, "a sampled run staged twice"
                        rec = it
        assert rec is not None, f"staged record not found for a sampled run ({p})"
        if out["schema"] is None:  # names only
            out["schema"] = {"top": sorted(rec), "call_row": sorted({k for c in rec.get("calls", []) for k in c}),
                             "kinds": sorted({str(c.get("kind")) for c in rec.get("calls", [])}),
                             "labels": (lambda v: v if len(v) <= 8 else f"{len(v)} distinct values (not printed)")(
                                 sorted({str(c.get("label")) for c in rec.get("calls", [])}))}
        if "ids" not in rec or "calls" not in rec:
            out["error"] = "staged record lacks ids or calls; schema only"
            return out
        ids = rec["ids"]
        at = {}  # readout position -> staged rows there
        for c in rec["calls"]:
            if c.get("readout") is not None:
                at.setdefault(int(c["readout"]), []).append(c)
        call_end = {(c["irid"], c["call_index"]): int(c["readout"]) for c in rec["calls"]
                    if c.get("kind") == "call_end" and c.get("readout") is not None}
        label = {}
        for c in rec["calls"]:
            if c.get("kind") == "command" and c.get("label") is not None:  # only the command row carries the call's own label
                label.setdefault((c["irid"], c["call_index"]), set()).add(str(c["label"]))

        def units(x, toks):
            """Map each row's token to the staged row of the wanted kind at that readout -> {(irid, call_index): token}."""
            got, bad, kinds_at = {}, 0, {}
            for t in toks:
                rs = [c for c in at.get(t, []) if c.get("kind") == WANT[x]]
                kinds_at[str(sorted(str(c.get("kind")) for c in at.get(t, [])))] = kinds_at.get(
                    str(sorted(str(c.get("kind")) for c in at.get(t, []))), 0) + 1
                if len(rs) != 1:
                    bad += 1
                    continue
                u = (rs[0]["irid"], rs[0]["call_index"])
                assert u not in got, f"{x}: two rows on one call"
                got[u] = t
            return got, bad, kinds_at

        U, info = {}, {}
        for x in ("p2", "sol", "oars"):
            U[x], bad, kinds_at = units(x, keys3[p][x][rk])
            info[x] = {"rows": len(keys3[p][x][rk]), "mapped": len(U[x]), "unmapped": bad, "staged_kinds_at_tokens": kinds_at}
        same_p2 = {pp: sorted(rows(g[f"p2_{pp}"]).get(rk, [])) == sorted(keys3[p]["p2"][rk]) for pp in P2[1:]}

        def describe(u):
            ce = call_end.get(u)
            return {"has_call_end_readout": ce is not None, "call_end_fixed_slot": ce is not None and ids[ce] in FIXED,
                    "label": sorted(v if v in ("impactful", "read_only") else "other" for v in label.get(u, [])) or None}

        def tally(us):
            d = {}
            for u in us:
                k = json.dumps(describe(u), sort_keys=True)
                d[k] = d.get(k, 0) + 1
            return d

        p2_not_sol = sorted(set(U["p2"]) - set(U["sol"]))
        sol_not_p2 = sorted(set(U["sol"]) - set(U["p2"]))
        out["runs"].append({
            "run_key": rk, "folder": g["_folder"], "map": g["_map"], "units": info, "p2_unit_sets_equal_across_probes": same_p2,
            "n_calls_with_call_end": len(call_end), "sol_vs_oars_same_calls": set(U["sol"]) == set(U["oars"]),
            "p2_not_in_sol": len(p2_not_sol), "p2_not_in_sol_by_class": tally(p2_not_sol),
            "sol_not_in_p2": len(sol_not_p2), "sol_not_in_p2_by_class": tally(sol_not_p2),
            "p2_not_in_oars": len(set(U["p2"]) - set(U["oars"])), "oars_not_in_p2": len(set(U["oars"]) - set(U["p2"])),
            "shared_by_class": tally(sorted(set(U["p2"]) & set(U["sol"])))})
    return out


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    res = check.remote()
    path = Path(__file__).parent / "results" / f"p2_unit_check_{stamp}.json"
    path.open("x").write(json.dumps(res, indent=1, default=str))
    print("wrote", path)
