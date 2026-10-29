"""Dry-run staging targets for the S_paper493 donor fix (09 via d9 10-15 ~23:15): the real arms' main and run2 manifests rebuilt from each box
read exactly as commit_targets now builds them (registered exclude_keys, then add_index_donors), and the g1/g2 sim-attack families' snapshot
manifests merged, each as a target `<name>_dry` staging to the NON-ROUTABLE set `<out>_dry` with donor_launches (read, never staged). Writes the
manifests and adds the targets to config/chain_targets.json (keys ending _dry only; an existing key stops). Keys and hashes only.

  python <this> <chain state dir>
"""
import json
import sys
from pathlib import Path

import chain as C

HERE = Path(__file__).resolve().parent


class St:
    def __init__(self, d: Path):
        self.d, self.s = d, json.loads((d / "state.json").read_text())

    def log(self, m: str):
        print(m)


def main():
    st = St(Path(sys.argv[1]))
    cfg = json.loads((HERE / "config" / "chain_batches.json").read_text())
    ct_p = HERE / "config" / "chain_targets.json"
    ct = json.loads(ct_p.read_text())
    new = {}
    for L in [x for x in cfg["launches"] if x.get("pool") == "sp493"]:
        b = dict(st.s["batches"][f"sp493-{L['q']}"]) | {"index_donors": L["index_donors"], "exclude_keys": L.get("exclude_keys", [])}
        d = json.loads((st.d / f"boxfiles_{b['id']}.json").read_text())
        mans = {}
        for name, t in C.targets_for(b | {"stage_donor_launches": None}).items():
            man = json.loads(json.dumps(d["manifests"]["run2" if t["set"] == "run2" else "main"]))
            ex = set(b["exclude_keys"])
            man["files"] = [f for f in man["files"] if f"{f['arm_dir']}|{f['task']}" not in ex]
            C.add_index_donors(b, name, man, st)
            mans[name] = (t, man)
        for name, (t, man) in mans.items():
            dl = sorted({f["launch"] for f in man["files"] if f["arm_dir"] in ("honest_a", "honest_b") and f["launch"] not in t["launch"]})
            new[f"{name}_dry"] = (t | {"manifest": f"manifest_{name}_dry.json", "out": f"{t['out']}_dry", "donor_launches": dl}, man)
    for g in ("g1", "g2"):
        fam = st.s["batches"][f"sp493-{g}"]
        snaps = sorted((x for x in st.s["batches"].values() if x.get("family") == fam["id"] and x["kind"] == "fus" and x.get("commit_targets")), key=lambda x: x["snap"])
        rows, srcs, no_eval = {}, [], []
        for x in snaps:
            (t0, tv0), = C.targets_for(x).items()
            p = HERE / "config" / tv0["manifest"]
            srcs.append(p.name)
            m0 = json.loads(p.read_text())
            no_eval += [x for x in m0["counted_attempts_without_eval"]["main"] if x not in no_eval]
            for f in m0["files"]:
                k = (f["arm_dir"], f["task"])
                if k in rows and rows[k]["path"] != f["path"]:
                    raise SystemExit(f"{g}: {k} has two different rows across snapshots")
                rows[k] = {kk: v for kk, v in f.items() if kk != "snapshot_id"}
        t = C.targets_for(fam)
        (name, t), = t.items()
        man = {"note": f"dry run: union of {', '.join(srcs)}", "counted_attempts_without_eval": {"main": no_eval},
               "files": sorted(rows.values(), key=lambda f: f["path"] + "|" + f["task"])}
        man["counts"] = {"files": len(man["files"]), "by_arm": {a: sum(f["arm_dir"] == a for f in man["files"]) for a in sorted({f["arm_dir"] for f in man["files"]})}}
        dl = sorted({f["launch"] for f in man["files"] if f["arm_dir"] in ("honest_a", "honest_b") and f["launch"] not in t["launch"]})
        new[f"{name}_dry"] = (t | {"manifest": f"manifest_{name}_dry.json", "out": f"{t['out']}_dry", "donor_launches": dl}, man)
    for k, (t, man) in new.items():
        assert k.endswith("_dry") and t["out"].endswith("_dry") and k not in ct, k
        (HERE / "config" / t["manifest"]).open("x").write(json.dumps(man, indent=1) + "\n")
        ct[k] = t
        print(k, {a: n for a, n in man["counts"]["by_arm"].items()}, "donor_launches", len(t["donor_launches"]))
    tmp = ct_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ct, indent=1) + "\n")
    tmp.replace(ct_p)


if __name__ == "__main__":
    main()
