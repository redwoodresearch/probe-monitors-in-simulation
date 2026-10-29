"""09 @ 3685810b4 item 11 for own_all_rs cutfull score records (APPEND-ONLY NOTE, 10-16): every run without rows, by key, with its ELIGIBLE readout units
counted from the run's extraction meta on the cutfull Volume (R: output_end reply rows; C: call_end command units): 0 eligible units = a structural zero,
KEPT as "not caught" for that readout; units present but no rows = a staging exclusion = STOP. Inputs: the score records (committed), their manifests and the
runs' meta files fetched from the Volumes (JSON only; each manifest sha-checked against its score record; each meta's key checked), and a run list
"<root> <meta arm> <meta stem> <folder> <key>" (the cutfull jobs rebuilt from the records' routing pins)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(records, man_dirs, run_list, meta_dir, out_rel):
    runs = {}
    for line in Path(run_list).read_text().splitlines():
        root, arm, stem, fo, key = line.split()
        runs[(fo, key)] = (root, arm, stem)
    out, stop = [], []
    for rel, md in zip(records.split(","), man_dirs.split(","), strict=True):
        b = (REPO / rel).read_bytes()
        if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True).stdout != b:
            raise SystemExit(f"{rel}: not committed as read")
        rec = json.loads(b)
        for r in rec["results"].values():
            for fo, x in r["folders"].items():
                if not any(c["runs_without_rows"] for c in x["counts"].values()):
                    continue
                mb = (Path(md) / f"{fo}.json").read_bytes()
                if hashlib.sha256(mb).hexdigest() != x["manifest_sha256"]:
                    raise SystemExit(f"{fo}: manifest differs from {rel}")
                man = json.loads(mb)
                zr = {u for c in man["counts"].values() for u in c["runs_without_rows"]}
                for u in sorted(zr):
                    cand = [(k, v) for (f_, k), v in runs.items() if f_ == fo and k.split("/")[-2].startswith(u.split("|", 1)[1])]
                    if len(cand) != 1:
                        raise SystemExit(f"{fo} {u}: not exactly one listed run ({len(cand)})")
                    key, (root, arm, stem) = cand[0]
                    mt_b = (Path(meta_dir) / f"{stem}.json").read_bytes()
                    mt = json.loads(mt_b)
                    if mt["key"] != key:
                        raise SystemExit(f"{stem}: meta key differs")
                    elig = {"R": sum(c["kind"] == "output_end" for c in mt["calls"]), "C": sum(c["kind"] == "call_end" for c in mt["calls"])}
                    rows = {"R": sum(c["kind"] == "output_end" and c.get("row_mean") is not None for c in mt["calls"]),
                            "C": sum(c["kind"] == "call_end" and c.get("row") is not None for c in mt["calls"])}
                    cls = {rd: ("structural zero (0 eligible units): KEPT as not caught" if elig[rd] == 0 else "STAGING EXCLUSION: STOP") for rd in ("R", "C")
                           if u in man["counts"][rd]["runs_without_rows"]}
                    if any(v.startswith("STAGING") for v in cls.values()):
                        stop.append(f"{fo} {u}")
                    out.append({"score_record": rel, "folder": fo, "unit": u, "key": key, "root": root, "meta": f"/{root}/meta/{arm}/{stem}.json",
                                "meta_sha256": hashlib.sha256(mt_b).hexdigest(), "meta_cut": mt.get("cut"), "tokens_total": mt.get("tokens_total"),
                                "eligible_units": elig, "rows": rows, "class": cls})
    note = {"label": "own_all_rs cutfull runs without rows, by key (09 @ 3685810b4 item 11); keys and counts only; APPEND-ONLY",
            "records": records.split(","), "runs": out, "structural_zero": sum(all(v.startswith("structural") for v in o["class"].values()) for o in out),
            "staging_exclusions": stop, "pass": not stop}
    (REPO / out_rel).open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"runs": len(out), "structural_zero": note["structural_zero"], "staging_exclusions": len(stop), "pass": note["pass"]}))
    if stop:
        raise SystemExit(f"STOP: staging exclusions {stop}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
