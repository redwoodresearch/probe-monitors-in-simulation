"""2e / 0d: an index (paths and sha256s only, NO values) of the per-unit score files of every own-trained probe on the corrected-B
cohort (T196), from the four pack manifests on the scores Volume (fetched locally, sha-pinned on the command line).

  python make_unit_file_index.py famexp_open.json:<sha> famexp_sealed.json:<sha> owns_open.json:<sha> owns_sealed.json:<sha>
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
VOL = "probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13"
MANIFESTS = {"famexp_open": "/famexp/2026-10-14_163027/pack_open_2026-10-14_180501/manifest.json",
             "famexp_sealed": "/famexp/2026-10-14_163027/pack_sim_sealed_2026-10-14_181123/manifest.json",
             "owns_open": "/owns/2026-10-14_231856/pack_open_2026-10-14_231946/manifest.json",
             "owns_sealed": "/owns/2026-10-14_231856/pack_sim_sealed_2026-10-14_233605/manifest.json"}
WORLDS = {"famexp_open": "run1, run2, honest_a, honest_b, honest_c", "famexp_sealed": "sim",
          "owns_open": "run1, run2, honest_a, honest_b, honest_c", "owns_sealed": "sim, simh_pilot, simh_full"}


def main():
    out = {"label": "paths and sha256s only; no values", "volume": VOL, "packs": {}, "files": []}
    for spec, name in zip(sys.argv[1:], MANIFESTS, strict=True):
        path, want = spec.rsplit(":", 1)
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit(f"{name}: manifest sha256 differs")
        m = json.loads(raw)
        out["packs"][name] = {"manifest": MANIFESTS[name], "sha256": want, "spec_stamp": m["spec_stamp"], "run_stamp": m["run_stamp"],
                              "part": m["part"], "worlds": WORLDS[name]}
        for k, v in sorted(m["variants"].items()):
            if k.split("|")[0].startswith("own_"):
                out["files"].append({"variant": k.split("|")[0], "readout": k.split("|")[1], "pack": name, "worlds": WORLDS[name],
                                     "path": "/" + v["npz"]["path"], "sha256": v["npz"]["sha256"], "rows": v["npz"]["rows"], "runs": v["npz"]["runs"],
                                     "complete": v["complete"]})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"own_family_unit_files_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"path": str(p.relative_to(HERE.parents[1])), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "files": len(out["files"])}))


if __name__ == "__main__":
    main()
