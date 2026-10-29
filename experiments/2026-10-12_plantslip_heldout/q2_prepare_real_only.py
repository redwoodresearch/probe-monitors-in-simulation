"""Row 2, step 2 (RULE.md here): the real-only preparation script, the only job allowed to load mixed score arrays before
step 3. For Olmo (run 2026-10-12_110549) and Llama (run 2026-10-12_143129), at the rule's selected layers, it copies only the
real arms' columns (real-honest, real-attack-e1, real-attack-e2) of each scores_L<l>.npz's #36-recipe readout
(output_end, mean over a run's outputs: `output_end__mean__task_nested` [5, 5, runs] and `__task_outer` [runs]) into new
real-only files, with a real-only runs.json. It computes nothing on any array, asserts that no sim-attack run is
written, checks each input against manifest part 1's sha256, and prints only paths and arm counts. Local, $0.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-12_plantslip_heldout/q2_prepare_real_only.py \\
      <manifest_part1 json> <olmo run_scores_2026-10-12_110549 dir> <llama run_scores_2026-10-12_143129 dir>
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
REAL = ("real-honest", "real-attack-e1", "real-attack-e2")
KEYS = ("output_end__mean__task_nested", "output_end__mean__task_outer")
PART1_SHA = "9cdd5d47289a78a618c533d37cb37a16e7e0299e5b94021e391ebb1f53336cac"  # results_manifest_v1.json's entry
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main(part1_p, olmo_dir, llama_dir):
    for f in (Path(__file__).resolve(), HERE / "RULE.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} is not committed as is")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if sha(part1_p) != PART1_SHA:
        raise SystemExit("not manifest part 1")
    part1 = json.loads(Path(part1_p).read_text())["existing_detector_artifacts"]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out_dir = HERE / "results" / f"real_only_{stamp}"; out_dir.mkdir(parents=True)
    rec = {"rule": "RULE.md (this folder), step 2", "git": head, "manifest part 1 sha256": PART1_SHA, "arms kept": list(REAL), "keys kept": list(KEYS), "readers": {}}
    for reader, d in (("olmo", Path(olmo_dir)), ("llama", Path(llama_dir))):
        path, commit = part1[reader]["run_record"].split("@")  # the fit's run record pins its runs.json (archive object data/runs.json)
        pinned = json.loads(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=True, cwd=HERE).stdout)["archive"]["objects"]["data/runs.json"]["sha256"]
        if sha(d / "runs.json") != pinned:
            raise SystemExit(f"{reader}: runs.json differs from the run record's pinned sha256")
        runs = json.loads((d / "runs.json").read_text())
        arms = Counter(r["arm"] for r in runs)
        if set(arms) != set(REAL) | {"sim-attack"}:
            raise SystemExit(f"{reader}: unexpected arms {sorted(arms)}")
        idx = np.array([i for i, r in enumerate(runs) if r["arm"] in REAL])
        real_runs = [runs[i] | {"source_index": int(i)} for i in idx]
        assert all(r["arm"] in REAL for r in real_runs) and not any(r["arm"] == "sim-attack" for r in real_runs)
        (out_dir / f"{reader}_runs.json").write_text(json.dumps(real_runs))
        rr = {"source runs.json sha256": sha(d / "runs.json"), "real runs written": dict(Counter(r["arm"] for r in real_runs)), "layers": {}}
        for layer in part1[reader]["layers"]:
            src = d / f"scores_L{layer}.npz"
            if sha(src) != part1[reader]["objects"][f"scores_L{layer}.npz"]["sha256"]:
                raise SystemExit(f"{reader} scores_L{layer}.npz differs from manifest part 1")
            with np.load(src) as z:  # mixed arrays: indexed to the real columns only, nothing computed
                arrays = {"output_end__mean__task_nested": z["output_end__mean__task_nested"][:, :, idx],
                          "output_end__mean__task_outer": z["output_end__mean__task_outer"][idx]}
            assert arrays["output_end__mean__task_nested"].shape[-1] == len(real_runs) == arrays["output_end__mean__task_outer"].shape[0]
            dst = out_dir / f"{reader}_scores_L{layer}_real_only.npz"
            np.savez(dst, source_index=idx, **arrays)
            rr["layers"][str(layer)] = {"source sha256": sha(src), "file": dst.name, "sha256": sha(dst)}
        rr["runs file"] = {"file": f"{reader}_runs.json", "sha256": sha(out_dir / f"{reader}_runs.json")}
        rec["readers"][reader] = rr
    with open(dest := out_dir / "real_only_record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(dest)
    for reader, rr in rec["readers"].items():
        print(reader, rr["real runs written"], "layers", sorted(rr["layers"], key=int))


if __name__ == "__main__":
    main(*sys.argv[1:4])
