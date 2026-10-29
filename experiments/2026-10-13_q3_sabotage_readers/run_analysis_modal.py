"""B3: run analyze_readers.py on each (score file, spec) pair and report_readers.py on the results, in one Modal CPU container
(Aswin, 17:24 PT 10-13: no compute on the laptop). The inputs are read from the B3 Volume (put there first by `modal volume
put`); the container returns the output files' bytes, and this entrypoint only writes them into results/. Numbers only.

  uv run --no-project --python 3.12 --with "modal==1.6.1" modal run run_analysis_modal.py \\
      --pairs "<scores on Volume>=<spec on Volume>,..." --part-a <grid_result_2026-10-13_163659.json on Volume>
"""

import os
import subprocess
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3sabr_analysis_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-q3sabr_acts_kill-2026-11-13"
HERE = Path(__file__).parent

app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
         .add_local_file(HERE / "analyze_readers.py", "/root/analyze_readers.py").add_local_file(HERE / "report_readers.py", "/root/report_readers.py"))


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={"/v": vol})
def analyze(pairs: list, part_a: str, commit: str) -> dict:
    import sys

    sys.path.insert(0, "/root")
    os.environ["B3_GIT_COMMIT"] = commit
    import analyze_readers
    import report_readers

    out = Path("/tmp/out")
    out.mkdir()
    for scores, spec in pairs:
        analyze_readers.main(f"/v/{scores}", f"/v/{spec}", str(out))
    results = sorted(str(p) for p in out.glob("grid_result_*.json"))
    report_readers.main(f"/v/{part_a}", results, str(out))
    files = {p.name: p.read_bytes() for p in out.iterdir()}
    Path("/v/analysis_v1").mkdir(exist_ok=True)
    for n, b in files.items():  # on the Volume first (d9: a lost client loses nothing)
        Path(f"/v/analysis_v1/{n}").write_bytes(b)
    vol.commit()
    return files


@app.local_entrypoint()
def main(pairs: str, part_a: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--", "analyze_readers.py", "report_readers.py", Path(__file__).name):
        raise SystemExit("commit the analysis scripts before they run")
    files = analyze.remote([p.split("=") for p in pairs.split(",")], part_a, git("rev-parse", "HEAD"))
    for name, b in sorted(files.items()):
        with (HERE / "results" / name).open("xb") as fh:
            fh.write(b)
        print("->", HERE / "results" / name)
