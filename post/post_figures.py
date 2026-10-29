"""Redraw the post's figures from the committed records, as the original render_all.py (not included) did: render only, no recomputation.

Each bundle in figures/ holds byte-identical copies of the figure scripts at the commit that drew the post's PNGs, the render log of that
run (its plan, its script commit and the sha256 of every output) and inputs.json (each record name in the plan -> a record in MANIFEST.json).
render_all.py's _draw is replayed: the scripts and the records go into a fresh folder with a results/ subfolder, every figure in the plan
runs with the plan's environment and `--git <script commit>`, and the new files in results/ are the outputs. Figure 3 is a schematic,
compared pixel by pixel with its committed reference render.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLES = {"fig2": "render_log_attack_v5_193930.json", "figs156": "render_log_figs156_routecA_012618.json",
           "figs78": "render_log_figs78_routecA_013135.json"}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def plan_env(pl):
    """render_all.py's environment block (L180-188 at 85204152c, the same at 6790ea75d and a9ca8a5c9)."""
    est = pl.get("estimator")
    return {"FIG_COMPARATOR_OVERRIDES": json.dumps(pl.get("comparator_overrides") or {}), "FIG_DRAW_EXTRA": json.dumps(pl.get("draw_extra") or []),
            "FIG_UNION_NOTE": pl.get("union_note") or "", "FIG_TITLE_BANNER": pl.get("title_banner") or "",
            "FIG_BUILD": json.dumps(pl.get("build") or {}), "FIG_KEEP_GOING": "1" if pl.get("keep_going") else "",
            "FIG_PARALLEL": "1" if pl.get("parallel") else "", "FIG_COMPARATOR": pl.get("comparator") or "",
            "FIG_ESTIMATOR": json.dumps(est) if isinstance(est, dict) else est or ""}


def pixels(png):
    from PIL import Image  # Pillow comes with matplotlib

    im = Image.open(io.BytesIO(png))
    return im.mode, im.size, im.tobytes()


def draw(b, pl, commit, inputs, records, figures, probs):
    """render_all.py's _draw: one fresh folder, the scripts and records in it, the figures run in order. -> {output name: bytes}."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "exp"
        (root / "results").mkdir(parents=True)
        for f in b.glob("*.py"):
            shutil.copy(f, root / f.name)
        for base in pl["records"]:
            (root / "results" / base).write_bytes(records.raw(inputs[base]))
        before = {p.name for p in (root / "results").iterdir()}
        for fg in figures:
            r = subprocess.run([sys.executable, fg["script"], *fg["args"], "--git", commit], cwd=root, capture_output=True, text=True,
                               env=os.environ | plan_env(pl))
            if r.returncode:
                probs.append(f"{b.name}: {fg['script']} exited {r.returncode}: {r.stderr[-1500:]}")
        return {p.name: p.read_bytes() for p in (root / "results").iterdir() if p.name not in before}


def render_bundle(name, records, out):
    """Run one bundle's plan; write its outputs to out/<name>/; -> list of problems (empty when every output matches)."""
    b = HERE / "figures" / name
    log = json.loads((b / BUNDLES[name]).read_text())
    pl, commit, want = log["plan"], log["script_commit"], log["outputs"]
    inputs = json.loads((b / "inputs.json").read_text())
    probs, outs = [], {}
    # plan field parallel: render_all ran each figure in its own container (its own fresh folder); otherwise all in one folder, in order
    for figs in ([[fg] for fg in pl["figures"]] if pl.get("parallel") else [pl["figures"]]):
        new = draw(b, pl, commit, inputs, records, figs, probs)
        probs += [f"{name}: {n} written twice" for n in set(outs) & set(new)]
        outs |= new
    (out / name).mkdir(parents=True, exist_ok=True)
    for n, data in outs.items():
        (out / name / n).write_bytes(data)
    for n in sorted(set(want) | set(outs)):
        if n not in outs:
            probs.append(f"{name}: {n} was not drawn")
        elif n not in want:
            probs.append(f"{name}: {n} was drawn but is not in the render log")
        elif sha256(outs[n]) != want[n]:
            ref = HERE / "expected" / "figures" / n
            if n.endswith(".png") and ref.exists() and pixels(outs[n]) == pixels(ref.read_bytes()):
                print(f"  {name}: {n}: PNG bytes differ but every pixel matches the committed PNG (Pillow / zlib build)")
            else:
                probs.append(f"{name}: {n}: sha256 {sha256(outs[n])[:16]} differs from the render log's {want[n][:16]}")
    return probs


def draw_fig3(records, out):
    """Figure 3 (the schematic, no data): drawn in the pinned environment and compared pixel by pixel with expected/figures/
    fig3_reference.png. The draft's own PNG was drawn with matplotlib 3.9.2, so the reference is the first render in the pinned environment,
    committed and pinned in MANIFEST.json ("pending"); without it this check fails with "fig3 reference missing"."""
    (out / "fig3").mkdir(parents=True, exist_ok=True)
    dest = out / "fig3" / "figure3_schematic_v5.png"
    r = subprocess.run([sys.executable, str(HERE / "figures" / "fig3" / "draw_fig3_schematic_v5.py"), str(dest)], capture_output=True, text=True)
    if r.returncode or not dest.exists():
        return [f"fig3: draw_fig3_schematic_v5.py exited {r.returncode}: {r.stderr[-1500:]}"]
    got = dest.read_bytes()
    ref = records.pending_file("fig3_reference")
    if ref is None:
        print(f"  fig3 reference missing: commit {dest} as expected/figures/fig3_reference.png and pin its sha256 in MANIFEST.json "
                f"(pending; this render's sha256 is {sha256(got)})")
        return []  # counted once, under MANIFEST pending
    if got == ref.read_bytes():
        print("  fig3: byte-identical to fig3_reference.png")
        return []
    if pixels(got) == pixels(ref.read_bytes()):
        print("  fig3: PNG bytes differ but every pixel matches fig3_reference.png (Pillow / zlib build)")
        return []
    return [f"fig3: the render's pixels differ from expected/figures/fig3_reference.png (render sha256 {sha256(got)[:16]})"]


def draft_is_render(exp):
    """The draft's PNG of each data figure is the PNG its render run wrote (expected/expected_figures.json vs the render logs)."""
    probs = []
    for f in exp["figures"]:
        if f["bundle"] in BUNDLES:
            want = json.loads((HERE / "figures" / f["bundle"] / BUNDLES[f["bundle"]]).read_text())["outputs"].get(f["stem"] + ".png")
            if want != f["draft_png_sha256"]:
                probs.append(f"{f['post']}: the draft's PNG is not the PNG in the {f['bundle']} render log")
    return probs


def redraw(records, out):
    """Redraw every bundle and Figure 3 into out/figures/; -> list of problems."""
    out = out / "figures"
    probs = draft_is_render(json.loads((HERE / "expected" / "expected_figures.json").read_text()))
    for name in BUNDLES:
        p = render_bundle(name, records, out)
        print(f"  {name}: {'OK' if not p else f'{len(p)} problem(s)'}")
        probs += p
    return probs + draw_fig3(records, out)
