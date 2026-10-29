"""One forest figure: every monitor's real − sim catch gap beside its two-real-runs gap, grouped by what it watches.

Reads every value by key with json.load, from a git commit (`git show ref:path`) or from S3 (sha256 checked against
the committed manifest's, which must list it), per rows.jsonl. Puts every gap on one convention:
- gap = real − sim, where "real" is the real run the row compares with the simulated one;
- yardstick = that same real run − the other real run (same pipeline, two real runs).
A row stored as sim − real (or as run 2 − run 1) is flipped, and the flip is recorded. Gaps are in percentage points of
catch at 1% false alarms unless the row says 5%. Positive = the monitor catches more on real runs.

Writes results/forest_values_<stamp>.json (every value read, its source, the flip) and results/forest_<stamp>.png.

  uv run --no-project --with matplotlib python plot_monitor_forest.py --rows rows.jsonl --repo <repo> [--s3-cache <dir>]
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SEALED = ("oct11c", "lean230", "zooc", "zoo_c", "zoo-c", "stage_records", "stage-records")
FLIP = {"sim - real", "run2 - run1", "real2 - real1"}  # stored conventions that point the other way
KEEP = {"real - sim", "run1 - run2", "real1 - real2"}
REAL, YARD, MARGIN = "#2f6eb5", "#8a919c", 10  # colors: real − sim, two real runs; dashed margin at ±10 points


def listed(man, url, want, base):
    """True if one manifest entry binds this object's full key (or URL) to this digest. A relative key in the manifest
    counts only when the manifest's own experiment folder (`base`) plus that key is exactly the object's key."""
    key = url.split("/", 3)[3]  # s3://bucket/<key>
    if isinstance(man, list):
        return any(listed(v, url, want, base) for v in man)
    if not isinstance(man, dict):
        return False
    if man.get("sha256") == want and ({key, url} & {man.get("key"), man.get("s3"), man.get("url")}):
        return True
    for k, v in man.items():
        if (v == want or isinstance(v, dict) and v.get("sha256") == want) and k in (key, url, key.removeprefix(base + "/")):
            return True
        if listed(v, url, want, base):
            return True
    return False


def load(src, repo, cache):
    if src.get("s3"):
        url = src["s3"]["url"]
        if any(w in url.lower() for w in SEALED):
            raise ValueError(f"sealed path refused: {url}")
        dst = Path(cache) / hashlib.sha256(url.encode()).hexdigest()[:16]
        if not dst.exists():
            subprocess.run(["aws", "s3", "cp", "--profile", "redwood-sandbox", "--only-show-errors", url, str(dst)], check=True)
        data = dst.read_bytes()
        want = src["s3"]["sha256"]
        man = json.loads(subprocess.run(["git", "-C", repo, "show", f"{src['s3']['manifest_ref']}:{src['s3']['manifest']}"],
                                        capture_output=True, text=True, check=True).stdout)
        bucket = url.split("/")[2]
        if isinstance(man, dict) and str(man.get("bucket", bucket)).removeprefix("s3://").strip("/") != bucket:
            raise ValueError(f"{url} is not in the manifest's bucket {man['bucket']}")
        if not listed(man, url, want, "/".join(src["s3"]["manifest"].split("/")[:2])):  # the committed manifest binds this object's key to this digest
            raise ValueError(f"the committed manifest does not list {url} with {want}")
        if hashlib.sha256(data).hexdigest() != want:
            raise ValueError(f"sha256 differs from the manifest: {url}")
        return json.loads(data), url
    if any(w in src["file"].lower() for w in SEALED):
        raise ValueError(f"sealed path refused: {src['file']}")
    out = subprocess.run(["git", "-C", repo, "show", f"{src['ref']}:{src['file']}"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout), f"{src['file']}@{src['ref']}"


def walk(obj, key):
    for k in key:
        obj = obj[k] if isinstance(obj, dict) else obj[int(k)]
    return float(obj)


def read(src, repo, cache):
    """(point, lo, hi) in points on the target convention, plus provenance. Fails loudly on any missing key."""
    data, where = load(src, repo, cache)
    p, lo, hi = (walk(data, src[k]) * src["scale"] for k in ("point_key", "lo_key", "hi_key"))
    conv = src["convention"]
    if conv not in FLIP | KEEP:
        raise ValueError(f"unknown convention {conv!r} in {where}")
    flipped = conv in FLIP
    if flipped:
        p, lo, hi = -p, -hi, -lo
    if not lo <= p <= hi:
        raise ValueError(f"point outside its interval in {where}: {p} [{lo}, {hi}]")
    ref = src.get("ref") or (src.get("s3") or {}).get("manifest_ref")  # an S3 file counts by its manifest's commit
    if not ref:
        raise ValueError(f"no commit to check against main for {where}")
    rc = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", ref, "origin/main"]).returncode
    if rc not in (0, 1):  # 1 = not an ancestor; anything else is a git error, not "branch only"
        raise RuntimeError(f"git merge-base failed ({rc}) for {ref}")
    on_main = rc == 0
    return {"point": p, "lo": lo, "hi": hi, "stored_convention": conv, "flipped": flipped, "source": where,
            "keys": [src["point_key"], src["lo_key"], src["hi_key"]], "scale": src["scale"], "on_main": on_main}


def f1(x):
    """One decimal with a sign, never "-0.0"."""
    return f"{0.0 if round(x, 1) == 0 else x:+.1f}"


def plot(rows, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("A", "Watches the agent (reads its commands)"), ("B", "Watches the simulated computer (reads its replies)")]
    lines = []  # (kind, payload) top to bottom
    for g, title in groups:
        lines.append(("head", title))
        lines += [("row", r) for r in rows if r["group"] == g]
    fig, ax = plt.subplots(figsize=(13, 0.62 * len(lines) + 1.6))
    lim = max(16.0, max(max(abs(r["gap"]["lo"]), abs(r["gap"]["hi"])) for r in rows) + 2)
    for i, (kind, x) in enumerate(lines):
        y = -i
        if kind == "head":
            ax.text(-lim - 0.6, y, x, fontsize=12, fontweight="bold", va="center", ha="right")
            continue
        g, c = x["gap"], x["control"]
        ax.errorbar(g["point"], y + 0.12, xerr=[[g["point"] - g["lo"]], [g["hi"] - g["point"]]], fmt="o", color=REAL,
                    ms=7, lw=2, capsize=3)
        txt = f"{f1(g['point'])} [{f1(g['lo'])}, {f1(g['hi'])}]"
        if c:
            ax.errorbar(c["point"], y - 0.16, xerr=[[c["point"] - c["lo"]], [c["hi"] - c["point"]]], fmt="o", mfc="white",
                        color=YARD, ms=6, lw=1.6, capsize=3)
            txt += f"   two real: {f1(c['point'])} [{f1(c['lo'])}, {f1(c['hi'])}]"
        else:
            txt += "   two real: none"
        ax.text(lim + 0.6, y, txt, fontsize=9.5, va="center", family="monospace")
        branch = any(v and v["on_main"] is False for v in (g, c))
        label = (f"{x['monitor']} — {x['unit']}" + (" (5%)" if x["alpha"] == 0.05 else "") + f"\n{x['setup']}, {x['tasks']}, {x['prereg']}"
                 + (", branch only" if branch else ""))
        ax.text(-lim - 0.6, y, label, fontsize=9, va="center", ha="right")
    for v in (-MARGIN, MARGIN):
        ax.axvline(v, ls="--", color="#555555", lw=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-len(lines) + 0.4, 0.6)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Real − simulated catch (points), at 1% false alarms unless marked 5%; what is caught is named per row.\n"
                  "Positive = catches more on real runs. Dashed: ±10.")
    ax.plot([], [], "o", color=REAL, label="real − simulated (95% range)")
    ax.plot([], [], "o", mfc="white", color=YARD, label="two real runs (95% range)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10)
    ax.set_title("Real − simulated catch gap for every monitor, beside the gap between two real runs", fontsize=13.5,
                 fontweight="bold", pad=30)
    fig.savefig(png, dpi=160, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(HERE / "rows.jsonl"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--s3-cache", default="/tmp/forest_s3_cache")
    ap.add_argument("--out", default=str(HERE / "results"), help="a scratch folder for dry runs; results/ is append-only")
    a = ap.parse_args()
    Path(a.s3_cache).mkdir(parents=True, exist_ok=True)
    with open(a.rows) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    out = []
    for r in rows:
        out.append({**{k: r[k] for k in ("row", "monitor", "unit", "group", "setup", "tasks", "alpha", "prereg", "notes")},
                    "gap": read(r["gap"], a.repo, a.s3_cache),
                    "control": read(r["control"], a.repo, a.s3_cache) if r.get("control") else None})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = Path(a.out)
    res.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "script_commit": git,
            "rows_sha256": hashlib.sha256(Path(a.rows).read_bytes()).hexdigest()}
    (res / f"forest_values_{stamp}.json").write_text(json.dumps({**meta, "rows": out}, indent=1) + "\n")
    plot(out, res / f"forest_{stamp}.png")
    for r in out:
        g, c = r["gap"], r["control"]
        print(f"{r['row']:4} {g['point']:+6.2f} [{g['lo']:+6.2f}, {g['hi']:+6.2f}] flip={g['flipped']!s:5} main={g['on_main']!s:5} | "
              + (f"{c['point']:+6.2f} [{c['lo']:+6.2f}, {c['hi']:+6.2f}] flip={c['flipped']}" if c else "no yardstick"))
    print(f"-> {res / f'forest_{stamp}.png'}")


if __name__ == "__main__":
    main()
