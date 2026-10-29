"""Main-text version of the monitor forest figure: short row labels, acting vs reading by colour, the as-is →
formatting-fixed pairs drawn as linked rows, and Aswin's title, order, labels and footnote (about 12:29 PT). The
matched-false-alarm row is in the appendix figure only. Full row details go to a caption file.

Same sources, checks and convention as plot_monitor_forest.py; writes forest_main_values/forest_main/caption_main_<stamp>.

  uv run --no-project --with matplotlib python plot_monitor_forest_main.py --repo <repo> [--s3-cache <dir>]
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
FLIP = {"sim - real", "run2 - run1", "real2 - real1"}
KEEP = {"real - sim", "run1 - run2", "real1 - real2"}
ACT, READ, YARD, MARGIN = "#c2651a", "#2f6eb5", "#8a919c", 10  # acting (on-policy), reading (off-policy), two real runs

# Presentation only: (row id, short label, acting?) per group, acting rows first; "pair" rows follow their as-is row.
LAYOUT = [
    ("Watches the agent (reads its commands)", [
        ("A3", "Command monitor, trained on real", True),
        ("A5", "Command monitor, trained on simulated", True),
        ("A4", "Text classifier on commands (no activations)", True),
        ("A7", "Whole-run attack detector, commands", True),
        ("A1", "Command monitor, Qwen3.5 reader", False),
        ("A2", "Command monitor, Qwen3-32B reader", False),
        ("A6", "Data-transfer command monitor", False),
    ]),
    ("Watches the simulated computer (reads its replies)", [
        ("B1a", "Reply detector, Qwen3-32B: simulator as-is", False),
        ("B1b", "→ formatting fixed", False),
        ("B2", "Reply detector, Olmo: simulator as-is", False),
        ("B2b", "→ formatting fixed", False),
        ("B3", "Reply detector, Qwen3.5 reader (within two-real-runs noise)", False),
        ("B4", "Whole-run attack detector, replies (mostly Qwen's reasoning)", True),
    ]),
]
APPENDIX_ONLY = {"B5": "matched false alarms (5%): in the appendix figure only (Aswin, about 12:29 PT)"}
TITLE = ("Monitors that watch the agent catch the same share in simulation; reply-reading monitors broke until two "
         "formatting slips were fixed.")  # Aswin's words, about 12:29 PT
PAIRS = [("B1a", "B1b"), ("B2", "B2b")]
FOOTNOTE = ("Not shown: the on-policy destructive-command and data-transfer monitors. They have no score yet: the human "
            "label sheet for destructive and transfer commands in Qwen3.5's acting runs is not filled in yet.\n"
            "The reading-set-up destructive monitor isn't shown because its two-real-runs control fails ±10.")


def listed(man, url, want, base):  # as in plot_monitor_forest.py: one entry binds the full key to the digest
    key = url.split("/", 3)[3]
    if isinstance(man, list):
        return any(listed(v, url, want, base) for v in man)
    if not isinstance(man, dict):
        return False
    if man.get("sha256") == want and ({key, url} & {man.get("key"), man.get("s3"), man.get("url")}):
        return True
    return any((v == want or isinstance(v, dict) and v.get("sha256") == want) and k in (key, url, key.removeprefix(base + "/"))
               or listed(v, url, want, base) for k, v in man.items())


def load(src, repo, cache):
    if src.get("s3"):
        url = src["s3"]["url"]
        if any(w in url.lower() for w in SEALED):
            raise ValueError(f"sealed path refused: {url}")
        dst = Path(cache) / hashlib.sha256(url.encode()).hexdigest()[:16]
        if not dst.exists():
            subprocess.run(["aws", "s3", "cp", "--profile", "redwood-sandbox", "--only-show-errors", url, str(dst)], check=True)
        data, want = dst.read_bytes(), src["s3"]["sha256"]
        man = json.loads(subprocess.run(["git", "-C", repo, "show", f"{src['s3']['manifest_ref']}:{src['s3']['manifest']}"],
                                        capture_output=True, text=True, check=True).stdout)
        if isinstance(man, dict) and str(man.get("bucket", url.split("/")[2])).removeprefix("s3://").strip("/") != url.split("/")[2]:
            raise ValueError(f"{url} is not in the manifest's bucket {man['bucket']}")
        if not listed(man, url, want, "/".join(src["s3"]["manifest"].split("/")[:2])):
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
    data, where = load(src, repo, cache)
    p, lo, hi = (walk(data, src[k]) * src["scale"] for k in ("point_key", "lo_key", "hi_key"))
    conv = src["convention"]
    if conv not in FLIP | KEEP:
        raise ValueError(f"unknown convention {conv!r} in {where}")
    if conv in FLIP:
        p, lo, hi = -p, -hi, -lo
    if not lo <= p <= hi:
        raise ValueError(f"point outside its interval in {where}: {p} [{lo}, {hi}]")
    ref = src.get("ref") or src["s3"]["manifest_ref"]
    rc = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", ref, "origin/main"]).returncode
    if rc not in (0, 1):
        raise RuntimeError(f"git merge-base failed ({rc}) for {ref}")
    return {"point": p, "lo": lo, "hi": hi, "stored_convention": conv, "flipped": conv in FLIP, "source": where,
            "keys": [src["point_key"], src["lo_key"], src["hi_key"]], "scale": src["scale"], "on_main": rc == 0}


def f1(x):
    return f"{0.0 if round(x, 1) == 0 else x:+.1f}"


def plot(vals, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = []
    for head, rows in LAYOUT:
        lines.append(("head", head))
        lines += [("row", r) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(lines) + 1.8))
    lim = 22.0
    ys = {}
    for i, (kind, x) in enumerate(lines):
        y = -i
        if kind == "head":
            ax.text(-lim - 0.6, y, x, fontsize=12, fontweight="bold", va="center", ha="right")
            continue
        rid, label, acting = x
        ys[rid] = y
        g, c = vals[rid]["gap"], vals[rid]["control"]
        col = ACT if acting else READ
        ax.errorbar(g["point"], y + 0.11, xerr=[[g["point"] - g["lo"]], [g["hi"] - g["point"]]], fmt="o", color=col, ms=7,
                    lw=2, capsize=3)
        txt = f"{f1(g['point'])} [{f1(g['lo'])}, {f1(g['hi'])}]"
        if c:
            ax.errorbar(c["point"], y - 0.15, xerr=[[c["point"] - c["lo"]], [c["hi"] - c["point"]]], fmt="o", mfc="white",
                        color=YARD, ms=5.5, lw=1.4, capsize=2.5)
            txt += f"   two real {f1(c['point'])}"
        else:
            txt += "   two real: none"
        ax.text(lim + 0.6, y, txt, fontsize=9.5, va="center", family="monospace")
        ax.text(-lim - 0.6, y, label, fontsize=10, va="center", ha="right")
    for a, b in PAIRS:  # as-is → fixed: an arrow between the two real − sim points
        pa, pb = vals[a]["gap"]["point"], vals[b]["gap"]["point"]
        ax.annotate("", xy=(pb, ys[b] + 0.11), xytext=(pa, ys[a] + 0.11),
                    arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2, "shrinkA": 5, "shrinkB": 5})
    for v in (-MARGIN, MARGIN):
        ax.axvline(v, ls="--", color="#555555", lw=1)
        ax.text(v, 0.45, "±10 margin", fontsize=9, color="#555555", ha="center", va="bottom")
    ax.annotate("→ simulation catches less", xy=(1, 0), xycoords="axes fraction", xytext=(0, -22),
                textcoords="offset points", ha="right", va="top", fontsize=9.5, color="#333333")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-len(lines) + 0.4, 0.6)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Real − simulated catch (points), at 1% false alarms. Positive = catches more on real runs. "
                  "Dashed: ±10.\n" + FOOTNOTE, fontsize=9.5, labelpad=22)
    ax.plot([], [], "o", color=ACT, label="real − simulated, acting (Qwen3.5 does the tasks)")
    ax.plot([], [], "o", color=READ, label="real − simulated, reading (a model reads Sol's runs)")
    ax.plot([], [], "o", mfc="white", color=YARD, label="two real runs")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9.5)
    ax.set_title(TITLE.replace("; ", ";\n"), fontsize=13, fontweight="bold", pad=28)
    fig.savefig(png, dpi=160, bbox_inches="tight")


def caption(rows, vals, labels):
    out = ["# Caption details for the main-text forest figure", "",
           "Each row: short label, then what is caught, set-up, tasks, alpha, pre-registered or post hoc, and source "
           "commits (branch only where marked). Values are real − sim [95% range] | two real runs [95% range].", ""]
    for head, rs in LAYOUT:
        out += [f"## {head}", ""]
        for rid, label, acting in rs:
            r, v = rows[rid], vals[rid]
            g, c = v["gap"], v["control"]
            branch = any(x and not x["on_main"] for x in (g, c))
            out.append(f"- **{label}** ({'acting' if acting else 'reading'}): {r['monitor']}; {r['unit']}; {r['setup']}; "
                       f"{r['tasks']}; {int(r['alpha'] * 100)}%; {r['prereg']}. {f1(g['point'])} [{f1(g['lo'])}, {f1(g['hi'])}]"
                       + (f" | {f1(c['point'])} [{f1(c['lo'])}, {f1(c['hi'])}]" if c else " | no two-real control")
                       + f". Source: `{g['source']}`" + (f"; two-real source: `{c['source']}`" if c and c["source"] != g["source"] else "")
                       + (" (branch only)" if branch else "") + ".")
        out.append("")
    out += ["## In the appendix figure only", ""] + [f"- {rid}: {why}" for rid, why in APPENDIX_ONLY.items()] + ["", FOOTNOTE, ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(HERE / "rows.jsonl"))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--s3-cache", default="/tmp/forest_s3_cache")
    ap.add_argument("--out", default=str(HERE / "results"), help="a scratch folder for dry runs; results/ is append-only")
    a = ap.parse_args()
    Path(a.s3_cache).mkdir(parents=True, exist_ok=True)
    with open(a.rows) as fh:
        rows = {r["row"]: r for r in (json.loads(line) for line in fh if line.strip())}
    shown = [rid for _, rs in LAYOUT for rid, _, _ in rs]
    if len(set(shown)) != len(shown) or sorted(shown + list(APPENDIX_ONLY)) != sorted(rows):
        raise ValueError(f"LAYOUT + APPENDIX_ONLY and rows.jsonl differ: {sorted(set(shown + list(APPENDIX_ONLY)) ^ set(rows))}")
    vals = {rid: {"gap": read(rows[rid]["gap"], a.repo, a.s3_cache),
                  "control": read(rows[rid]["control"], a.repo, a.s3_cache) if rows[rid].get("control") else None}
            for rid in shown}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = Path(a.out)
    res.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "script_commit": git,
            "rows_sha256": hashlib.sha256(Path(a.rows).read_bytes()).hexdigest()}
    (res / f"forest_main_values_{stamp}.json").write_text(json.dumps({**meta, "rows": vals}, indent=1) + "\n")
    (res / f"caption_main_{stamp}.md").write_text(caption(rows, vals, LAYOUT))
    plot(vals, res / f"forest_main_{stamp}.png")
    print(f"-> {res / f'forest_main_{stamp}.png'}")


if __name__ == "__main__":
    main()
