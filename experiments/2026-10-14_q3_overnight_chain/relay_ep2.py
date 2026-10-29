"""Epoch-2 real-honest relay, decoupled from the chain's family build (d9 10-15 ~16:10): box -> laptop cache -> chain Volume
evals_v1/<run>/<rel>, by an ALLOWLIST only: the .eval files the launch's own upload manifests list, plus launch.json, gate1_final.json and
the manifests themselves. cells.json and any approach or attack file are never copied (approach files move only via
relay_approaches.sh). Every file is sha256- and size-checked against the box manifest on the laptop and again on the Volume before any
laptop copy is deleted; nothing is opened beyond hashing (the manifests, keys and hashes only, are parsed).

  plan:  python3 relay_ep2.py plan  <devbox> <run dir on box, e.g. phase2/runs_ep2/2026-10-15_115153_ep2_t196> <manifest names, e.g. e2a,e2b,common>
                                    <box hashes json {tsv name: sha256} (the end record's sha256_on_box)> <allowlist.json>
  relay: python3 relay_ep2.py relay <allowlist.json> <record.json>
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEVBOX_SH = REPO / "experiments/2026-10-07_phase2_attack_check/devbox.sh"
VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
HASHER = ("probe-shift_aswin-surya_sub-q3chain_volhash_kill-2026-11-14", "hash_on_volume")
CACHE = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/chain_run_state/cache")
META = ("launch.json", "gate1_final.json")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rsync(devbox: str, run_dir: str, rels: list, dest: Path, lst: Path):
    lst.write_text("".join(r + "\n" for r in rels))
    dest.mkdir(parents=True, exist_ok=True)
    for rel in rels:
        assert safe(rel), rel
    r = subprocess.run(["bash", str(DEVBOX_SH), "rsync", "-t", "--no-links", "--no-devices", "--no-specials", f"--files-from={lst}", f"box:{run_dir}/", str(dest) + "/"],
                       env={**__import__("os").environ, "DEVBOX": devbox}, capture_output=True, timeout=3600)
    if r.returncode != 0:
        raise SystemExit(f"rsync rc {r.returncode}: {r.stderr.decode()[-300:]}")


def safe(rel: str) -> bool:
    """A plain relative path inside the run dir: no absolute path, no '..', no empty or '.' parts."""
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and all(x not in ("", ".", "..") for x in parts)


def local(cache: Path, rel: str) -> Path:
    """The cached copy: a regular file (not a link) inside the cache folder."""
    f = cache / rel
    assert safe(rel) and not f.is_symlink() and f.is_file() and f.resolve().is_relative_to(cache.resolve()), rel
    return f


def allowed(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    low = rel.lower()
    return safe(rel) and (name.endswith(".eval") or name in META or (name.startswith("upload_manifest_") and name.endswith(".tsv"))) and \
        name != "cells.json" and "approach" not in low and "attack" not in low


def plan(devbox: str, run_dir: str, names: str, box_hashes: str, out: str):
    run = run_dir.rstrip("/").rsplit("/", 1)[-1]
    cache = CACHE / f"ep2_{run}"
    tsvs = [f"upload_manifest_{n}.tsv" for n in names.split(",")]
    rsync(devbox, run_dir, tsvs, cache, cache.parent / f"ep2_{run}_manifests.lst")
    files, seen = [], set()
    want = json.loads(Path(box_hashes).read_text())
    for t in tsvs:  # each manifest equals its box copy by sha256 before any of its lines is trusted
        if sha(local(cache, t)) != want[t]:
            raise SystemExit(f"{t}: laptop copy sha256 != box sha256 {want[t]}")
    for t in tsvs:
        for line in (cache / t).read_text().splitlines():
            rel, nbytes, h = line.split("\t")
            assert rel not in seen, rel
            seen.add(rel)
            files.append({"rel": rel, "bytes": int(nbytes), "sha256": h, "manifest": t})
    for t in tsvs:  # the manifests themselves (their box hash: the end record's, checked by the caller against this one)
        files.append({"rel": t, "bytes": (cache / t).stat().st_size, "sha256": want[t], "manifest": "self"})
    bad = [f["rel"] for f in files if not allowed(f["rel"])]
    if bad:
        raise SystemExit(f"not on the allowlist (nothing copied beyond the manifests): {bad}")
    metas = sorted(f["rel"] for f in files if f["rel"].rsplit("/", 1)[-1] in META)
    assert metas == sorted(META), metas  # launch.json and gate1_final.json come from the common manifest
    doc = {"what": "epoch-2 relay allowlist (names, bytes, sha256 from the launch's own upload manifests); no other path is copied",
           "devbox": devbox, "run_dir": run_dir, "box_hashes_of_manifests": want, "run": run, "volume": VOL, "dest_prefix": f"evals_v1/{run}/",
           "counts": {m: sum(f["manifest"] == m for f in files) for m in [*tsvs, "self"]}, "files": files}
    Path(out).open("x").write(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"run": run, "counts": doc["counts"], "total": len(files)}))


def relay(allow: str, out: str):
    import modal

    a = json.loads(Path(allow).read_text())
    ap = Path(allow).resolve()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(ap)], cwd=HERE, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(ap)], cwd=HERE).returncode == 0
    if not (tracked and clean):
        raise SystemExit("the allowlist must be committed, unchanged, before the relay runs")
    bad = [f["rel"] for f in a["files"] if not allowed(f["rel"])]
    if bad:
        raise SystemExit(f"allowlist entries outside the rule: {bad}")
    cache = CACHE / f"ep2_{a['run']}"
    if shutil.disk_usage(CACHE).free < 20 * 1024**3:
        raise SystemExit("laptop free disk < 20 GB")
    items = [f | {"dest": a["dest_prefix"] + f["rel"]} for f in a["files"]]
    hasher = modal.Function.from_name(*HASHER)
    on_vol = hasher.remote([x["dest"] for x in items])
    todo = [x for x in items if on_vol.get(x["dest"]) != x["sha256"]]
    rsync(a["devbox"], a["run_dir"], sorted(x["rel"] for x in todo if x["manifest"] != "self"), cache, cache.parent / f"ep2_{a['run']}_relay.lst")
    for x in todo:
        f = local(cache, x["rel"])
        if f.stat().st_size != x["bytes"] or sha(f) != x["sha256"]:
            raise SystemExit(f"{x['rel']}: laptop copy differs from the box manifest; nothing uploaded")
    if todo:
        vol = modal.Volume.from_name(VOL, version=2)
        with vol.batch_upload(force=True) as batch:
            for x in todo:
                batch.put_file(str(cache / x["rel"]), "/" + x["dest"])
    on_vol = hasher.remote([x["dest"] for x in items])
    bad = [x["dest"] for x in items if on_vol.get(x["dest"]) != x["sha256"]]  # sha256 equality fixes the bytes (and so the size)
    if bad:
        raise SystemExit(f"{len(bad)} destinations differ on the Volume; nothing deleted: {bad[:3]}")
    doc = {"what": "epoch-2 relay record: every allowlisted file verified by sha256 on the Volume; laptop copies deleted",
           "allowlist": Path(allow).name, "allowlist_sha256": sha(Path(allow)), "run": a["run"], "volume": VOL, "dest_prefix": a["dest_prefix"],
           "files": len(items), "uploaded_now": len(todo), "already_on_volume": len(items) - len(todo), "verified_on_volume": len(items) - len(bad),
           "relayed_at": subprocess.run(["date", "+%Y-%m-%d %H:%M:%S %Z"], capture_output=True, text=True).stdout.strip()}
    Path(out).open("x").write(json.dumps(doc | {"laptop_copies": "deleted after this record was written (each one verified on the Volume)"}, indent=1) + "\n")
    for x in items:  # recorded and every destination verified: delete the laptop copies
        if (cache / x["rel"]).exists() or (cache / x["rel"]).is_symlink():
            f = local(cache, x["rel"])  # a regular, non-link file inside the cache only; anything else raises, nothing deleted
            if f.stat().st_size == x["bytes"] and sha(f) == x["sha256"]:  # only a copy identical to the verified Volume file is deleted
                f.unlink()
            else:
                print(json.dumps({"kept_mismatched_laptop_copy": x["rel"]}))
    print(json.dumps(doc))


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan(*sys.argv[2:7])
    elif sys.argv[1] == "relay":
        relay(*sys.argv[2:4])
    else:
        raise SystemExit(__doc__)
