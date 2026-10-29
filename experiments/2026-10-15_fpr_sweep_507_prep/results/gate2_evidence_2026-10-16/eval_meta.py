"""Gate-2 evidence (09's ruling (a')), METADATA ONLY: for the honest / sim-honest .eval files of the flagged cells, the file sha256 (checked against
the frozen inventories), the eval run / task ids and creation time, and per sample its id, epoch, uuid and the NAMES of its metadata fields, with
the VALUES of fork / reference fields only when they are short identifiers (keys, paths, shas, ids). No message or tool text is read or printed;
no attack run is opened. Usage: modal run eval_meta.py --out <local json>"""
import json

import modal

VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
app = modal.App("probe-shift_aswin-surya_sub-pooledworld_gate2evid_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("zipfile-zstd==0.0.4")
vol = modal.Volume.from_name(VOL)
FILES = {  # run key -> (Volume path under evals_v1, expected sha256 from the frozen inventories)
    "simh_nm360|nm172_shared_dotfiles": ("2026-10-15_simh_nm_s1/sim_honest@september/nm172_shared_dotfiles/2026-10-15T19-11-26-00-00_la-shared-dotfiles-sim-honest-1_kqJxHhCUmqLXYxy5NAvzE8.eval",
                                         "ca1f248f29f8009eb197bc2a6486a5400a61ee70391ca7bcebc470434941aec1"),
    "honest_a_e2|nm172_shared_dotfiles": ("2026-10-15_123630_ep2_nm/honest_a/nm172_shared_dotfiles/2026-10-15T19-50-05-00-00_la-shared-dotfiles-real-honest-1_6tdbiHJWkAsZSWLmGcTfHh.eval",
                                          "941fedbb06c36c2cf1ff798e0ebb44c52e8d643bc63d7fe5c4d1d3ab08a154fe"),
    "honest_a|nm172_shared_dotfiles": ("2026-10-14_newmains_s1/honest_a/nm172_shared_dotfiles/2026-10-15T02-55-31-00-00_la-shared-dotfiles-real-honest-1_SPbeytrDXXVJ7zqRws67nr.eval",
                                       "4427114ca0d12e0289867a87710de5be31239407cf4925a7fab3ea0b262bed66"),
    "honest_b|nm172_shared_dotfiles": ("2026-10-14_newmains_s1/honest_b/nm172_shared_dotfiles/2026-10-15T02-58-42-00-00_la-shared-dotfiles-real-honest-1_f59TfDgXc4mNRS8MjGyzTY.eval",
                                       "f90d6ecda291cf3939f14fe7bfd3011013ce671d261ce6e0e23ddaafc12e299e"),
    "honest_a_e2|nm305_user_tracking": ("2026-10-15_123630_ep2_nm/honest_a/nm305_user_tracking/2026-10-15T19-56-07-00-00_la-user-tracking-real-honest-1_crX9Ne8kQ4iG23reVT3FuV.eval",
                                        "1149c9c2839761fa90efb7d101b3b3f66beda90fda42daa0b62a792e323b7118"),
    "honest_b_e2|nm305_user_tracking": ("2026-10-15_123630_ep2_nm/honest_b/nm305_user_tracking/2026-10-15T19-57-38-00-00_la-user-tracking-real-honest-1_bcsuxEYJFsR5sJnSASzfAy.eval",
                                        "e8a8edee4424cc2d2486d4803f47055c316d0d0dc1ab76cb78e413d1c6f0ae33"),
    "honest_b|nm305_user_tracking": ("2026-10-14_newmains_s1/honest_b/nm305_user_tracking/2026-10-15T03-12-28-00-00_la-user-tracking-real-honest-1_CtLkbkiGRHthy4rebscHWc.eval",
                                     "74bf7712a2d433ccf9b53031c9af60df29f4c88a2d521edf265e69c20a3ab9cf"),
}


def short_id(v):  # a value is printed only if it is a short identifier-like scalar (no free text)
    import re

    return isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 160 and re.fullmatch(r"[A-Za-z0-9_@|:./#+=,\- ]*", v) is not None
                                                  and v.count(" ") <= 2)


@app.function(image=image, cpu=1, memory=2048, timeout=900, retries=0, volumes={"/ev": vol})
def meta() -> dict:
    import hashlib
    import zipfile

    import zipfile_zstd  # noqa: F401  (.eval members are zstd-compressed)
    from pathlib import Path

    out = {}
    for key, (rel, sha) in FILES.items():
        p = Path("/ev/evals_v1") / rel
        b = p.read_bytes()
        r = {"path": f"evals_v1/{rel}", "sha256": hashlib.sha256(b).hexdigest(), "sha_matches_inventory": hashlib.sha256(b).hexdigest() == sha}
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            r["zip_members"] = sorted(n for n in names if not n.startswith("samples/"))[:20]
            if "header.json" in names:
                h = json.loads(z.read("header.json"))
                ev = h.get("eval") or {}
                r["eval"] = {k: ev.get(k) for k in ("run_id", "task_id", "created", "task", "eval_id") if short_id(ev.get(k))}
                r["eval_metadata_fields"] = sorted((ev.get("metadata") or {}).keys())
            sams = [n for n in names if n.startswith("samples/") and n.endswith(".json")]
            r["samples"] = []
            for n in sams:
                s = json.loads(z.read(n))
                md = s.get("metadata") or {}
                fk = {k: (v if short_id(v) else {"type": type(v).__name__, "len": len(json.dumps(v, default=str))})
                      for k, v in md.items() if any(t in k.lower() for t in ("fork", "ref", "donor", "source", "parent", "prefix", "launch", "epoch"))}
                r["samples"].append({"member": n, "id": s.get("id") if short_id(s.get("id")) else "?", "epoch": s.get("epoch"), "uuid": s.get("uuid") if short_id(s.get("uuid")) else "?",
                                     "metadata_fields": sorted(md.keys()), "fork_reference_fields": fk,
                                     "n_messages": len(s.get("messages") or [])})
        out[key] = r
    return out


@app.local_entrypoint()
def main(out: str):
    r = meta.remote()
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1, default=str)
    print(json.dumps({k: {"sha_ok": v["sha_matches_inventory"], "samples": len(v["samples"])} for k, v in r.items()}))
