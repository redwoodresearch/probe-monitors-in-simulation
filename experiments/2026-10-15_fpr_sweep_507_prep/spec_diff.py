"""Drive-spec diff (09's spec-only controls, via 507-prep, 10-17): two specs as dispatch() builds them ({mode, jobs, meta, IN, cfg, label, dest})
compared by key. Output is paths and counts only, never a value. The diff runs in a Modal CPU container (no spec value is read on the laptop).

  modal run spec_diff.py --a <spec> --b <spec> [--plant-b] [--out results/spec_diff/<name>.json]
    <spec> = a local .pkl from run_peraction311.py --spec-only, or vol:<path> for a spec on the solavg Volume (a real run's drivers/<stem>.pkl)
    --plant-b: the POSITIVE control: one synthetic S_old H_sim key is added to one S_old job of b first; the diff must list exactly that path

Ignored (the run's own identity): IGNORE below, listed under "ignored_differing" whenever they differ or are present on one side only; and
each side's own run stamp (meta.stamp) inside any string, and its own HEAD sha (meta.git) inside a field named "commit" (the pins stamped with
HEAD at build), read as <stamp> / <git>, listed under "normalized_differing". Dict key order is not compared; dicts whose order differs are listed under "order_differing". None of the three
lists counts as a difference; anything else (any other code-identity field) does.
"""

import json
import math
import pickle
import re
from pathlib import Path

import modal

IGNORE = frozenset({"meta/git", "meta/stamp", "meta/read_started_pt", "meta/script_sha256", "meta/_submit_unix", "meta/shipped_inputs_sent_now",
                    "meta/pooled_world/modules_sha256/experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py"})  # the runner's own sha only:
# every other module's sha (full_override, inputs_full, ...) is compared, so a code change there shows
PLANT_KEY = "sim_honest_plant|positive_control_planted_key"
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"  # = run_common455.VOLS["solavg"] (the test asserts it)
app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_specdiff_kill-2026-11-14")
vol = modal.Volume.from_name(VOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def _key(k):
    return k if isinstance(k, str) else repr(k)


class Diff:
    """same(a, b): deep equality (NaN equals NaN, numpy by shape/dtype/bytes, types strict, dict key ORDER not compared), memoized by object
    identity (jobs share most of their cfg). With norm, each side's own run stamp is read as <stamp> inside any string; its own HEAD sha (full,
    or a 7+ hex prefix as a whole word) is read as <git> ONLY in a field named "commit" (the pins stamped with HEAD at build: honest_b_relaunch.pin,
    the fold-drops pin; Claude review r1, Astra r1 opt. (a)): HEAD anywhere else stays a difference. walk(): the paths where two specs differ, at
    the highest differing node; strings equal only after that reading are listed separately."""

    def __init__(self, a, b, norm=True):
        ma, mb = a.get("meta") or {}, b.get("meta") or {}
        self.id_ = ((ma.get("stamp") or "", ma.get("git") or ""), (mb.get("stamp") or "", mb.get("git") or "")) if norm else (("", ""), ("", ""))
        self.memo = {}
        self.raw = Diff(a, b, norm=False) if norm else self
        self.out = {"only_a": [], "only_b": [], "changed": [], "ignored_differing": [], "normalized_differing": [], "order_differing": []}

    @staticmethod
    def _norm(s, stamp, git):
        if stamp:
            s = s.replace(stamp, "<stamp>")
        if git:
            s = re.sub(r"\b[0-9a-f]{7,40}\b", lambda m: "<git>" if git.startswith(m.group(0)) else m.group(0), s)
        return s

    def same(self, a, b, commit=False):
        if isinstance(a, str) and isinstance(b, str):
            (sa, ga), (sb, gb) = self.id_
            return a == b or (self.raw is not self and self._norm(a, sa, ga if commit else "") == self._norm(b, sb, gb if commit else ""))
        if type(a) is not type(b):
            return False
        if isinstance(a, float):
            return a == b or (math.isnan(a) and math.isnan(b))
        if type(a).__module__ == "numpy" and hasattr(a, "tobytes"):
            return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        if not isinstance(a, (dict, list, tuple, set, frozenset)):
            return a == b
        k = (id(a), id(b))
        if k not in self.memo:
            if isinstance(a, dict):
                r = a.keys() == b.keys() and all(self.same(a[x], b[x], x == "commit") for x in a)
            elif isinstance(a, (set, frozenset)):
                r = a == b
            else:
                r = len(a) == len(b) and all(self.same(x, y) for x, y in zip(a, b))
            self.memo[k] = r
        return self.memo[k]

    def listed(self, a, b, path, into):  # noqa: C901
        """Paths under an equal-after-reading node where the raw values differ (strings), and dicts whose key order differs. In this pass a
        container shared by many jobs is listed once, at the first path it is met on (walk() itself may list a shared dict's order at each job
        path where the jobs differ elsewhere; Astra r1 opt. (b))."""
        if path in IGNORE:  # Astra r1 R2: an ignored field is filed as ignored, never as normalized
            if not self.raw.same(a, b):
                self.out["ignored_differing"].append(path)
            return
        k = ("listed", id(a), id(b)) if isinstance(a, (dict, list, tuple)) else None
        if k in self.memo:
            return
        if k:
            self.memo[k] = True
        if isinstance(a, str) and isinstance(b, str):
            if a != b and self.same(a, b, path.rsplit("/", 1)[-1] == "commit"):
                self.out[into].append(path)
        elif isinstance(a, dict) and isinstance(b, dict):
            if list(a) != list(b) and a.keys() == b.keys():
                self.out["order_differing"].append(path or "<root>")
            for x in a:
                if x in b:
                    self.listed(a[x], b[x], f"{path}/{_key(x)}".lstrip("/"), into)
        elif isinstance(a, (list, tuple)) and type(a) is type(b):
            for i, (x, y) in enumerate(zip(a, b)):
                self.listed(x, y, f"{path}/{i}".lstrip("/"), into)

    def walk(self, a, b, path=""):
        if path in IGNORE:  # compared raw (no reading), so every differing ignored field is listed
            if not self.raw.same(a, b):
                self.out["ignored_differing"].append(path)
            return
        if self.same(a, b, path.rsplit("/", 1)[-1] == "commit"):
            if not self.raw.same(a, b) or isinstance(a, (dict, list, tuple)):
                self.listed(a, b, path, "normalized_differing")
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if list(a) != list(b) and a.keys() == b.keys():
                self.out["order_differing"].append(path or "<root>")
            for x in a:
                px = f"{path}/{_key(x)}".lstrip("/")
                if x not in b:  # Astra r1 R1: an ignored field present on one side only is ignored (listed), not a difference
                    self.out["ignored_differing" if px in IGNORE else "only_a"].append(px)
                else:
                    self.walk(a[x], b[x], px)
            for x in b:
                px = f"{path}/{_key(x)}".lstrip("/")
                if x not in a:
                    self.out["ignored_differing" if px in IGNORE else "only_b"].append(px)
        elif isinstance(a, (list, tuple)) and type(a) is type(b):
            for i, (x, y) in enumerate(zip(a, b)):
                self.walk(x, y, f"{path}/{i}".lstrip("/"))
            self.out["only_a"] += [f"{path}/{i}".lstrip("/") for i in range(len(b), len(a))]
            self.out["only_b"] += [f"{path}/{i}".lstrip("/") for i in range(len(a), len(b))]
        else:
            self.out["changed"].append(path or "<root>")


def diff(a, b):
    """-> {only_a, only_b, changed (counted in n_differences); ignored_differing, normalized_differing (strings equal once each side's stamp
    and HEAD are read as <stamp> / <git>), order_differing (dicts with the same keys in another order): listed, not counted}."""
    d = Diff(a, b)
    d.walk(a, b)
    return d.out | {"n_differences": len(d.out["only_a"]) + len(d.out["only_b"]) + len(d.out["changed"])}


PIN_PATHS = (("meta", "full_length", "s_old_simh_extra_roots", "*", "zero_unit_list"),)  # route C's zero-unit pin (metadata, not a result)


def pins_at(spec):
    """The {path, sha256} pins at PIN_PATHS ('*' = any key), with where each sits; only those two fields leave the container."""
    out = []

    def go(x, segs, where):
        if not segs:
            if isinstance(x, dict) and isinstance(x.get("path"), str) and isinstance(x.get("sha256"), str):
                out.append({"at": "/".join(where), "path": x["path"], "sha256": x["sha256"]})
            return
        if isinstance(x, dict):
            for k in (list(x) if segs[0] == "*" else [segs[0]] if segs[0] in x else []):
                go(x[k], segs[1:], where + [_key(k)])

    for ps in PIN_PATHS:
        go(spec, list(ps), [])
    return out


ROUTEC_TAG = "s_old_only"  # = full_override.S_OLD_ONLY (the test asserts it)


def routec_entries(spec):
    """Route C's tagged S_old H_sim entries, per job path: {at, file_sha256, zero_unit_ok} (pilot run keys and a file sha; no score)."""
    out = []
    for i, job in enumerate(spec.get("jobs") or []):
        for lb, es in sorted(((job[1] or {}).get("simh_old") or {}).items(), key=lambda kv: _key(kv[0])):  # by label, not insertion order
            for n, e in enumerate(es):
                if isinstance(e, dict) and e.get(ROUTEC_TAG):
                    out.append({"at": f"jobs/{i}/1/simh_old/{_key(lb)}/{n}", "file_sha256": e.get("sha256"), "zero_unit_ok": e.get("zero_unit_ok")})
    return out


def side_meta(spec):
    """Launch metadata of a spec, no value (2e / 507-prep option A, R2: nothing is unpickled on the laptop): its build commit, record path,
    pooled flag, the content-addressed shipped inputs {name: {path, sha256}}, and keys_only (a boolean: every job is a keys-only census row)."""
    m = spec.get("meta") or {}
    jobs = spec.get("jobs") or []
    return {"git": m.get("git"), "dest": spec.get("dest"), "pooled": bool(m.get("pooled_world")),
            "shipped_inputs": {k: {"path": v.get("path"), "sha256": v.get("sha256")} for k, v in (m.get("shipped_inputs") or {}).items()},
            "keys_only": bool(jobs) and all(isinstance(j[1], dict) and j[1].get("keys_only") is True for j in jobs)}


def plant(spec):
    """The positive control: one synthetic key appended to the S_old H_sim admitted list (e12 fill_S_old[0]) of the first S_old job, copied on
    write along that path only (other jobs sharing the list are untouched). -> (new spec, the one path the diff must list). STOPs if absent."""
    for i, job in enumerate(spec["jobs"]):
        c = job[1]
        fill = ((c.get("e12") or {}).get("fill_S_old") or []) if c.get("origin") == "S_old" else []
        if fill and isinstance(fill[0].get("admitted"), list):
            f0 = dict(fill[0]) | {"admitted": list(fill[0]["admitted"]) + [PLANT_KEY]}
            c2 = dict(c) | {"e12": dict(c["e12"]) | {"fill_S_old": [f0] + list(fill[1:])}}
            jobs = list(spec["jobs"])
            jobs[i] = type(job)((job[0], c2, *job[2:]))
            return dict(spec) | {"jobs": jobs}, f"jobs/{i}/1/e12/fill_S_old/0/admitted/{len(fill[0]['admitted'])}"
    raise SystemExit("plant: no S_old job with an e12 fill_S_old admitted list in this spec")


@app.function(image=image, cpu=(2, 4), memory=(4096, 32768), timeout=1800, retries=0, volumes={"/sol": vol.read_only()})
def diff_remote(a_src, b_src, plant_b: bool) -> dict:
    def load(s):
        return pickle.loads((Path("/sol") / s[1]).read_bytes() if s[0] == "vol" else s[1])

    a, b = load(a_src), load(b_src)
    planted = None
    if plant_b:
        b, planted = plant(b)
    out = diff(a, b) | {"a_pins": pins_at(a), "b_pins": pins_at(b), "a_routec_entries": routec_entries(a), "b_routec_entries": routec_entries(b),
                        "a_meta": side_meta(a), "b_meta": side_meta(b)}  # 507-prep: the gate checks the paid spec pins THIS pre-flight record
    if planted is not None:
        out["positive_control"] = {"planted_path": planted, "pass": out["n_differences"] == 1 and out["only_b"] == [planted]}
    return out


@app.local_entrypoint()
def main(a: str, b: str, plant_b: bool = False, out: str = ""):
    import hashlib
    import subprocess

    def src(s):
        if s.startswith("vol:"):
            return ("vol", s[4:]), {"volume_path": s[4:]}
        blob = Path(s).read_bytes()
        return ("bytes", blob), {"local": s, "sha256": hashlib.sha256(blob).hexdigest()}

    (sa, ia), (sb, ib) = src(a), src(b)
    res = diff_remote.remote(sa, sb, plant_b)
    rec = {"what": "drive-spec diff by key (paths only)", "a": ia, "b": ib, "plant_b": plant_b, "ignored_paths": sorted(IGNORE),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, cwd=Path(__file__).parent).stdout.decode().strip()} | res
    blob = json.dumps(rec, indent=1).encode()
    if out:
        dest = Path(__file__).parent / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as fh:
            fh.write(blob)
    print(json.dumps({k: (v[:50] if isinstance(v, list) else v) for k, v in rec.items()} | {"out": out or None,
                     "sha256": hashlib.sha256(blob).hexdigest() if out else None}, indent=1))
