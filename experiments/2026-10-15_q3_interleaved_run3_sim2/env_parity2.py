# Read-only, ON a box (d9 08:13, q4c fallback on xbox3): what fixes a 311 completion's per-environment compose/digest state
# besides the harness itself: images.json + the local image IDs, the settings path, the settings working tree OUTSIDE .git,
# and the settings git HEAD/refs. Same fields as env_parity.py plus the outside-git tree and git state used in 0d's 04:56 pass.
import hashlib, json, os, subprocess
IM = "/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json"
SET = "/home/aswin/phase2/settings"
out = {"images_json_sha256": hashlib.sha256(open(IM, "rb").read()).hexdigest()}
imgs = json.load(open(IM))["images"]
ids = {}
for ref, meta in sorted(imgs.items()):
    p = subprocess.run(["docker", "image", "inspect", ref, "--format", "{{.Id}}"], capture_output=True, text=True)
    ids[ref] = {"local_id": p.stdout.strip() or None, "matches_manifest": p.stdout.strip() == meta.get("id")}
out["images_total"] = len(ids)
out["images_local_matching_manifest"] = sum(v["matches_manifest"] for v in ids.values())
out["image_ids_sha256"] = hashlib.sha256(json.dumps({k: v["local_id"] for k, v in ids.items()}, sort_keys=True).encode()).hexdigest()
files = []
for root, dirs, fs in os.walk(SET):
    dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".git"))
    for f in sorted(fs):
        p = os.path.join(root, f)
        if os.path.islink(p): files.append((os.path.relpath(p, SET), "link:" + os.readlink(p))); continue
        files.append((os.path.relpath(p, SET), hashlib.sha256(open(p, "rb").read()).hexdigest()))
out["settings_path"] = SET
out["settings_realpath"] = os.path.realpath(SET)
out["settings_files_outside_git"] = len(files)
out["settings_tree_outside_git_sha256"] = hashlib.sha256(json.dumps(files).encode()).hexdigest()
g = lambda *a: subprocess.run(["git", "-C", SET, *a], capture_output=True, text=True).stdout  # noqa: E731
out["settings_git_head"] = g("rev-parse", "HEAD").strip()
out["settings_git_refs_sha256"] = hashlib.sha256(g("for-each-ref", "--format=%(refname) %(objectname)").encode()).hexdigest()
print(json.dumps(out))
