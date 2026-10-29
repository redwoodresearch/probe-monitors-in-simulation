"""A harness freeze manifest, as the boxes check it: `find . -type f` minus results/, __pycache__, attempt/results/ and the
manifests, LC_ALL=C sorted paths, `sha256sum` lines. Runs on the laptop's harness copy (hashes only).
  python make_manifest.py <harness dir> <out tsv> <excluded manifest path>..."""
import hashlib, sys
from pathlib import Path
h, out, skip = Path(sys.argv[1]), Path(sys.argv[2]), {f"./{p}" for p in sys.argv[3:]}
paths = sorted((f"./{p.relative_to(h)}" for p in h.rglob("*") if p.is_file()), key=lambda s: s.encode())
paths = [p for p in paths if not (p.startswith("./results/") or "/__pycache__/" in p or p.startswith("./attempt/results/")) and p not in skip]
out.write_text("".join(f"{hashlib.sha256((h / p).read_bytes()).hexdigest()}  {p}\n" for p in paths))
print(f"{len(paths)} files -> {out}")
