"""sha256 and size of one file on this job's Volume, streamed (the laptop hashes; it never parses or prints contents).
    uv run --no-project --python 3.12 --with modal==1.6.1 python vol_sha.py <path on the Volume>"""
import hashlib
import sys

import modal

v = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14")
h, n = hashlib.sha256(), 0
for chunk in v.read_file(sys.argv[1]):
    h.update(chunk)
    n += len(chunk)
print(h.hexdigest(), n)
