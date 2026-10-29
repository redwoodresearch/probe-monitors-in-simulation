"""Every gitleaks finding must be a line `"<world>|<cell>": "<64 hex>"` (a sha256 under a join key); print the fingerprints if all are."""
import re
import sys

fps = re.findall(r"Fingerprint:\s+(\S+)", open(sys.argv[1]).read())
bad = []
for fp in fps:
    path, rule, line = fp.rsplit(":", 2)
    text = open(path).read().splitlines()[int(line) - 1]
    if rule != "generic-api-key" or not re.fullmatch(r'\s*"[a-z0-9_]+\|[A-Za-z0-9_]+": "[0-9a-f]{64}",?', text):
        bad.append(fp)
print(f"{len(fps)} findings, {len(set(fps))} unique, {len(bad)} not a join-key sha256", file=sys.stderr)
if bad:
    sys.exit("\n".join(bad))
print("\n".join(sorted(set(fps))))
