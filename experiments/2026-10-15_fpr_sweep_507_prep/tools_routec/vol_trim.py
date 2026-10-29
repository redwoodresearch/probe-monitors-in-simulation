"""M1's trim (CAT/PS NOTE, the TRIM variable; A8's reviewed form), as a committed file so no launch line runs inline Python (2e, 10-18).

`modal volume get <vol> <path> -` (modal 1.6.1) writes the file's bytes to stdout followed by a 41-byte CLI trailer. This reads that stream
from stdin and writes exactly the first <size> bytes to stdout, only if the stream is <size> content bytes plus that exact trailer; otherwise
it exits non-zero with a STOP message and writes nothing. It never writes to disk and never unpickles.

  modal volume get <vol> <path> - | uv run --offline --no-project --python 3.12 python -I vol_trim.py <size> | shasum -a 256
"""

import sys

TRAILER = "✓ Finished downloading files to local!\n".encode()


def main():
    z = int(sys.argv[1])
    b = sys.stdin.buffer.read()
    if not (z >= 1 and len(b) == z + len(TRAILER) and b[z:] == TRAILER):
        sys.exit("STOP: the stream is not <size> content bytes + the 41-byte CLI trailer")
    sys.stdout.buffer.write(b[:z])


if __name__ == "__main__":
    main()
