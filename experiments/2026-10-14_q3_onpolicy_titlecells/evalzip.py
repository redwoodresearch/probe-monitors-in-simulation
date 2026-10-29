"""Let Python 3.13's zipfile read Inspect .eval files, whose members are zstd-compressed (zip method 93; zipfile reads it
natively only from Python 3.14, and the devbox's uv 0.5.0 can't fetch 3.14). Importing this module registers method 93
through `zstandard` (in the September sims venv). Reading only; nothing is written."""
import zipfile

import zstandard

ZSTD = 93
_check, _get = zipfile._check_compression, zipfile._get_decompressor


def _check_compression(ct):
    if ct != ZSTD:
        _check(ct)


def _get_decompressor(ct):
    return zstandard.ZstdDecompressor().decompressobj() if ct == ZSTD else _get(ct)


if not hasattr(zipfile, "ZIP_ZSTANDARD"):
    zipfile._check_compression, zipfile._get_decompressor = _check_compression, _get_decompressor
