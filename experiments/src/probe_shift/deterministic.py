"""Bit-reproducible GPU probe fits.

Refitting the same Q2 probes twice on an A10G gave slightly different scores, and a few folds then chose another C
(experiments/2026-10-06_qwen35_replication README, "Q2 fit grouping": 21 of 48 refits identical). Two causes:
  - cuBLAS reductions that are not deterministic run to run. Importing this module sets CUBLAS_WORKSPACE_CONFIG to
    cuBLAS's deterministic workspace. It only takes effect before the process's first cuBLAS call, so probes.py and
    torch_probe.py import it: import either before putting a tensor on the GPU, or set the variable in the image.
  - C values whose validation AUROCs tie within float noise. `pick` breaks such ties toward the first C in grid order.
`scope()` turns on torch's deterministic algorithms for the duration of a fit (an op without a deterministic
implementation raises), restores the previous setting afterwards, and refuses to run if CUDA started before the
workspace setting existed. torch is imported lazily, so this module imports anywhere.
"""

import contextlib
import os
import sys

CUBLAS_WORKSPACE = ":4096:8"
C_TIE_TOL = 1e-6  # validation AUROCs closer than this are float noise (a few pair flips), not a reason to change C

_LATE = ("torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
         and "CUBLAS_WORKSPACE_CONFIG" not in os.environ)  # CUDA up before the setting existed: cuBLAS may ignore it
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)


@contextlib.contextmanager
def scope():
    """Deterministic torch for the enclosed fit; the process's earlier setting afterwards."""
    import torch

    if _LATE or os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("cuBLAS may be running without a deterministic workspace: import probe_shift.probes (or "
                           "torch_probe) before the first CUDA tensor, or set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the image")
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def pick(aucs: list[float], tol: float = C_TIE_TOL) -> int:
    """The index of the first value within `tol` of the largest: the C chosen from validation AUROCs in grid order."""
    top = max(aucs)
    return next(i for i, a in enumerate(aucs) if a >= top - tol)
