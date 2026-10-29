"""Hugging Face models the project reads with, each pinned to one revision so reruns load the same files.

QWEN3_32B: the revision behind every split and extraction so far. It is the only Qwen3-32B snapshot on the HF cache
Volume (`probe-shift_aswin-surya_hf-cache`) and in the local cache, and the hub's main since 2025-07-26.
"""

QWEN3_32B = {"name": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137"}

# QWEN35_27B: the reader for the Qwen3.5 redo (2026-10-06), the same model as Phase 2's agent. The only snapshot on the
# HF cache Volume (refs/main there), loaded text-only as Qwen3_5ForCausalLM (transformers >= 5.19).
QWEN35_27B = {"name": "Qwen/Qwen3.5-27B", "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654"}
