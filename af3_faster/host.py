"""Host-side row levers applied to native run_alphafold without copying it.

Native 3.1.7 already has FIX1 (one tokamax autotune attempt per process). Fast
mode injects kernel-tile ``--buckets`` so FPF N is a multiple of 64.
Featurisation prefetch / output writer are named skips unless native grows those
hooks — wrapping ``__main__``'s fold loop is not a signature-stable patch.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from af3_faster import PREFIX

MAX_BUCKET = 5120


def kernel_tile_buckets(tile: int = 64, hi: int = MAX_BUCKET) -> List[int]:
    return list(range(tile, hi + 1, tile))


def buckets_flag(tile: int = 64) -> str:
    return "--buckets=" + ",".join(str(b) for b in kernel_tile_buckets(tile))


def argv_has_buckets(argv: Sequence[str]) -> bool:
    return any(t == "--buckets" or t.startswith("--buckets=") for t in argv)


def inject_buckets(argv: List[str], tile: int = 64) -> Tuple[List[str], str]:
    """Insert kernel-tile buckets unless the caller already set --buckets."""
    if argv_has_buckets(argv):
        return list(argv), "caller"
    flag = buckets_flag(tile)
    print(f"{PREFIX} HOST buckets=kernel_tile tile={tile} {flag}", flush=True)
    return [flag, *argv], "kernel_tile"


def describe_fix1(script: str | None = None) -> str:
    from af3_faster import compat
    if compat.native_has_fix1(script):
        print(f"{PREFIX} SKIP FIX1 reason=native_autotune_once", flush=True)
        return "skipped:native_autotune_once"
    print(f"{PREFIX} SKIP FIX1 reason=no_autotune_hook", flush=True)
    return "skipped:no_autotune_hook"


def describe_row_levers() -> dict:
    print(f"{PREFIX} SKIP L1 reason=native_run_loop", flush=True)
    print(f"{PREFIX} SKIP WRITER reason=native_run_loop", flush=True)
    return {"L1": "skipped:native_run_loop", "WRITER": "skipped:native_run_loop"}
