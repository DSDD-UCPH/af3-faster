"""Native AlphaFold 3 shims (colabfold 3.1.7+ or google-deepmind orig).

The kit keys OF3 layout on ``global_config.of3_weights``. Colabfold keys it on
``global_config.model`` and ``model_config.TRANSPOSED_COLUMN_PAIR_BIAS``.
Orig has no OF3 / ``--model`` flag.
"""
from __future__ import annotations

import inspect
import os
import re
from typing import Any, FrozenSet, Optional

from af3_faster import FAST_MODELS, OF3_MODELS, PREFIX

ENV_REPO = "AF3_FASTER_REPO"
ENV_RUN = "AF3_RUN_ALPHAFOLD"
MIN_NATIVE = (3, 1, 7)
FLAVOR_COLABFOLD = "colabfold"
FLAVOR_ORIG = "orig"
FLAVOR_MISSING = "missing"


def model_name(gc: Any) -> str:
    name = getattr(gc, "model", None)
    if name:
        return str(name)
    if getattr(gc, "of3_weights", False):
        return "openfold3"
    return "alphafold3"


def is_of3(gc: Any) -> bool:
    if getattr(gc, "of3_weights", False):
        return True
    return model_name(gc) in OF3_MODELS


def per_block_pair_ln(gc: Any) -> bool:
    """OF3-style per-block pair LayerNorm in the diffusion transformer (not AF3 / openbind)."""
    try:
        from alphafold3.model import model_config
        names = getattr(model_config, "PER_BLOCK_PAIR_LAYER_NORM", ())
        if names:
            return model_name(gc) in names or getattr(gc, "model", None) == "chai1"
    except Exception:
        pass
    return is_of3(gc) and model_name(gc) not in ("openbind0",)


def ending_bias_transposed(gc: Any, transpose: bool) -> bool:
    """Whether GridSelfAttention ending-node pair bias is swapped (native OF3 convention)."""
    if not transpose:
        return False
    try:
        from alphafold3.model import model_config
        names = getattr(model_config, "TRANSPOSED_COLUMN_PAIR_BIAS", ())
        if names:
            return model_name(gc) in names
    except Exception:
        pass
    return is_of3(gc)


def native_has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def call_native(fn, *args, **kwargs):
    """Call ``fn``, dropping kwargs the native signature does not accept."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return fn(*args, **{k: v for k, v in kwargs.items() if k in accepted})


def native_has_conditioning_only() -> bool:
    try:
        from alphafold3.model.network import diffusion_head as DH
        return native_has_param(DH.DiffusionHead.__call__, "conditioning_only")
    except Exception:
        return False


def native_has_atom_cond() -> bool:
    try:
        from alphafold3.model.network import atom_cross_attention as ACA
        return native_has_param(ACA.atom_cross_att_encoder, "conditioning_only")
    except Exception:
        return False


def native_has_cond_share() -> bool:
    """Native sample() already takes noise_level_prev as an unbatched scan input."""
    try:
        from alphafold3.model.network import diffusion_head as DH
        src = inspect.getsource(DH.sample)
        return "noise_level_prev" in src and "step_noise" in src
    except Exception:
        return False


def native_has_fix1(script: Optional[str] = None) -> bool:
    """Native 3.1.7 already one-shots tokamax autotune. Read the script; do not import it."""
    try:
        path = script or find_run_alphafold()
        with open(path, encoding="utf-8") as f:
            return "_autotune_attempted" in f.read()
    except Exception:
        return False


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for tok in re.split(r"[^0-9]+", str(text or "").strip()):
        if tok.isdigit():
            parts.append(int(tok))
    return tuple(parts or (0,))


def native_version() -> str:
    """Installed alphafold3 version (colabfold dist, orig dist, or alphafold3.version)."""
    try:
        import importlib.metadata as md
        names = (
            ("alphafold3", "alphafold3-colabfold")
            if not native_has_decoded_ccd()
            else ("alphafold3-colabfold", "alphafold3")
        )
        for name in names:
            try:
                return md.version(name)
            except Exception:
                continue
    except Exception:
        pass
    try:
        from alphafold3.version import __version__ as ver
        if ver:
            return str(ver)
    except Exception:
        pass
    try:
        import alphafold3
        ver = getattr(alphafold3, "__version__", None)
        if ver:
            return str(ver)
    except Exception:
        pass
    return "0"


def native_has_decoded_ccd() -> bool:
    try:
        from alphafold3.constants import decoded_ccd  # noqa: F401
        return True
    except Exception:
        return False


def native_flavor() -> str:
    """``colabfold`` (sokrypton OF3 fork), ``orig`` (google-deepmind), or ``missing``."""
    try:
        import alphafold3  # noqa: F401
    except Exception:
        return FLAVOR_MISSING
    if native_has_decoded_ccd():
        return FLAVOR_COLABFOLD
    return FLAVOR_ORIG


def of3_supported(flavor: Optional[str] = None) -> bool:
    """OF3 / openbind0 exist only on alphafold3-colabfold (``--model`` + decoded_ccd)."""
    return (flavor or native_flavor()) == FLAVOR_COLABFOLD


def native_stack_ok() -> tuple[bool, str]:
    """Importable alphafold3 with cpp + jax/haiku/tokamax. Colabfold also needs decoded_ccd at >= 3.1.7."""
    import importlib
    try:
        import alphafold3  # noqa: F401
    except Exception as e:
        return False, f"alphafold3_missing:{type(e).__name__}:{e}"
    flavor = native_flavor()
    ver = native_version()
    if flavor == FLAVOR_COLABFOLD and _version_tuple(ver) < MIN_NATIVE:
        return False, f"alphafold3_too_old:{ver}<3.1.7"
    try:
        import alphafold3.cpp  # noqa: F401
    except Exception as e:
        return False, f"alphafold3_cpp_missing:{type(e).__name__}"
    for name in ("jax", "haiku", "tokamax"):
        try:
            importlib.import_module(name)
        except Exception as e:
            return False, f"{name}_missing:{type(e).__name__}:{e}"
    return True, f"{flavor}:{getattr(alphafold3, '__file__', 'imported')}"


def native_bfloat16_all(gc: Any) -> bool:
    return str(getattr(gc, "bfloat16", "") or "") == "all"


def fast_allowed(model: str) -> bool:
    return model in FAST_MODELS


def parse_model_flag(argv: list[str], default: str = "alphafold3") -> str:
    for i, tok in enumerate(argv):
        if tok.startswith("--model="):
            return tok.split("=", 1)[1] or default
        if tok == "--model" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            return argv[i + 1]
    return default


def find_run_alphafold(environ=os.environ) -> str:
    """Native ``run_alphafold.py``. Editable install: walk up from alphafold3.__file__."""
    explicit = (environ.get(ENV_RUN) or "").strip()
    if explicit:
        p = os.path.abspath(explicit)
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"{PREFIX} {ENV_RUN}={explicit} is not a file")
    for key in (ENV_REPO, "AF3_FAST_REPO", "AF3_JAX_REPO"):
        root = (environ.get(key) or "").strip()
        if root:
            p = os.path.join(os.path.abspath(root), "run_alphafold.py")
            if os.path.isfile(p):
                return p
    try:
        import alphafold3
        here = os.path.dirname(os.path.abspath(alphafold3.__file__))
    except Exception as e:
        raise FileNotFoundError(
            f"{PREFIX} no alphafold3 import and no {ENV_REPO}: {e}"
        ) from e
    for _ in range(8):
        cand = os.path.join(here, "run_alphafold.py")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise FileNotFoundError(
        f"{PREFIX} cannot find run_alphafold.py next to alphafold3; set {ENV_RUN} to the native script (or {ENV_REPO} to the checkout)"
    )


def of3_weights_present(model: str, model_dir: Optional[str]) -> tuple[bool, str]:
    """Whether OF3/AF3 weights look loadable. Missing dir is a named miss, not a crash."""
    if not model_dir:
        return False, "no --model_dir"
    root = os.path.abspath(model_dir)
    if not os.path.isdir(root):
        return False, f"model_dir_missing:{root}"
    hits = sorted(
        os.path.join(root, fn)
        for fn in os.listdir(root)
        if fn.endswith((".bin.zst", ".npz", ".pkl", ".bin"))
    )
    if not hits:
        return False, f"no_weights_in:{root}"

    def _of3_name(path: str) -> bool:
        n = os.path.basename(path).lower()
        return "openfold" in n or n.startswith("of3")

    of3_hits = [p for p in hits if _of3_name(p)]
    af3_hits = [p for p in hits if p not in of3_hits]
    if model in OF3_MODELS:
        return True, (of3_hits or hits)[0]
    return True, (af3_hits or hits)[0]


def parse_model_dir(argv: list[str]) -> Optional[str]:
    for i, tok in enumerate(argv):
        if tok.startswith("--model_dir="):
            return tok.split("=", 1)[1] or None
        if tok == "--model_dir" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            return argv[i + 1]
    return None


def drop_model_flag(argv: list[str]) -> list[str]:
    """Orig has no ``--model``; strip it so absl does not see an unknown flag."""
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--model=") and not tok.startswith("--model_dir"):
            i += 1
            continue
        if tok == "--model" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            i += 2
            continue
        out.append(tok)
        i += 1
    return out
