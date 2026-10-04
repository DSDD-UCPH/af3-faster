"""Install fast levers into this process, then native run_alphafold can be imported."""
from __future__ import annotations

import atexit
import importlib
import os
import sys

from af3_faster import PREFIX
from af3_faster import compat
from af3_faster import report as R

# Same order as af3_jax_opt.fpf_launch.TREE_LEVERS. Hoists are on by default.
# Colabfold native 3.1.7+ already hoists pair/atom conditioning and shares
# per-step noise_level, so those kit copies are skipped there. Orig gets them.
# SAMPLER_BF16 is the kit hybrid (f32 residual, bf16 GEMM operands), not native
# bfloat16='all'.
TREE_LEVERS = (
    ("cond_share", "COND_SHARE"),
    ("atom_cond_hoist", "ATOM_COND_HOIST"),
    ("dattn", "DATTN"),
    ("ttr", "TTR"),
    ("triatt_xla", "TRIATT_XLA"),
    ("sampler_bf16", "SAMPLER_BF16"),
    ("atom_attn", "ATOM_ATTN"),
    ("trimul_cd", "TRIMUL_CD"),
    ("lnp", "LNP"),
    ("hoist_logits", "HOIST_LOGITS"),
)
INSTALL_AFTER_ADDON = ("hoist_logits",)

# Native 3.1.7 already hoists pair/atom conditioning and shares per-step noise_level;
# installing the kit copies would replace native DiffusionHead.__call__
# (conditioning_only / pair_cond / atom_cond). Pair-logit hoist is a native-safe
# subclass that calls super().__call__ then stashes loop-invariant projections.
# Orig has no conditioning_only; HOIST_LOGITS wraps Model._sample_diffusion instead.
NATIVE_SKIP = {
    "COND_SHARE": "native_sample_scan_share",
    "ATOM_COND_HOIST": "native_atom_cond",
}

FAST_ENV = {
    "AF3_FLASHPAIRFORMER": "both",
    "AF3_DIFFUSION_HOIST": "1",
    "AF3_JAX_DATTN": "1",
    "AF3_JAX_TTR": "fast",
    "AF3_JAX_TRIATT_XLA": "fast",
    "AF3_JAX_SAMPLER_BF16": "1",
    "AF3_JAX_ATOM_ATTN": "1",
    "AF3_JAX_TRIMUL_CD": "fast",
    "AF3_JAX_LNP": "fast",
    "AF3_JAX_HOIST_LOGITS": "1",
    "AF3_JAX_COND_SHARE": "1",
    "AF3_JAX_ATOM_COND_HOIST": "1",
}

_TREE: dict = {}
_SKIPS: dict = {}
_HOIST = {"installed": False, "counted": False, "calls": 0, "probe_error": None}


def apply_fast_env(environ=None) -> dict:
    env = environ if environ is not None else os.environ
    for k, v in FAST_ENV.items():
        env.setdefault(k, v)
    off = {t.strip() for t in (env.get("MODEL_OPT_LEVERS_OFF") or "").split(",") if t.strip()}
    for lever, key in (
        ("SAMPLER_BF16", "AF3_JAX_SAMPLER_BF16"),
        ("HOIST_LOGITS", "AF3_JAX_HOIST_LOGITS"),
        ("COND_SHARE", "AF3_JAX_COND_SHARE"),
        ("ATOM_COND_HOIST", "AF3_JAX_ATOM_COND_HOIST"),
        ("DIFFUSION_HOIST", "AF3_DIFFUSION_HOIST"),
    ):
        if lever in off:
            env[key] = "0"
    return dict(FAST_ENV)


def _alias_fpf() -> None:
    import af3_faster.fpf as fpf
    import af3_faster.fpf.patch as patch
    import af3_faster.fpf.diffusion_hoist as diffusion_hoist
    sys.modules.setdefault("af3_flashpairformer", fpf)
    sys.modules.setdefault("af3_flashpairformer.patch", patch)
    sys.modules.setdefault("af3_flashpairformer.diffusion_hoist", diffusion_hoist)


def _load_lever(name: str):
    mod = importlib.import_module(f"af3_faster.levers.{name}")
    _TREE[name] = mod
    return mod


def _skip_reason(lever: str) -> str | None:
    if lever == "COND_SHARE" and compat.native_has_cond_share():
        return NATIVE_SKIP[lever]
    if lever == "ATOM_COND_HOIST" and compat.native_has_atom_cond():
        return NATIVE_SKIP[lever]
    if lever == "DIFFUSION_HOIST" and compat.native_has_conditioning_only():
        return "native_pair_atom_cond"
    return None


def _install_tree(late: bool = False) -> None:
    for name, lever in TREE_LEVERS:
        try:
            mod = _TREE.get(name) if late else None
            mod = mod or _load_lever(name)
        except Exception as e:
            from af3_faster._core.oom import is_oom
            if is_oom(e):
                raise
            print(f"{PREFIX} SERVED warning=levers/{name}.py failed to load: {type(e).__name__}: {e}", flush=True)
            continue
        if bool(getattr(mod, "INSTALL_AFTER_ADDON", False)) != late:
            continue
        why = _skip_reason(lever)
        if why:
            _SKIPS[lever] = why
            print(f"{PREFIX} SKIP {lever} reason={why}", flush=True)
            continue
        if not mod.wanted():
            continue
        try:
            mod.install()
        except Exception as e:
            from af3_faster._core.oom import is_oom
            if is_oom(e):
                raise
            print(f"{PREFIX} SERVED warning={lever} install failed: {type(e).__name__}: {e}", flush=True)


def _install_cache_key() -> None:
    try:
        _load_lever("portable_cache_key").main_guard()
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        print(f"{PREFIX} CACHEKEY accelerator=error reason={type(e).__name__}", flush=True)


def _install_templates() -> None:
    try:
        _load_lever("templates").install()
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        print(f"{PREFIX} TEMPLATES guard=failed:{type(e).__name__}:{str(e).replace(' ', '_')[:200]}", flush=True)
    else:
        print(f"{PREFIX} TEMPLATES guard=installed site=alphafold3.model.features.Templates.compute_features", flush=True)


def install_fast() -> None:
    """Cache key, templates, tree levers, FlashPairformer. Registers the SERVED line at exit."""
    atexit.register(lambda: R.print_report(_TREE, _HOIST, _SKIPS))
    _install_cache_key()
    _install_templates()
    _install_tree(late=False)
    why = _skip_reason("DIFFUSION_HOIST")
    if why:
        _SKIPS["DIFFUSION_HOIST"] = why
        os.environ["AF3_DIFFUSION_HOIST"] = "0"
        print(f"{PREFIX} SKIP DIFFUSION_HOIST reason={why}", flush=True)
    _alias_fpf()
    try:
        import af3_faster.fpf  # noqa: F401  — install() from AF3_FLASHPAIRFORMER
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        print(f"{PREFIX} SERVED warning=af3_flashpairformer import failed: {type(e).__name__}: {e}", flush=True)
        return
    _install_tree(late=True)
    _count_hoist()
    R.refuse_held()


def _count_hoist() -> None:
    if os.environ.get("AF3_DIFFUSION_HOIST", "0").strip() in ("", "0", "off", "false"):
        _HOIST["installed"] = False
        return
    try:
        from af3_faster.fpf import diffusion_hoist as dh
        from alphafold3.model import model as af3_model
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        _HOIST["probe_error"] = type(e).__name__
        return
    stock = (getattr(dh, "_STOCK", None) or {}).get("sample")
    current = af3_model.Model._sample_diffusion
    if stock is None or current is stock:
        return
    _HOIST["installed"] = True

    def _sample_diffusion_counted(self, *args, **kwargs):
        _HOIST["calls"] += 1
        return current(self, *args, **kwargs)

    try:
        import haiku as hk
        _sample_diffusion_counted = hk.transparent(_sample_diffusion_counted)
    except Exception as e:
        from af3_faster._core.oom import is_oom
        if is_oom(e):
            raise
        return
    af3_model.Model._sample_diffusion = _sample_diffusion_counted
    _HOIST["counted"] = True
