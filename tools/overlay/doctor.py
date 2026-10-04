"""af3-faster doctor: native stack, GPU tiles, Pallas probe, lever preview."""
from __future__ import annotations

import os
import sys

from af3_faster import PREFIX, __version__, compat
from af3_faster.install import FAST_ENV, TREE_LEVERS, apply_fast_env


def _kv(fields: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())


def _emit(tag: str, fields: dict) -> None:
    print(f"{PREFIX} {tag} {_kv(fields)}", flush=True)


def _pkg_ver(name: str) -> str:
    try:
        mod = __import__(name)
        v = getattr(mod, "__version__", None)
        if v:
            return str(v)
    except Exception as e:
        return f"missing:{type(e).__name__}"
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"


def main(_argv: list[str] | None = None) -> None:
    _emit("DOCTOR", {"af3_faster": __version__, "pid": os.getpid()})
    ok, where = compat.native_stack_ok()
    try:
        import alphafold3
        path = getattr(alphafold3, "__file__", "unknown")
        ver = compat.native_version()
    except Exception as e:
        ver, path = "unimportable", type(e).__name__
    _emit(
        "DOCTOR",
        {
            "alphafold3": ver,
            "flavor": compat.native_flavor(),
            "of3": int(compat.of3_supported()),
            "ok": int(ok),
            "where": str(where).replace(" ", "_"),
            "path": path,
        },
    )
    try:
        script = compat.find_run_alphafold()
    except FileNotFoundError as e:
        script = f"missing:{e}"
    _emit("DOCTOR", {"run_alphafold": script})
    _emit(
        "DOCTOR",
        {
            "jax": _pkg_ver("jax"),
            "haiku": _pkg_ver("haiku"),
            "tokamax": _pkg_ver("tokamax"),
        },
    )
    cc = "none"
    tiles = "none"
    probe = {"ok": False, "kind": "unrun", "detail": ""}
    try:
        import jax
        devs = jax.devices()
        d0 = devs[0]
        cc = str(getattr(d0, "compute_capability", "") or "unknown")
        _emit(
            "DOCTOR",
            {
                "backend": jax.default_backend(),
                "device": str(getattr(d0, "device_kind", d0)).replace(" ", "_"),
                "cc": cc,
                "n_devices": len(devs),
            },
        )
        from af3_faster._core.kernels import fpf_pallas_serve as S
        tiles = S.tiles_label(cc)
        probe = S.probe(require_gpu=True)
    except Exception as e:
        _emit("DOCTOR", {"gpu": f"error:{type(e).__name__}:{str(e)[:160].replace(' ', '_')}"})
    _emit("DOCTOR", {"tiles": tiles, "cc": cc, "probe_ok": int(bool(probe.get("ok"))), "probe_kind": probe.get("kind") or "ok", "probe_detail": str(probe.get("detail") or "none").replace(" ", "_")[:120]})
    apply_fast_env()
    levers = []
    off = {t.strip() for t in (os.environ.get("MODEL_OPT_LEVERS_OFF") or "").split(",") if t.strip()}
    for name, lever in TREE_LEVERS:
        why = None
        if lever in off:
            why = "levers_off"
        elif lever == "COND_SHARE" and compat.native_has_cond_share():
            why = "native_sample_scan_share"
        elif lever == "ATOM_COND_HOIST" and compat.native_has_atom_cond():
            why = "native_atom_cond"
        elif lever == "DIFFUSION_HOIST" and compat.native_has_conditioning_only():
            why = "native_pair_atom_cond"
        envn = FAST_ENV.get(f"AF3_JAX_{lever}") or FAST_ENV.get(f"AF3_{lever}") or os.environ.get(f"AF3_JAX_{lever}", "")
        levers.append(f"{lever}:{'skip:'+why if why else envn or 'on'}")
    why_dh = None
    if "DIFFUSION_HOIST" in off:
        why_dh = "levers_off"
    elif compat.native_has_conditioning_only():
        why_dh = "native_pair_atom_cond"
    levers.append(
        f"DIFFUSION_HOIST:{'skip:' + why_dh if why_dh else FAST_ENV.get('AF3_DIFFUSION_HOIST') or 'on'}"
    )
    _emit("DOCTOR", {"mode": "fast", "fpf": os.environ.get("AF3_FLASHPAIRFORMER", "both"), "levers": ",".join(levers)})
    if not ok:
        flavor = compat.native_flavor()
        if flavor == compat.FLAVOR_ORIG:
            hint = "pip install -e the google-deepmind/alphafold3 checkout (and build its cpp extension)"
        else:
            hint = "pip install alphafold3-colabfold >= 3.1.7 into this environment"
        print(f"{PREFIX} DOCTOR native stack is not ready; {hint}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main(sys.argv[1:])
